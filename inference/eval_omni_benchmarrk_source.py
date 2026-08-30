# -*- coding: utf-8 -*-
"""
Image-to-image inference script (Batch processing via JSONL)
"""
import os
import json
import argparse
import time
import random  # [新增] 引入 random
from PIL import Image
import torch
from transformers import AutoConfig, AutoTokenizer
import sys

# Ensure this path is correct relative to where you run the script
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import SPECIAL_TOKENS
from model import LLaDAForMultiModalGeneration
from utils.generation_utils import setup_seed
from utils.image_utils import (
    preprocess_image, decode_vq_to_image, calculate_vq_params,
    generate_crop_size_list, var_center_crop, add_break_line,
    encode_img_with_breaks
)
from generators.image_to_image_generator_source import generate_i2i
from utils.prompt_utils import generate_image_to_image_prompt, create_prompt_templates
from diffusers import VQModel


def process_single_entry(entry, args, model, tokenizer, vqvae, templates, device, SPECIAL_TOKENS):
    """
    Process a single JSONL entry.
    """
    # Unpack Special Tokens
    MASK = SPECIAL_TOKENS["mask_token"]
    NEW_LINE = SPECIAL_TOKENS["newline_token"]
    BOA = SPECIAL_TOKENS["answer_start"]
    EOA = SPECIAL_TOKENS["answer_end"]
    BOI = SPECIAL_TOKENS["boi"]
    EOI = SPECIAL_TOKENS["eoi"]

    # Extract data from JSON entry
    input_image_path = entry.get("input_image_path")
    # Take the first prompt if it's a list, otherwise use as string
    prompt_text_list = entry.get("text", [""])
    prompt_text = prompt_text_list[0] if isinstance(prompt_text_list, list) and len(prompt_text_list) > 0 else str(
        prompt_text_list)
    omni_id = entry.get("omni_edit_id", "unknown_id")

    # Get target path to extract filename later
    target_image_path = entry.get("target_image_path")

    print(f"\n[Processing] ID: {omni_id} | Prompt: {prompt_text[:50]}...")

    if not os.path.exists(input_image_path):
        print(f"[Error] Image not found: {input_image_path}")
        return

    if not target_image_path:
        print(
            f"[Error] 'target_image_path' missing in JSONL entry for ID: {omni_id}. Cannot determine output filename.")
        return

    # Generate prompts
    edit_type = args.edit_type
    input_prompt, uncon_text, system_prompt = generate_image_to_image_prompt(
        prompt_text, edit_type, templates
    )

    # Handle reference image transfer logic (if applicable)
    if "image_ref_transfer" in edit_type:
        input_ref = args.ref_image_path
        if not input_ref:
            print("[Warning] image_ref_transfer selected but no ref_image_path provided in args. Skipping.")
            return

        img_ref = Image.open(input_image_path).convert("RGB")
        crop_size_list = generate_crop_size_list((args.size // 32) ** 2, 32)
        img_ref = var_center_crop(img_ref, crop_size_list=crop_size_list)
        img_token_input = encode_img_with_breaks(img_ref, vqvae)
        img = Image.open(input_ref).convert("RGB")
    else:
        img = Image.open(input_image_path).convert("RGB")

    prompt_ids = tokenizer(input_prompt)["input_ids"]
    uncon_text_ids = tokenizer(uncon_text)["input_ids"]

    # Preprocess image
    crop_size_list = generate_crop_size_list((args.size // 32) ** 2, 32)
    img = var_center_crop(img, crop_size_list=crop_size_list)

    image_width, image_height = img.size
    vae_scale = 2 ** (len(vqvae.config.block_out_channels) - 1)
    seq_len, newline_every, token_grid_height, token_grid_width = calculate_vq_params(image_height, image_width,
                                                                                      vae_scale)

    # Encode image (Source Tokens)
    input_img_token = encode_img_with_breaks(img, vqvae)
    input_img_token_without_BOT_EOI = input_img_token[1: -1]

    # Construct Inputs
    if "image_ref_transfer" in edit_type:
        con_input = prompt_ids[:-1] + img_token_input + input_img_token + prompt_ids[-1:]
        uncon_input_text = uncon_text_ids[:-1] + img_token_input + input_img_token + uncon_text_ids[-1:]
    else:
        con_input = prompt_ids[:-1] + input_img_token + prompt_ids[-1:]
        uncon_input_text = uncon_text_ids[:-1] + input_img_token + uncon_text_ids[-1:]

    uncon_input_image = prompt_ids

    # --- [修改逻辑] Build mask with Source-in-Mask strategy ---
    if args.mix_ratio > 0:
        # 按照 mix_ratio 混合 Source Token 和 Mask Token
        # 必须确保 input_img_token 里的结构（NEW_LINE）不被破坏
        mixed_tokens = []
        for token in input_img_token:
            if token == NEW_LINE:
                mixed_tokens.append(NEW_LINE)
            elif token == BOI:
                continue
            elif token == EOI:
                continue
            else:
                # 随机决定：保留 Source 还是使用 Mask
                if random.random() < args.mix_ratio:
                    mixed_tokens.append(token)
                else:
                    mixed_tokens.append(MASK)
        img_mask_token = mixed_tokens
    else:
        # 原始逻辑：全 Mask
        img_mask_token = add_break_line([MASK] * seq_len, token_grid_height, token_grid_width, new_number=NEW_LINE)
    # -------------------------------------------------------

    img_pred_token = [BOA] + [BOI] + img_mask_token + [EOI] + [EOA]
    code_start = len(con_input) + 2

    # To Tensor
    con_input = torch.tensor(con_input + img_pred_token, device=device).unsqueeze(0)
    uncon_input_text = torch.tensor(uncon_input_text, device=device).unsqueeze(0)
    uncon_input_image = torch.tensor(uncon_input_image, device=device).unsqueeze(0)

    # Generate
    start_time = time.time()
    vq_tokens = generate_i2i(
        model,
        con_input,
        seq_len=seq_len,
        newline_every=newline_every,
        timesteps=args.timesteps,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        cfg_img=args.cfg_img,
        uncon_text=uncon_input_text,
        uncon_image=uncon_input_image,
        code_start=code_start,
        # [修改] 传入 mix 参数和 source tokens
        mix_ratio=args.mix_ratio,
        source_tokens=input_img_token_without_BOT_EOI
    )

    # --- FILE PATH SETUP ---
    filename = os.path.basename(target_image_path)

    # 1. Prepare save path
    save_path = os.path.join(args.output_dir, filename)

    # Decode VQ tokens to PIL Image
    # Note: Passed None for save_path inside the function as we will save manually
    out_img = decode_vq_to_image(
        vq_tokens, None,
        vae_ckpt=args.vae_ckpt,
        image_height=image_height,
        image_width=image_width,
        vqvae=vqvae
    )

    # --- EXPLICITLY SAVE THE IMAGE ---
    out_img.save(save_path)

    # 2. Save Concat Image in omni_edit_images/concat
    w1, h1 = img.size
    w2, h2 = out_img.size
    canvas = Image.new("RGB", (w1 + w2, max(h1, h2)), "white")
    canvas.paste(img, (0, 0))
    canvas.paste(out_img, (w1, 0))

    # Construct concat filename (same name but with _concat before extension)
    base, ext = os.path.splitext(filename)
    concat_filename = f"{base}_concat{ext}"
    concat_path = os.path.join(args.concat_dir, concat_filename)

    canvas.save(concat_path)

    elapsed = time.time() - start_time
    print(f"[✓] Saved {save_path}")
    print(f"    Saved Comparison: {concat_path} (Time {elapsed:.2f}s)")


def main():
    parser = argparse.ArgumentParser(description="Image-to-image inference (JSONL Batch)")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Fine-tuned checkpoint path")
    parser.add_argument("--input_jsonl", type=str,
                        default="",
                        help="Path to input JSONL file")
    parser.add_argument("--edit_type", type=str, default="edit_add", help="Edit type (e.g. edit_add, edit_replace)")

    # Only relevant if using image_ref_transfer edit type
    parser.add_argument("--ref_image_path", type=str, default=None, help="Reference image path (if global for all)")

    # Generation params
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--size", type=int, default=512, help="Image size")
    parser.add_argument("--timesteps", type=int, default=64, help="Number of timesteps")
    parser.add_argument("--cfg_scale", type=float, default=5.5, help="CFG scale")
    parser.add_argument("--cfg_img", type=float, default=0.0, help="Image CFG scale")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--vae_ckpt", type=str, default="./Alpha-VLLM/Lumina-DiMOO", help="VAE checkpoint path")
    # [新增参数]
    parser.add_argument("--mix_ratio", type=float, default=0.5, help="Source in Mask mix ratio for inference (0.0 = pure mask)")

    args = parser.parse_args()

    # 1. Setup
    if args.seed != 0:
        setup_seed(args.seed)

    # --- MODIFIED SECTION: Output Directory Setup ---
    # 1. omni_edit_images (root for outputs)
    checkpoint_dir = os.path.abspath(args.checkpoint)
    timesteps = args.timesteps
    mix_ratio = args.mix_ratio
    cfg_scale = args.cfg_scale
    cfg_img = args.cfg_img
    temp = args.temperature
    output_dir = os.path.join(checkpoint_dir, f"omni_edit_images_t{timesteps}_m{mix_ratio}_c{cfg_scale}_ci{cfg_img}_temp{temp}")

    # 2. concat subfolder
    concat_dir = os.path.join(output_dir, "concat")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(concat_dir, exist_ok=True)

    print(f"[Setup] Output directory: {output_dir}")
    print(f"[Setup] Concat directory: {concat_dir}")
    print(f"[Setup] Mix Ratio: {args.mix_ratio}")

    # Update args so process_single_entry can access both paths
    args.output_dir = output_dir
    args.concat_dir = concat_dir
    # ------------------------------------------------

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 2. Load Models (Once)
    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = LLaDAForMultiModalGeneration.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, device_map="auto",
    )
    vqvae = VQModel.from_pretrained(args.vae_ckpt, subfolder="vqvae").to(device)

    # Prompt Templates
    templates = create_prompt_templates()

    # 3. Process JSONL
    print(f"Reading from {args.input_jsonl}...")
    with open(args.input_jsonl, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            process_single_entry(
                entry, args, model, tokenizer, vqvae, templates, device, SPECIAL_TOKENS
            )
        except json.JSONDecodeError:
            print(f"[Error] Failed to parse JSON on line {i + 1}")
        except Exception as e:
            print(f"[Error] Failed processing line {i + 1}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
