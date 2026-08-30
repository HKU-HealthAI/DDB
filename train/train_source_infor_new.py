import pickle
from typing import List, Tuple
import random
import torch
import os
import numpy as np
import torch.nn.functional as F
import math
from transformers import AutoTokenizer, AutoConfig
import sys
import matplotlib.pyplot as plt
import time
import uuid
from PIL import Image
from torchvision import transforms


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model import LLaDAForMultiModalGeneration
from xllmx.data.item_processor import ItemProcessorBase
from xllmx.solvers.finetune import FinetuneSolverBase


def add_break_line(sequence, H, W, new_number=0):
    result = []
    for i in range(H):
        start = i * W
        end = start + W
        row = sequence[start:end]
        result.extend(row + [new_number])
    return result


def load_and_preprocess_image(image_path, target_h_tokens, target_w_tokens):
    """
    读取并预处理图像，使其与 Token 的空间排布严格对齐。
    Returns:
        Tensor: [1, 3, H_px, W_px], 归一化到 [0, 1]
    """
    try:
        # 计算目标像素尺寸 (VQVAE 倍率为 16)
        req_h_px = target_h_tokens * 16
        req_w_px = target_w_tokens * 16

        # 读取图像
        img = Image.open(image_path).convert('RGB')
        w_orig, h_orig = img.size

        # --- 复现 Pre-token 的 Center Crop 逻辑 ---
        target_aspect = req_w_px / req_h_px
        orig_aspect = w_orig / h_orig

        if orig_aspect > target_aspect:
            # 原图更宽，按高度缩放，裁剪宽度
            new_h = h_orig
            new_w = int(target_aspect * new_h)
            left = (w_orig - new_w) // 2
            top = 0
            right = left + new_w
            bottom = h_orig
        else:
            # 原图更高，按宽度缩放，裁剪高度
            new_w = w_orig
            new_h = int(new_w / target_aspect)
            left = 0
            top = (h_orig - new_h) // 2
            right = w_orig
            bottom = top + new_h

        img = img.crop((left, top, right, bottom))
        img = img.resize((req_w_px, req_h_px), Image.BICUBIC)

        # 转为 Tensor [C, H, W]
        tensor = transforms.ToTensor()(img)
        return tensor.unsqueeze(0)  # [1, C, H, W]

    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def get_image_info_map_from_pixels(
        height_tokens: int,
        width_tokens: int,
        image_path: str = None,
        edit_path: str = None,  # 仅用于 edit_diff
        metric="random",
        visualize=False,
        vis_dir="./vis_maps"
):
    """
    基于原始 RGB 图像计算信息量，并映射回 Token 空间。
    包含增强的可视化功能。
    """
    if metric == "random":
        return None

    # 如果需要计算图像内容，image_path 是必须的 (对于 edit 任务，image_path 对应 edit_path 即目标图)
    target_path = edit_path if edit_path else image_path

    if not target_path or not os.path.exists(target_path):
        return None

    # 1. 加载目标图像 (Pixel Space)
    tgt_tensor = load_and_preprocess_image(target_path, height_tokens, width_tokens)
    if tgt_tensor is None:
        return None

    device = torch.device("cpu")
    tgt_tensor = tgt_tensor.to(device)

    # 如果是 edit_diff 模式，我们需要保留 src_tensor 用于计算和可视化
    src_tensor = None
    if image_path and os.path.exists(image_path) and metric == "edit_diff":
        src_tensor = load_and_preprocess_image(image_path, height_tokens, width_tokens)
        if src_tensor is not None:
            src_tensor = src_tensor.to(device)

    info_map_pixel = None  # [1, 1, H_px, W_px]

    # 2. 计算 Pixel 级信息量
    with torch.no_grad():
        if metric == "variance":
            # 局部方差
            gray = 0.299 * tgt_tensor[:, 0, :, :] + 0.587 * tgt_tensor[:, 1, :, :] + 0.114 * tgt_tensor[:, 2, :, :]
            gray = gray.unsqueeze(1)
            mean = F.avg_pool2d(gray, 3, stride=1, padding=1)
            mean_sq = F.avg_pool2d(gray ** 2, 3, stride=1, padding=1)
            var = mean_sq - mean ** 2
            info_map_pixel = var

        elif metric == "gradient":
            # Sobel 梯度
            gray = 0.299 * tgt_tensor[:, 0, :, :] + 0.587 * tgt_tensor[:, 1, :, :] + 0.114 * tgt_tensor[:, 2, :, :]
            gray = gray.unsqueeze(1)
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
            grad_x = F.conv2d(gray, sobel_x, padding=1)
            grad_y = F.conv2d(gray, sobel_y, padding=1)
            info_map_pixel = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        elif metric == "frequency":
            # 高频分量
            blurred = F.avg_pool2d(tgt_tensor, kernel_size=5, stride=1, padding=2)
            high_freq = (tgt_tensor - blurred).abs().mean(dim=1, keepdim=True)
            info_map_pixel = high_freq

        elif metric == "edit_diff":
            if src_tensor is None:
                return None
            # Pixel Difference (L1 Loss)
            diff = (tgt_tensor - src_tensor).abs().mean(dim=1, keepdim=True)
            info_map_pixel = diff

    if info_map_pixel is None:
        return None

    # 3. 将 Pixel Map 映射到 Token Map (Downsample)
    info_map_token = F.avg_pool2d(info_map_pixel, kernel_size=16, stride=16)
    info_map = info_map_token.squeeze()  # [H, W]

    # 4. 归一化
    min_v = info_map.min()
    max_v = info_map.max()
    if max_v - min_v > 1e-6:
        info_map = (info_map - min_v) / (max_v - min_v)
    else:
        info_map = torch.ones_like(info_map)

    # ==========================================
    # 可视化保存 (增强版：包含原图和目标图)
    # ==========================================
    if visualize and vis_dir:
        try:
            os.makedirs(vis_dir, exist_ok=True)
            plt.switch_backend('Agg')

            # 设置画布：1行4列 (Source, Target, Pixel Diff, Token Map)
            fig, ax = plt.subplots(1, 4, figsize=(20, 5))

            # 1. Source Image
            if src_tensor is not None:
                src_img_np = src_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                ax[0].imshow(src_img_np)
                ax[0].set_title("Source Image")
            else:
                ax[0].text(0.5, 0.5, "No Source Image", ha='center')
                ax[0].set_title("Source Image (None)")
            ax[0].axis('off')

            # 2. Target Image
            tgt_img_np = tgt_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            ax[1].imshow(tgt_img_np)
            ax[1].set_title("Target Image")
            ax[1].axis('off')

            # 3. Pixel Level Metric
            pixel_data = info_map_pixel.squeeze().cpu().numpy()
            im1 = ax[2].imshow(pixel_data, cmap='jet')
            ax[2].set_title(f"Pixel Metric: {metric}")
            ax[2].axis('off')
            plt.colorbar(im1, ax=ax[2], fraction=0.046, pad=0.04)

            # 4. Token Level Importance
            token_data = info_map.cpu().numpy()
            im2 = ax[3].imshow(token_data, cmap='jet')
            ax[3].set_title(f"Token Importance ({height_tokens}x{width_tokens})")
            ax[3].axis('off')
            plt.colorbar(im2, ax=ax[3], fraction=0.046, pad=0.04)

            timestamp = int(time.time() * 1000)
            uid = str(uuid.uuid4())[:6]
            save_name = f"{metric}_vis_{timestamp}_{uid}.png"
            plt.savefig(os.path.join(vis_dir, save_name), bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"Vis error: {e}")
            import traceback
            traceback.print_exc()
    # ==========================================

    return info_map.view(-1)  # Flatten [H*W]


def mask_codes(
        codes,
        source_codes=None,
        sch="cosine",
        mask=False,
        editing=False,
        mix_ratio=0.5,
        info_metric="random",
        mask_random_ratio=0.2,
        image_path=None,
        edit_path=None,
        image_size=(32, 32),  # (H, W) in tokens
        visualize=False,
        vis_dir="./vis_maps"
):
    r = random.uniform(0, 1)
    if len(codes) <= 5 and mask == False:
        mask_ratio = 1.0
    elif sch == "cosine":
        mask_ratio = math.cos(r * math.pi / 2)
    elif sch == "linear":
        if r < 0.05:
            r = r + 0.05
        mask_ratio = r
    else:
        mask_ratio = 1.0

    num_to_mask = int(len(codes) * mask_ratio)
    if num_to_mask < 1:
        num_to_mask = 1

    # ==========================================
    # 计算权重 (基于 Pixel + 混合随机性)
    # ==========================================
    weights = None
    if info_metric != "random" and not mask:
        weights = get_image_info_map_from_pixels(
            height_tokens=image_size[0],
            width_tokens=image_size[1],
            image_path=image_path,
            edit_path=edit_path,
            metric=info_metric,
            visualize=visualize,
            vis_dir=vis_dir
        )

        # 长度校验
        if weights is not None and len(weights) != len(codes):
            print("len(weights) != len(codes)")
            weights = None

    if weights is not None:
        # [修改] 实现混合随机性
        # weights 当前是重要性分数 (0~1)，我们将其转化为概率分布

        # 1. 先加上极小值防止全0，并归一化为概率 (Sum = 1)
        w_sum = weights.sum() + 1e-8
        prob_info = weights / w_sum

        # 2. 如果启用 mask_random_ratio，则混合均匀分布
        if mask_random_ratio > 0.0:
            # 均匀分布概率 (Pure Random)
            prob_uniform = torch.ones_like(weights) / len(weights)

            # 混合: (1 - ratio) * Info + ratio * Random
            # 这样保证了有 probability = ratio 的概率是完全随机选择的
            final_prob = (1.0 - mask_random_ratio) * prob_info + mask_random_ratio * prob_uniform
        else:
            final_prob = prob_info

        # 3. 采样
        # multinomial 不需要输入和为1，但前面归一化为了方便加权控制
        indices_to_mask = torch.multinomial(final_prob, num_to_mask, replacement=False).tolist()
        # print("indices_to_mask from weight")
    else:
        # 完全随机回退
        print("indices_to_mask from random")
        indices_to_mask = random.sample(range(len(codes)), num_to_mask)
    # ==========================================

    masked_codes = codes[:]
    labels = [-100] * len(codes)

    if mix_ratio == -1:
        mix_ratio = torch.rand(1).item()

    for index in indices_to_mask:
        labels[index] = codes[index]
        if editing and source_codes is not None:
            if index < len(source_codes):
                if random.random() < mix_ratio:
                    masked_codes[index] = source_codes[index]
                else:
                    masked_codes[index] = 126336
            else:
                masked_codes[index] = 126336
        else:
            masked_codes[index] = 126336

    return masked_codes, labels


def load_image_tokens(image_path):
    with open(image_path, "rb") as f:
        data_pkl = pickle.load(f)
    assert data_pkl["height"] % 16 == 0 and data_pkl["width"] % 16 == 0
    height, width = data_pkl["width"] // 16, data_pkl["height"] // 16
    tokens = add_break_line(data_pkl["input_ids"], height, width, new_number=126084)
    return tokens


class ItemProcessor(ItemProcessorBase):
    def __init__(self, tokenizer, max_len, mix_ratio=0.5, info_metric="random", mask_random_ratio=0.2, vq_ckpt=None,
                 visualize=False, vis_dir="./vis_maps", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mix_ratio = mix_ratio
        self.info_metric = info_metric
        self.mask_random_ratio = mask_random_ratio
        self.visualize = visualize
        self.vis_dir = vis_dir

        if info_metric != "random":
            print(f"[ItemProcessor] Using Pixel-based Metric: {info_metric} with {mask_random_ratio * 100}% random mix")

    def process_item(self, data_item: dict, training_mode=False) -> Tuple[List, List]:
        # --- Understanding Data ---
        if data_item["user_image"] != "" and data_item["answer_image"] == "":
            instruction = "<system>" + data_item["system_prompt"] + "</system>" + "<user>" + data_item[
                "user_prompt"] + "</user>"
            instruction_token = \
                self.tokenizer(instruction, truncation=True, max_length=1024, padding=False,
                               return_tensors="pt").input_ids[0].tolist()

            image_tokens = load_image_tokens(data_item["user_image"])
            instruction_token = instruction_token[:-1] + [126349] + image_tokens + [126350] + instruction_token[-1:]
            instruction_label = [-100] * len(instruction_token)

            answer = data_item["answer_text"] + "</answer>"
            answer_token = \
                self.tokenizer(answer, truncation=True, max_length=1024, padding=False, return_tensors="pt").input_ids[
                    0].tolist()
            answer_token, answer_label = mask_codes(answer_token)
            padding_len = 1024 - len(answer_token)
            padding_token = [126339] * padding_len
            padding_token, padding_label = mask_codes(padding_token, mask=True)
            all_token = instruction_token + [126354] + answer_token + padding_token
            all_label = instruction_label + [-100] + answer_label + padding_label

        # --- Text-to-Image Data ---
        elif data_item["user_image"] == "" and data_item["answer_image"] != "":
            # print("t2i task")
            if np.random.rand() < 0.1:
                instruction = "<system>" + data_item[
                    "system_prompt"] + "</system>" + "<user>" + "<uncondition>" + "</user>"
            else:
                instruction = "<system>" + data_item["system_prompt"] + "</system>" + "<user>" + data_item[
                    "user_prompt"] + "</user>"
            instruction_token = \
                self.tokenizer(instruction, truncation=True, max_length=1024, padding=False,
                               return_tensors="pt").input_ids[0].tolist()
            instruction_label = [-100] * len(instruction_token)

            with open(data_item["answer_image"], "rb") as f:
                data_pkl = pickle.load(f)
            image_tokens = data_pkl["input_ids"]
            image_height, image_width = data_pkl["width"] // 16, data_pkl["height"] // 16

            img_path = data_item.get("image_path", None)

            image_masked_codes, image_labels = mask_codes(
                image_tokens,
                info_metric=self.info_metric,
                mask_random_ratio=self.mask_random_ratio,
                image_path=img_path,
                edit_path=None,
                image_size=(image_height, image_width),
                visualize=self.visualize,
                vis_dir=self.vis_dir
            )

            image_tokens = add_break_line(image_masked_codes, image_height, image_width, new_number=126084)
            image_labels = add_break_line(image_labels, image_height, image_width, new_number=-100)
            all_token = instruction_token + [126354] + [126349] + image_tokens + [126350] + [126355]
            all_label = instruction_label + [-100] + [-100] + image_labels + [-100] + [-100]

        # --- Image-to-Image Data ---
        elif data_item["user_image"] != "" and data_item["answer_image"] != "":
            with open(data_item["user_image"], "rb") as f:
                source_pkl = pickle.load(f)
            source_tokens_raw = source_pkl["input_ids"]

            rand_val = np.random.rand()
            use_source_image_in_input = True
            if rand_val < 0.1:
                instruction = "<system>" + data_item[
                    "system_prompt"] + "</system>" + "<user>" + "<uncondition>" + "</user>"
                use_source_image_in_input = True
            else:
                instruction = "<system>" + data_item["system_prompt"] + "</system>" + "<user>" + data_item[
                    "user_prompt"] + "</user>"
                use_source_image_in_input = True

            instruction_token = \
                self.tokenizer(instruction, truncation=True, max_length=1024, padding=False,
                               return_tensors="pt").input_ids[0].tolist()

            if use_source_image_in_input:
                image_tokens = source_tokens_raw
                image_height, image_width = source_pkl["width"] // 16, source_pkl["height"] // 16
                image_tokens_with_break = add_break_line(image_tokens, image_height, image_width, new_number=126084)
                instruction_token = instruction_token[:-1] + [126349] + image_tokens_with_break + [
                    126350] + instruction_token[-1:]

            instruction_label = [-100] * len(instruction_token)

            with open(data_item["answer_image"], "rb") as f:
                target_pkl = pickle.load(f)
            target_tokens_raw = target_pkl["input_ids"]
            image_height, image_width = target_pkl["width"] // 16, target_pkl["height"] // 16

            ori_img_path = data_item.get("image_path", None)
            edit_img_path = data_item.get("edit_path", None)

            image_masked_codes, image_labels = mask_codes(
                target_tokens_raw,
                source_codes=source_tokens_raw if use_source_image_in_input else None,
                editing=True,
                mix_ratio=self.mix_ratio,
                info_metric=self.info_metric,
                mask_random_ratio=self.mask_random_ratio,
                image_path=ori_img_path,
                edit_path=edit_img_path,
                image_size=(image_height, image_width),
                visualize=self.visualize,
                vis_dir=self.vis_dir
            )

            image_tokens = add_break_line(image_masked_codes, image_height, image_width, new_number=126084)
            image_labels = add_break_line(image_labels, image_height, image_width, new_number=-100)

            all_token = instruction_token + [126354] + [126349] + image_tokens + [126350] + [126355]
            all_label = instruction_label + [-100] + [-100] + image_labels + [-100] + [-100]

        return all_token, all_label

    def predict_item_token_length(self, data_item: dict) -> int:
        if "token" in data_item:
            return len(data_item["token"])
        elif "len" in data_item:
            return data_item["len"]
        else:
            raise ValueError()


class Solver(FinetuneSolverBase):
    @classmethod
    def get_args_parser(cls):
        parser = super().get_args_parser()
        parser.add_argument("--max_seq_len", default=1024, type=int, help="max token length")
        parser.add_argument("--dropout", type=float, default=0.05)
        parser.add_argument("--mix_ratio", type=float, default=0.5,
                            help="Ratio of source tokens mixed in mask for editing tasks")

        parser.add_argument("--info_metric", type=str, default="edit_diff",
                            choices=["random", "variance", "frequency", "gradient", "edit_diff"],
                            help="Masking strategy based on image information.")

        parser.add_argument("--mask_random_ratio", type=float, default=0.3, help="Ratio of random noise (0.0 to 1.0)")

        parser.add_argument("--vae_ckpt", type=str, default=None, help="Deprecated in this version")

        parser.add_argument("--visualize", default=False, help="Visualize and save info maps.")
        parser.add_argument("--vis_dir", type=str, default="./vis_maps_Medical_cdd", help="Directory to save visualizations.")

        return parser

    def _model_func(self, init_from: str) -> (LLaDAForMultiModalGeneration, None):
        tokenizer = AutoTokenizer.from_pretrained(init_from, trust_remote_code=True)
        model = LLaDAForMultiModalGeneration.from_pretrained(init_from, torch_dtype=torch.bfloat16, device_map="cpu")
        model.model.set_activation_checkpointing("whole_layer")
        return model, tokenizer

    def _item_processor_func(self, tokenizer=None, max_len=None) -> ItemProcessorBase:
        return ItemProcessor(
            tokenizer,
            max_len,
            mix_ratio=self.args.mix_ratio,
            info_metric=self.args.info_metric,
            mask_random_ratio=self.args.mask_random_ratio,
            vq_ckpt=None,  # No VQ Needed
            visualize=self.args.visualize,
            vis_dir=self.args.vis_dir
        )

    def _make_and_save_starting_point(self, save_path: str) -> None:
        tokenizer = AutoTokenizer.from_pretrained(self.args.init_from, trust_remote_code=True)
        base_config = AutoConfig.from_pretrained(self.args.init_from)
        model = LLaDAForMultiModalGeneration(base_config)
        model.resize_token_embeddings(len(tokenizer))
        model.model.transformer.ff_out = torch.nn.Linear(4096, len(tokenizer), bias=False)


if __name__ == "__main__":
    args = Solver.get_args_parser().parse_args()
    solver = Solver(args)
    solver.run()
