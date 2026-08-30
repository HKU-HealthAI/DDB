import torch
import math
import os
import numpy as np
from PIL import Image
from typing import Callable, Optional, List
from utils.generation_utils import cosine_schedule, gumbel_max_sample, mask_by_random_topk


@torch.no_grad()
def generate_i2i(
        model,
        prompt: torch.LongTensor,
        *,
        seq_len: int = 1024,
        newline_every: int = 16,
        timesteps: int = 18,
        mask_token_id: int = 126336,
        newline_id: int = 126084,
        temperature: float = 1.0,
        cfg_scale: float = 0.0,
        cfg_img: float = 0.0,
        uncon_text: torch.LongTensor,
        uncon_image: torch.LongTensor,
        code_start: Optional[int] = None,
        codebook_size: int = 8192,
        noise_schedule: Callable[[torch.Tensor], torch.Tensor] = cosine_schedule,
        text_vocab_size: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        mix_ratio: float = 0.0,
        source_tokens: Optional[List[int]] = None,
        # ================= [新增] 中间过程保存相关参数 =================
        save_intermediate: bool = False,
        save_dir: Optional[str] = None,
        sample_id: Optional[str] = None,
        input_image_pil: Optional[Image.Image] = None,
        decode_func: Optional[Callable] = None,
        image_height: int = 512,
        image_width: int = 512,
        save_step_freq: int = 4,  # <--- [新增] 默认每 1 步保存一次
        # ==========================================================
) -> torch.LongTensor:
    device = next(model.parameters()).device
    prompt = prompt.to(device)
    B, P = prompt.shape
    assert B == 1, "batch>1 not supported"

    source_tokens_tensor = None
    if source_tokens is not None:
        source_tokens_tensor = torch.tensor(source_tokens, device=device, dtype=torch.long)

    x = prompt

    # 1. 初始化 vq_mask (仅 Target 区域)
    vq_mask = torch.zeros_like(x, dtype=torch.bool)
    target_end_idx = x.shape[1] - 2
    target_seq = x[0, code_start:target_end_idx]
    is_newline = (target_seq == newline_id)
    vq_mask[0, code_start:target_end_idx] = ~is_newline

    unknown_cnt = vq_mask.sum(dim=1, keepdim=True)
    vq_len = unknown_cnt

    if text_vocab_size is None:
        vocab_total = model(torch.zeros(1, 1, dtype=torch.long, device=device), infer=True).logits.size(-1)
        text_vocab_size = vocab_total - codebook_size
    vocab_offset = text_vocab_size

    for step in range(timesteps):
        if unknown_cnt.item() == 0:
            break

        should_save_this_step = save_intermediate and (step % save_step_freq == 0 or step == timesteps - 1)

        if step < timesteps - 1:
            frac = noise_schedule(torch.tensor([(step + 1) / timesteps], device=device))
            num_to_mask = (vq_len.float() * frac).floor().clamp_min(1).long()
        else:
            num_to_mask = torch.zeros_like(unknown_cnt)

        # Forward pass
        if cfg_scale > 0 or cfg_img > 0:
            uncond_text = torch.cat((uncon_text.to(x.device), x[:, code_start - 2:]), dim=1)
            uncond_text_vq_mask = torch.cat(
                (torch.zeros((1, uncon_text.size(1)), dtype=torch.bool, device=x.device), vq_mask[:, code_start - 2:]),
                dim=1)
            uncond_img = torch.cat((uncon_image.to(x.device), x[:, code_start - 2:]), dim=1)
            uncond_img_vq_mask = torch.cat(
                (torch.zeros((1, uncon_image.size(1)), dtype=torch.bool, device=x.device), vq_mask[:, code_start - 2:]),
                dim=1)

            cond_logits = model(x, infer=True).logits[:, vq_mask[0], vocab_offset: vocab_offset + codebook_size]
            uncond_logits_text = model(uncond_text, infer=True).logits[:, uncond_text_vq_mask[0],
                                 vocab_offset: vocab_offset + codebook_size]
            uncond_logits_img = model(uncond_img, infer=True).logits[:, uncond_img_vq_mask[0],
                                vocab_offset: vocab_offset + codebook_size]
            logits = cond_logits + cfg_scale * (cond_logits - uncond_logits_text) + cfg_img * (
                    cond_logits - uncond_logits_img)
        else:
            logits = model(x, infer=True).logits[:, vq_mask[0], vocab_offset: vocab_offset + codebook_size]

        # Sampling
        sampled = gumbel_max_sample(logits, temperature, generator=generator)
        sampled_full = sampled + vocab_offset
        probs = torch.softmax(logits, dim=-1)
        conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

        # 填入预测值
        flat_idx = vq_mask.nonzero(as_tuple=False)[:, 1]
        x.view(-1)[flat_idx] = sampled_full.view(-1)

        # ================= [新增] 1. 保存当前 Step 的完整采样结果 (Sample) =================
        if should_save_this_step and decode_func is not None and save_dir is not None:
            tmp_target = x[0, code_start:target_end_idx]
            tmp_valid = tmp_target[tmp_target != newline_id]
            # 把当前的所有 token 都解码（这一步所有 unknown 位置已经被上面填入了 sampled 结果）
            sample_img_pil = decode_func(tmp_valid.unsqueeze(0))
            sample_dir = os.path.join(save_dir, "step_sample")
            os.makedirs(sample_dir, exist_ok=True)
            sample_img_pil.save(os.path.join(sample_dir, f"{sample_id}_step{step:02d}.png"))
        # ==============================================================================

        # 更新置信度图
        conf_map = torch.full_like(x, -math.inf, dtype=probs.dtype)
        conf_map.view(-1)[flat_idx] = conf.view(-1)

        # Select tokens to re-mask
        mask_sel = mask_by_random_topk(num_to_mask.squeeze(1), conf, temperature=temperature, generator=generator)

        # 获取需要被 "加噪/回退" 的全局索引
        tokens_to_mask_global_indices = flat_idx[mask_sel.view(-1)]

        # Mix Edit Strategy
        if mix_ratio > 0 and source_tokens_tensor is not None:
            rand_probs = torch.rand(tokens_to_mask_global_indices.shape, device=device)
            relative_indices = tokens_to_mask_global_indices - code_start

            valid_mask = (relative_indices >= 0) & (relative_indices < len(source_tokens_tensor))

            new_values = torch.full_like(tokens_to_mask_global_indices, mask_token_id)
            use_source = (rand_probs < mix_ratio) & valid_mask
            if use_source.any():
                source_vals = source_tokens_tensor[relative_indices[use_source]]
                new_values[use_source] = source_vals

            x.view(-1)[tokens_to_mask_global_indices] = new_values
        else:
            x.view(-1)[tokens_to_mask_global_indices] = mask_token_id

        # Update vq_mask
        vq_mask = torch.zeros_like(x, dtype=torch.bool)
        vq_mask.view(-1)[tokens_to_mask_global_indices] = True

        unknown_cnt = vq_mask.sum(dim=1, keepdim=True)

        # Restore Newline Integrity
        x[0, code_start:target_end_idx][is_newline] = newline_id
        vq_mask[0, code_start:target_end_idx][is_newline] = False

        # ================= [新增] 2. 保存进入下一步前的真实状态图和覆盖图 (State & Mask Vis) =================
        if should_save_this_step and decode_func is not None and save_dir is not None:
            target_x = x[0, code_start:target_end_idx]
            target_vq_mask = vq_mask[0, code_start:target_end_idx]

            valid_mask_idx = (target_x != newline_id)
            filtered_x = target_x[valid_mask_idx]
            filtered_vq_mask = target_vq_mask[valid_mask_idx]

            # [修复] 为了防止纯 Mask Token 导致解码越界，将其限制在合法的图像 token 范围内
            safe_tokens = filtered_x.clone()
            safe_tokens[safe_tokens == mask_token_id] = vocab_offset
            safe_tokens = torch.clamp(safe_tokens, vocab_offset, vocab_offset + codebook_size - 1)

            # 2.1 保存状态图
            state_img_pil = decode_func(safe_tokens.unsqueeze(0))
            state_dir = os.path.join(save_dir, "step_state")
            os.makedirs(state_dir, exist_ok=True)
            state_img_pil.save(os.path.join(state_dir, f"{sample_id}_step{step:02d}.png"))

            # 2.2 生成 Mask 索引可视化和叠加融合可视化
            if input_image_pil is not None:
                mask_index_dir = os.path.join(save_dir, "mask_vis_indexed")
                composite_dir = os.path.join(save_dir, "mask_vis_composite")
                os.makedirs(mask_index_dir, exist_ok=True)
                os.makedirs(composite_dir, exist_ok=True)

                input_img_np = np.array(input_image_pil.resize((image_width, image_height)))
                state_img_np = np.array(state_img_pil)

                # 生成状态图：0=Unmasked(已生成), 1=Mask, 2=Replace(保留源图)
                state_map_1d = torch.zeros_like(filtered_x, dtype=torch.uint8)
                is_pure_mask = filtered_vq_mask & (filtered_x == mask_token_id)
                is_replace = filtered_vq_mask & (filtered_x != mask_token_id)

                state_map_1d[is_pure_mask] = 1
                state_map_1d[is_replace] = 2

                # 映射到网格分辨率 (通常为下采样 32 倍)
                # 动态映射到网格分辨率 (例如: 1024 tokens -> 32x32)
                num_tokens = len(state_map_1d)
                aspect_ratio = image_height / image_width

                # 自动推导宽高网格
                H_tokens = int(math.sqrt(num_tokens * aspect_ratio))
                W_tokens = num_tokens // H_tokens

                state_map_2d = state_map_1d.view(H_tokens, W_tokens).cpu().numpy()

                # 上采样回原图分辨率
                scale_h = image_height // H_tokens
                scale_w = image_width // W_tokens
                state_map_up = np.repeat(np.repeat(state_map_2d, scale_h, axis=0), scale_w, axis=1)

                # --- 保存 Mask Index 索引图 ---
                vis_img = np.zeros((image_height, image_width, 3), dtype=np.uint8)
                vis_img[state_map_up == 1] = [0, 0, 0]  # 红色: Mask
                vis_img[state_map_up == 2] = [0, 0, 255]  # 蓝色: Replace
                vis_img[state_map_up == 0] = [128, 128, 128]  # 灰色: Unmasked
                Image.fromarray(vis_img).save(os.path.join(mask_index_dir, f"{sample_id}_step{step:02d}.png"))

                # --- 保存 Mask 综合融合图 ---
                # 用半透明红色混合纯 Mask 区
                alpha = 1
                blend = input_img_np.copy().astype(np.float32)
                blend[state_map_up == 1] = alpha * np.array([0, 0, 0]) + (1 - alpha) * blend[state_map_up == 1]
                blend = np.clip(blend, 0, 255).astype(np.uint8)

                final_vis = blend.copy()
                final_vis[state_map_up == 2] = input_img_np[state_map_up == 2]  # Replace区 填原图
                final_vis[state_map_up == 0] = state_img_np[state_map_up == 0]  # Unmask区 填模型解码图
                Image.fromarray(final_vis).save(os.path.join(composite_dir, f"{sample_id}_step{step:02d}.png"))

                # ================= [新增] 3. 保存纯粹的 Mask vs Unmask 状态图 =================
                unmask_status_dir = os.path.join(save_dir, "step_unmask_status")
                unmask_emerge_dir = os.path.join(save_dir, "step_unmask_emerge")
                os.makedirs(unmask_status_dir, exist_ok=True)
                os.makedirs(unmask_emerge_dir, exist_ok=True)

                # filtered_vq_mask 中 True 代表仍被 Mask，False 代表已经 Unmask
                mask_bool_2d = filtered_vq_mask.view(H_tokens, W_tokens).cpu().numpy()
                mask_bool_up = np.repeat(np.repeat(mask_bool_2d, scale_h, axis=0), scale_w, axis=1)

                # 1. 黑白二值图：白色表示已生成 (Unmasked)，黑色表示待生成 (Masked)
                binary_vis = np.zeros((image_height, image_width), dtype=np.uint8)
                binary_vis[~mask_bool_up] = 255  # False (Unmasked) 涂白
                binary_vis[mask_bool_up] = 0  # True (Masked) 涂黑
                Image.fromarray(binary_vis).save(os.path.join(unmask_status_dir, f"{sample_id}_step{step:02d}.png"))

                # 2. 图像渐显图：在当前解码的图像上，将还处于 Mask 的区域压暗（呈现出图像一点点“浮现”的直观效果）
                emerge_vis = state_img_np.copy()
                emerge_vis[mask_bool_up] = emerge_vis[mask_bool_up] // 3  # 将 Mask 区域亮度降低至 1/3
                Image.fromarray(emerge_vis).save(os.path.join(unmask_emerge_dir, f"{sample_id}_step{step:02d}.png"))
                # ==============================================================================
        # ==============================================================================

    vq_ids = x[0, code_start:-2]
    vq_ids = vq_ids[vq_ids != newline_id].view(1, seq_len)
    return vq_ids