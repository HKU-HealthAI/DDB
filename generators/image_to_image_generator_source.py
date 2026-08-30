# -*- coding: utf-8 -*-
"""
Image-to-image generator (supports DDP)
"""
import torch
import math
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

        # 更新置信度图
        conf_map = torch.full_like(x, -math.inf, dtype=probs.dtype)
        conf_map.view(-1)[flat_idx] = conf.view(-1)

        # Select tokens to re-mask
        # 从 Target 区域中选出置信度最低的 num_to_mask 个 Token
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

    vq_ids = x[0, code_start:-2]
    vq_ids = vq_ids[vq_ids != newline_id].view(1, seq_len)
    return vq_ids


# @torch.no_grad()
# def generate_i2i(
#         model,
#         prompt: torch.LongTensor,
#         *,
#         seq_len: int = 1024,
#         newline_every: int = 16,
#         timesteps: int = 18,
#         mask_token_id: int = 126336,
#         newline_id: int = 126084,
#         temperature: float = 1.0,
#         cfg_scale: float = 0.0,
#         cfg_img: float = 0.0,
#         uncon_text: torch.LongTensor,
#         uncon_image: torch.LongTensor,
#         code_start: Optional[int] = None,
#         codebook_size: int = 8192,
#         noise_schedule: Callable[[torch.Tensor], torch.Tensor] = cosine_schedule,
#         text_vocab_size: Optional[int] = None,
#         generator: Optional[torch.Generator] = None,
#         mix_ratio: float = 0.0,
#         source_tokens: Optional[List[int]] = None,
# ) -> torch.LongTensor:
#     device = next(model.parameters()).device
#     prompt = prompt.to(device)
#     B, P = prompt.shape
#     assert B == 1, "batch>1 not supported"
#
#     source_tokens_tensor = None
#     if source_tokens is not None:
#         source_tokens_tensor = torch.tensor(source_tokens, device=device, dtype=torch.long)
#
#     x = prompt
#
#     # 1. 初始化 vq_mask (仅 Target 区域)
#     vq_mask = torch.zeros_like(x, dtype=torch.bool)
#     target_end_idx = x.shape[1] - 2
#     target_seq = x[0, code_start:target_end_idx]
#     is_newline = (target_seq == newline_id)
#     vq_mask[0, code_start:target_end_idx] = ~is_newline
#
#     unknown_cnt = vq_mask.sum(dim=1, keepdim=True)
#     vq_len = unknown_cnt
#
#     if text_vocab_size is None:
#         vocab_total = model(torch.zeros(1, 1, dtype=torch.long, device=device), infer=True).logits.size(-1)
#         text_vocab_size = vocab_total - codebook_size
#     vocab_offset = text_vocab_size
#
#     for step in range(timesteps):
#         if unknown_cnt.item() == 0:
#             break
#
#         if step < timesteps - 1:
#             frac = noise_schedule(torch.tensor([(step + 1) / timesteps], device=device))
#             num_to_mask = (vq_len.float() * frac).floor().clamp_min(1).long()
#         else:
#             num_to_mask = torch.zeros_like(unknown_cnt)
#
#         # Forward pass
#         if cfg_scale > 0 or cfg_img > 0:
#             # [核心修复] 构造用于无条件分支的纯 Mask 输入
#             # 训练时 Uncond 分支看到的是纯 Mask，而此时 x 中混合了 Source Token。
#             # 为了匹配训练分布，我们需要手动构造一个将"待预测位置"全部填为 Mask 的输入。
#             x_pure_mask = x.clone()
#
#             # 找出所有当前需要预测的位置 (vq_mask 为 True)
#             flat_idx_current = vq_mask.nonzero(as_tuple=False)[:, 1]
#             # 强制将这些位置填为 mask_token_id (消除 Source Hint)
#             x_pure_mask.view(-1)[flat_idx_current] = mask_token_id
#
#             # 使用 x_pure_mask 构建无条件输入
#             uncond_text = torch.cat((uncon_text.to(x.device), x_pure_mask[:, code_start - 2:]), dim=1)
#             uncond_text_vq_mask = torch.cat(
#                 (torch.zeros((1, uncon_text.size(1)), dtype=torch.bool, device=x.device), vq_mask[:, code_start - 2:]),
#                 dim=1)
#
#             uncond_img = torch.cat((uncon_image.to(x.device), x_pure_mask[:, code_start - 2:]), dim=1)
#             uncond_img_vq_mask = torch.cat(
#                 (torch.zeros((1, uncon_image.size(1)), dtype=torch.bool, device=x.device), vq_mask[:, code_start - 2:]),
#                 dim=1)
#
#             # 条件分支使用 x (包含 Mix Hint)
#             cond_logits = model(x, infer=True).logits[:, vq_mask[0], vocab_offset: vocab_offset + codebook_size]
#
#             # 无条件分支使用 x_pure_mask (纯 Mask)
#             uncond_logits_text = model(uncond_text, infer=True).logits[:, uncond_text_vq_mask[0],
#                                  vocab_offset: vocab_offset + codebook_size]
#             uncond_logits_img = model(uncond_img, infer=True).logits[:, uncond_img_vq_mask[0],
#                                 vocab_offset: vocab_offset + codebook_size]
#
#             logits = cond_logits + cfg_scale * (cond_logits - uncond_logits_text) + cfg_img * (
#                     cond_logits - uncond_logits_img)
#         else:
#             logits = model(x, infer=True).logits[:, vq_mask[0], vocab_offset: vocab_offset + codebook_size]
#
#         # Sampling
#         sampled = gumbel_max_sample(logits, temperature, generator=generator)
#         sampled_full = sampled + vocab_offset
#         probs = torch.softmax(logits, dim=-1)
#         conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
#         # 填入预测值
#         flat_idx = vq_mask.nonzero(as_tuple=False)[:, 1]
#         x.view(-1)[flat_idx] = sampled_full.view(-1)
#
#         # 更新置信度图
#         conf_map = torch.full_like(x, -math.inf, dtype=probs.dtype)
#         conf_map.view(-1)[flat_idx] = conf.view(-1)
#
#         # Select tokens to re-mask
#         # 从 Target 区域中选出置信度最低的 num_to_mask 个 Token
#         mask_sel = mask_by_random_topk(num_to_mask.squeeze(1), conf, temperature=temperature, generator=generator)
#
#         # 获取需要被 "加噪/回退" 的全局索引
#         tokens_to_mask_global_indices = flat_idx[mask_sel.view(-1)]
#
#         # Mix Edit Strategy
#         if mix_ratio > 0 and source_tokens_tensor is not None:
#             rand_probs = torch.rand(tokens_to_mask_global_indices.shape, device=device)
#             relative_indices = tokens_to_mask_global_indices - code_start
#
#             valid_mask = (relative_indices >= 0) & (relative_indices < len(source_tokens_tensor))
#
#             new_values = torch.full_like(tokens_to_mask_global_indices, mask_token_id)
#             use_source = (rand_probs < mix_ratio) & valid_mask
#             if use_source.any():
#                 source_vals = source_tokens_tensor[relative_indices[use_source]]
#                 new_values[use_source] = source_vals
#
#             x.view(-1)[tokens_to_mask_global_indices] = new_values
#         else:
#             x.view(-1)[tokens_to_mask_global_indices] = mask_token_id
#
#         # Update vq_mask
#         vq_mask = torch.zeros_like(x, dtype=torch.bool)
#         vq_mask.view(-1)[tokens_to_mask_global_indices] = True
#
#         unknown_cnt = vq_mask.sum(dim=1, keepdim=True)
#
#         # Restore Newline Integrity
#         x[0, code_start:target_end_idx][is_newline] = newline_id
#         vq_mask[0, code_start:target_end_idx][is_newline] = False
#
#     vq_ids = x[0, code_start:-2]
#     vq_ids = vq_ids[vq_ids != newline_id].view(1, seq_len)
#     return vq_ids