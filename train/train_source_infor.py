import pickle
from typing import List, Tuple
import random
from accelerate import init_empty_weights
import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoTokenizer, AutoConfig
import sys
import matplotlib.pyplot as plt
import time
import uuid

from diffusers import VQModel

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


def get_image_info_map(
        tokens: List[int],
        height: int,
        width: int,
        metric="random",
        vq_model=None,
        source_tokens: List[int] = None,
        visualize=False,
        vis_dir="./vis_maps"
):
    """
    计算图像 Token 的信息量热力图 (Latent 空间计算以提升效率)
    Returns:
        weights: tensor shape [H*W], normalized probability
    """
    # 1. 基础检查：如果是随机或无模型，直接返回 None
    if metric == "random" or vq_model is None:
        return None

    device = vq_model.device

    # 2. 将 Token 转换为 Latent Indices
    # 假设 Image Token Offset 是 126356 (根据原代码推断)
    tokens_tensor = torch.tensor(tokens, device=device)
    code_indices = (tokens_tensor - 126356).long()

    # 检查长度
    if len(code_indices) != height * width:
        return None

    # Reshape: [1, H, W]
    code_map = code_indices.view(1, height, width)

    # 3. 获取 Latent Embeddings (Feature Map)
    try:
        # [1, H, W, D] -> [1, D, H, W]
        embedding = vq_model.quantize.get_codebook_entry(code_map, shape=(1, height, width, -1))
        feat_map = embedding.permute(0, 3, 1, 2)
    except AttributeError:
        # 如果 vq_model 结构不同，尝试直接使用 embedding layer
        try:
            embedding = vq_model.quantize.embedding(code_map)
            feat_map = embedding.permute(0, 3, 1, 2)
        except:
            # 兜底：如果获取不到 embedding，直接归一化 indices 作为一个通道使用 (粗略近似)
            feat_map = code_map.float().unsqueeze(1) / 8192.0

    info_map = torch.ones((height, width), device=device)

    # 4. 根据 Metric 计算 Info Map
    with torch.no_grad():
        if metric == "variance":
            # 计算局部方差 (3x3 滑窗)
            # Var(X) = E[X^2] - (E[X])^2
            mean = F.avg_pool2d(feat_map, 3, stride=1, padding=1)
            mean_sq = F.avg_pool2d(feat_map ** 2, 3, stride=1, padding=1)
            var = mean_sq - mean ** 2
            # 对所有通道求和作为该位置的信息量
            info_map = var.sum(dim=1).squeeze(0)

        elif metric == "gradient":
            # Sobel 算子计算梯度
            C = feat_map.shape[1]
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=feat_map.dtype).view(1, 1,
                                                                                                                   3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=feat_map.dtype).view(1, 1,
                                                                                                                   3, 3)

            # 对每个通道独立计算梯度
            sobel_x = sobel_x.repeat(C, 1, 1, 1)
            sobel_y = sobel_y.repeat(C, 1, 1, 1)

            grad_x = F.conv2d(feat_map, sobel_x, padding=1, groups=C)
            grad_y = F.conv2d(feat_map, sobel_y, padding=1, groups=C)

            gradient = torch.sqrt(grad_x ** 2 + grad_y ** 2)
            info_map = gradient.mean(dim=1).squeeze(0)

        elif metric == "frequency":
            # 简单的高频能量检测：原图 - 低通滤波图
            blurred = F.avg_pool2d(feat_map, kernel_size=3, stride=1, padding=1)
            high_freq = (feat_map - blurred).abs()
            info_map = high_freq.sum(dim=1).squeeze(0)

        elif metric == "edit_diff":
            if source_tokens is None:
                return None

            # 处理 Source
            src_indices = (torch.tensor(source_tokens, device=device) - 126356).long()
            if len(src_indices) != len(code_indices): return None

            src_map = src_indices.view(1, height, width)

            # 计算 Latent 差异
            try:
                src_emb = vq_model.quantize.get_codebook_entry(src_map, shape=(1, height, width, -1)).permute(0, 3, 1,
                                                                                                              2)
                tgt_emb = feat_map
                # L1 Distance
                diff = (tgt_emb - src_emb).abs().sum(dim=1).squeeze(0)
            except:
                diff = (code_map.float() - src_map.float()).abs().squeeze(0).squeeze(0)

            info_map = diff

    # 5. 归一化
    # 我们希望 Mask 概率与 info_map 成正比
    min_v = info_map.min()
    max_v = info_map.max()
    if max_v - min_v > 1e-6:
        info_map = (info_map - min_v) / (max_v - min_v)
    else:
        # 如果全图一样（比如纯色），退化为均匀分布
        info_map = torch.ones_like(info_map)

    # ==========================================
    # [新增] 可视化保存逻辑
    # ==========================================
    if visualize and vis_dir:
        try:
            # 确保目录存在
            os.makedirs(vis_dir, exist_ok=True)

            # 使用 Agg 后端，防止在无 GUI 服务器上报错
            plt.switch_backend('Agg')

            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111)

            # 绘制热力图 (Latent Space)
            # print(info_map.shape)
            map_data = info_map.detach().cpu().numpy()
            im = ax.imshow(map_data, cmap='jet')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax.set_title(f"Metric: {metric}\nSize: {height}x{width}")
            ax.axis('off')

            # 生成唯一文件名: metric_timestamp_uuid.png
            timestamp = int(time.time() * 1000)
            uid = str(uuid.uuid4())[:6]
            save_name = f"{metric}_{timestamp}_{uid}.png"
            save_path = os.path.join(vis_dir, save_name)

            plt.savefig(save_path, bbox_inches='tight', dpi=100)
            plt.close(fig)

            # Optional: 打印保存信息
            # print(f"[InfoMap] Saved visualization to {save_path}")

        except Exception as e:
            print(f"[Warning] Visualization failed: {e}")
    # ==========================================

    # 展平为 1D 权重 [H*W]
    return info_map.view(-1).cpu()


def mask_codes(
        codes,
        source_codes=None,
        sch="cosine",
        mask=False,
        editing=False,
        mix_ratio=0.5,
        # 新增参数
        info_metric="random",
        mask_random_ratio=0.2,
        vq_model=None,
        image_size=(32, 32),  # (H, W) in latent space
        # [新增] 可视化参数透传
        visualize=False,
        vis_dir="./vis_maps"
):
    r = random.uniform(0, 1)
    # --- 调度器逻辑保持不变 ---
    if len(codes) <= 5 and mask == False:
        mask_ratio = 1.0
    elif sch == "cosine":
        mask_ratio = math.cos(r * math.pi / 2)
    elif sch == "linear":
        if r < 0.05:
            r = r + 0.05
        mask_ratio = r
    else:
        # print("Not Implement")
        mask_ratio = 1.0  # Fallback
    num_to_mask = int(len(codes) * mask_ratio)
    if num_to_mask < 1:
        num_to_mask = 1
    # ==========================================
    # [修改] 选择 Mask Indices 的策略 (Weighted Masking)
    # ==========================================
    weights = None
    # 仅在非纯 mask 模式且有 VQModel 时计算
    if info_metric != "random" and vq_model is not None and not mask:
        try:
            # print("get_image_info_map......")
            weights = get_image_info_map(
                codes,
                image_size[0],
                image_size[1],
                metric=info_metric,
                vq_model=vq_model,
                source_tokens=source_codes,
                # 传递可视化参数
                visualize=visualize,
                vis_dir=vis_dir
            )
            # print(weights.shape)
            # 长度校验
            if weights is not None and len(weights) != len(codes):
                print("len(weights) != len(codes)")
                weights = None
        except Exception as e:
            # print(f"[Warning] Info map calculation failed: {e}")
            weights = None

    if weights is not None:
        # 使用多项式分布进行加权采样 (Hard Negative Mining)
        # 加上 epsilon 防止概率为 0
        weights = weights + 1e-5
        indices_to_mask = torch.multinomial(weights, num_to_mask, replacement=False).tolist()
    else:
        # 原始随机逻辑
        indices_to_mask = random.sample(range(len(codes)), num_to_mask)
    # ==========================================
    masked_codes = codes[:]
    labels = [-100] * len(codes)

    if mix_ratio == -1:
        mix_ratio = torch.rand(1).item()

    for index in indices_to_mask:
        labels[index] = codes[index]  # 标签永远是 Target (Ground Truth)
        if editing and source_codes is not None:
            if index < len(source_codes):
                if random.random() < mix_ratio:
                    masked_codes[index] = source_codes[index]
                else:
                    masked_codes[index] = 126336
            else:
                masked_codes[index] = 126336
        else:
            masked_codes[index] = 126336  # <|mdm_mask|>

    return masked_codes, labels

def load_image_tokens(image_path):
    with open(image_path, "rb") as f:
        data_pkl = pickle.load(f)
    assert data_pkl["height"] % 16 == 0 and data_pkl["width"] % 16 == 0
    height, width = data_pkl["width"] // 16, data_pkl["height"] // 16
    # add breakline for image
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
        # [新增] 存储可视化配置
        self.visualize = visualize
        self.vis_dir = vis_dir

        # [新增] 初始化 VQ Model 用于计算 Metric
        self.vq_model = None
        if info_metric != "random" and vq_ckpt is not None:
            try:
                # 这里的 VQModel 类名取决于你的具体实现
                self.vq_model = VQModel.from_pretrained(vq_ckpt, subfolder="vqvae").eval()
                # print(f"Loaded VQModel for metric: {info_metric}")
            except Exception as e:
                print(f"Warning: Failed to load VQ Model from {vq_ckpt}: {e}")

    def process_item(self, data_item: dict, training_mode=False) -> Tuple[List, List]:
        # Understanding Data (保持不变)
        if data_item["user_image"] != "" and data_item["answer_image"] == "":
            instruction = "<system>" + data_item["system_prompt"] + "</system>" + "<user>" + data_item[
                "user_prompt"] + "</user>"
            instruction_token = \
                self.tokenizer(instruction, truncation=True, max_length=1024, padding=False,
                               return_tensors="pt").input_ids[
                    0].tolist()

            image_tokens = load_image_tokens(data_item["user_image"])
            # 126349 --> <Image>, 126350 --> </Image>
            instruction_token = instruction_token[:-1] + [126349] + image_tokens + [126350] + instruction_token[-1:]
            instruction_label = [-100] * len(instruction_token)

            """
                Final answer template: 
                <answer> answer token </answer> <padding> <padding> <padding> <padding> .....
            """
            answer = data_item["answer_text"] + "</answer>"
            answer_token = \
                self.tokenizer(answer, truncation=True, max_length=1024, padding=False, return_tensors="pt").input_ids[
                    0].tolist()
            # Answer mask 不使用 info metric
            answer_token, answer_label = mask_codes(answer_token)
            padding_len = 1024 - len(answer_token)
            # 126339 for padding
            padding_token = [126339] * padding_len
            padding_token, padding_label = mask_codes(padding_token, mask=True)
            # 126354 --> <answer>
            all_token = instruction_token + [126354] + answer_token + padding_token
            all_label = instruction_label + [-100] + answer_label + padding_label

        # Text-to-Image Data (修改支持 Info Mask)
        elif data_item["user_image"] == "" and data_item["answer_image"] != "":
            # CFG --> drop 10% text prompt
            if np.random.rand() < 0.1:
                instruction = "<system>" + data_item[
                    "system_prompt"] + "</system>" + "<user>" + "<uncondition>" + "</user>"
            else:
                instruction = "<system>" + data_item["system_prompt"] + "</system>" + "<user>" + data_item[
                    "user_prompt"] + "</user>"
            instruction_token = \
                self.tokenizer(instruction, truncation=True, max_length=1024, padding=False,
                               return_tensors="pt").input_ids[
                    0].tolist()
            instruction_label = [-100] * len(instruction_token)

            with open(data_item["answer_image"], "rb") as f:
                data_pkl = pickle.load(f)
            image_tokens = data_pkl["input_ids"]
            assert data_pkl["height"] % 16 == 0 and data_pkl["width"] % 16 == 0
            image_height, image_width = data_pkl["width"] // 16, data_pkl["height"] // 16

            # [调用 mask_codes 并传入 metric]
            image_masked_codes, image_labels = mask_codes(
                image_tokens,
                info_metric=self.info_metric,
                mask_random_ratio=self.mask_random_ratio,
                vq_model=self.vq_model,
                image_size=(image_height, image_width),
                # 传递可视化参数
                visualize=self.visualize,
                vis_dir=self.vis_dir
            )

            image_tokens = add_break_line(image_masked_codes, image_height, image_width, new_number=126084)
            image_labels = add_break_line(image_labels, image_height, image_width, new_number=-100)
            all_token = instruction_token + [126354] + [126349] + image_tokens + [126350] + [126355]
            all_label = instruction_label + [-100] + [-100] + image_labels + [-100] + [-100]

        # Image-to-Image Data (修改支持 Info Mask & Edit Diff)
        elif data_item["user_image"] != "" and data_item["answer_image"] != "":
            # 1. 预先加载 Source Tokens
            with open(data_item["user_image"], "rb") as f:
                source_pkl = pickle.load(f)
            source_tokens_raw = source_pkl["input_ids"]

            # 2. 决定 Drop 策略 (Text CFG vs Image CFG)
            rand_val = np.random.rand()
            use_source_image_in_input = True

            if rand_val < 0.1:
                # [Case 1] Text CFG
                instruction = "<system>" + data_item[
                    "system_prompt"] + "</system>" + "<user>" + "<uncondition>" + "</user>"
                use_source_image_in_input = True

            # elif rand_val < 0.2:
            #     # [Case 2] Image CFG
            #     instruction = "<system>" + data_item["system_prompt"] + "</system>" + "<user>" + data_item[
            #         "user_prompt"] + "</user>"
            #     use_source_image_in_input = False

            else:
                # [Case 3] Normal
                instruction = "<system>" + data_item["system_prompt"] + "</system>" + "<user>" + data_item[
                    "user_prompt"] + "</user>"
                use_source_image_in_input = True

            # 3. 构建 Instruction Token
            instruction_token = \
                self.tokenizer(instruction, truncation=True, max_length=1024, padding=False,
                               return_tensors="pt").input_ids[
                    0].tolist()

            if use_source_image_in_input:
                image_tokens = source_tokens_raw
                assert source_pkl["height"] % 16 == 0 and source_pkl["width"] % 16 == 0
                image_height, image_width = source_pkl["width"] // 16, source_pkl["height"] // 16
                image_tokens_with_break = add_break_line(image_tokens, image_height, image_width, new_number=126084)
                instruction_token = instruction_token[:-1] + [126349] + image_tokens_with_break + [
                    126350] + instruction_token[-1:]

            instruction_label = [-100] * len(instruction_token)

            # 4. 处理 Target Image (Answer)
            with open(data_item["answer_image"], "rb") as f:
                target_pkl = pickle.load(f)
            target_tokens_raw = target_pkl["input_ids"]
            assert target_pkl["height"] % 16 == 0 and target_pkl["width"] % 16 == 0
            image_height, image_width = target_pkl["width"] // 16, target_pkl["height"] // 16

            # 5. Mask Codes (Mix Edit + Info Metric)
            # 只有当 Input 包含 Source Image 时，edit_diff 才有意义
            image_masked_codes, image_labels = mask_codes(
                target_tokens_raw,
                source_codes=source_tokens_raw if use_source_image_in_input else None,
                editing=True,
                mix_ratio=self.mix_ratio,
                # [新增参数传递]
                info_metric=self.info_metric,
                vq_model=self.vq_model,
                image_size=(image_height, image_width),
                # 传递可视化参数
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
        # task-specific parameters
        parser.add_argument("--max_seq_len", default=1024, type=int, help="max token length")
        parser.add_argument("--dropout", type=float, default=0.05)
        parser.add_argument("--mix_ratio", type=float, default=0.5,
                            help="Ratio of source tokens mixed in mask for editing tasks")

        # [新增] 参数支持信息量 Mask
        parser.add_argument("--info_metric", type=str, default="edit_diff",
                            choices=["random", "variance", "frequency", "gradient", "edit_diff"],
                            help="Masking strategy based on image information.")
        parser.add_argument("--mask_random_ratio", type=float, default=0.2,
                            help="Ratio of random noise")
        parser.add_argument("--vae_ckpt", type=str, default="./Alpha-VLLM/Lumina-DiMOO",
                            help="Path to VQVAE checkpoint, required if info_metric is not random.")

        # [新增] 可视化参数
        parser.add_argument("--visualize", default=False, help="Visualize and save info maps.")
        parser.add_argument("--vis_dir", type=str, default="./vis_maps", help="Directory to save visualizations.")

        return parser

    def _model_func(
            self,
            init_from: str,
    ) -> (LLaDAForMultiModalGeneration, None):
        # Final SFT
        tokenizer = AutoTokenizer.from_pretrained(init_from, trust_remote_code=True)
        model = LLaDAForMultiModalGeneration.from_pretrained(init_from, torch_dtype=torch.bfloat16, device_map="cpu")
        model.model.set_activation_checkpointing("whole_layer")
        return model, tokenizer

    def _item_processor_func(self, tokenizer=None, max_len=None) -> ItemProcessorBase:
        # [修改] 传递新的参数
        return ItemProcessor(
            tokenizer,
            max_len,
            mix_ratio=self.args.mix_ratio,
            info_metric=self.args.info_metric,
            mask_random_ratio=self.args.mask_random_ratio,
            vq_ckpt=self.args.vae_ckpt,
            visualize=self.args.visualize,
            vis_dir=self.args.vis_dir
        )

    def _make_and_save_starting_point(self, save_path: str) -> None:
        tokenizer = AutoTokenizer.from_pretrained(self.args.init_from, trust_remote_code=True)
        base_config = AutoConfig.from_pretrained(self.args.init_from)
        model = LLaDAForMultiModalGeneration(base_config)
        model.resize_token_embeddings(len(tokenizer))
        model.model.transformer.ff_out = torch.nn.Linear(4096, len(tokenizer), bias=False)  # model dim --> 4096


if __name__ == "__main__":
    args = Solver.get_args_parser().parse_args()
    solver = Solver(args)
    solver.run()
