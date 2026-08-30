import pickle
from typing import List, Tuple
import random
from accelerate import init_empty_weights
import torch
import os
import numpy as np
import torch.nn as nn
import math
from transformers import AutoTokenizer, AutoConfig
import sys

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


def mask_codes(codes, source_codes=None, sch="cosine", mask=False, editing=False, mix_ratio=0.5):
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
        print("Not Implement")
        mask_ratio = 1.0  # Fallback

    num_to_mask = int(len(codes) * mask_ratio)
    if num_to_mask < 1:
        num_to_mask = 1

    indices_to_mask = random.sample(range(len(codes)), num_to_mask)
    masked_codes = codes[:]
    labels = [-100] * len(codes)

    # --- 新增 Mix Edit 逻辑 ---
    # 定义替换比例，例如 0.5 的概率使用 Source Token，0.5 使用 Mask Token
    # 你也可以像你提供的 snippet 一样将其设为随机数 r_mix = random.random()
    # mix_ratio = 0.5
    if mix_ratio == -1:
        mix_ratio = torch.rand(1).item()
        # print(mix_ratio)
    # else:
    #     mix_ratio = mix_ratio.float()

    for index in indices_to_mask:
        labels[index] = codes[index]  # 标签永远是 Target (Ground Truth)

        # 核心修改：如果是编辑模式且提供了源 Token，进行概率混合
        if editing and source_codes is not None:
            # 确保 source_codes 长度一致，防止越界
            if index < len(source_codes):
                # 随机决定是用 Mask 还是用 Source
                if random.random() < mix_ratio:
                    # print("mix_ratio", mix_ratio)
                    # print("index", index)
                    # 此时保留 Source Token (Source in Mask)
                    masked_codes[index] = source_codes[index]
                else:
                    # 此时使用 Mask Token
                    masked_codes[index] = 126336
            else:
                # 长度不匹配时的兜底策略
                masked_codes[index] = 126336
        else:
            # 原始逻辑：全部 Mask
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


# class ItemProcessor(ItemProcessorBase):
#     def __init__(self, tokenizer, max_len, mix_ratio=0.5, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.tokenizer = tokenizer
#         self.max_len = max_len
#         self.mix_ratio = mix_ratio  # 保存参数
#
#     def process_item(self, data_item: dict, training_mode=False) -> Tuple[List, List]:
#         # ... (Understanding Data 和 Text-to-Image Data 分支保持不变) ...
#
#         # Image-to-Image Data
#         if data_item["user_image"] != "" and data_item["answer_image"] != "":
#             if np.random.rand() < 0.1:
#                 instruction = "<system>" + data_item[
#                     "system_prompt"] + "</system>" + "<user>" + "<uncondition>" + "</user>"
#                 instruction_token = self.tokenizer(instruction, truncation=True, max_length=1024, padding=False,
#                                                    return_tensors="pt").input_ids[0].tolist()
#                 instruction_label = [-100] * len(instruction_token)
#             else:
#                 instruction = "<system>" + data_item["system_prompt"] + "</system>" + "<user>" + data_item[
#                     "user_prompt"] + "</user>"
#                 instruction_token = self.tokenizer(instruction, truncation=True, max_length=1024, padding=False,
#                                                    return_tensors="pt").input_ids[0].tolist()
#
#                 # --- 处理 Source Image (User Image) ---
#                 with open(data_item["user_image"], "rb") as f:
#                     source_pkl = pickle.load(f)  # 改个名方便区分
#
#                 # [修改 1] 获取原始 Source Tokens (用于 mix_edit)
#                 source_tokens_raw = source_pkl["input_ids"]
#
#                 image_tokens = source_tokens_raw  # 此时 image_tokens 用于构建 instruction
#                 assert source_pkl["height"] % 16 == 0 and source_pkl["width"] % 16 == 0
#                 image_height, image_width = source_pkl["width"] // 16, source_pkl["height"] // 16
#
#                 # 这里的 image_tokens 加入了换行符，用于输入
#                 image_tokens_with_break = add_break_line(image_tokens, image_height, image_width, new_number=126084)
#
#                 instruction_token = instruction_token[:-1] + [126349] + image_tokens_with_break + [
#                     126350] + instruction_token[-1:]
#                 instruction_label = [-100] * len(instruction_token)
#
#             # --- 处理 Target Image (Answer Image) ---
#             with open(data_item["answer_image"], "rb") as f:
#                 target_pkl = pickle.load(f)
#             target_tokens_raw = target_pkl["input_ids"]
#
#             assert target_pkl["height"] % 16 == 0 and target_pkl["width"] % 16 == 0
#             image_height, image_width = target_pkl["width"] // 16, target_pkl["height"] // 16
#
#             # [修改 2] 调用 mask_codes 时传入 source_tokens_raw 并开启 editing=True
#             # 注意：这里假设 Source 和 Target 分辨率一致，token 数量一致
#             image_masked_codes, image_labels = mask_codes(
#                 target_tokens_raw,
#                 source_codes=source_tokens_raw if 'source_tokens_raw' in locals() else None,
#                 # 只有在非 CFG (有User Image) 时才传
#                 editing=True,
#                 mix_ratio=self.mix_ratio  # 传入超参数
#             )
#
#             image_tokens = add_break_line(image_masked_codes, image_height, image_width, new_number=126084)
#             image_labels = add_break_line(image_labels, image_height, image_width, new_number=-100)
#
#             all_token = instruction_token + [126354] + [126349] + image_tokens + [126350] + [126355]
#             all_label = instruction_label + [-100] + [-100] + image_labels + [-100] + [-100]
#
#         return all_token, all_label
#
#     def predict_item_token_length(self, data_item: dict) -> int:
#         # breakpoint()
#         if "token" in data_item:
#             return len(data_item["token"])
#         elif "len" in data_item:
#             return data_item["len"]
#         else:
#             raise ValueError()


class ItemProcessor(ItemProcessorBase):
    def __init__(self, tokenizer, max_len, mix_ratio=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mix_ratio = mix_ratio

    def process_item(self, data_item: dict, training_mode=False) -> Tuple[List, List]:
        # Understanding Data (保持不变)
        if data_item["user_image"] != "" and data_item["answer_image"] == "":
            instruction = "<system>" + data_item["system_prompt"] + "</system>" + "<user>" + data_item[
                "user_prompt"] + "</user>"
            instruction_token = \
            self.tokenizer(instruction, truncation=True, max_length=1024, padding=False, return_tensors="pt").input_ids[
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
            answer_token, answer_label = mask_codes(answer_token)
            padding_len = 1024 - len(answer_token)
            # 126339 for padding
            padding_token = [126339] * padding_len
            padding_token, padding_label = mask_codes(padding_token, mask=True)
            # 126354 --> <answer>
            all_token = instruction_token + [126354] + answer_token + padding_token
            all_label = instruction_label + [-100] + answer_label + padding_label

        # Text-to-Image Data (保持不变)
        elif data_item["user_image"] == "" and data_item["answer_image"] != "":
            # CFG --> drop 10% text prompt
            if np.random.rand() < 0.1:
                instruction = "<system>" + data_item[
                    "system_prompt"] + "</system>" + "<user>" + "<uncondition>" + "</user>"
            else:
                instruction = "<system>" + data_item["system_prompt"] + "</system>" + "<user>" + data_item[
                    "user_prompt"] + "</user>"
            instruction_token = \
            self.tokenizer(instruction, truncation=True, max_length=1024, padding=False, return_tensors="pt").input_ids[
                0].tolist()
            instruction_label = [-100] * len(instruction_token)

            with open(data_item["answer_image"], "rb") as f:
                data_pkl = pickle.load(f)
            image_tokens = data_pkl["input_ids"]
            assert data_pkl["height"] % 16 == 0 and data_pkl["width"] % 16 == 0
            image_height, image_width = data_pkl["width"] // 16, data_pkl["height"] // 16
            image_masked_codes, image_labels = mask_codes(image_tokens)
            image_tokens = add_break_line(image_masked_codes, image_height, image_width, new_number=126084)
            image_labels = add_break_line(image_labels, image_height, image_width, new_number=-100)
            all_token = instruction_token + [126354] + [126349] + image_tokens + [126350] + [126355]
            all_label = instruction_label + [-100] + [-100] + image_labels + [-100] + [-100]

        # Image-to-Image Data (修改支持 Dual CFG)
        elif data_item["user_image"] != "" and data_item["answer_image"] != "":
            # 1. 预先加载 Source Tokens
            with open(data_item["user_image"], "rb") as f:
                source_pkl = pickle.load(f)
            source_tokens_raw = source_pkl["input_ids"]

            # 2. 决定 Drop 策略 (Text CFG vs Image CFG)
            rand_val = np.random.rand()
            use_source_image_in_input = True

            if rand_val < 0.1:
                # [Case 1] Text CFG (Drop Text): 支持 cfg_scale
                # Instruction: <uncondition> + Image
                instruction = "<system>" + data_item[
                    "system_prompt"] + "</system>" + "<user>" + "<uncondition>" + "</user>"
                use_source_image_in_input = True

            elif rand_val < 0.2:
                # [Case 2] Image CFG (Drop Image): 支持 cfg_img
                # Instruction: Prompt (No Image)
                # 此时任务退化为 Text-to-Image
                instruction = "<system>" + data_item["system_prompt"] + "</system>" + "<user>" + data_item[
                    "user_prompt"] + "</user>"
                use_source_image_in_input = False

            else:
                # [Case 3] Normal (Keep Both)
                instruction = "<system>" + data_item["system_prompt"] + "</system>" + "<user>" + data_item[
                    "user_prompt"] + "</user>"
                use_source_image_in_input = True

            # 3. 构建 Instruction Token
            instruction_token = \
            self.tokenizer(instruction, truncation=True, max_length=1024, padding=False, return_tensors="pt").input_ids[
                0].tolist()

            # 如果需要 Source Image，则插入到指令中
            if use_source_image_in_input:
                image_tokens = source_tokens_raw
                assert source_pkl["height"] % 16 == 0 and source_pkl["width"] % 16 == 0
                image_height, image_width = source_pkl["width"] // 16, source_pkl["height"] // 16
                image_tokens_with_break = add_break_line(image_tokens, image_height, image_width, new_number=126084)

                # 插入 <Image>...</Image>
                instruction_token = instruction_token[:-1] + [126349] + image_tokens_with_break + [
                    126350] + instruction_token[-1:]

            instruction_label = [-100] * len(instruction_token)

            # 4. 处理 Target Image (Answer)
            with open(data_item["answer_image"], "rb") as f:
                target_pkl = pickle.load(f)
            target_tokens_raw = target_pkl["input_ids"]
            assert target_pkl["height"] % 16 == 0 and target_pkl["width"] % 16 == 0
            image_height, image_width = target_pkl["width"] // 16, target_pkl["height"] // 16

            # 5. Mask Codes (Mix Edit 策略)
            # 如果我们在 Input 中丢弃了 Image (Case 2)，则在 Target 中也不应该使用 Source Hint (mix_edit)，
            # 否则模型会作弊（从 Target 噪声中恢复图片，而不是从 Text 生成）。
            # 因此：仅当 use_source_image_in_input 为 True 时，才传入 source_codes。
            image_masked_codes, image_labels = mask_codes(
                target_tokens_raw,
                source_codes=source_tokens_raw if use_source_image_in_input else None,
                editing=True,
                mix_ratio=self.mix_ratio
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


    def predict_item_token_length(self, data_item: dict) -> int:
        # breakpoint()
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
                            help="Ratio of source tokens mixed in mask for editing tasks (0.0 means pure mask)")
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
        return ItemProcessor(tokenizer, max_len, mix_ratio=self.args.mix_ratio)

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
