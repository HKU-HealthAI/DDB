

import json


def convert_format(input_file, output_file):
    # 1. 读取数据
    # 注意：你提供的输入看起来是一个包含多个对象的 JSON 列表 [...]
    # 如果你的源文件本身就是每行一个对象的 jsonl，请在下方注释处切换读取方式
    with open(input_file, 'r', encoding='utf-8') as f:
        try:
            # 尝试作为整个 JSON 列表读取 (符合你提供的示例)
            data = json.load(f)
        except json.JSONDecodeError:
            # 如果失败，尝试作为 JSONL (每行一个对象) 读取
            f.seek(0)
            data = [json.loads(line) for line in f]

    # 2. 打开输出文件准备写入
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item in data:
            # --- 提取 image_path ---
            # 原数据中 image 是一个列表，我们需要取第一个元素
            image_path = ""
            if "image" in item and len(item["image"]) > 0:
                image_path = item["image"][0]

            # --- 提取 prompt ---
            # 遍历 conversations，找到 'from' 为 'human' 的那句话
            prompt = ""
            if "conversations" in item:
                for turn in item["conversations"]:
                    if turn.get("from") == "human":
                        prompt = turn.get("value", "")
                        # 找到后跳出循环，防止有多个轮次（通常取第一句）
                        break

            # --- 构建新的对象 ---
            # 根据你的要求：保留 image_path, prompt，并去除 edit_path
            new_obj = {
                "image_path": image_path,
                "prompt": prompt
            }

            # --- 写入文件 ---
            # ensure_ascii=False 保证中文字符正常显示
            f_out.write(json.dumps(new_obj, ensure_ascii=False) + '\n')

    print(f"转换完成！已保存至 {output_file}")


# 使用示例
if __name__ == "__main__":
    # 请确保目录下有 input.json 文件，内容为你提供的源数据
    convert_format('input.json', 'output.jsonl')