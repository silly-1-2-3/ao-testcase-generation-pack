#!/usr/bin/env python3
"""
prepare_sft_data.py —— 将切分好的 train/eval/test JSONL 转换为 SFT 训练所需的 messages 格式。

输入：
    - ./data/train.jsonl   (split_dataset.py 输出)
    - ./data/eval.jsonl
    - ./data/test.jsonl

输出：
    - ./data/train_sft.jsonl   (messages 格式，可直接用于 train_swift_sft.py)
    - ./data/eval_sft.jsonl
    - ./data/test_sft.jsonl

转换逻辑：
    system prompt: "你是一个精通波音737 AMM 到结构化测试用例转换的专家..."
    user prompt:   AO 原文 (content 字段)
    assistant:     JSON 格式的 rows 列表

使用：
    conda activate transport_vllm
    python prepare_sft_data.py
"""
import json
import os
import argparse
from pathlib import Path

# 从文件加载 SYSTEM_PROMPT（避免 Python 字符串逃逸问题）
_PROMPT_FILE = os.path.join(os.path.dirname(__file__), 'system_prompt_v4.txt')
with open(_PROMPT_FILE, 'r', encoding='utf-8') as _f:
    SYSTEM_PROMPT = _f.read()

OUTPUT_HEADERS = [
    "步骤层级",        # 用例/子用例/步骤/判据/执行步骤
    "说明",            # 步骤名称
    "注意事项",
    "操作内容",
    "操作对象",
    "操作目的",
    "是否同时发送",
    "多判据组合条件",
    "是否使用设备",
    "操作类型",
    "判据类型",
    "判据范围",
    "判据描述",
    "左值",
    "右值",
    "单位",
    "设备类型",
    "设备单元号",
    "设备指令号",
    "设备参数",
    "判据关联标志",
]


def parse_args():
    p = argparse.ArgumentParser(description="Prepare SFT data in messages format")
    p.add_argument("--input-dir", default="./data", help="输入目录（含 train/eval/test.jsonl）")
    p.add_argument("--output-dir", default="./data", help="输出目录")
    p.add_argument("--no-shuffle", action="store_true", help="不打乱数据顺序")
    return p.parse_args()


def convert_item(item: dict) -> dict:
    """将单条样本转为 messages 格式"""
    content = item.get("content", "")
    rows = item.get("rows", [])

    # assistant 回复：紧凑 JSON
    assistant_text = json.dumps(rows, ensure_ascii=False)

    return {
        "id": item.get("id"),
        "source": item.get("source", ""),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
            {"role": "assistant", "content": assistant_text},
        ],
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for split in ["train", "eval", "test"]:
        in_path = os.path.join(args.input_dir, f"{split}.jsonl")
        out_path = os.path.join(args.output_dir, f"{split}_sft.jsonl")

        if not os.path.exists(in_path):
            print(f"[WARN] 跳过不存在的文件: {in_path}")
            continue

        converted = []
        with open(in_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    converted.append(convert_item(item))
                except json.JSONDecodeError:
                    continue

        with open(out_path, "w", encoding="utf-8") as f:
            for item in converted:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"[INFO] {split}: {len(converted)} 条 -> {out_path}")

    print("[INFO] SFT 数据准备完成")


if __name__ == "__main__":
    main()