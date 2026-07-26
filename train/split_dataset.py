#!/usr/bin/env python3
"""
split_dataset.py —— 将 all.jsonl（Ground Truth）切分为 train / eval / test 三部分。

输入：
    - ./mock_data/outputs/all.jsonl    (Ground Truth，6331 条)
    - ./mock_data/mock_data_ch/ao.jsonl   (AO 原文，6296 条)

输出：
    - ./data/train.jsonl    (80%)
    - ./data/eval.jsonl     (10%)
    - ./data/test.jsonl     (10%)

策略：
    1. 以 all.jsonl 为基准，通过 id 关联 ao.jsonl 的 AO 原文
    2. 按照 source（ATA 章节）做分层抽样（stratified split），保证各章节比例一致
    3. 固定随机种子确保可复现

使用：
    python split_dataset.py [--train-ratio 0.8] [--eval-ratio 0.1] [--seed 42]
"""
import os
import json
import argparse
import random
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Split dataset into train/eval/test")
    p.add_argument("--output-all", default="../mock_data/outputs/all.jsonl",
                   help="Ground Truth JSONL file")
    p.add_argument("--cleaned-ao", default="../mock_data/mock_data_ch/ao.jsonl",
                   help="Cleaned AO translations JSONL file")
    p.add_argument("--out-dir", default="./data", help="Output directory")
    p.add_argument("--train-ratio", type=float, default=0.80)
    p.add_argument("--eval-ratio", type=float, default=0.10)
    p.add_argument("--test-ratio", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-rows", type=int, default=1,
                   help="过滤 rows 数量 < min_rows 的样本")
    return p.parse_args()


def load_jsonl(path: str) -> dict:
    """加载 JSONL，返回 {id: obj} 字典"""
    data = {}
    if not os.path.exists(path):
        print(f"[WARN] 文件不存在: {path}")
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                id_ = obj.get("id")
                if id_ is not None:
                    data[id_] = obj
            except json.JSONDecodeError:
                continue
    return data


def main():
    args = parse_args()

    # 1. 加载数据
    print(f"[INFO] 加载 Ground Truth: {args.output_all}")
    gt_data = load_jsonl(args.output_all)
    print(f"  -> {len(gt_data)} 条")

    print(f"[INFO] 加载 AO 原文: {args.cleaned_ao}")
    ao_data = load_jsonl(args.cleaned_ao)
    print(f"  -> {len(ao_data)} 条")

    # 2. 合并：以 GT 为准，关联 AO 原文
    merged = []
    skipped_no_ao = 0
    skipped_few_rows = 0
    for id_, gt_obj in gt_data.items():
        rows = gt_obj.get("rows", [])
        if len(rows) < args.min_rows:
            skipped_few_rows += 1
            continue
        ao_obj = ao_data.get(id_)
        if ao_obj is None:
            skipped_no_ao += 1
            continue
        merged.append({
            "id": id_,
            "source": gt_obj.get("source", ""),
            "content": ao_obj.get("content", ""),   # AO 原文
            "rows": rows,                            # Ground Truth 结构化表格
        })

    print(f"[INFO] 合并后有效样本: {len(merged)}")
    print(f"  跳过 (无AO): {skipped_no_ao}, 跳过 (rows太少): {skipped_few_rows}")

    # 3. 按 source 分层
    strata = defaultdict(list)
    for item in merged:
        src = item.get("source", "unknown")
        strata[src].append(item)

    print(f"[INFO] 分层数 (source): {len(strata)}")
    for src, items in sorted(strata.items(), key=lambda x: -len(x[1])):
        print(f"  source={src}: {len(items)} 条")

    # 4. 分层抽样
    random.seed(args.seed)
    train_all, eval_all, test_all = [], [], []

    for src, items in strata.items():
        random.shuffle(items)
        n = len(items)
        n_train = max(1, int(n * args.train_ratio))
        n_eval = max(1, int(n * args.eval_ratio))
        n_test = n - n_train - n_eval
        if n_test < 1:
            # 样本太少时优先保证 train
            n_train = max(1, n - 2)
            n_eval = max(1, n - n_train - 1)
            n_test = n - n_train - n_eval

        train_all.extend(items[:n_train])
        eval_all.extend(items[n_train:n_train + n_eval])
        test_all.extend(items[n_train + n_eval:])

    random.shuffle(train_all)
    random.shuffle(eval_all)
    random.shuffle(test_all)

    # 5. 写入输出
    os.makedirs(args.out_dir, exist_ok=True)

    for name, data in [("train", train_all), ("eval", eval_all), ("test", test_all)]:
        path = os.path.join(args.out_dir, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[INFO] 写入 {path}: {len(data)} 条")

    # 6. 打印统计
    print("\n" + "=" * 60)
    print(f"数据集切分完成 (seed={args.seed})")
    print(f"  Train: {len(train_all)} ({100*len(train_all)/len(merged):.1f}%)")
    print(f"  Eval:  {len(eval_all)} ({100*len(eval_all)/len(merged):.1f}%)")
    print(f"  Test:  {len(test_all)} ({100*len(test_all)/len(merged):.1f}%)")
    print(f"  Total: {len(merged)}")
    print("=" * 60)


if __name__ == "__main__":
    main()