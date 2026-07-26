#!/usr/bin/env python3
"""统计 SFT 数据经过真实模型 chat template 后的 token 长度。

该脚本只读取模型 tokenizer 和 JSONL 数据，不会修改或截断原始数据。
总长度的计算方式与 train_sft_final.py 保持一致：

1. tokenizer.apply_chat_template(..., tokenize=False)
2. tokenizer(..., add_special_tokens=True, truncation=False)

同时将 system/user prompt 与 assistant 答案分开统计，用于判断右截断会
丢失多少答案 token。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILES = [
    PROJECT_ROOT / "data" / "train_sft.jsonl",
    PROJECT_ROOT / "data" / "eval_sft.jsonl",
]
DEFAULT_THRESHOLDS = [4096, 6144, 8192, 12288, 16384]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "使用训练模型的 tokenizer 和 chat template 统计 SFT 数据长度，"
            "并评估指定 max_seq_length 下的右截断影响。"
        )
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="本地 Qwen3.5 模型目录；只加载 tokenizer，不加载模型权重。",
    )
    parser.add_argument(
        "--files",
        type=Path,
        nargs="+",
        default=DEFAULT_FILES,
        help="待统计的一个或多个 JSONL 文件。默认统计 data/train_sft.jsonl 和 eval_sft.jsonl。",
    )
    parser.add_argument(
        "--training-max-length",
        type=int,
        default=6144,
        help="训练计划采用的 max_seq_length，用于模拟右截断影响。默认 6144。",
    )
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
        help="额外统计超过这些 token 长度的样本数。",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="展示每个文件中最长的 N 条样本。默认 20。",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="可选：将完整统计结果写入 JSON 文件。不会改写数据集。",
    )
    return parser.parse_args()


def resolve_existing_path(path: Path, description: str) -> Path:
    """优先按当前目录解析，未找到时再按项目根目录解析。"""
    path = path.expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, PROJECT_ROOT / path]
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate
    checked = "\n  - ".join(str(candidate.resolve()) for candidate in candidates)
    raise FileNotFoundError(f"{description}不存在，已检查：\n  - {checked}")


def nearest_rank(values: list[int], percentile: float) -> int:
    """返回 nearest-rank 百分位数；values 必须已经升序排列。"""
    if not values:
        return 0
    rank = max(1, math.ceil(percentile * len(values)))
    return values[rank - 1]


def length_summary(values: list[int]) -> dict[str, int | float]:
    if not values:
        return {
            "min": 0,
            "mean": 0.0,
            "p50": 0,
            "p75": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "max": 0,
        }

    ordered = sorted(values)
    return {
        "min": ordered[0],
        "mean": round(mean(ordered), 2),
        "p50": nearest_rank(ordered, 0.50),
        "p75": nearest_rank(ordered, 0.75),
        "p90": nearest_rank(ordered, 0.90),
        "p95": nearest_rank(ordered, 0.95),
        "p99": nearest_rank(ordered, 0.99),
        "max": ordered[-1],
    }


def encode_like_training(tokenizer: Any, text: str) -> list[int]:
    """严格复现 train_sft_final.py 中 chat template 之后的 tokenizer 调用。"""
    return tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )["input_ids"]


def render_and_measure(
    tokenizer: Any,
    messages: list[dict[str, str]],
) -> tuple[int, int, int, bool]:
    """返回总长度、prompt 长度、assistant 部分长度、前缀是否一致。"""
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    full_ids = encode_like_training(tokenizer, full_text)

    # AO 结构化输出任务关闭思考。对完整 assistant 消息，Qwen3.5 模板会
    # 生成相同的空 think 块；这个生成前缀用于确定答案 token 的起点。
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = encode_like_training(tokenizer, prompt_text)

    text_prefix_matches = full_text.startswith(prompt_text)
    token_prefix_matches = full_ids[: len(prompt_ids)] == prompt_ids
    prefix_matches = text_prefix_matches and token_prefix_matches

    if prefix_matches:
        assistant_length = len(full_ids) - len(prompt_ids)
    else:
        # 保留总长度统计，但不能安全推断答案在完整序列中的 token 边界。
        assistant_length = 0

    return len(full_ids), len(prompt_ids), assistant_length, prefix_matches


def validate_messages(messages: Any) -> tuple[bool, str]:
    if not isinstance(messages, list) or not messages:
        return False, "messages 不是非空列表"

    expected_roles = ["system", "user", "assistant"]
    roles: list[Any] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            return False, f"messages[{index}] 不是对象"
        roles.append(message.get("role"))
        if not isinstance(message.get("content"), str):
            return False, f"messages[{index}].content 不是字符串"

    if roles != expected_roles:
        return False, f"角色顺序为 {roles!r}，期望 {expected_roles!r}"
    return True, ""


def inspect_file(
    tokenizer: Any,
    filename: Path,
    training_max_length: int,
    thresholds: list[int],
    top_n: int,
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    ids: list[str] = []
    non_empty_line_count = 0
    valid_assistant_json = 0
    assistant_json_arrays = 0
    prefix_mismatch_count = 0
    with filename.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            non_empty_line_count += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(
                    {
                        "line": line_number,
                        "id": None,
                        "error": f"JSON 解析失败: {exc.msg}",
                    }
                )
                continue

            if not isinstance(record, dict):
                errors.append(
                    {
                        "line": line_number,
                        "id": None,
                        "error": "顶层 JSON 不是对象",
                    }
                )
                continue

            sample_id = str(record.get("id", f"line-{line_number}"))
            ids.append(sample_id)
            messages = record.get("messages")
            valid, error = validate_messages(messages)
            if not valid:
                errors.append({"line": line_number, "id": sample_id, "error": error})
                continue

            assistant_content = messages[-1]["content"]
            try:
                parsed_assistant = json.loads(assistant_content)
                valid_assistant_json += 1
                if isinstance(parsed_assistant, list):
                    assistant_json_arrays += 1
                else:
                    errors.append(
                        {
                            "line": line_number,
                            "id": sample_id,
                            "error": "assistant 内容是合法 JSON，但顶层不是数组",
                        }
                    )
            except json.JSONDecodeError as exc:
                errors.append(
                    {
                        "line": line_number,
                        "id": sample_id,
                        "error": f"assistant 内容不是合法 JSON: {exc.msg}",
                    }
                )

            try:
                total_length, prompt_length, assistant_length, prefix_matches = (
                    render_and_measure(tokenizer, messages)
                )
            except Exception as exc:
                errors.append(
                    {
                        "line": line_number,
                        "id": sample_id,
                        "error": f"chat template/tokenizer 处理失败: {exc}",
                    }
                )
                continue

            if not prefix_matches:
                prefix_mismatch_count += 1

            truncated_tokens = max(0, total_length - training_max_length)
            prompt_is_truncated = prompt_length > training_max_length
            answer_capacity = max(0, training_max_length - prompt_length)
            answer_tokens_removed = (
                max(0, assistant_length - answer_capacity) if prefix_matches else None
            )

            samples.append(
                {
                    "line": line_number,
                    "id": sample_id,
                    "source": record.get("source", ""),
                    "total_tokens": total_length,
                    "prompt_tokens": prompt_length,
                    "assistant_tokens": assistant_length if prefix_matches else None,
                    "prefix_matches": prefix_matches,
                    "excess_tokens": truncated_tokens,
                    "prompt_is_truncated": prompt_is_truncated,
                    "assistant_tokens_removed": answer_tokens_removed,
                }
            )

    duplicate_ids = sorted(
        sample_id for sample_id, count in Counter(ids).items() if count > 1
    )
    total_lengths = [sample["total_tokens"] for sample in samples]
    prompt_lengths = [sample["prompt_tokens"] for sample in samples]
    assistant_lengths = [
        sample["assistant_tokens"]
        for sample in samples
        if sample["assistant_tokens"] is not None
    ]

    over_training_limit = [
        sample for sample in samples if sample["total_tokens"] > training_max_length
    ]
    prompt_truncated = [sample for sample in samples if sample["prompt_is_truncated"]]
    assistant_truncated = [
        sample
        for sample in samples
        if sample["assistant_tokens_removed"] is not None
        and sample["assistant_tokens_removed"] > 0
    ]
    total_removed = sum(sample["excess_tokens"] for sample in over_training_limit)
    assistant_removed = sum(
        sample["assistant_tokens_removed"] for sample in assistant_truncated
    )

    longest = sorted(
        samples,
        key=lambda sample: sample["total_tokens"],
        reverse=True,
    )[:top_n]
    invalid_line_count = len({error["line"] for error in errors})

    return {
        "file": str(filename),
        "non_empty_records": non_empty_line_count,
        "parsed_records": len(ids),
        "measured_records": len(samples),
        "invalid_records": invalid_line_count,
        "error_count": len(errors),
        "errors": errors,
        "duplicate_id_count": len(duplicate_ids),
        "duplicate_ids": duplicate_ids,
        "assistant_json_valid_count": valid_assistant_json,
        "assistant_json_array_count": assistant_json_arrays,
        "template_prefix_mismatch_count": prefix_mismatch_count,
        "lengths": {
            "total": length_summary(total_lengths),
            "prompt": length_summary(prompt_lengths),
            "assistant": length_summary(assistant_lengths),
        },
        "thresholds": {
            str(threshold): {
                "count": sum(length > threshold for length in total_lengths),
                "percent": round(
                    100 * sum(length > threshold for length in total_lengths)
                    / len(total_lengths),
                    2,
                )
                if total_lengths
                else 0.0,
            }
            for threshold in thresholds
        },
        "training_truncation": {
            "max_seq_length": training_max_length,
            "samples_over_limit": len(over_training_limit),
            "samples_over_limit_percent": round(
                100 * len(over_training_limit) / len(samples), 2
            )
            if samples
            else 0.0,
            "samples_with_prompt_truncated": len(prompt_truncated),
            "samples_with_assistant_truncated": len(assistant_truncated),
            "total_tokens_removed": total_removed,
            "assistant_tokens_removed": assistant_removed,
        },
        "longest_samples": longest,
    }


def print_length_block(title: str, values: dict[str, int | float]) -> None:
    print(
        f"  {title:<10}"
        f" min={values['min']}"
        f" mean={values['mean']}"
        f" P50={values['p50']}"
        f" P75={values['p75']}"
        f" P90={values['p90']}"
        f" P95={values['p95']}"
        f" P99={values['p99']}"
        f" max={values['max']}"
    )


def print_report(report: dict[str, Any], top_n: int) -> None:
    print("\n" + "=" * 88)
    print(f"文件: {report['file']}")
    print(
        f"非空记录={report['non_empty_records']}，"
        f"成功解析={report['parsed_records']}，"
        f"成功统计={report['measured_records']}，"
        f"异常记录={report['invalid_records']}（错误项={report['error_count']}），"
        f"重复 ID={report['duplicate_id_count']}"
    )
    print(
        f"assistant 内容为合法 JSON={report['assistant_json_valid_count']}，"
        f"其中 JSON 数组={report['assistant_json_array_count']}，"
        f"模板前缀不一致={report['template_prefix_mismatch_count']}"
    )

    print("\nToken 长度分布:")
    print_length_block("总序列", report["lengths"]["total"])
    print_length_block("Prompt", report["lengths"]["prompt"])
    print_length_block("Assistant", report["lengths"]["assistant"])

    print("\n超过长度阈值:")
    for threshold, result in report["thresholds"].items():
        print(f"  > {threshold:>5}: {result['count']:>5} ({result['percent']:.2f}%)")

    truncation = report["training_truncation"]
    print(f"\n按 max_seq_length={truncation['max_seq_length']} 进行右截断时:")
    print(
        f"  超长样本: {truncation['samples_over_limit']} "
        f"({truncation['samples_over_limit_percent']:.2f}%)"
    )
    print(f"  Prompt 自身被截断的样本: {truncation['samples_with_prompt_truncated']}")
    print(f"  Assistant 答案被截断的样本: {truncation['samples_with_assistant_truncated']}")
    print(f"  总共会删除的 token: {truncation['total_tokens_removed']}")
    print(f"  其中属于 Assistant 的 token: {truncation['assistant_tokens_removed']}")

    print(f"\n最长的 {min(top_n, len(report['longest_samples']))} 条样本:")
    print(
        f"  {'ID':<16} {'总长度':>8} {'Prompt':>8} "
        f"{'Assistant':>10} {'超出':>8} {'行号':>8}"
    )
    for sample in report["longest_samples"]:
        assistant_length = sample["assistant_tokens"]
        assistant_text = "N/A" if assistant_length is None else str(assistant_length)
        print(
            f"  {sample['id'][:16]:<16} "
            f"{sample['total_tokens']:>8} "
            f"{sample['prompt_tokens']:>8} "
            f"{assistant_text:>10} "
            f"{sample['excess_tokens']:>8} "
            f"{sample['line']:>8}"
        )

    if report["errors"]:
        print("\n异常记录（最多显示 20 条）:")
        for error in report["errors"][:20]:
            print(f"  line={error['line']} id={error['id']}: {error['error']}")


def main() -> int:
    args = parse_args()
    if args.training_max_length <= 0:
        raise ValueError("--training-max-length 必须大于 0")
    if args.top_n < 0:
        raise ValueError("--top-n 不能小于 0")
    if any(threshold <= 0 for threshold in args.thresholds):
        raise ValueError("--thresholds 中的所有值都必须大于 0")

    model_dir = resolve_existing_path(args.model_dir, "模型目录")
    files = [resolve_existing_path(path, "数据文件") for path in args.files]
    thresholds = sorted(set(args.thresholds + [args.training_max_length]))

    print(f"[INFO] tokenizer: {model_dir}")
    print("[INFO] 只加载 tokenizer，不会加载 Qwen3.5 模型权重。")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
    )

    reports = [
        inspect_file(
            tokenizer=tokenizer,
            filename=filename,
            training_max_length=args.training_max_length,
            thresholds=thresholds,
            top_n=args.top_n,
        )
        for filename in files
    ]

    for report in reports:
        print_report(report, args.top_n)

    output = {
        "model_dir": str(model_dir),
        "tokenizer_class": tokenizer.__class__.__name__,
        "training_max_length": args.training_max_length,
        "reports": reports,
    }

    if args.output_json:
        output_path = args.output_json.expanduser()
        if not output_path.is_absolute():
            output_path = (Path.cwd() / output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n[INFO] 完整报告已写入: {output_path}")

    invalid_count = sum(report["invalid_records"] for report in reports)
    mismatch_count = sum(
        report["template_prefix_mismatch_count"] for report in reports
    )
    duplicate_count = sum(report["duplicate_id_count"] for report in reports)
    if invalid_count or mismatch_count or duplicate_count:
        print(
            f"\n[ERROR] 发现 {invalid_count} 条异常记录、"
            f"{mismatch_count} 条模板前缀不一致记录、"
            f"{duplicate_count} 个重复 ID；请先处理再训练。",
            file=sys.stderr,
        )
        return 1

    over_limit_count = sum(
        report["training_truncation"]["samples_over_limit"] for report in reports
    )
    if over_limit_count:
        print(
            f"\n[WARN] 共有 {over_limit_count} 条样本会在当前 "
            f"max_seq_length={args.training_max_length} 下被截断；"
            "请根据报告决定提高长度、拆分或剔除异常长样本。"
        )

    print("\n[INFO] 数据格式和模板前缀检查通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
