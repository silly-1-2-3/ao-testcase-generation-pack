#!/usr/bin/env python3
"""
使用同一个 vLLM 引擎，对本地 Qwen3.5 Base 与 LoRA adapter 做成对评测。

设计约定：
    1. 严格离线，不访问 Hugging Face Hub，也不发送 vLLM 使用统计。
    2. Prompt 与 SFT 训练保持一致，显式设置 enable_thinking=False。
    3. 使用完全相同的 tokenized prompt 和生成参数依次评测 Base、LoRA。
    4. 不使用 guided/structured decoding，避免约束解码掩盖模型格式能力差异。
    5. 严格校验数据；任何坏行立即报错，不再静默跳过。
    6. 保存逐样本原始输出、解析状态、停止原因、token 数和评分，便于审计。

推荐先在验证集上冒烟：
    python train/eval_model_vllm.py \
        --base_model ../qwen3_5_9b_deploy/models/Qwen3.5-9B \
        --adapter outputs/qwen35_lora_full_v1 \
        --data_file data/eval_sft.jsonl \
        --output_dir eval_results/qwen35_lora_full_v1_eval_smoke \
        --max_samples 8

验证生成参数冻结后，再用 test_sft.jsonl 做最终测试。
"""


from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# 必须在导入 transformers / torch / vllm 之前设置。
STRICT_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "HF_HUB_DISABLE_UPDATE_CHECK": "1",
    "HF_HUB_DISABLE_XET": "1",
    "DO_NOT_TRACK": "1",
    "VLLM_DO_NOT_TRACK": "1",
    "VLLM_NO_USAGE_STATS": "1",
    # conda-forge 的 vLLM CUDA 构建未必同时提供 flashinfer Python 包。
    # 显式使用 vLLM 官方支持的原生采样回退，避免引擎初始化时导入失败。
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
}
for environment_name, environment_value in STRICT_OFFLINE_ENV.items():
    os.environ[environment_name] = environment_value

for credential_name in (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "WANDB_API_KEY",
):
    os.environ.pop(credential_name, None)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _normalize_omp_num_threads() -> None:
    """在 vLLM/PyTorch 加载前修复 AutoDL 可能注入的非法线程数。"""
    raw_value = os.environ.get("OMP_NUM_THREADS")
    if raw_value is None:
        return
    try:
        parsed_value = int(raw_value.strip())
        if parsed_value <= 0:
            raise ValueError
        os.environ["OMP_NUM_THREADS"] = str(parsed_value)
    except (TypeError, ValueError):
        os.environ["OMP_NUM_THREADS"] = "1"
        print(
            f"[WARN] 检测到非法 OMP_NUM_THREADS={raw_value!r}，"
            "已在导入 vLLM/PyTorch 前修正为 1。"
        )


_normalize_omp_num_threads()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import UseCaseTableMetric as LegacyUseCaseTableMetric
from metrics_v2 import UseCaseTableMetricV2

STEP_LIST_KEY = "步骤列表"
ALLOWED_LEVELS = {"用例", "子用例", "步骤", "判据", "执行步骤"}
REQUIRED_FIELDS = {
    "用例": ["步骤层级", "说明", "注意事项"],
    "子用例": ["步骤层级", "说明", "注意事项"],
    "步骤": [
        "步骤层级",
        "说明",
        "注意事项",
        "操作内容",
        "操作对象",
        "操作目的",
        "是否同时发送",
        "多判据组合条件",
    ],
    "判据": [
        "步骤层级",
        "是否使用设备",
        "操作类型",
        "判据类型",
        "判据范围",
        "判据描述",
        "左值",
        "右值",
        "单位",
    ],
    "执行步骤": [
        "步骤层级",
        "设备类型",
        "设备单元号",
        "设备指令号",
        "设备参数",
        "判据关联标志",
    ],
}
ENUM_FIELDS = {
    "是否同时发送": {"是", "否"},
    "多判据组合条件": {"全部成功", "任一成功"},
    "是否使用设备": {"是", "否"},
    "操作类型": {"仅操作", "选值", "录值"},
    "判据类型": {"其他", "判据关联"},
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline paired vLLM evaluation for Qwen3.5 Base and LoRA"
    )
    parser.add_argument(
        "--base_model",
        "--base-model",
        dest="base_model",
        required=True,
        help="本地 Qwen3.5 Base 模型目录",
    )
    parser.add_argument(
        "--adapter",
        required=True,
        help="训练结束后输出目录根部的 LoRA adapter",
    )
    parser.add_argument(
        "--data_file",
        "--data-file",
        "--eval_file",
        "--eval-file",
        "--test_file",
        "--test-file",
        dest="data_file",
        default="data/eval_sft.jsonl",
        help="SFT 格式评测集；默认先使用 eval_sft.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        default="eval_results/qwen35_base_vs_lora",
    )
    parser.add_argument(
        "--max_samples",
        "--max-samples",
        dest="max_samples",
        type=int,
        default=None,
        help="固定随机种子抽取 N 条，仅用于冒烟；正式评测不要设置",
    )
    parser.add_argument(
        "--max_model_len",
        "--max-model-len",
        dest="max_model_len",
        type=int,
        default=16384,
        help="vLLM 最大上下文，必须容纳 prompt + max_new_tokens",
    )
    parser.add_argument(
        "--max_new_tokens",
        "--max-new-tokens",
        dest="max_new_tokens",
        type=int,
        default=8192,
        help="最大生成长度；当前训练答案最大约 6198 token",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="正式可复现评测默认使用贪心解码 temperature=0",
    )
    parser.add_argument("--top_p", "--top-p", dest="top_p", type=float, default=1.0)
    parser.add_argument(
        "--gpu_memory",
        "--gpu-memory",
        dest="gpu_memory",
        type=float,
        default=0.85,
        help="vLLM 可使用的单卡显存比例",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        "--tensor-parallel-size",
        dest="tensor_parallel_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--metric_version",
        "--metric-version",
        dest="metric_version",
        choices=("v2", "legacy"),
        default="v2",
        help="默认使用具有单调全局行对齐和对称漏行/幻觉惩罚的 v2 指标",
    )
    metric_fields = parser.add_mutually_exclusive_group()
    metric_fields.add_argument(
        "--include_device_fields",
        "--include-device-fields",
        dest="include_device_fields",
        action="store_true",
        help="设备字段参与评分（默认）",
    )
    metric_fields.add_argument(
        "--skip_device_fields",
        "--skip-device-fields",
        dest="include_device_fields",
        action="store_false",
        help="忽略设备字段，仅用于复现旧实验",
    )
    parser.set_defaults(include_device_fields=True)
    parser.add_argument(
        "--overwrite_output_dir",
        "--overwrite-output-dir",
        dest="overwrite_output_dir",
        action="store_true",
        help="允许覆盖输出目录中的本评测结果文件",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    positive_values = {
        "max_model_len": args.max_model_len,
        "max_new_tokens": args.max_new_tokens,
        "tensor_parallel_size": args.tensor_parallel_size,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"--{name} 必须大于 0，当前值为 {value}")
    if args.max_new_tokens >= args.max_model_len:
        raise ValueError(
            "--max_new_tokens 必须小于 --max_model_len，"
            "还需要为 system/user prompt 预留上下文"
        )
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max_samples 必须大于 0")
    if not 0.0 <= args.temperature:
        raise ValueError("--temperature 必须大于等于 0")
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("--top_p 必须位于 (0, 1]")
    if not 0.0 < args.gpu_memory <= 1.0:
        raise ValueError("--gpu_memory 必须位于 (0, 1]")
    if args.temperature > 0:
        print(
            "[WARN] temperature > 0 会引入采样波动；"
            "正式 Base/LoRA 对比建议使用默认值 0。"
        )


def resolve_project_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"输出路径不是目录: {path}")
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"输出目录已存在且非空: {path}\n"
            "请换一个 --output_dir，或确认后显式添加 --overwrite_output_dir。"
        )
    path.mkdir(parents=True, exist_ok=True)


def load_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"JSON 文件格式错误: {path}，行 {exc.lineno}，列 {exc.colno}: {exc.msg}"
        ) from exc


def validate_base_model_dir(path: Path) -> Dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(f"Base 模型目录不存在: {path}")
    config_path = path / "config.json"
    tokenizer_config_path = path / "tokenizer_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Base 模型缺少 config.json: {path}")
    if not tokenizer_config_path.is_file():
        raise FileNotFoundError(f"Base 模型缺少 tokenizer_config.json: {path}")

    config = load_json_file(config_path)
    weight_files = sorted(path.glob("model*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"Base 模型目录中没有 safetensors 权重: {path}")
    return {
        "config": config,
        "weight_files": weight_files,
        "config_path": config_path,
        "tokenizer_config_path": tokenizer_config_path,
    }


def validate_adapter_dir(path: Path) -> Dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(f"LoRA adapter 目录不存在: {path}")
    config_path = path / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"LoRA adapter 缺少 adapter_config.json: {path}\n"
            "请使用训练完成后 output_dir 根目录，而不是 train_logs 目录。"
        )
    weight_candidates = [
        path / "adapter_model.safetensors",
        path / "adapter_model.bin",
    ]
    weight_path = next((item for item in weight_candidates if item.is_file()), None)
    if weight_path is None:
        raise FileNotFoundError(f"LoRA adapter 缺少权重文件: {path}")

    config = load_json_file(config_path)
    if str(config.get("peft_type", "")).upper() != "LORA":
        raise ValueError(
            f"adapter_config.json 的 peft_type 不是 LORA: "
            f"{config.get('peft_type')!r}"
        )
    rank = config.get("r")
    if not isinstance(rank, int) or rank <= 0:
        raise ValueError(f"adapter_config.json 中的 r 非法: {rank!r}")

    ranks = [rank]
    rank_pattern = config.get("rank_pattern") or {}
    if not isinstance(rank_pattern, dict):
        raise TypeError("adapter_config.json 中的 rank_pattern 必须是对象")
    for module_name, module_rank in rank_pattern.items():
        if not isinstance(module_rank, int) or module_rank <= 0:
            raise ValueError(
                f"rank_pattern[{module_name!r}] 的 rank 非法: {module_rank!r}"
            )
        ranks.append(module_rank)

    return {
        "config": config,
        "config_path": config_path,
        "weight_path": weight_path,
        "max_rank": max(ranks),
    }


def validate_rows(rows: Any, context: str) -> List[str]:
    """返回结构错误列表；调用方决定是立即报错还是作为模型输出质量记录。"""
    errors: List[str] = []
    if not isinstance(rows, list):
        return [f"{context}: 根节点必须是 JSON 数组"]
    if not rows:
        errors.append(f"{context}: JSON 数组为空")

    for row_index, row in enumerate(rows):
        prefix = f"{context}[{row_index}]"
        if not isinstance(row, dict):
            errors.append(f"{prefix}: 每个元素必须是对象，实际为 {type(row).__name__}")
            continue

        for field_name, field_value in row.items():
            if not isinstance(field_value, str):
                errors.append(
                    f"{prefix}.{field_name}: 字段值必须是字符串，"
                    f"实际为 {type(field_value).__name__}"
                )

        level = row.get("步骤层级")
        if not isinstance(level, str) or level not in ALLOWED_LEVELS:
            errors.append(f"{prefix}.步骤层级: 非法取值 {level!r}")
            continue

        missing_fields = [
            field_name
            for field_name in REQUIRED_FIELDS[level]
            if field_name not in row
        ]
        if missing_fields:
            errors.append(f"{prefix}: 缺少必选字段 {missing_fields}")

        for field_name, allowed_values in ENUM_FIELDS.items():
            if field_name not in row:
                continue
            field_value = row[field_name]
            if isinstance(field_value, str) and field_value != "" and field_value not in allowed_values:
                errors.append(
                    f"{prefix}.{field_name}: 非法取值 {field_value!r}，"
                    f"允许值为 {sorted(allowed_values)}"
                )
    return errors


def validate_messages(
    record: Dict[str, Any],
    *,
    path: Path,
    line_number: int,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    sample_id = record.get("id")
    messages = record.get("messages")
    location = f"{path}:{line_number}（ID={sample_id!r}）"
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"{location}: messages 必须至少包含 prompt 和 assistant")

    normalized: List[Dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError(f"{location}: messages[{index}] 必须是对象")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"{location}: messages[{index}].role 非法: {role!r}")
        if not isinstance(content, str):
            raise TypeError(f"{location}: messages[{index}].content 必须是字符串")
        normalized.append({"role": role, "content": content})

    if normalized[-1]["role"] != "assistant":
        raise ValueError(f"{location}: 最后一条消息必须是 assistant ground truth")
    if any(message["role"] == "assistant" for message in normalized[:-1]):
        raise ValueError(f"{location}: 当前单轮评测数据不允许 prompt 中提前出现 assistant")
    if not any(message["role"] == "user" for message in normalized[:-1]):
        raise ValueError(f"{location}: prompt 中缺少 user 消息")
    for index, message in enumerate(normalized):
        if message["role"] == "system" and index != 0:
            raise ValueError(f"{location}: system 消息只能出现在首位")

    try:
        ground_truth = json.loads(normalized[-1]["content"])
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{location}: assistant ground truth 不是合法 JSON，"
            f"列 {exc.colno}: {exc.msg}"
        ) from exc
    schema_errors = validate_rows(ground_truth, f"{location}.assistant")
    if schema_errors:
        preview = "\n".join(f"  - {item}" for item in schema_errors[:10])
        raise ValueError(f"{location}: ground truth 结构不合法：\n{preview}")
    return normalized, ground_truth


def load_evaluation_data(
    path: Path,
    max_samples: Optional[int],
    seed: int,
) -> Tuple[List[Dict[str, Any]], int]:
    if not path.is_file():
        raise FileNotFoundError(f"评测数据不存在: {path}")

    records: List[Dict[str, Any]] = []
    seen_ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: JSONL 解析失败，"
                    f"列 {exc.colno}: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise TypeError(f"{path}:{line_number}: 每条 JSONL 记录必须是对象")
            if "id" not in record or record["id"] is None:
                raise ValueError(f"{path}:{line_number}: 缺少非空 id")

            id_key = json.dumps(
                record["id"], ensure_ascii=False, sort_keys=True
            )
            if id_key in seen_ids:
                raise ValueError(f"{path}:{line_number}: 重复 id={record['id']!r}")
            seen_ids.add(id_key)

            messages, ground_truth = validate_messages(
                record,
                path=path,
                line_number=line_number,
            )
            normalized_record = dict(record)
            normalized_record["messages"] = messages
            normalized_record["_gt_rows"] = ground_truth
            normalized_record["_line_number"] = line_number
            records.append(normalized_record)

    if not records:
        raise ValueError(f"评测数据没有有效记录: {path}")
    total_records = len(records)

    if max_samples is not None and max_samples < total_records:
        rng = random.Random(seed)
        selected_indices = sorted(rng.sample(range(total_records), max_samples))
        records = [records[index] for index in selected_indices]
        print(
            f"[WARN] 冒烟模式：使用固定 seed={seed} 抽取 "
            f"{len(records)}/{total_records} 条记录。"
        )
    print(f"[INFO] 评测数据校验通过：选择 {len(records)}/{total_records} 条。")
    return records, total_records


def _normalize_token_ids(value: Any) -> List[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("tokenizer 意外返回多个 batch")
        value = value[0]
    return [int(token_id) for token_id in value]


def build_generation_inputs(
    tokenizer: Any,
    records: Sequence[Dict[str, Any]],
    *,
    max_model_len: int,
    max_new_tokens: int,
) -> Tuple[List[Dict[str, List[int]]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    使用与训练相同的两阶段模板编码，并把 token IDs 直接交给 vLLM。

    这样既避免 Transformers 与 vLLM 再次分词产生差异，也能在推理前严格
    检查 prompt + 最大生成长度是否超过上下文。
    """
    generation_inputs: List[Dict[str, List[int]]] = []
    metadatas: List[Dict[str, Any]] = []
    prompt_lengths: List[int] = []

    for record in records:
        messages = record["messages"]
        prompt_messages = messages[:-1]
        rendered_prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = tokenizer(
            rendered_prompt,
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        prompt_token_ids = _normalize_token_ids(encoded["input_ids"])
        if not prompt_token_ids:
            raise ValueError(f"样本 ID={record['id']!r} 的 prompt token 为空")

        required_length = len(prompt_token_ids) + max_new_tokens
        if required_length > max_model_len:
            raise ValueError(
                f"样本 ID={record['id']!r} 的 prompt={len(prompt_token_ids)} token，"
                f"加上 max_new_tokens={max_new_tokens} 后为 {required_length}，"
                f"超过 max_model_len={max_model_len}。脚本不会静默截断。"
            )

        generation_inputs.append({"prompt_token_ids": prompt_token_ids})
        prompt_lengths.append(len(prompt_token_ids))
        metadatas.append(
            {
                "id": record["id"],
                "source": record.get("source", ""),
                "line_number": record["_line_number"],
                "gt_rows": record["_gt_rows"],
                "expected_prompt_tokens": len(prompt_token_ids),
            }
        )

    summary = length_summary(prompt_lengths)
    print(
        "[INFO] Prompt token: "
        f"min={summary['min']} mean={summary['mean']} "
        f"P95={summary['p95']} max={summary['max']}；"
        f"最大预留总长度={summary['max'] + max_new_tokens}/{max_model_len}"
    )
    return generation_inputs, metadatas, summary


def _response_content(response: Any) -> str:
    """兼容本地 vLLM 文本及少量 API 风格 payload。"""
    if isinstance(response, dict):
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
        for key in ("content", "text"):
            if isinstance(response.get(key), str):
                return response[key]
    return str(response or "")


def _remove_recovery_wrappers(text: str) -> str:
    """仅用于诊断性恢复；是否为纯 JSON 仍根据原始输出单独记录。"""
    text = text.strip()
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    closing = re.search(r"</think>", text, flags=re.IGNORECASE)
    if closing:
        text = text[closing.end() :]
    text = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def _iter_json_values(text: str) -> Iterable[Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        yield value


def _rows_from_value(value: Any) -> Tuple[Optional[List[Any]], Optional[str]]:
    if isinstance(value, list):
        return value, "array"
    if isinstance(value, dict):
        for key in (STEP_LIST_KEY, "rows", "steps"):
            candidate = value.get(key)
            if isinstance(candidate, list):
                return candidate, f"object_wrapper:{key}"
    return None, None


def parse_output_with_diagnostics(response: Any) -> Dict[str, Any]:
    """
    解析模型输出并保留格式诊断。

    指标可以对可恢复的字典行评分，但 pure JSON array 合规率会单独统计，
    因此 Markdown、解释文字或对象包裹不会被悄悄掩盖。
    """
    raw_text = _response_content(response)
    stripped_raw = raw_text.strip()
    exact_value: Any = None
    exact_json_ok = False
    exact_error: Optional[str] = None
    try:
        exact_value = json.loads(stripped_raw)
        exact_json_ok = True
    except json.JSONDecodeError as exc:
        exact_error = f"原始输出不是完整 JSON：列 {exc.colno}: {exc.msg}"

    rows: Optional[List[Any]] = None
    parse_mode = "failed"
    strict_json_array = False

    if exact_json_ok:
        rows, wrapper_kind = _rows_from_value(exact_value)
        if rows is not None:
            if wrapper_kind == "array":
                parse_mode = "strict_json_array"
                strict_json_array = True
            else:
                parse_mode = wrapper_kind or "json_object_wrapper"

    if rows is None:
        recovery_text = _remove_recovery_wrappers(raw_text)
        for value in _iter_json_values(recovery_text):
            candidate, wrapper_kind = _rows_from_value(value)
            if candidate is not None:
                rows = candidate
                parse_mode = f"recovered_{wrapper_kind}"
                break

    errors: List[str] = []
    if exact_error is not None:
        errors.append(exact_error)
    if rows is None:
        errors.append("没有找到 JSON 数组或受支持的数组包装对象")
        return {
            "rows": [],
            "parse_success": False,
            "strict_json_array": False,
            "schema_valid": False,
            "parse_mode": parse_mode,
            "errors": errors,
        }

    all_rows_are_objects = all(isinstance(row, dict) for row in rows)
    schema_errors = validate_rows(rows, "prediction")
    errors.extend(schema_errors)
    return {
        "rows": rows if all_rows_are_objects else [],
        "parse_success": all_rows_are_objects,
        "strict_json_array": strict_json_array,
        "schema_valid": all_rows_are_objects and not schema_errors,
        "parse_mode": parse_mode,
        "errors": errors,
    }


def parse_output(response: Any) -> List[Dict[str, Any]]:
    """保留旧调用方式：只返回可安全交给指标函数的字典行。"""
    return parse_output_with_diagnostics(response)["rows"]


def run_vllm_eval(
    llm: Any,
    sampling_params: Any,
    generation_inputs: Sequence[Dict[str, List[int]]],
    metadatas: Sequence[Dict[str, Any]],
    adapter_path: Optional[Path],
    label: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from vllm.lora.request import LoRARequest

    print(f"\n{'=' * 72}")
    print(f"[{label}] vLLM 离线批量推理：{len(generation_inputs)} 条")
    print(f"{'=' * 72}")

    # 每次传入新的列表，确保 Base 与 LoRA 使用完全相同且未被前一次调用修改的 IDs。
    request_inputs = [
        {"prompt_token_ids": list(item["prompt_token_ids"])}
        for item in generation_inputs
    ]
    start = time.time()
    if adapter_path is None:
        outputs = llm.generate(request_inputs, sampling_params)
    else:
        lora_request = LoRARequest(
            "ao_testcase_eval_adapter",
            1,
            str(adapter_path),
        )
        outputs = llm.generate(
            request_inputs,
            sampling_params,
            lora_request=lora_request,
        )
    elapsed = time.time() - start

    if len(outputs) != len(metadatas):
        raise RuntimeError(
            f"[{label}] vLLM 返回 {len(outputs)} 条，"
            f"但输入为 {len(metadatas)} 条"
        )

    results: List[Dict[str, Any]] = []
    for output, metadata in zip(outputs, metadatas):
        if not getattr(output, "outputs", None):
            raise RuntimeError(f"[{label}] ID={metadata['id']!r} 没有生成候选")
        completion = output.outputs[0]
        raw_output = str(getattr(completion, "text", "") or "")
        diagnostics = parse_output_with_diagnostics(raw_output)
        predicted_rows = diagnostics.pop("rows")
        stop_reason = getattr(completion, "stop_reason", None)
        if stop_reason is not None and not isinstance(stop_reason, (str, int, float, bool)):
            stop_reason = str(stop_reason)

        actual_prompt_ids = getattr(output, "prompt_token_ids", None)
        prompt_token_count = (
            len(actual_prompt_ids)
            if actual_prompt_ids is not None
            else metadata["expected_prompt_tokens"]
        )
        output_token_ids = getattr(completion, "token_ids", None)
        output_token_count = len(output_token_ids) if output_token_ids is not None else None

        results.append(
            {
                "id": metadata["id"],
                "source": metadata["source"],
                "data_line_number": metadata["line_number"],
                "gt_rows": metadata["gt_rows"],
                "pred_rows": predicted_rows,
                "raw_output": raw_output,
                "parse": diagnostics,
                "finish_reason": getattr(completion, "finish_reason", None),
                "stop_reason": stop_reason,
                "prompt_token_count": prompt_token_count,
                "output_token_count": output_token_count,
            }
        )

    runtime = {
        "elapsed_seconds": round(elapsed, 4),
        "samples_per_second": round(
            len(generation_inputs) / max(elapsed, 1e-9),
            6,
        ),
    }
    diagnostics_summary = summarize_generation(results)
    print(
        f"[{label}] 完成：{runtime['elapsed_seconds']:.2f}s，"
        f"{runtime['samples_per_second']:.4f} samples/s；"
        f"纯 JSON={diagnostics_summary['strict_json_array_count']}/"
        f"{len(results)}，schema 合法={diagnostics_summary['schema_valid_count']}/"
        f"{len(results)}，长度截断={diagnostics_summary['length_truncated_count']}"
    )
    return results, runtime


def create_metric(
    metric_version: str,
    include_device_fields: bool,
) -> Any:
    common_kwargs = {
        "alpha": 0.4,
        "skip_device_fields": not include_device_fields,
        "verbose": False,
    }
    if metric_version == "v2":
        return UseCaseTableMetricV2(**common_kwargs)
    if metric_version == "legacy":
        return LegacyUseCaseTableMetric(**common_kwargs)
    raise ValueError(f"未知 metric_version: {metric_version}")


def compute_metrics(
    results: List[Dict[str, Any]],
    *,
    label: str,
    metric_version: str,
    include_device_fields: bool,
) -> Dict[str, Any]:
    metric = create_metric(metric_version, include_device_fields)
    per_sample: List[Dict[str, Any]] = []
    all_field_scores: Dict[str, List[float]] = defaultdict(list)

    for result in results:
        score = metric.compute(result["gt_rows"], result["pred_rows"])
        score["id"] = result["id"]
        score["source"] = result["source"]
        result["scores"] = score
        per_sample.append(score)
        for field_name, field_score in score.get("field_scores", {}).items():
            all_field_scores[field_name].append(float(field_score))

    sample_count = len(per_sample)
    field_avg = {
        field_name: round(sum(scores) / len(scores), 4)
        for field_name, scores in all_field_scores.items()
    }
    batch = {
        "metric_version": metric_version,
        "include_device_fields": include_device_fields,
        "num_samples": sample_count,
        "avg_overall": round(
            sum(item["overall_score"] for item in per_sample) / sample_count,
            4,
        ),
        "avg_structure": round(
            sum(item["structure_score"] for item in per_sample) / sample_count,
            4,
        ),
        "avg_content": round(
            sum(item["content_score"] for item in per_sample) / sample_count,
            4,
        ),
        "field_avg": field_avg,
        "per_sample": per_sample,
    }
    print(
        f"\n--- [{label}] metric={metric_version} ---\n"
        f"overall={batch['avg_overall']:.4f} | "
        f"structure={batch['avg_structure']:.4f} | "
        f"content={batch['avg_content']:.4f}"
    )
    return batch


def _percentile(values: Sequence[int], percentile: float) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * percentile)))
    return int(ordered[index])


def length_summary(values: Sequence[int]) -> Dict[str, Any]:
    if not values:
        return {"min": 0, "mean": 0.0, "p95": 0, "max": 0}
    return {
        "min": int(min(values)),
        "mean": round(sum(values) / len(values), 2),
        "p95": _percentile(values, 0.95),
        "max": int(max(values)),
    }


def summarize_generation(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    sample_count = len(results)
    parse_success_count = sum(
        bool(item["parse"]["parse_success"]) for item in results
    )
    strict_json_count = sum(
        bool(item["parse"]["strict_json_array"]) for item in results
    )
    schema_valid_count = sum(
        bool(item["parse"]["schema_valid"]) for item in results
    )
    truncated_count = sum(
        item.get("finish_reason") == "length" for item in results
    )
    recovered_count = sum(
        item["parse"]["parse_mode"].startswith("recovered_") for item in results
    )
    finish_reasons = Counter(
        str(item.get("finish_reason") or "none") for item in results
    )
    output_lengths = [
        int(item["output_token_count"])
        for item in results
        if item.get("output_token_count") is not None
    ]
    prompt_lengths = [
        int(item["prompt_token_count"])
        for item in results
        if item.get("prompt_token_count") is not None
    ]

    def rate(count: int) -> float:
        return round(count / sample_count, 6) if sample_count else 0.0

    return {
        "parse_success_count": parse_success_count,
        "parse_success_rate": rate(parse_success_count),
        "strict_json_array_count": strict_json_count,
        "strict_json_array_rate": rate(strict_json_count),
        "schema_valid_count": schema_valid_count,
        "schema_valid_rate": rate(schema_valid_count),
        "recovered_json_count": recovered_count,
        "recovered_json_rate": rate(recovered_count),
        "length_truncated_count": truncated_count,
        "length_truncated_rate": rate(truncated_count),
        "finish_reason_counts": dict(sorted(finish_reasons.items())),
        "prompt_tokens": length_summary(prompt_lengths),
        "output_tokens": length_summary(output_lengths),
    }


def aggregate_metrics(batch: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in batch.items()
        if key != "per_sample"
    }


def build_paired_comparison(
    base_results: Sequence[Dict[str, Any]],
    lora_results: Sequence[Dict[str, Any]],
    base_metrics: Dict[str, Any],
    lora_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    if len(base_results) != len(lora_results):
        raise RuntimeError("Base 与 LoRA 的结果数量不同，无法做成对比较")

    deltas: List[float] = []
    improved = 0
    degraded = 0
    tied = 0
    for base_item, lora_item in zip(base_results, lora_results):
        if base_item["id"] != lora_item["id"]:
            raise RuntimeError(
                f"Base/LoRA 样本顺序不一致: "
                f"{base_item['id']!r} != {lora_item['id']!r}"
            )
        delta = (
            float(lora_item["scores"]["overall_score"])
            - float(base_item["scores"]["overall_score"])
        )
        deltas.append(delta)
        if delta > 0:
            improved += 1
        elif delta < 0:
            degraded += 1
        else:
            tied += 1

    base_overall = float(base_metrics["avg_overall"])
    lora_overall = float(lora_metrics["avg_overall"])
    relative_improvement = (
        round((lora_overall - base_overall) / base_overall * 100, 2)
        if base_overall != 0
        else None
    )
    return {
        "overall_absolute_delta": round(lora_overall - base_overall, 4),
        "overall_relative_improvement_pct": relative_improvement,
        "mean_paired_overall_delta": round(
            sum(deltas) / len(deltas),
            4,
        ) if deltas else 0.0,
        "lora_better_samples": improved,
        "lora_worse_samples": degraded,
        "tied_samples": tied,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for package_name in ("vllm", "torch", "transformers"):
        try:
            versions[package_name] = version(package_name)
        except PackageNotFoundError:
            versions[package_name] = "not-installed"
    return versions


def gpu_summary(torch_module: Any) -> Dict[str, Any]:
    if not torch_module.cuda.is_available():
        raise RuntimeError("PyTorch 未检测到 CUDA GPU，不能运行 vLLM GPU 评测")
    device_index = torch_module.cuda.current_device()
    properties = torch_module.cuda.get_device_properties(device_index)
    major, minor = torch_module.cuda.get_device_capability(device_index)
    return {
        "device_index": device_index,
        "device_name": properties.name,
        "total_memory_gib": round(properties.total_memory / (1024**3), 2),
        "compute_capability": f"{major}.{minor}",
        "torch_cuda": torch_module.version.cuda,
    }


def write_json(path: Path, payload: Any) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary_path, path)


def main() -> None:
    args = parse_args()
    validate_args(args)

    base_model_dir = resolve_project_path(args.base_model)
    adapter_dir = resolve_project_path(args.adapter)
    data_file = resolve_project_path(args.data_file)
    output_dir = resolve_project_path(args.output_dir)

    base_info = validate_base_model_dir(base_model_dir)
    adapter_info = validate_adapter_dir(adapter_dir)
    records, total_records = load_evaluation_data(
        data_file,
        args.max_samples,
        args.seed,
    )
    prepare_output_dir(output_dir, args.overwrite_output_dir)

    print("[INFO] 严格离线模式已启用：HF Hub 与 vLLM 使用统计均已禁用。")
    print(f"[INFO] project_root: {PROJECT_ROOT}")
    print(f"[INFO] base_model: {base_model_dir}")
    print(f"[INFO] adapter: {adapter_dir}")
    print(f"[INFO] data_file: {data_file}")
    print(f"[INFO] output_dir: {output_dir}")
    print(f"[INFO] LoRA max rank: {adapter_info['max_rank']}")

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    cuda_info = gpu_summary(torch)
    print(
        f"[INFO] GPU: {cuda_info['device_name']}，"
        f"{cuda_info['total_memory_gib']} GiB，"
        f"capability={cuda_info['compute_capability']}，"
        f"PyTorch CUDA={cuda_info['torch_cuda']}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model_dir),
        local_files_only=True,
        trust_remote_code=False,
    )
    generation_inputs, metadatas, prompt_length_summary = build_generation_inputs(
        tokenizer,
        records,
        max_model_len=args.max_model_len,
        max_new_tokens=args.max_new_tokens,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    adapter_config = adapter_info["config"]
    base_config = base_info["config"]
    manifest_path = output_dir / "evaluation_manifest.json"
    model_weight_bytes = sum(
        path.stat().st_size for path in base_info["weight_files"]
    )
    manifest: Dict[str, Any] = {
        "status": "initializing",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args),
        "resolved_paths": {
            "base_model": str(base_model_dir),
            "adapter": str(adapter_dir),
            "data_file": str(data_file),
            "output_dir": str(output_dir),
        },
        "versions": {
            **package_versions(),
            "python": platform.python_version(),
        },
        "strict_offline_environment": dict(STRICT_OFFLINE_ENV),
        "cuda": cuda_info,
        "base_model": {
            "architectures": base_config.get("architectures"),
            "model_type": base_config.get("model_type"),
            "config_sha256": sha256_file(base_info["config_path"]),
            "tokenizer_config_sha256": sha256_file(
                base_info["tokenizer_config_path"]
            ),
            "weight_file_count": len(base_info["weight_files"]),
            "weight_bytes": model_weight_bytes,
            "language_model_only": True,
        },
        "adapter": {
            "peft_type": adapter_config.get("peft_type"),
            "base_model_name_or_path": adapter_config.get(
                "base_model_name_or_path"
            ),
            "rank": adapter_config.get("r"),
            "max_rank": adapter_info["max_rank"],
            "target_modules": adapter_config.get("target_modules"),
            "config_sha256": sha256_file(adapter_info["config_path"]),
            "weight_file": adapter_info["weight_path"].name,
            "weight_bytes": adapter_info["weight_path"].stat().st_size,
            "weight_sha256": sha256_file(adapter_info["weight_path"]),
        },
        "dataset": {
            "selected_samples": len(records),
            "total_samples": total_records,
            "selected_ids": [record["id"] for record in records],
            "file_sha256": sha256_file(data_file),
        },
        "generation": {
            "enable_thinking": False,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "max_model_len": args.max_model_len,
            "seed": args.seed,
            "dtype": args.dtype,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory,
            "generation_config": "vllm",
            "guided_or_structured_decoding": False,
            "prompt_tokens": prompt_length_summary,
        },
        "metric": {
            "version": args.metric_version,
            "alpha": 0.4,
            "include_device_fields": args.include_device_fields,
        },
    }
    write_json(manifest_path, manifest)

    try:
        print("[INFO] 初始化单个 vLLM Base 引擎，并启用动态 LoRA……")
        llm = LLM(
            model=str(base_model_dir),
            tokenizer=str(base_model_dir),
            enable_lora=True,
            max_lora_rank=adapter_info["max_rank"],
            max_loras=1,
            gpu_memory_utilization=args.gpu_memory,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
            seed=args.seed,
            trust_remote_code=False,
            generation_config="vllm",
            language_model_only=True,
        )

        base_results, base_runtime = run_vllm_eval(
            llm,
            sampling_params,
            generation_inputs,
            metadatas,
            None,
            "BASE",
        )
        base_metrics = compute_metrics(
            base_results,
            label="BASE",
            metric_version=args.metric_version,
            include_device_fields=args.include_device_fields,
        )
        base_report = {
            **aggregate_metrics(base_metrics),
            "generation": summarize_generation(base_results),
            "runtime": base_runtime,
        }
        write_jsonl(output_dir / "base_predictions.jsonl", base_results)
        write_json(output_dir / "base_metrics.json", base_report)
        print("[INFO] Base 逐样本结果和指标已落盘。")

        lora_results, lora_runtime = run_vllm_eval(
            llm,
            sampling_params,
            generation_inputs,
            metadatas,
            adapter_dir,
            "LORA",
        )
        lora_metrics = compute_metrics(
            lora_results,
            label="LORA",
            metric_version=args.metric_version,
            include_device_fields=args.include_device_fields,
        )
        lora_report = {
            **aggregate_metrics(lora_metrics),
            "generation": summarize_generation(lora_results),
            "runtime": lora_runtime,
        }
        write_jsonl(output_dir / "lora_predictions.jsonl", lora_results)
        write_json(output_dir / "lora_metrics.json", lora_report)
        print("[INFO] LoRA 逐样本结果和指标已落盘。")

        comparison = build_paired_comparison(
            base_results,
            lora_results,
            base_metrics,
            lora_metrics,
        )
        summary = {
            "metric_version": args.metric_version,
            "num_samples": len(records),
            "base": base_report,
            "lora": lora_report,
            "comparison": comparison,
            # 保留旧汇总中的便捷字段名。
            "improvement_pct": comparison[
                "overall_relative_improvement_pct"
            ],
        }
        summary_path = output_dir / "vllm_compare_summary.json"
        write_json(summary_path, summary)

        manifest["status"] = "completed"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["output_files"] = [
            "evaluation_manifest.json",
            "base_predictions.jsonl",
            "base_metrics.json",
            "lora_predictions.jsonl",
            "lora_metrics.json",
            "vllm_compare_summary.json",
        ]
        write_json(manifest_path, manifest)

        print("\n[INFO] Base/LoRA 成对评测完成。")
        print(f"[INFO] 汇总结果: {summary_path}")
        print(
            "[INFO] overall delta="
            f"{comparison['overall_absolute_delta']:+.4f}，"
            "relative="
            f"{comparison['overall_relative_improvement_pct']}%"
        )
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
