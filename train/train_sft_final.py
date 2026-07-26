#!/usr/bin/env python3
"""
使用 PEFT/LoRA 对本地 Qwen3.5-9B 进行 AO 指令到结构化测试用例的 SFT。

训练约定：
    1. 使用 AutoModelForCausalLM，仅加载语言模型，不训练视觉编码器。
    2. 输入包含 system、user 和 assistant；损失只计算最终 assistant 答案。
    3. 数据超过 max_seq_length 时立即报错，不进行静默截断。
    4. 相对路径统一相对于项目根目录 ao-testcase-generation 解析。
    5. 强制严格离线：禁止 Hugging Face Hub、遥测、W&B 和模型上传。

推荐从项目根目录运行：
    python train/train_sft_final.py \
        --model_dir ../qwen3_5_9b_deploy/models/Qwen3.5-9B \
        --train_file data/train_sft.jsonl \
        --eval_file data/eval_sft.jsonl \
        --output_dir outputs/qwen35_lora
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from datetime import datetime, timezone
from importlib.util import find_spec
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, List, Sequence

# ============================================================
# 严格离线模式
# 必须在导入 torch / datasets / peft / transformers 之前设置。
# 使用直接赋值而不是 setdefault，避免外部环境中的 "0" 覆盖安全配置。
# ============================================================
STRICT_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "HF_HUB_DISABLE_UPDATE_CHECK": "1",
    "HF_HUB_DISABLE_XET": "1",
    "DO_NOT_TRACK": "1",
    "DISABLE_TELEMETRY": "1",
    "WANDB_DISABLED": "true",
    "WANDB_MODE": "disabled",
}
for environment_name, environment_value in STRICT_OFFLINE_ENV.items():
    os.environ[environment_name] = environment_value

# 防止当前训练进程意外继承外部平台凭据。
for credential_name in (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "WANDB_API_KEY",
):
    os.environ.pop(credential_name, None)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def _normalize_omp_num_threads() -> None:
    """在 libgomp 随 PyTorch 加载前修复继承到的非法线程数。"""
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
            "已在导入 PyTorch 前修正为 1。"
        )


_normalize_omp_num_threads()

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_DIR = "../qwen3_5_9b_deploy/models/Qwen3.5-9B"
DEFAULT_TRAIN_FILE = "data/train_sft.jsonl"
DEFAULT_EVAL_FILE = "data/eval_sft.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/qwen35_lora"

# Qwen3.5 是混合架构：既有标准全注意力层，也有 Gated DeltaNet 层。
# 前四项覆盖全注意力层，接下来的五项覆盖 Gated DeltaNet，最后三项覆盖所有 MLP。
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3.5-9B LoRA SFT for AO structured testcase generation"
    )
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--train_file", type=str, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--eval_file", type=str, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache_dir", type=str, default=None)

    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument(
        "--eval_accumulation_steps",
        type=int,
        default=8,
        help="分批把验证结果移到 CPU，降低验证阶段的显存峰值",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=8192,
        help="当前数据最大为 7780 token；超长样本会报错，不会被静默截断",
    )
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_steps", type=int, default=-1)

    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--gradient_checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
        help="启用梯度检查点（默认启用）",
    )
    checkpoint_group.add_argument(
        "--no_gradient_checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="关闭梯度检查点",
    )
    parser.set_defaults(gradient_checkpointing=True)

    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="允许在非空输出目录中开始一次新训练；请谨慎使用",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="仅取前 N 条训练样本，适合启动前冒烟测试",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="仅取前 N 条验证样本，适合启动前冒烟测试",
    )

    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def validate_args(args: argparse.Namespace) -> None:
    positive_values = {
        "max_seq_length": args.max_seq_length,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "logging_steps": args.logging_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"--{name} 必须大于 0，当前值为 {value}")

    if args.num_train_epochs <= 0 and args.max_steps <= 0:
        raise ValueError("--num_train_epochs 必须大于 0；使用 --max_steps 时后者必须大于 0")
    if not 0.0 <= args.lora_dropout < 1.0:
        raise ValueError("--lora_dropout 必须位于 [0, 1) 区间")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup_ratio 必须位于 [0, 1) 区间")
    if args.save_steps % args.eval_steps != 0:
        raise ValueError(
            "启用 load_best_model_at_end 时，--save_steps 必须是 --eval_steps 的整数倍"
        )
    for name in ("max_train_samples", "max_eval_samples"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name} 必须大于 0")


def resolve_project_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def package_versions() -> Dict[str, str]:
    result: Dict[str, str] = {}
    for package in ("torch", "transformers", "datasets", "peft", "accelerate"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def verify_strict_offline_mode() -> Dict[str, str]:
    mismatches = {
        name: {
            "expected": expected,
            "actual": os.environ.get(name),
        }
        for name, expected in STRICT_OFFLINE_ENV.items()
        if os.environ.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(
            "严格离线环境变量被修改："
            f"{json.dumps(mismatches, ensure_ascii=False)}"
        )

    leaked_credentials = [
        name
        for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "WANDB_API_KEY")
        if os.environ.get(name)
    ]
    if leaked_credentials:
        raise RuntimeError(
            f"训练进程中仍存在外部平台凭据：{leaked_credentials}"
        )

    print(
        "[INFO] 严格离线模式已启用：Hugging Face Hub、遥测、Xet、"
        "更新检查和 W&B 均已禁用。"
    )
    return dict(STRICT_OFFLINE_ENV)


def optional_kernel_status() -> Dict[str, bool]:
    status = {
        "causal_conv1d": find_spec("causal_conv1d") is not None,
        "flash_linear_attention": find_spec("fla") is not None,
    }
    if all(status.values()):
        print("[INFO] Qwen3.5 Gated DeltaNet 快速训练内核已安装。")
    else:
        missing = [name for name, available in status.items() if not available]
        print(
            "[WARN] 缺少 Qwen3.5 Gated DeltaNet 可选快速内核："
            f"{missing}。Transformers 会回退到 PyTorch 实现，训练可能明显变慢；"
            "不要在未确认 CUDA/PyTorch 兼容性的情况下盲目安装二进制包。"
        )
    return status


class JsonlStatusLoggerCallback(TrainerCallback):
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(item) for item in value]
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        return str(value)

    def _append(self, payload: Dict[str, Any]) -> None:
        with self.log_file.open("a", encoding="utf-8") as file:
            file.write(
                json.dumps(self._json_safe(payload), ensure_ascii=False) + "\n"
            )

    def on_train_begin(self, args, state, control, **kwargs):
        self._append(
            {
                "event": "train_begin",
                "time": self._utc_now(),
                "global_step": state.global_step,
                "epoch": state.epoch,
            }
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self._append(
                {
                    "event": "log",
                    "time": self._utc_now(),
                    "global_step": state.global_step,
                    "epoch": state.epoch,
                    "logs": logs,
                }
            )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            self._append(
                {
                    "event": "eval",
                    "time": self._utc_now(),
                    "global_step": state.global_step,
                    "epoch": state.epoch,
                    "metrics": metrics,
                }
            )

    def on_train_end(self, args, state, control, **kwargs):
        self._append(
            {
                "event": "train_end",
                "time": self._utc_now(),
                "global_step": state.global_step,
                "epoch": state.epoch,
                "best_metric": state.best_metric,
                "best_model_checkpoint": state.best_model_checkpoint,
            }
        )


def sample_identifier(example: Dict[str, Any]) -> str:
    sample_id = example.get("id")
    return str(sample_id) if sample_id is not None else "<无 id>"


def to_messages(example: Dict[str, Any]) -> List[Dict[str, str]]:
    """提取 messages，兼容 instruction/input/output 格式，并做严格校验。"""
    if "messages" in example and isinstance(example["messages"], list):
        messages = example["messages"]
    else:
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output_text = example.get("output", "")
        user_content = (
            f"{instruction}\n\n{input_text}" if input_text else instruction
        )
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output_text},
        ]

    sample_id = sample_identifier(example)
    if len(messages) < 2:
        raise ValueError(f"样本 {sample_id} 至少需要 user 和 assistant 两条消息")

    normalized: List[Dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError(f"样本 {sample_id} 的 messages[{index}] 不是对象")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(
                f"样本 {sample_id} 的 messages[{index}].role 非法: {role!r}"
            )
        if not isinstance(content, str):
            raise TypeError(
                f"样本 {sample_id} 的 messages[{index}].content 必须是字符串"
            )
        normalized.append({"role": role, "content": content})

    if normalized[-1]["role"] != "assistant":
        raise ValueError(f"样本 {sample_id} 的最后一条消息必须是 assistant")
    if not normalized[-1]["content"].strip():
        raise ValueError(f"样本 {sample_id} 的 assistant 答案为空")
    return normalized


def render_chat_ids(
    tokenizer,
    messages: Sequence[Dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> List[int]:
    # 与 count_dataset.py 使用完全相同的两阶段编码方式，确保其长度报告仍有效。
    rendered_text = tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )
    token_ids = tokenizer(
        rendered_text,
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )["input_ids"]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ValueError("聊天模板意外返回了多个 batch")
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def build_preprocess_fn(tokenizer, max_seq_length: int):
    """
    构建 assistant-only loss：
      - input_ids 保留完整 system + user + assistant；
      - prompt 对应的 labels 设为 -100；
      - 仅 assistant JSON 内容和对话结束标记参与交叉熵损失。
    """

    def preprocess(example: Dict[str, Any]) -> Dict[str, List[int]]:
        messages = to_messages(example)
        full_ids = render_chat_ids(
            tokenizer,
            messages,
            add_generation_prompt=False,
        )
        prompt_ids = render_chat_ids(
            tokenizer,
            messages[:-1],
            add_generation_prompt=True,
        )
        sample_id = sample_identifier(example)

        if len(full_ids) > max_seq_length:
            raise ValueError(
                f"样本 {sample_id} 的真实长度为 {len(full_ids)}，超过 "
                f"--max_seq_length={max_seq_length}。请提高长度，或在确认后剔除/拆分该样本；"
                "脚本不会静默截断 assistant 答案。"
            )
        if len(prompt_ids) >= len(full_ids):
            raise ValueError(
                f"样本 {sample_id} 没有可监督的 assistant token："
                f"prompt={len(prompt_ids)}, total={len(full_ids)}"
            )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                f"样本 {sample_id} 的完整对话 token 不是 prompt token 的前缀，"
                "无法安全确定 assistant loss 边界。请检查 chat_template。"
            )

        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    return preprocess


def build_data_collator(tokenizer):
    def data_collator(features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        input_ids = [
            torch.tensor(feature["input_ids"], dtype=torch.long)
            for feature in features
        ]
        attention_mask = [
            torch.tensor(feature["attention_mask"], dtype=torch.long)
            for feature in features
        ]
        labels = [
            torch.tensor(feature["labels"], dtype=torch.long)
            for feature in features
        ]
        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=tokenizer.pad_token_id,
            ),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                attention_mask,
                batch_first=True,
                padding_value=0,
            ),
            "labels": torch.nn.utils.rnn.pad_sequence(
                labels,
                batch_first=True,
                padding_value=-100,
            ),
        }

    return data_collator


def summarize_tokenized_dataset(dataset, name: str) -> Dict[str, Any]:
    if len(dataset) == 0:
        raise ValueError(f"{name} 数据集为空")

    lengths: List[int] = []
    supervised_lengths: List[int] = []
    for row in dataset:
        labels = row["labels"]
        sequence_length = len(row["input_ids"])
        supervised_length = sum(label != -100 for label in labels)
        if supervised_length == 0:
            raise ValueError(f"{name} 中存在没有监督 token 的样本")
        lengths.append(sequence_length)
        supervised_lengths.append(supervised_length)

    summary = {
        "samples": len(lengths),
        "sequence_min": min(lengths),
        "sequence_mean": round(sum(lengths) / len(lengths), 2),
        "sequence_max": max(lengths),
        "supervised_min": min(supervised_lengths),
        "supervised_mean": round(
            sum(supervised_lengths) / len(supervised_lengths), 2
        ),
        "supervised_max": max(supervised_lengths),
        "supervised_token_ratio": round(
            sum(supervised_lengths) / sum(lengths), 6
        ),
    }
    print(
        f"[INFO] {name}: samples={summary['samples']}, "
        f"sequence(min/mean/max)="
        f"{summary['sequence_min']}/{summary['sequence_mean']}/{summary['sequence_max']}, "
        f"supervised(min/mean/max)="
        f"{summary['supervised_min']}/{summary['supervised_mean']}/"
        f"{summary['supervised_max']}, "
        f"supervised_ratio={summary['supervised_token_ratio']:.2%}"
    )
    return summary


def get_training_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "未检测到可用 CUDA GPU。Qwen3.5-9B LoRA 训练不应回退到 CPU；"
            "请检查 nvidia-smi、CUDA_VISIBLE_DEVICES 和 PyTorch CUDA 版本。"
        )
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def get_cuda_summary() -> Dict[str, Any]:
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    capability = torch.cuda.get_device_capability(device_index)
    arch = f"sm_{capability[0]}{capability[1]}"
    supported_arches = torch.cuda.get_arch_list()
    summary = {
        "device_index": device_index,
        "device_name": properties.name,
        "total_memory_gib": round(properties.total_memory / (1024**3), 2),
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "required_arch": arch,
        "torch_cuda": torch.version.cuda,
        "torch_supported_arches": supported_arches,
    }
    print(
        f"[INFO] GPU: {summary['device_name']}, "
        f"{summary['total_memory_gib']} GiB, capability={summary['compute_capability']}, "
        f"PyTorch CUDA={summary['torch_cuda']}"
    )
    if supported_arches and arch not in supported_arches:
        print(
            f"[WARN] 当前 GPU 需要 {arch}，但 torch.cuda.get_arch_list() 未明确包含它："
            f"{supported_arches}。请先做 1 step 冒烟测试，确认当前 PyTorch wheel 支持该显卡。"
        )
    return summary


def validate_text_only_causal_model(model) -> None:
    class_name = model.__class__.__name__
    model_type = str(getattr(model.config, "model_type", "unknown"))
    print(f"[INFO] 实际模型类: {class_name}, model_type={model_type}")

    if not class_name.endswith("ForCausalLM"):
        raise TypeError(
            f"期望加载纯语言因果模型，但实际得到 {class_name}。"
            "请勿使用 *ForConditionalGeneration 进行本任务训练。"
        )
    if not model_type.startswith("qwen3_5"):
        raise TypeError(
            f"当前脚本的 LoRA 层清单按 Qwen3.5 设计，但模型类型是 {model_type!r}。"
        )

    vision_markers = (
        "visual.",
        ".visual.",
        "vision_model",
        "vision_tower",
        "image_encoder",
        "patch_embed",
    )
    vision_parameters = [
        name
        for name, _ in model.named_parameters()
        if any(marker in name.lower() for marker in vision_markers)
    ]
    if vision_parameters:
        preview = ", ".join(vision_parameters[:5])
        raise TypeError(
            "模型中仍检测到视觉参数，说明没有加载成纯语言模型。示例参数："
            f"{preview}"
        )
    print("[INFO] 纯语言模型检查通过：参数中未发现视觉编码器。")


def find_lora_target_counts(
    model, target_modules: Sequence[str]
) -> Dict[str, int]:
    counts = {target: 0 for target in target_modules}
    for module_name, _ in model.named_modules():
        leaf_name = module_name.rsplit(".", 1)[-1]
        if leaf_name in counts:
            counts[leaf_name] += 1
    return counts


def validate_lora_targets(model) -> Dict[str, int]:
    counts = find_lora_target_counts(model, LORA_TARGET_MODULES)
    missing = [name for name, count in counts.items() if count == 0]
    if missing:
        raise ValueError(
            "下列 LoRA 目标层在当前模型中不存在："
            f"{missing}。这通常表示 Transformers 版本、模型类型或层命名不匹配。"
        )

    text_config = getattr(model.config, "text_config", model.config)
    layer_types = list(getattr(text_config, "layer_types", []))
    num_layers = int(getattr(text_config, "num_hidden_layers", len(layer_types)))
    if layer_types:
        full_attention_layers = layer_types.count("full_attention")
        linear_attention_layers = layer_types.count("linear_attention")
        expected_counts = {
            "q_proj": full_attention_layers,
            "k_proj": full_attention_layers,
            "v_proj": full_attention_layers,
            "o_proj": full_attention_layers,
            "in_proj_qkv": linear_attention_layers,
            "in_proj_z": linear_attention_layers,
            "in_proj_b": linear_attention_layers,
            "in_proj_a": linear_attention_layers,
            "out_proj": linear_attention_layers,
            "gate_proj": num_layers,
            "up_proj": num_layers,
            "down_proj": num_layers,
        }
        mismatches = {
            name: {"expected": expected_counts[name], "actual": counts[name]}
            for name in LORA_TARGET_MODULES
            if counts[name] != expected_counts[name]
        }
        if mismatches:
            raise ValueError(
                "LoRA 目标层数量与 Qwen3.5 配置不一致："
                f"{json.dumps(mismatches, ensure_ascii=False)}"
            )
    print(f"[INFO] LoRA 目标层匹配数量: {json.dumps(counts, ensure_ascii=False)}")
    return counts


def validate_trainable_parameters(model) -> Dict[str, Any]:
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    unexpected = [name for name in trainable_names if "lora_" not in name.lower()]
    if not trainable_names:
        raise RuntimeError("LoRA 注入后没有任何可训练参数")
    if unexpected:
        raise RuntimeError(
            "检测到 LoRA 之外的参数被设为可训练，示例："
            + ", ".join(unexpected[:10])
        )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    summary = {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_ratio": trainable / total,
        "trainable_tensor_count": len(trainable_names),
    }
    print(
        f"[INFO] 可训练参数: {trainable:,} / {total:,} "
        f"({summary['trainable_ratio']:.4%})，且全部属于 LoRA。"
    )
    return summary


def set_training_cache_mode(model, use_cache: bool) -> None:
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = use_cache
    text_config = getattr(model.config, "text_config", None)
    if text_config is not None and hasattr(text_config, "use_cache"):
        text_config.use_cache = use_cache


def select_samples(dataset, maximum: int | None, name: str):
    if maximum is None or maximum >= len(dataset):
        return dataset
    print(f"[WARN] {name} 仅使用前 {maximum}/{len(dataset)} 条样本（冒烟测试模式）")
    return dataset.select(range(maximum))


def build_training_arguments(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    use_bf16: bool,
) -> TrainingArguments:
    """
    兼容 Transformers 4.x/5.x 的 TrainingArguments 参数变化。

    输出目录覆盖由本脚本自行检查，因此不再依赖已从新版删除的
    overwrite_output_dir；模型和 adapter 本身默认以 safetensors 保存，
    因此也不传新版已删除的 save_safetensors。
    """
    supported_parameters = set(
        inspect.signature(TrainingArguments.__init__).parameters
    )

    training_kwargs: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "eval_accumulation_steps": args.eval_accumulation_steps,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_total_limit": 2,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_torch",
        "bf16": use_bf16,
        "fp16": not use_bf16,
        "report_to": "none",
        "push_to_hub": False,
        "run_name": args.run_name,
        "logging_strategy": "steps",
        "save_strategy": "steps",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "prediction_loss_only": True,
        "gradient_checkpointing": args.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "remove_unused_columns": False,
        "label_names": ["labels"],
        "seed": args.seed,
        "data_seed": args.seed,
        "dataloader_num_workers": 0,
    }

    if "eval_strategy" in supported_parameters:
        training_kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in supported_parameters:
        training_kwargs["evaluation_strategy"] = "steps"
    else:
        raise RuntimeError(
            "当前 Transformers 的 TrainingArguments 同时缺少 "
            "eval_strategy/evaluation_strategy，无法配置训练中验证。"
        )

    if "warmup_ratio" in supported_parameters:
        training_kwargs["warmup_ratio"] = args.warmup_ratio
        warmup_api = "warmup_ratio"
    elif "warmup_steps" in supported_parameters:
        # 新版允许 warmup_steps 使用 [0, 1) 的浮点数表示比例。
        training_kwargs["warmup_steps"] = args.warmup_ratio
        warmup_api = "warmup_steps(float ratio)"
    else:
        raise RuntimeError(
            "当前 Transformers 的 TrainingArguments 不支持 warmup 配置。"
        )

    unsupported = sorted(
        name for name in training_kwargs if name not in supported_parameters
    )
    if unsupported:
        transformers_version = package_versions()["transformers"]
        raise RuntimeError(
            f"Transformers {transformers_version} 的 TrainingArguments "
            f"不支持这些必要参数：{unsupported}。"
        )

    print(
        f"[INFO] TrainingArguments API 检查通过："
        f"Transformers={package_versions()['transformers']}，"
        f"warmup 接口={warmup_api}。"
    )
    return TrainingArguments(**training_kwargs)


def main() -> None:
    args = parse_args()
    validate_args(args)
    offline_environment = verify_strict_offline_mode()

    model_dir = resolve_project_path(args.model_dir)
    train_file = resolve_project_path(args.train_file)
    eval_file = resolve_project_path(args.eval_file)
    output_dir = resolve_project_path(args.output_dir)
    cache_dir = (
        resolve_project_path(args.cache_dir) if args.cache_dir is not None else None
    )
    resume_from_checkpoint = (
        resolve_project_path(args.resume_from_checkpoint)
        if args.resume_from_checkpoint
        else None
    )

    if not model_dir.is_dir():
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")
    if not train_file.is_file():
        raise FileNotFoundError(f"训练文件不存在: {train_file}")
    if not eval_file.is_file():
        raise FileNotFoundError(
            f"验证文件不存在: {eval_file}。当前脚本依赖验证集选择最佳 checkpoint。"
        )
    if resume_from_checkpoint is not None and not resume_from_checkpoint.is_dir():
        raise FileNotFoundError(
            f"resume checkpoint 目录不存在: {resume_from_checkpoint}"
        )

    if (
        output_dir.exists()
        and any(path.is_file() for path in output_dir.rglob("*"))
        and resume_from_checkpoint is None
        and not args.overwrite_output_dir
    ):
        raise FileExistsError(
            f"输出目录已存在且非空: {output_dir}\n"
            "请换一个 --output_dir、使用 --resume_from_checkpoint，"
            "或确认后显式添加 --overwrite_output_dir。"
        )
    status_log_dir = output_dir / "train_logs"

    print(f"[INFO] project_root: {PROJECT_ROOT}")
    print(f"[INFO] model_dir: {model_dir}")
    print(f"[INFO] train_file: {train_file}")
    print(f"[INFO] eval_file: {eval_file}")
    print(f"[INFO] output_dir: {output_dir}")
    print(
        f"[INFO] CUDA_VISIBLE_DEVICES: "
        f"{os.environ.get('CUDA_VISIBLE_DEVICES', '未设置（由系统选择）')}"
    )

    training_dtype = get_training_dtype()
    cuda_summary = get_cuda_summary()
    kernel_status = optional_kernel_status()
    print(f"[INFO] 训练 dtype: {training_dtype}")
    use_bf16 = training_dtype == torch.bfloat16
    training_args = build_training_arguments(
        args,
        output_dir,
        use_bf16=use_bf16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=False,
        local_files_only=True,
        token=False,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    if tokenizer.chat_template is None:
        raise ValueError("tokenizer 未提供 chat_template，无法确定对话与 loss 边界")
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("tokenizer 同时缺少 pad_token 和 eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        dtype=training_dtype,
        trust_remote_code=False,
        local_files_only=True,
        token=False,
        cache_dir=str(cache_dir) if cache_dir else None,
        low_cpu_mem_usage=True,
    )
    validate_text_only_causal_model(model)

    model_context_length = getattr(model.config, "max_position_embeddings", None)
    text_config = getattr(model.config, "text_config", None)
    if model_context_length is None and text_config is not None:
        model_context_length = getattr(text_config, "max_position_embeddings", None)
    if (
        isinstance(model_context_length, int)
        and args.max_seq_length > model_context_length
    ):
        raise ValueError(
            f"--max_seq_length={args.max_seq_length} 超过模型原生上下文长度 "
            f"{model_context_length}"
        )
    print(f"[INFO] 模型原生上下文长度: {model_context_length or '配置中未声明'}")

    lora_target_counts = validate_lora_targets(model)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    trainable_summary = validate_trainable_parameters(model)

    # 训练期间不需要 KV cache；梯度检查点由 Trainer 统一启用，避免重复调用。
    set_training_cache_mode(model, use_cache=False)

    raw_train = load_dataset("json", data_files=str(train_file), split="train")
    raw_eval = load_dataset("json", data_files=str(eval_file), split="train")
    raw_train = select_samples(raw_train, args.max_train_samples, "训练集")
    raw_eval = select_samples(raw_eval, args.max_eval_samples, "验证集")

    preprocess = build_preprocess_fn(tokenizer, args.max_seq_length)
    tokenized_train = raw_train.map(
        preprocess,
        remove_columns=raw_train.column_names,
        desc="Tokenizing train (assistant-only labels)",
        load_from_cache_file=False,
    )
    tokenized_eval = raw_eval.map(
        preprocess,
        remove_columns=raw_eval.column_names,
        desc="Tokenizing eval (assistant-only labels)",
        load_from_cache_file=False,
    )
    train_data_summary = summarize_tokenized_dataset(tokenized_train, "训练集")
    eval_data_summary = summarize_tokenized_dataset(tokenized_eval, "验证集")

    status_log_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args),
        "resolved_paths": {
            "model_dir": str(model_dir),
            "train_file": str(train_file),
            "eval_file": str(eval_file),
            "output_dir": str(output_dir),
            "resume_from_checkpoint": (
                str(resume_from_checkpoint) if resume_from_checkpoint else None
            ),
        },
        "versions": package_versions(),
        "strict_offline_environment": offline_environment,
        "cuda": cuda_summary,
        "optional_fast_kernels": kernel_status,
        "model": {
            "class": model.base_model.model.__class__.__name__,
            "model_type": getattr(model.config, "model_type", None),
            "context_length": model_context_length,
            "dtype": str(training_dtype),
            "language_model_only": True,
        },
        "loss": {
            "type": "causal_language_model_cross_entropy",
            "supervision": "final_assistant_content_and_end_marker_only",
            "ignore_index": -100,
            "enable_thinking": False,
        },
        "lora_target_counts": lora_target_counts,
        "trainable_parameters": trainable_summary,
        "train_data": train_data_summary,
        "eval_data": eval_data_summary,
    }
    (status_log_dir / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=build_data_collator(tokenizer),
        callbacks=[
            JsonlStatusLoggerCallback(status_log_dir / "train_status.jsonl")
        ],
    )

    effective_batch_size = (
        args.per_device_train_batch_size
        * args.gradient_accumulation_steps
        * training_args.world_size
    )
    print(f"[INFO] 有效训练 batch size: {effective_batch_size}")
    print("[INFO] 开始训练……")
    train_result = trainer.train(
        resume_from_checkpoint=(
            str(resume_from_checkpoint) if resume_from_checkpoint else None
        )
    )

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    summary = {
        "time": datetime.now(timezone.utc).isoformat(),
        "run_name": args.run_name,
        "global_step": trainer.state.global_step,
        "epoch": trainer.state.epoch,
        "best_metric": trainer.state.best_metric,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "effective_batch_size": effective_batch_size,
        "train_metrics": train_result.metrics,
    }
    (status_log_dir / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


    print(f"[INFO] 最佳 LoRA adapter 已保存到: {output_dir}")
    print(f"[INFO] 训练日志与配置清单: {status_log_dir}")


if __name__ == "__main__":
    main()
