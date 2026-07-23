#!/usr/bin/env python3
"""
train_sft_final.py —— 基于 Qwen3-7B（或 Qwen2.5-7B）使用 PEFT/LoRA 微调。

功能：
    1. 加载本地 base model（参考 vllm/ 目录下的下载方式）
    2. 注入 LoRA adapter
    3. 使用 ./data/train_sft.jsonl 训练
    4. 使用 ./data/eval_sft.jsonl 做验证
    5. 保存 adapter 权重到 ./outputs/qwen3_7b_use_case_lora

参考代码：
    - task1/sft/train_swift_sft.py   (SFT 训练模板)
    - task1/vllm/start_vllm_server.py (模型路径配置)
    - task1/sft/scripts/download_model.py (模型下载)

环境要求：
    conda create -n sft_train python=3.10
    conda activate sft_train
    pip install torch transformers peft accelerate datasets
    pip install modelscope  # 如需从 ModelScope 下载模型

使用：
    # 1. 先切分数据集
    cd task1
    python split_dataset.py --out-dir ./data --seed 42

    # 2. 准备 SFT 格式数据
    python prepare_sft_data.py

    # 3. 开始训练
    python train_sft_final.py

    # 或自定义参数：
    python train_sft_final.py \
        --model_dir ../models/Qwen3-7B-Instruct \
        --train_file ./data/train_sft.jsonl \
        --eval_file ./data/eval_sft.jsonl \
        --output_dir ./outputs/qwen3_7b_use_case_lora \
        --num_train_epochs 5 \
        --learning_rate 1e-4 \
        --lora_r 32 \
        --max_seq_length 4096
"""
import os
import sys
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# ============================================================
# 默认配置（与 vllm/ 目录代码一致的本地模型路径）
# ============================================================
DEFAULT_MODEL_DIR = "../models/ModelScope_Qwen2.5-7B-instruct/modelscope_cache/Qwen/Qwen2___5-7B-Instruct"
DEFAULT_TRAIN_FILE = "data/train_sft.jsonl"
DEFAULT_EVAL_FILE = "data/eval_sft.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/qwen3_7b_use_case_lora"

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def parse_args():
    p = argparse.ArgumentParser(description="LoRA SFT for use case table generation")
    p.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR)
    p.add_argument("--train_file", type=str, default=DEFAULT_TRAIN_FILE)
    p.add_argument("--eval_file", type=str, default=DEFAULT_EVAL_FILE)
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--num_train_epochs", type=float, default=5.0)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--eval_accumulation_steps", type=int, default=8, help="eval 也做梯度累积避免 logits OOM")
    p.add_argument("--max_seq_length", type=int, default=6144)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--use_wandb", action="store_true",
                    help="启用 Weights & Biases 远程监控")
    p.add_argument("--wandb_project", type=str, default="qwen25-lora-sft",
                    help="W&B 项目名")
    p.add_argument("--wandb_entity", type=str, default=None,
                    help="W&B 用户名/团队名（可选）")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------- callback ----------
class JsonlStatusLoggerCallback(TrainerCallback):
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _json_safe(self, value):
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(v) for v in value]
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        return str(value)

    def _append(self, payload: dict):
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(self._json_safe(payload), ensure_ascii=False) + "\n")

    def on_train_begin(self, args, state, control, **kwargs):
        self._append({"event": "train_begin", "time": self._utc_now(),
                       "global_step": state.global_step, "epoch": state.epoch})

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self._append({"event": "log", "time": self._utc_now(),
                           "global_step": state.global_step, "epoch": state.epoch, "logs": logs})

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            self._append({"event": "eval", "time": self._utc_now(),
                           "global_step": state.global_step, "epoch": state.epoch, "metrics": metrics})

    def on_train_end(self, args, state, control, **kwargs):
        self._append({"event": "train_end", "time": self._utc_now(),
                       "global_step": state.global_step, "epoch": state.epoch,
                       "best_metric": state.best_metric,
                       "best_model_checkpoint": state.best_model_checkpoint})


# ---------- 数据处理 ----------
def to_messages(example: dict) -> List[Dict[str, str]]:
    """提取 messages 字段或从 instruction/input/output 构建"""
    if "messages" in example and isinstance(example["messages"], list):
        return example["messages"]
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output_text = example.get("output", "")
    user_content = f"{instruction}\n\n{input_text}" if input_text else instruction
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output_text},
    ]


def build_preprocess_fn(tokenizer, max_seq_length: int):
    def preprocess(example: dict) -> dict:
        messages = to_messages(example)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokenized = tokenizer(text, truncation=True, max_length=max_seq_length, padding=False)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    return preprocess


def build_data_collator(tokenizer):
    def data_collator(features: List[dict]) -> dict:
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True,
                                                           padding_value=tokenizer.pad_token_id),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0),
            "labels": torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100),
        }
    return data_collator


# ---------- 模型加载 ----------
def get_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def main():
    args = parse_args()
    root_dir = Path(__file__).resolve().parent

    # 路径解析
    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.exists():
        # 尝试相对路径
        model_dir = (root_dir / args.model_dir).resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"模型目录不存在: {model_dir}\n请修改 --model_dir 参数或 DEFAULT_MODEL_DIR")

    train_file = Path(args.train_file)
    if not train_file.is_absolute():
        train_file = root_dir / train_file
    if not train_file.exists():
        raise FileNotFoundError(f"训练文件不存在: {train_file}")

    eval_file = Path(args.eval_file)
    if not eval_file.is_absolute():
        eval_file = root_dir / eval_file
    if not eval_file.exists():
        print(f"[WARN] 验证文件不存在: {eval_file}，将不使用验证集")
        eval_file = None

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    status_log_dir = output_dir / "train_logs"
    status_log_dir.mkdir(parents=True, exist_ok=True)

    # 环境变量
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    print(f"[INFO] model_dir: {model_dir}")
    print(f"[INFO] train_file: {train_file}")
    print(f"[INFO] eval_file: {eval_file}")
    print(f"[INFO] output_dir: {output_dir}")
    print(f"[INFO] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'auto')}")

    # 加载 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        dtype=get_dtype(),
        trust_remote_code=True,
        local_files_only=True,
    )

    # 注入 LoRA
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

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # 加载数据
    raw_dataset = load_dataset("json", data_files=str(train_file), split="train")
    preprocess = build_preprocess_fn(tokenizer, args.max_seq_length)
    tokenized_dataset = raw_dataset.map(
        preprocess,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing",
    )

    # 验证集（可选）
    eval_dataset = None
    if eval_file and eval_file.exists():
        raw_eval = load_dataset("json", data_files=str(eval_file), split="train")
        eval_dataset = raw_eval.map(
            preprocess,
            remove_columns=raw_eval.column_names,
            desc="Tokenizing eval",
        )

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=2,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to="wandb" if args.use_wandb else "none",
        run_name=args.run_name,
        logging_strategy="steps",
        # logging_dir 已由 report_to 自动处理
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        seed=args.seed,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=eval_dataset,
        data_collator=build_data_collator(tokenizer),
        callbacks=[JsonlStatusLoggerCallback(status_log_dir / "train_status.jsonl")],
    )

    # W&B 初始化
    if args.use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.run_name or f"qwen25-lora-{args.num_train_epochs}ep",
                config={
                    "model": "Qwen2.5-7B-Instruct",
                    "method": "LoRA",
                    "lora_r": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                    "epochs": args.num_train_epochs,
                    "lr": args.learning_rate,
                    "max_seq_length": args.max_seq_length,
                    "batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
                }
            )
            print(f"[INFO] W&B: https://wandb.ai/{args.wandb_entity or 'your-account'}/{args.wandb_project}")
        except Exception as e:
            print(f"[WARN] W&B 初始化失败: {e}")
            print("[WARN] 继续训练但不使用 W&B 监控。请检查: wandb login --relogin")
            os.environ["WANDB_MODE"] = "disabled"

    # 训练
    print("[INFO] 开始训练...")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # 保存
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # 记录总结
    summary = {
        "time": datetime.now(timezone.utc).isoformat(),
        "run_name": args.run_name,
        "global_step": trainer.state.global_step,
        "epoch": trainer.state.epoch,
        "best_metric": trainer.state.best_metric,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
    }
    (status_log_dir / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[INFO] LoRA adapter 已保存到: {output_dir}")
    print(f"[INFO] 训练日志: {status_log_dir}")


if __name__ == "__main__":
    main()