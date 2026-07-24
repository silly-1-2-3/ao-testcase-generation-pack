#!/usr/bin/env python3
"""Evaluate a base model and a LoRA adapter through vLLM."""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import UseCaseTableMetric

STEP_LIST_KEY = "\u6b65\u9aa4\u5217\u8868"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--adapter", required=True, help="LoRA adapter directory")
    p.add_argument("--test_file", default="./data/test_sft.jsonl")
    p.add_argument("--output_dir", default="./eval_results")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument(
        "--enable_thinking",
        action="store_true",
        help="enable Qwen3.5 thinking in the chat template; disabled by default",
    )
    p.add_argument(
        "--max_model_len",
        type=int,
        default=8192,
        help="vLLM context limit; keep it >= the SFT prompt length plus generated thinking/output",
    )
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--gpu_memory", type=float, default=0.85)
    return p.parse_args()


def load_test(path, max_samples=None):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if max_samples and len(data) > max_samples:
        import random
        random.seed(42)
        data = random.sample(data, max_samples)
    return data


def build_prompt(messages):
    return [m for m in messages if m.get("role") != "assistant"]


def extract_gt(messages):
    for message in messages:
        if message.get("role") == "assistant":
            try:
                value = json.loads(message.get("content", "[]"))
            except json.JSONDecodeError:
                return []
            return value if isinstance(value, list) else []
    return []


def _response_content(response) -> str:
    """Read final content from an API payload or raw local generation."""
    if isinstance(response, dict):
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
        if isinstance(response.get("content"), str):
            return response["content"]
        if isinstance(response.get("text"), str):
            return response["text"]
    return str(response or "")


def _remove_thinking(text: str) -> str:
    """Remove Qwen-style reasoning before parsing the structured answer."""
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    closing = re.search(r"</think>", text, flags=re.IGNORECASE)
    if closing:
        text = text[closing.end():]
    text = re.sub(r"^\s*\x60\x60\x60(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\x60\x60\x60\s*$", "", text)
    return text.strip()


def _iter_json_values(text: str):
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        yield value


def parse_output(response):
    """Parse row arrays from thinking-model or JSON-wrapped responses."""
    text = _remove_thinking(_response_content(response))
    try:
        values = [json.loads(text)]
    except json.JSONDecodeError:
        values = list(_iter_json_values(text))

    rows = None
    for value in values:
        if isinstance(value, list):
            if value or rows is None:
                rows = value
        elif isinstance(value, dict):
            candidate = value.get(STEP_LIST_KEY, value.get("rows", value.get("steps", [])))
            if isinstance(candidate, list) and (candidate or rows is None):
                rows = candidate
    return rows if rows is not None else []


def run_vllm_eval(llm, sampling_params, test_data, tokenizer, adapter_path, label,
                  enable_thinking=False):
    from vllm.lora.request import LoRARequest

    print(f"\n{'=' * 60}")
    print(f"[{label}] vLLM batch inference: {len(test_data)} samples")
    print(f"{'=' * 60}")

    prompts = []
    metadatas = []
    for sample in test_data:
        messages = build_prompt(sample.get("messages", []))
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        prompts.append(prompt)
        metadatas.append({
            "id": sample.get("id"),
            "source": sample.get("source", ""),
            "gt_rows": extract_gt(sample.get("messages", [])),
        })

    start = time.time()
    if adapter_path:
        lora_request = LoRARequest("eval_adapter", 1, adapter_path)
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    else:
        outputs = llm.generate(prompts, sampling_params)

    results = []
    for output, metadata in zip(outputs, metadatas):
        raw = output.outputs[0].text
        results.append({
            "id": metadata["id"],
            "source": metadata["source"],
            "gt_rows": metadata["gt_rows"],
            "pred_rows": parse_output(raw),
        })

    elapsed = time.time() - start
    rate = len(prompts) / max(elapsed, 1e-6)
    print(f"[{label}] completed: {len(prompts)} samples, {rate:.2f} samples/s")
    return results


def compute_metrics(results, label):
    metric = UseCaseTableMetric(alpha=0.4, verbose=False)
    samples = [
        {
            "id": result["id"],
            "source": result["source"],
            "gt_rows": result["gt_rows"],
            "pred_rows": result["pred_rows"],
        }
        for result in results
    ]
    batch = metric.batch_evaluate(samples)
    print(
        f"\n--- [{label}] ---\n"
        f"overall={batch['avg_overall']:.4f} | "
        f"structure={batch['avg_structure']:.4f} | "
        f"content={batch['avg_content']:.4f}"
    )
    return batch


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    test_data = load_test(args.test_file, args.max_samples)
    print(f"Test samples: {len(test_data)}")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    llm = LLM(
        model=args.base_model,
        enable_lora=True,
        max_lora_rank=32,
        gpu_memory_utilization=args.gpu_memory,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.9,
        max_tokens=args.max_new_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    base_results = run_vllm_eval(
        llm, sampling_params, test_data, tokenizer, None, "BASE", args.enable_thinking
    )
    base_metrics = compute_metrics(base_results, "BASE")

    adapter_path = str(Path(args.adapter).resolve())
    lora_results = run_vllm_eval(
        llm, sampling_params, test_data, tokenizer, adapter_path, "LORA", args.enable_thinking
    )
    lora_metrics = compute_metrics(lora_results, "LORA")

    improvement = (
        (lora_metrics["avg_overall"] - base_metrics["avg_overall"])
        / max(base_metrics["avg_overall"], 0.001)
        * 100
    )
    summary = {
        "base": {
            "avg_overall": base_metrics["avg_overall"],
            "avg_structure": base_metrics["avg_structure"],
            "avg_content": base_metrics["avg_content"],
            "field_avg": base_metrics["field_avg"],
        },
        "lora": {
            "avg_overall": lora_metrics["avg_overall"],
            "avg_structure": lora_metrics["avg_structure"],
            "avg_content": lora_metrics["avg_content"],
            "field_avg": lora_metrics["field_avg"],
        },
        "improvement_pct": round(improvement, 2),
        "num_samples": len(test_data),
    }
    output_path = os.path.join(args.output_dir, "vllm_compare_summary.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
