#!/usr/bin/env python3
"""Annotate or safely replace device commands in generated test-case JSONL."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from retrieval.production.device_retrieval_engine import DeviceRetriever

DEVICE_FIELDS = ("设备类型", "设备单元号", "设备指令号", "设备参数")
QUERY_FIELDS = (
    "说明",
    "操作内容",
    "操作对象",
    "操作目的",
    "判据描述",
    "操作类型",
    "设备类型",
    "设备单元号",
    "设备参数",
)
EMPTY_VALUES = {"", "[]", "null", "none", "无", "否"}


def meaningful(value: Any) -> bool:
    return str(value if value is not None else "").strip().lower() not in EMPTY_VALUES


def locate_rows(record: Any) -> list[dict[str, Any]]:
    if isinstance(record, list):
        return record
    if not isinstance(record, dict):
        return []
    for key in ("rows", "pred_rows", "步骤列表"):
        value = record.get(key)
        if isinstance(value, list):
            return value
    return []


def should_search(row: dict[str, Any], all_execution_steps: bool) -> bool:
    if str(row.get("步骤层级", "")).strip() != "执行步骤":
        return False
    if any(meaningful(row.get(field)) for field in DEVICE_FIELDS):
        return True
    return all_execution_steps


def build_query(rows: list[dict[str, Any]], row_index: int, context_rows: int = 3) -> str:
    parts: list[str] = []
    start = max(0, row_index - context_rows)
    for index in range(start, row_index):
        row = rows[index]
        if str(row.get("步骤层级", "")).strip() == "执行步骤":
            continue
        for field in QUERY_FIELDS:
            value = str(row.get(field, "") or "").strip()
            if meaningful(value) and value not in parts:
                parts.append(value)

    row = rows[row_index]
    for field in QUERY_FIELDS:
        value = str(row.get(field, "") or "").strip()
        if meaningful(value) and value not in parts:
            parts.append(value)
    return "。".join(parts)


def replacement_decision(
    original_identifier: str,
    candidates: list[dict[str, Any]],
    retriever: DeviceRetriever,
    mode: str,
    bge_threshold: float,
    bge_margin: float,
) -> tuple[str, bool]:
    if retriever.is_known(original_identifier):
        if not candidates:
            return "review_known_identifier_no_candidate", False

        identifier = str(original_identifier or "").strip()

        def matches(candidate: dict[str, Any]) -> bool:
            return identifier in {
                str(candidate.get("设备指令主键", "") or "").strip(),
                str(candidate.get("设备指令号", "") or "").strip(),
            }

        if matches(candidates[0]):
            return "kept_known_identifier_top1", False
        if any(matches(candidate) for candidate in candidates[1:]):
            return "review_known_identifier_not_top1", False
        return "review_known_identifier_not_retrieved", False
    if not candidates:
        return "no_candidate", False
    if mode == "annotate":
        return "review_unknown_identifier", False
    top = candidates[0]
    if top["bge_score"] is None:
        return "review_bm25_only_not_auto_replaced", False
    if top.get("bge_rank") != 1:
        return "review_rrf_top_not_bge_top1", False
    if top["bge_score"] < bge_threshold:
        return "review_bge_below_threshold", False
    margin = top.get("bge_margin_to_second")
    if margin is None:
        return "review_bge_margin_unavailable", False
    if margin < bge_margin:
        return "review_bge_margin_too_small", False
    return "replaced_unknown_identifier", True


def validate_io_paths(
    input_path: Path,
    output_path: Path,
    audit_path: Path,
    force: bool,
) -> None:
    resolved = {
        "input": input_path.resolve(),
        "output": output_path.resolve(),
        "audit": audit_path.resolve(),
    }
    if len(set(resolved.values())) != len(resolved):
        raise ValueError(
            "Input, output, and audit paths must be three different files: "
            f"{resolved}"
        )
    existing = [
        str(path)
        for path in (output_path, audit_path)
        if path.exists()
    ]
    if existing and not force:
        raise FileExistsError(
            f"Outputs already exist; use --force only for intentional overwrite: {existing}"
        )


def process_record(
    record: Any,
    record_index: int,
    retriever: DeviceRetriever,
    mode: str,
    top_k: int,
    candidate_pool: int,
    bge_threshold: float,
    bge_margin: float,
    all_execution_steps: bool,
) -> tuple[Any, list[dict[str, Any]]]:
    output = copy.deepcopy(record)
    rows = locate_rows(output)
    audit_rows = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict) or not should_search(row, all_execution_steps):
            continue
        query = build_query(rows, row_index)
        candidates = retriever.search(query, top_k, candidate_pool) if query else []
        original = {
            "设备类型": row.get("设备类型", ""),
            "设备单元号": row.get("设备单元号", ""),
            "设备指令号": row.get("设备指令号", ""),
            "设备参数": row.get("设备参数", ""),
        }
        decision, replace = replacement_decision(
            str(original["设备指令号"]), candidates, retriever,
            mode, bge_threshold, bge_margin,
        )
        if replace:
            row["设备指令号"] = candidates[0]["设备指令号"]
            row["设备类型"] = candidates[0]["设备类型"]
        annotation = {
            "query": query,
            "bge_query_instruction": (
                retriever.bge_query_instruction
                if retriever.bge_model is not None
                else None
            ),
            "original": original,
            "decision": decision,
            "replaced": replace,
            "candidates": candidates,
        }
        row["_device_retrieval"] = annotation
        audit_rows.append({
            "record_index": record_index,
            "record_id": (
                output.get("id", output.get("sample_id", ""))
                if isinstance(output, dict) else ""
            ),
            "row_index": row_index,
            **annotation,
        })
    return output, audit_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply device retrieval to JSONL cases.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--devices", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--bge-model", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mode", choices=("annotate", "replace-invalid"), default="annotate")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-pool", type=int, default=50)
    parser.add_argument("--bge-threshold", type=float, default=0.68)
    parser.add_argument("--bge-margin", type=float, default=0.05)
    parser.add_argument("--all-execution-steps", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Intentionally overwrite existing output and audit files",
    )
    args = parser.parse_args()

    validate_io_paths(args.input, args.output, args.audit, args.force)
    retriever = DeviceRetriever(
        args.devices, args.index_dir, args.bge_model, args.device
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    records = processed_rows = replacements = 0
    with (
        args.input.open(encoding="utf-8-sig") as source,
        args.output.open("w", encoding="utf-8", newline="\n") as destination,
        args.audit.open("w", encoding="utf-8", newline="\n") as audit_stream,
    ):
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{args.input}:{line_number}: invalid JSON: {exc}") from exc
            output, audits = process_record(
                value, records, retriever, args.mode, args.top_k,
                args.candidate_pool, args.bge_threshold, args.bge_margin,
                args.all_execution_steps,
            )
            destination.write(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n")
            for audit in audits:
                audit_stream.write(
                    json.dumps(audit, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                replacements += int(audit["replaced"])
            records += 1
            processed_rows += len(audits)
    print(f"[ok] records: {records}")
    print(f"[ok] device rows: {processed_rows}")
    print(f"[ok] replacements: {replacements} (mode={args.mode})")
    print(f"[ok] output: {args.output}")
    print(f"[ok] audit:  {args.audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
