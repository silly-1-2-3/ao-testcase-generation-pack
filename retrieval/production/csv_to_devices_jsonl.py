#!/usr/bin/env python3
"""Join device-command and device-category CSV files into canonical JSONL."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

COMMAND_REQUIRED = {
    "device_command_id",
    "dev_cat_id",
    "command_desc",
    "command_code",
    "param_type",
    "param_desc",
}
CATEGORY_REQUIRED = {
    "dev_cat_id",
    "dev_cat_name",
    "dev_cat_desc",
    "device_cat_code",
    "device_cat_alias",
}
TRUE_VALUES = {"1", "true", "yes", "y", "是", "已删除", "deleted"}


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\u0000", "").strip()


def is_deleted(value: Any) -> bool:
    return clean(value).lower() in TRUE_VALUES


def open_csv(path: Path) -> tuple[list[dict[str, str]], str, str, list[str]]:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            text = path.read_text(encoding=encoding)
            sample = text[:65536]
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
                delimiter = dialect.delimiter
            except csv.Error:
                delimiter = ","
            reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
            headers = [clean(value) for value in (reader.fieldnames or [])]
            rows = []
            for raw in reader:
                row = {clean(key): clean(value) for key, value in raw.items() if key is not None}
                if any(row.values()):
                    rows.append(row)
            return rows, encoding, delimiter, headers
        except UnicodeError as exc:
            last_error = exc
    raise ValueError(f"Unable to decode CSV {path}: {last_error}")


def ensure_headers(path: Path, actual: list[str], required: set[str]) -> None:
    missing = sorted(required - set(actual))
    if missing:
        raise ValueError(f"{path}: missing required columns: {missing}")


def searchable_text(parts: list[str]) -> str:
    result: list[str] = []
    seen: set[str] = set()
    for part in parts:
        value = clean(part)
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return " | ".join(result)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(
    command_csv: Path,
    category_csv: Path,
    output_jsonl: Path,
    report_json: Path,
    rejected_jsonl: Path,
    include_deleted: bool = False,
) -> dict[str, Any]:
    command_rows, command_encoding, command_delimiter, command_headers = open_csv(command_csv)
    category_rows, category_encoding, category_delimiter, category_headers = open_csv(category_csv)
    ensure_headers(command_csv, command_headers, COMMAND_REQUIRED)
    ensure_headers(category_csv, category_headers, CATEGORY_REQUIRED)

    category_by_id: dict[str, dict[str, str]] = {}
    duplicate_category_ids: list[str] = []
    deleted_categories = 0
    for row in category_rows:
        category_id = clean(row.get("dev_cat_id"))
        if not category_id:
            continue
        if is_deleted(row.get("deleted")) and not include_deleted:
            deleted_categories += 1
            continue
        if category_id in category_by_id:
            duplicate_category_ids.append(category_id)
            continue
        category_by_id[category_id] = row

    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_command_ids: set[str] = set()
    duplicate_command_ids: list[str] = []
    deleted_commands = 0
    orphan_commands = 0
    missing_business_codes = 0

    for row_number, command in enumerate(command_rows, 2):
        command_id = clean(command.get("device_command_id"))
        category_id = clean(command.get("dev_cat_id"))
        if is_deleted(command.get("deleted")) and not include_deleted:
            deleted_commands += 1
            continue
        if not command_id:
            rejected.append({"row": row_number, "reason": "missing_device_command_id"})
            continue
        if command_id in seen_command_ids:
            duplicate_command_ids.append(command_id)
            rejected.append({
                "row": row_number,
                "reason": "duplicate_device_command_id",
                "device_command_id": command_id,
            })
            continue
        seen_command_ids.add(command_id)
        category = category_by_id.get(category_id)
        if category is None:
            orphan_commands += 1
            rejected.append({
                "row": row_number,
                "reason": "dev_cat_id_not_found_or_deleted",
                "device_command_id": command_id,
                "dev_cat_id": category_id,
            })
            continue

        command_code = clean(command.get("command_code"))
        if not command_code:
            missing_business_codes += 1
        category_name = clean(category.get("dev_cat_name"))
        category_alias = clean(category.get("device_cat_alias"))
        category_code = clean(category.get("device_cat_code"))
        category_desc = clean(category.get("dev_cat_desc"))
        command_desc = clean(command.get("command_desc"))
        param_type = clean(command.get("param_type"))
        param_desc = clean(command.get("param_desc"))
        device_type = category_name or category_alias or category_code or category_id
        instruction_code = command_code or command_id
        parameter = json.dumps(
            {"param_type": param_type, "param_desc": param_desc},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        text = searchable_text([
            category_name,
            category_alias,
            category_desc,
            category_code,
            category_id,
            command_desc,
            command_code,
            command_id,
            param_type,
            param_desc,
        ])
        records.append({
            "设备指令主键": command_id,
            "设备指令号": instruction_code,
            "设备类型": device_type,
            "设备单元号": "",
            "设备指令功能说明": command_desc,
            "设备参数": parameter,
            "设备类别主键": category_id,
            "设备类别编码": category_code,
            "设备类别别名": category_alias,
            "设备类别描述": category_desc,
            "参数类型": param_type,
            "参数描述": param_desc,
            "_text": text,
        })

    records.sort(key=lambda value: (value["设备类别主键"], value["设备指令主键"]))
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    with rejected_jsonl.open("w", encoding="utf-8", newline="\n") as stream:
        for item in rejected:
            stream.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")

    report = {
        "format": "device-csv-conversion-v1",
        "command_csv": {
            "path": str(command_csv.resolve()),
            "sha256": file_sha256(command_csv),
            "encoding": command_encoding,
            "delimiter": command_delimiter,
            "rows": len(command_rows),
            "headers": command_headers,
        },
        "category_csv": {
            "path": str(category_csv.resolve()),
            "sha256": file_sha256(category_csv),
            "encoding": category_encoding,
            "delimiter": category_delimiter,
            "rows": len(category_rows),
            "headers": category_headers,
        },
        "output": {
            "path": str(output_jsonl.resolve()),
            "records": len(records),
            "sha256": file_sha256(output_jsonl),
        },
        "statistics": {
            "active_categories": len(category_by_id),
            "deleted_categories_skipped": deleted_categories,
            "deleted_commands_skipped": deleted_commands,
            "orphan_commands_rejected": orphan_commands,
            "missing_command_code_fallback_to_id": missing_business_codes,
            "duplicate_category_ids": sorted(set(duplicate_category_ids)),
            "duplicate_command_ids": sorted(set(duplicate_command_ids)),
            "rejected_records": len(rejected),
        },
    }
    report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert device CSV tables to canonical JSONL.")
    parser.add_argument("--commands", type=Path, required=True, help="设备指令 CSV")
    parser.add_argument("--categories", type=Path, required=True, help="设备类型 CSV")
    parser.add_argument("--output", type=Path, required=True, help="Canonical devices.jsonl")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--rejected", type=Path)
    parser.add_argument("--include-deleted", action="store_true")
    args = parser.parse_args()
    report = args.report or args.output.with_suffix(".conversion_report.json")
    rejected = args.rejected or args.output.with_suffix(".rejected.jsonl")
    result = convert(
        args.commands, args.categories, args.output, report, rejected, args.include_deleted
    )
    stats = result["statistics"]
    print(f"[ok] devices: {result['output']['records']} -> {args.output}")
    print(f"[ok] report:  {report}")
    print(f"[ok] rejected:{stats['rejected_records']} -> {rejected}")
    if stats["duplicate_category_ids"] or stats["duplicate_command_ids"]:
        print("[warning] duplicate primary keys found; inspect conversion report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
