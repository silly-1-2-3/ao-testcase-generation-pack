"""AO 结构化测试用例研发平台的离线 Web/API 服务。

数据流：

    browser -> server.py:8081 -> vLLM:8000
                                      ├─ qwen35-base
                                      └─ qwen35-lora (Base + LoRA)

设计约束：

1. 服务不加载模型权重，只调用本机 OpenAI-compatible vLLM。
2. 每条 AO 都是独立请求，消息中只有 System Prompt 和本次 AO。
3. Base/LoRA 实时对比使用相同 Prompt 与生成参数，只改变 model 字段。
4. 默认读取 eval_results 下样本数最多的成对评测目录，也可显式指定。
5. 默认仅监听 127.0.0.1，适合通过 SSH 端口转发访问。
6. 外部 Qwen API、Hugging Face 下载和运行时 LoRA 更新均不参与此服务。
7. 设备检索默认关闭；开启后固定为 annotate，只返回候选和审计信息，
   不改写 Base 或 Base+LoRA 的原始结构化输出。
8. 用户可在网页副本中显式应用/撤销候选；Excel 同时保存最终结果、
   原始模型结果与人工修改审计，服务端不长期保存该副本。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional


import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / "train"
EVAL_ROOT = PROJECT_ROOT / "eval_results"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
try:
    from pre.xlsx_export import build_ao_workbook
except ModuleNotFoundError:
    # 兼容 ``python pre/server.py``：此时脚本目录本身位于 sys.path。
    from xlsx_export import build_ao_workbook


def _parse_cors_origins(raw_value: str) -> list[str]:
    """解析逗号分隔的 CORS 来源；只有显式配置 ``*`` 才允许任意来源。"""

    origins = [
        origin.strip()
        for origin in raw_value.split(",")
        if origin.strip()
    ]
    return origins or ["http://127.0.0.1:8081", "http://localhost:8081"]


def _resolve_project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _resolve_optional_project_path(raw_path: str | Path | None) -> Path | None:
    value = str(raw_path or "").strip()
    return _resolve_project_path(value) if value else None


def _safe_load_json_file(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _discover_eval_dir() -> Path:
    """选择当前最完整的 Base/LoRA 成对评测目录。

    优先级：
    1. AO_EVAL_DIR 显式指定；
    2. eval_results 中 num_samples 最大的完整成对评测目录；
    3. 旧版评测目录（仅用于兼容）；
    4. eval_results 根目录。
    """

    configured = os.environ.get("AO_EVAL_DIR", "").strip()
    if configured:
        return _resolve_project_path(configured)

    candidates: list[tuple[int, int, Path]] = []
    if EVAL_ROOT.is_dir():
        for directory in EVAL_ROOT.iterdir():
            if not directory.is_dir():
                continue
            required = (
                directory / "vllm_compare_summary.json",
                directory / "base_predictions.jsonl",
                directory / "lora_predictions.jsonl",
            )
            if not all(path.is_file() for path in required):
                continue
            summary = _safe_load_json_file(required[0])
            sample_count = int(summary.get("num_samples", 0) or 0)
            modified = max(path.stat().st_mtime_ns for path in required)
            candidates.append((sample_count, modified, directory.resolve()))

    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return candidates[0][2]

    legacy = TRAIN_DIR / "eval_results_did_v4_v2" / "batch"
    if legacy.exists():
        return legacy.resolve()
    return EVAL_ROOT.resolve()


CORS_ORIGINS = _parse_cors_origins(
    os.environ.get(
        "AO_CORS_ORIGINS",
        "http://127.0.0.1:8081,http://localhost:8081",
    )
)

EVAL_DIR = _discover_eval_dir()
DB_PATH = _resolve_project_path(
    os.environ.get(
        "AO_RETRIEVAL_DB",
        str(PROJECT_ROOT / "retrieval" / "device_corpus.jsonl"),
    )
)
DEVICE_RETRIEVAL_MODE = os.environ.get(
    "AO_DEVICE_RETRIEVAL_MODE",
    "off",
).strip().lower()
RETRIEVAL_INDEX_DIR = _resolve_optional_project_path(
    os.environ.get("AO_RETRIEVAL_INDEX_DIR")
)
RETRIEVAL_BGE_MODEL = _resolve_optional_project_path(
    os.environ.get("AO_RETRIEVAL_BGE_MODEL")
)
RETRIEVAL_DEVICE = os.environ.get(
    "AO_RETRIEVAL_DEVICE",
    "cpu",
).strip()
RETRIEVAL_DATA_KIND = os.environ.get(
    "AO_RETRIEVAL_DATA_KIND",
    "example",
).strip().lower()
RETRIEVAL_DATA_LABEL = os.environ.get(
    "AO_RETRIEVAL_DATA_LABEL",
    (
        "示例设备库（2 条，仅研发验证）"
        if RETRIEVAL_DATA_KIND == "example"
        else "生产设备库"
    ),
).strip()
RETRIEVAL_TOP_K = int(os.environ.get("AO_RETRIEVAL_TOP_K", "5"))
RETRIEVAL_CANDIDATE_POOL = int(
    os.environ.get("AO_RETRIEVAL_CANDIDATE_POOL", "50")
)
RETRIEVAL_BGE_THRESHOLD = float(
    os.environ.get("AO_RETRIEVAL_BGE_THRESHOLD", "0.68")
)
RETRIEVAL_BGE_MARGIN = float(
    os.environ.get("AO_RETRIEVAL_BGE_MARGIN", "0.05")
)

VLLM_URL = os.environ.get(
    "VLLM_URL",
    "http://127.0.0.1:8000/v1",
).rstrip("/")
VLLM_BASE_MODEL = os.environ.get("VLLM_BASE_MODEL", "qwen35-base").strip()
VLLM_LORA_MODEL = os.environ.get(
    "VLLM_LORA_MODEL",
    os.environ.get("VLLM_MODEL", "qwen35-lora"),
).strip()
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "").strip()

INFER_TEMPERATURE = float(os.environ.get("AO_TEMPERATURE", "0"))
INFER_TOP_P = float(os.environ.get("AO_TOP_P", "1"))
INFER_MAX_TOKENS = int(os.environ.get("AO_MAX_TOKENS", "8192"))
VLLM_REQUEST_TIMEOUT = float(os.environ.get("AO_VLLM_TIMEOUT", "600"))
DEFAULT_PROMPT_MODE = os.environ.get(
    "AO_PROMPT_MODE",
    "compressed",
).strip().lower()
SYSTEM_PROMPT_OVERRIDE = os.environ.get("AO_SYSTEM_PROMPT", "").strip()

STEP_LIST_KEY = "步骤列表"
STEP_LEVEL_KEY = "步骤层级"
DEVICE_COMMAND_KEY = "设备指令号"
DEVICE_TYPE_KEY = "设备类型"
DEVICE_UNIT_KEY = "设备单元号"

_http_client: Optional[httpx.AsyncClient] = None
_file_cache: dict[str, tuple[tuple[int, int] | None, Any]] = {}
_device_retriever: Any | None = None
_device_retrieval_error: str | None = None
_retrieval_initialization_attempted = False
_device_retrieval_lock = asyncio.Lock()


app = FastAPI(title="AO Test Case R&D Platform")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        headers: dict[str, str] = {}
        if VLLM_API_KEY:
            headers["Authorization"] = f"Bearer {VLLM_API_KEY}"
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(VLLM_REQUEST_TIMEOUT),
            headers=headers,
        )
    return _http_client


def _file_fingerprint(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size
    except OSError:
        return None


def _load_json(path: Path) -> Any:
    cache_key = f"json::{path.resolve()}"
    fingerprint = _file_fingerprint(path)
    cached = _file_cache.get(cache_key)
    if cached and cached[0] == fingerprint:
        return cached[1]

    data: Any = None
    if fingerprint is not None:
        try:
            with path.open(encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError):
            data = None
    _file_cache[cache_key] = (fingerprint, data)
    return data


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    cache_key = f"jsonl::{path.resolve()}"
    fingerprint = _file_fingerprint(path)
    cached = _file_cache.get(cache_key)
    if cached and cached[0] == fingerprint:
        return cached[1]

    records: list[dict[str, Any]] = []
    if fingerprint is not None:
        try:
            with path.open(encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"{path.name} 第 {line_number} 行不是合法 JSON"
                        ) from exc
                    if isinstance(item, dict):
                        records.append(item)
        except OSError as exc:
            raise ValueError(f"无法读取评测文件: {path}") from exc

    _file_cache[cache_key] = (fingerprint, records)
    return records


def _eval_layout() -> str:
    paired_files = (
        EVAL_DIR / "vllm_compare_summary.json",
        EVAL_DIR / "base_predictions.jsonl",
        EVAL_DIR / "lora_predictions.jsonl",
    )
    if all(path.is_file() for path in paired_files):
        return "paired_vllm"
    if (EVAL_DIR / "did_batch_summary.json").is_file():
        return "legacy"
    return "missing"


def _paired_eval_maps() -> tuple[
    list[str],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    if _eval_layout() != "paired_vllm":
        return [], {}, {}

    base_records = _load_jsonl(EVAL_DIR / "base_predictions.jsonl")
    lora_records = _load_jsonl(EVAL_DIR / "lora_predictions.jsonl")
    base_map = {str(item.get("id")): item for item in base_records}
    lora_map = {str(item.get("id")): item for item in lora_records}
    ordered_ids = [
        str(item.get("id"))
        for item in base_records
        if str(item.get("id")) in lora_map
    ]
    return ordered_ids, base_map, lora_map


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _record_metrics(record: dict[str, Any]) -> dict[str, Any]:
    scores = record.get("scores")
    scores = scores if isinstance(scores, dict) else {}
    return {
        "overall": _as_float(scores.get("overall_score")),
        "structure": _as_float(scores.get("structure_score")),
        "content": _as_float(scores.get("content_score")),
        "row_coverage": _as_float(scores.get("row_coverage")),
        "row_precision": _as_float(scores.get("row_precision")),
        "row_match_f1": _as_float(scores.get("row_match_f1")),
        "row_count_match": bool(scores.get("row_count_match", False)),
        "matched_rows": int(scores.get("matched_rows", 0) or 0),
        "unmatched_gt": int(scores.get("unmatched_gt", 0) or 0),
        "unmatched_pred": int(scores.get("unmatched_pred", 0) or 0),
        "field_scores": (
            scores.get("field_scores")
            if isinstance(scores.get("field_scores"), dict)
            else {}
        ),
    }

def _parse_json(raw: Any) -> list[dict[str, Any]]:
    """从 OpenAI 响应、纯 JSON 或 fenced JSON 中提取结果行。"""

    if isinstance(raw, dict):
        choices = raw.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            raw = message.get("content", raw.get("content", ""))
        else:
            raw = raw.get("content", raw.get("text", ""))

    text = str(raw or "").strip()
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    closing = re.search(r"</think>", text, flags=re.IGNORECASE)
    if closing:
        text = text[closing.end():]
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text).strip()

    try:
        values = [json.loads(text)]
    except json.JSONDecodeError:
        values = []
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            values.append(value)

    rows: list[dict[str, Any]] | None = None
    for value in values:
        candidate: Any = None
        if isinstance(value, list):
            candidate = value
        elif isinstance(value, dict):
            candidate = value.get(
                STEP_LIST_KEY,
                value.get("rows", value.get("steps", [])),
            )
        if isinstance(candidate, list):
            normalized = [row for row in candidate if isinstance(row, dict)]
            if normalized or rows is None:
                rows = normalized
    return rows if rows is not None else []


def _extract_response_text(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message")
    if isinstance(message, dict):
        return str(message.get("content", "") or "")
    return str(choices[0].get("text", "") or "")


def _load_device_codes() -> set[str]:
    codes: set[str] = set()
    if not DB_PATH.is_file():
        return codes
    try:
        records = _load_jsonl(DB_PATH)
    except ValueError:
        return codes
    for item in records:
        code = str(item.get(DEVICE_COMMAND_KEY, "") or "").strip()
        if code and code not in {"[]", "null"}:
            codes.add(code)
    return codes


def _check_device_rows(rows: Any) -> dict[str, Any] | None:
    if not isinstance(rows, list):
        return None
    codes = _load_device_codes()
    details = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        level = str(row.get(STEP_LEVEL_KEY, "") or "").strip()
        if level not in {"步骤", "执行步骤"}:
            continue
        code = str(row.get(DEVICE_COMMAND_KEY, "") or "").strip()
        if not code or code in {"[]", "null"}:
            continue
        details.append(
            {
                "row_idx": index,
                "code": code,
                "type": row.get(DEVICE_TYPE_KEY, ""),
                "unit": row.get(DEVICE_UNIT_KEY, ""),
                "in_db": code in codes if codes else None,
            }
        )
    if not details:
        return None
    known = [item for item in details if item["in_db"] is not None]
    found = sum(1 for item in known if item["in_db"])
    return {
        "db_available": bool(codes),
        "total": len(details),
        "checked": len(known),
        "found": found,
        "not_found": len(known) - found,
        "details": details,
    }


def _retrieval_warning() -> str | None:
    if DEVICE_RETRIEVAL_MODE == "off":
        return None
    if RETRIEVAL_DATA_KIND == "example":
        return (
            "当前使用示例设备库，仅用于验证检索链路；候选结果不能作为"
            "现场设备指令依据，且网页不会自动改写模型输出。"
        )
    return (
        "检索以 annotate 模式运行：只提供候选与审计信息，"
        "不会自动改写模型输出。"
    )


def _retrieval_corpus_records() -> int:
    if _device_retriever is not None:
        devices = getattr(_device_retriever, "devices", [])
        return len(devices) if isinstance(devices, list) else 0
    if not DB_PATH.is_file():
        return 0
    try:
        return len(_load_jsonl(DB_PATH))
    except ValueError:
        return 0


def _device_retrieval_status() -> dict[str, Any]:
    enabled = DEVICE_RETRIEVAL_MODE == "annotate"
    ready = (
        enabled
        and _device_retriever is not None
        and _device_retrieval_error is None
    )
    return {
        "enabled": enabled,
        "ready": ready,
        "mode": DEVICE_RETRIEVAL_MODE,
        "data_kind": RETRIEVAL_DATA_KIND,
        "data_label": RETRIEVAL_DATA_LABEL,
        "corpus_records": _retrieval_corpus_records(),
        "devices_path": str(DB_PATH),
        "index_dir": (
            str(RETRIEVAL_INDEX_DIR)
            if RETRIEVAL_INDEX_DIR is not None
            else None
        ),
        "bge_model": (
            str(RETRIEVAL_BGE_MODEL)
            if RETRIEVAL_BGE_MODEL is not None
            else None
        ),
        "device": RETRIEVAL_DEVICE,
        "top_k": RETRIEVAL_TOP_K,
        "candidate_pool": RETRIEVAL_CANDIDATE_POOL,
        "error": _device_retrieval_error,
        "warning": _retrieval_warning(),
    }


def _initialize_device_retrieval_sync() -> None:
    """按需加载 BM25+BGE；失败时保留 Web 服务并通过健康接口报告。"""

    global _device_retriever
    global _device_retrieval_error
    global _retrieval_initialization_attempted

    _retrieval_initialization_attempted = True
    _device_retriever = None
    _device_retrieval_error = None
    if DEVICE_RETRIEVAL_MODE == "off":
        return

    try:
        if DEVICE_RETRIEVAL_MODE != "annotate":
            raise ValueError(
                "设备检索只允许 off 或 annotate；Web 服务禁止自动替换模式"
            )
        if not DB_PATH.is_file():
            raise FileNotFoundError(f"设备语料不存在: {DB_PATH}")
        if RETRIEVAL_INDEX_DIR is None:
            raise ValueError("已开启设备检索，但未设置检索索引目录")
        if not RETRIEVAL_INDEX_DIR.is_dir():
            raise FileNotFoundError(
                f"检索索引目录不存在: {RETRIEVAL_INDEX_DIR}"
            )
        required_index_files = (
            "bm25_devices.pkl",
            "index_manifest.json",
            "bge_vectors.npy",
            "bge_meta.json",
        )
        missing = [
            name
            for name in required_index_files
            if not (RETRIEVAL_INDEX_DIR / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "检索索引缺少文件: " + ", ".join(missing)
            )
        if RETRIEVAL_BGE_MODEL is None:
            raise ValueError("已开启设备检索，但未设置本地 BGE 模型目录")
        if not RETRIEVAL_BGE_MODEL.is_dir():
            raise FileNotFoundError(
                f"本地 BGE 模型目录不存在: {RETRIEVAL_BGE_MODEL}"
            )

        from retrieval.production.device_retrieval_engine import (
            DeviceRetriever,
        )

        _device_retriever = DeviceRetriever(
            DB_PATH,
            RETRIEVAL_INDEX_DIR,
            RETRIEVAL_BGE_MODEL,
            RETRIEVAL_DEVICE,
        )
    except Exception as exc:
        _device_retriever = None
        _device_retrieval_error = f"{type(exc).__name__}: {str(exc)[:500]}"


def _annotate_device_rows_sync(rows: list[dict[str, Any]]) -> dict[str, Any]:
    started = time.perf_counter()
    if not _retrieval_initialization_attempted:
        _initialize_device_retrieval_sync()

    status = _device_retrieval_status()
    result = {
        **status,
        "processed_rows": 0,
        "elapsed_ms": 0.0,
        "query_instruction": None,
        "rows": [],
    }
    if not status["enabled"] or not status["ready"]:
        return result

    try:
        from retrieval.production.apply_device_retrieval import (
            build_query,
            replacement_decision,
            should_search,
        )

        retriever = _device_retriever
        audits: list[dict[str, Any]] = []
        for row_index, row in enumerate(rows):
            if not isinstance(row, dict) or not should_search(row, False):
                continue
            query = build_query(rows, row_index)
            candidates = (
                retriever.search(
                    query,
                    RETRIEVAL_TOP_K,
                    RETRIEVAL_CANDIDATE_POOL,
                )
                if query
                else []
            )
            original = {
                "设备类型": row.get("设备类型", ""),
                "设备单元号": row.get("设备单元号", ""),
                "设备指令号": row.get("设备指令号", ""),
                "设备参数": row.get("设备参数", ""),
            }
            decision, replaced = replacement_decision(
                str(original["设备指令号"]),
                candidates,
                retriever,
                "annotate",
                RETRIEVAL_BGE_THRESHOLD,
                RETRIEVAL_BGE_MARGIN,
            )
            if replaced:
                raise RuntimeError(
                    "annotate 模式意外产生了替换操作，已中止检索后处理"
                )
            audits.append(
                {
                    "row_index": row_index,
                    "query": query,
                    "original": original,
                    "decision": decision,
                    "replaced": False,
                    "candidates": candidates,
                }
            )

        result.update(
            {
                "processed_rows": len(audits),
                "elapsed_ms": round(
                    (time.perf_counter() - started) * 1000,
                    1,
                ),
                "query_instruction": getattr(
                    retriever,
                    "bge_query_instruction",
                    None,
                ),
                "rows": audits,
            }
        )
        return result
    except Exception as exc:
        result["ready"] = False
        result["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        result["elapsed_ms"] = round(
            (time.perf_counter() - started) * 1000,
            1,
        )
        return result


async def _annotate_device_rows(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    # SentenceTransformer 的同一实例不并发调用；Base/LoRA 的 vLLM 请求仍并行。
    async with _device_retrieval_lock:
        return await asyncio.to_thread(_annotate_device_rows_sync, rows)


def _normalize_prompt_mode(variant: str = "") -> str:
    normalized = str(variant or "").strip().lower()
    if normalized.endswith("+full") or normalized == "full":
        return "full"
    if normalized.endswith("+compressed") or normalized == "compressed":
        return "compressed"
    return DEFAULT_PROMPT_MODE


def _system_prompt(variant: str = "") -> str:
    """读取与请求模式对应的训练 Prompt。"""

    if SYSTEM_PROMPT_OVERRIDE:
        prompt_path = _resolve_project_path(SYSTEM_PROMPT_OVERRIDE)
    else:
        mode = _normalize_prompt_mode(variant)
        filename = (
            "system_prompt_v4_full.txt"
            if mode == "full"
            else "system_prompt_v4.txt"
        )
        prompt_path = TRAIN_DIR / filename

    if not prompt_path.is_file():
        raise FileNotFoundError(f"System Prompt 文件不存在: {prompt_path}")
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"System Prompt 文件为空: {prompt_path}")
    return prompt


def _validate_runtime_config() -> None:
    if not VLLM_BASE_MODEL:
        raise ValueError("VLLM_BASE_MODEL/--base-model 不能为空")
    if not VLLM_LORA_MODEL:
        raise ValueError("VLLM_LORA_MODEL/--lora-model 不能为空")
    if INFER_TEMPERATURE < 0:
        raise ValueError("AO_TEMPERATURE/--temperature 必须大于等于 0")
    if not 0 < INFER_TOP_P <= 1:
        raise ValueError("AO_TOP_P/--top-p 必须位于 (0, 1]")
    if INFER_MAX_TOKENS <= 0:
        raise ValueError("AO_MAX_TOKENS/--max-tokens 必须大于 0")
    if VLLM_REQUEST_TIMEOUT <= 0:
        raise ValueError("AO_VLLM_TIMEOUT/--request-timeout 必须大于 0")
    if DEFAULT_PROMPT_MODE not in {"compressed", "full"}:
        raise ValueError("AO_PROMPT_MODE/--prompt-mode 只能是 compressed 或 full")
    if DEVICE_RETRIEVAL_MODE not in {"off", "annotate"}:
        raise ValueError(
            "AO_DEVICE_RETRIEVAL_MODE/--device-retrieval-mode "
            "只能是 off 或 annotate"
        )
    if RETRIEVAL_DATA_KIND not in {"example", "production"}:
        raise ValueError(
            "AO_RETRIEVAL_DATA_KIND/--retrieval-data-kind "
            "只能是 example 或 production"
        )
    if not RETRIEVAL_DATA_LABEL:
        raise ValueError("AO_RETRIEVAL_DATA_LABEL/--retrieval-data-label 不能为空")
    if not RETRIEVAL_DEVICE:
        raise ValueError("AO_RETRIEVAL_DEVICE/--retrieval-device 不能为空")
    if RETRIEVAL_TOP_K <= 0:
        raise ValueError("AO_RETRIEVAL_TOP_K/--retrieval-top-k 必须大于 0")
    if RETRIEVAL_CANDIDATE_POOL < RETRIEVAL_TOP_K:
        raise ValueError(
            "candidate-pool 必须大于等于 top-k"
        )
    _system_prompt(DEFAULT_PROMPT_MODE)


class InferReq(BaseModel):
    ao_text: str
    variant: str = "compare+compressed"


class ExportWorkbookReq(BaseModel):
    variant: str
    model_name: str
    ao_text: str
    prompt_mode: str
    original_rows: list[dict[str, Any]]
    final_rows: list[dict[str, Any]]
    modifications: list[dict[str, Any]]
    generation: dict[str, Any]


def _validate_ao_text(raw_text: str) -> str:
    ao_text = raw_text.strip()
    if len(ao_text) < 20:
        raise HTTPException(400, "AO 文本至少需要 20 个字符")
    return ao_text


async def _infer_one(
    *,
    model_name: str,
    ao_text: str,
    prompt_mode: str,
    system_prompt: str,
) -> dict[str, Any]:
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ao_text},
        ],
        "temperature": INFER_TEMPERATURE,
        "top_p": INFER_TOP_P,
        "max_tokens": INFER_MAX_TOKENS,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.perf_counter()
    try:
        response = await _get_client().post(
            f"{VLLM_URL}/chat/completions",
            json=payload,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        if response.status_code != 200:
            return {
                "success": False,
                "model": model_name,
                "prompt_mode": prompt_mode,
                "elapsed_ms": elapsed_ms,
                "error": f"vLLM {response.status_code}: {response.text[:500]}",
            }

        data = response.json()
        raw_output = _extract_response_text(data)
        table = _parse_json(raw_output)
        usage = data.get("usage")
        device_retrieval = await _annotate_device_rows(table)
        return {
            "success": True,
            "model": model_name,
            "prompt_mode": prompt_mode,
            "elapsed_ms": elapsed_ms,
            "table": table,
            "row_count": len(table),
            "parse_success": bool(table),
            "device_check": _check_device_rows(table),
            "device_retrieval": device_retrieval,
            "finish_reason": (
                data.get("choices", [{}])[0].get("finish_reason")
                if isinstance(data.get("choices"), list) and data["choices"]
                else None
            ),
            "usage": usage if isinstance(usage, dict) else {},
            "generation": {
                "temperature": INFER_TEMPERATURE,
                "top_p": INFER_TOP_P,
                "max_tokens": INFER_MAX_TOKENS,
                "enable_thinking": False,
            },
            "raw_output": raw_output,
        }
    except Exception as exc:
        return {
            "success": False,
            "model": model_name,
            "prompt_mode": prompt_mode,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "error": f"vLLM: {str(exc)[:500]}",
        }


@app.get("/api/eval/summary")
async def eval_summary():
    layout = _eval_layout()
    if layout == "paired_vllm":
        data = _load_json(EVAL_DIR / "vllm_compare_summary.json") or {}
        table = []
        for variant, label in (
            ("base", "Qwen3.5-9B Base"),
            ("lora", "Qwen3.5-9B Base + LoRA"),
        ):
            values = data.get(variant)
            values = values if isinstance(values, dict) else {}
            table.append(
                {
                    "variant": variant,
                    "label": label,
                    "overall": _as_float(values.get("avg_overall")),
                    "structure": _as_float(values.get("avg_structure")),
                    "content": _as_float(values.get("avg_content")),
                    "generation": values.get("generation", {}),
                    "runtime": values.get("runtime", {}),
                }
            )
        return {
            "layout": layout,
            "metric_version": data.get("metric_version", "v2"),
            "num_samples": int(data.get("num_samples", 0) or 0),
            "table": table,
            "comparison": data.get("comparison", {}),
            "eval_dir": str(EVAL_DIR),
        }

    if layout == "legacy":
        data = _load_json(EVAL_DIR / "did_batch_summary.json") or {}
        table = []
        aggregated = data.get("aggregated")
        aggregated = aggregated if isinstance(aggregated, dict) else {}
        for variant, values in aggregated.items():
            values = values if isinstance(values, dict) else {}

            def average(name: str) -> float:
                value = values.get(name, 0)
                if isinstance(value, dict):
                    value = value.get("avg", 0)
                return _as_float(value)

            table.append(
                {
                    "variant": variant,
                    "label": variant,
                    "overall": average("avg_overall"),
                    "structure": average("avg_structure"),
                    "content": average("avg_content"),
                }
            )
        return {
            "layout": layout,
            "num_samples": int(data.get("num_samples", 0) or 0),
            "table": table,
            "comparison": {},
            "eval_dir": str(EVAL_DIR),
        }

    return {
        "layout": "missing",
        "num_samples": 0,
        "table": [],
        "comparison": {},
        "eval_dir": str(EVAL_DIR),
    }


@app.get("/api/eval/field_scores")
async def field_scores():
    layout = _eval_layout()
    if layout == "paired_vllm":
        data = _load_json(EVAL_DIR / "vllm_compare_summary.json") or {}
        variants: dict[str, dict[str, float]] = {}
        fields: set[str] = set()
        for variant in ("base", "lora"):
            values = data.get(variant)
            values = values if isinstance(values, dict) else {}
            field_values = values.get("field_avg")
            field_values = field_values if isinstance(field_values, dict) else {}
            variants[variant] = {
                str(field): _as_float(score)
                for field, score in field_values.items()
            }
            fields.update(variants[variant])
        return {
            "layout": layout,
            "variants": variants,
            "fields": sorted(fields),
            "variant_labels": {
                "base": "Qwen3.5-9B Base",
                "lora": "Qwen3.5-9B Base + LoRA",
            },
        }

    if layout == "legacy":
        data = _load_json(EVAL_DIR / "field_aggregate.json") or {}
        variants: dict[str, dict[str, float]] = {}
        fields: set[str] = set()
        for variant, values in data.items():
            if not isinstance(values, dict):
                continue
            variants[variant] = {}
            for field, value in values.items():
                if isinstance(value, dict):
                    value = value.get("avg", value)
                variants[variant][field] = _as_float(value)
                fields.add(field)
        return {
            "layout": layout,
            "variants": variants,
            "fields": sorted(fields),
            "variant_labels": {key: key for key in variants},
        }

    return {
        "layout": "missing",
        "variants": {},
        "fields": [],
        "variant_labels": {},
    }


@app.get("/api/eval/manifest")
async def eval_manifest():
    if _eval_layout() != "paired_vllm":
        return {"available": False, "eval_dir": str(EVAL_DIR)}
    data = _load_json(EVAL_DIR / "evaluation_manifest.json")
    return {
        "available": isinstance(data, dict),
        "eval_dir": str(EVAL_DIR),
        "manifest": data if isinstance(data, dict) else {},
    }


@app.get("/api/eval/samples")
async def list_samples(
    search: str = "",
    page: int = 0,
    page_size: int = 200,
    sort: str = "delta_desc",
):
    page = max(page, 0)
    page_size = max(1, min(page_size, 1000))

    if _eval_layout() == "paired_vllm":
        ordered_ids, base_map, lora_map = _paired_eval_maps()
        normalized_search = search.strip().lower()
        samples = []
        for sample_id in ordered_ids:
            base = base_map[sample_id]
            lora = lora_map[sample_id]
            source = str(lora.get("source", base.get("source", "")) or "")
            if normalized_search and (
                normalized_search not in sample_id.lower()
                and normalized_search not in source.lower()
            ):
                continue

            base_metrics = _record_metrics(base)
            lora_metrics = _record_metrics(lora)
            delta = lora_metrics["overall"] - base_metrics["overall"]
            gt_rows = lora.get("gt_rows", base.get("gt_rows", []))
            gt_rows = gt_rows if isinstance(gt_rows, list) else []
            base_parse = base.get("parse")
            lora_parse = lora.get("parse")
            samples.append(
                {
                    "id": sample_id,
                    "source": source,
                    "base_score": base_metrics["overall"],
                    "lora_score": lora_metrics["overall"],
                    "delta": delta,
                    "best_score": max(
                        base_metrics["overall"],
                        lora_metrics["overall"],
                    ),
                    "gt_rows_count": len(gt_rows),
                    "base_schema_valid": bool(
                        base_parse.get("schema_valid", False)
                        if isinstance(base_parse, dict)
                        else False
                    ),
                    "lora_schema_valid": bool(
                        lora_parse.get("schema_valid", False)
                        if isinstance(lora_parse, dict)
                        else False
                    ),
                }
            )

        sorters = {
            "delta_desc": lambda item: (-item["delta"], str(item["id"])),
            "delta_asc": lambda item: (item["delta"], str(item["id"])),
            "lora_desc": lambda item: (-item["lora_score"], str(item["id"])),
            "base_desc": lambda item: (-item["base_score"], str(item["id"])),
            "id": lambda item: str(item["id"]),
        }
        samples.sort(key=sorters.get(sort, sorters["delta_desc"]))
        start = page * page_size
        return {
            "layout": "paired_vllm",
            "total": len(samples),
            "page": page,
            "page_size": page_size,
            "samples": samples[start:start + page_size],
        }

    if _eval_layout() == "legacy":
        directory = EVAL_DIR / "per_sample"
        if not directory.exists():
            return {"layout": "legacy", "total": 0, "samples": []}
        samples = []
        for path in directory.glob("*.json"):
            data = _load_json(path)
            if not isinstance(data, dict):
                continue
            sample_id = str(data.get("id", path.stem))
            if search and search not in sample_id:
                continue
            scores = []
            metrics = data.get("metrics")
            metrics = metrics if isinstance(metrics, dict) else {}
            for metric in metrics.values():
                if not isinstance(metric, dict):
                    continue
                value = metric.get("overall", metric.get("avg_overall", 0))
                if isinstance(value, dict):
                    value = value.get("avg", 0)
                scores.append(_as_float(value))
            gt_rows = data.get("gt_rows", [])
            samples.append(
                {
                    "id": sample_id,
                    "source": data.get("source", ""),
                    "best_score": max(scores, default=0),
                    "gt_rows_count": len(gt_rows) if isinstance(gt_rows, list) else 0,
                }
            )
        start = page * page_size
        return {
            "layout": "legacy",
            "total": len(samples),
            "page": page,
            "page_size": page_size,
            "samples": samples[start:start + page_size],
        }

    return {"layout": "missing", "total": 0, "samples": []}


@app.get("/api/eval/samples/{sample_id}")
async def get_sample(sample_id: str):
    layout = _eval_layout()
    if layout == "paired_vllm":
        _, base_map, lora_map = _paired_eval_maps()
        base = base_map.get(str(sample_id))
        lora = lora_map.get(str(sample_id))
        if base is None or lora is None:
            raise HTTPException(404, f"Sample {sample_id} not found")

        base_metrics = _record_metrics(base)
        lora_metrics = _record_metrics(lora)
        gt_rows = lora.get("gt_rows", base.get("gt_rows", []))
        gt_rows = gt_rows if isinstance(gt_rows, list) else []

        def variant_payload(
            record: dict[str, Any],
            metrics: dict[str, Any],
            label: str,
        ) -> dict[str, Any]:
            rows = record.get("pred_rows", [])
            rows = rows if isinstance(rows, list) else []
            parse = record.get("parse")
            return {
                "label": label,
                "rows": rows,
                "metrics": metrics,
                "parse": parse if isinstance(parse, dict) else {},
                "finish_reason": record.get("finish_reason"),
                "prompt_token_count": int(
                    record.get("prompt_token_count", 0) or 0
                ),
                "output_token_count": int(
                    record.get("output_token_count", 0) or 0
                ),
                "raw_output": str(record.get("raw_output", "") or ""),
            }

        return {
            "layout": layout,
            "id": sample_id,
            "source": lora.get("source", base.get("source", "")),
            "data_line_number": lora.get(
                "data_line_number",
                base.get("data_line_number"),
            ),
            "gt_rows": gt_rows,
            "gt_rows_count": len(gt_rows),
            "variants": {
                "base": variant_payload(
                    base,
                    base_metrics,
                    "Qwen3.5-9B Base",
                ),
                "lora": variant_payload(
                    lora,
                    lora_metrics,
                    "Qwen3.5-9B Base + LoRA",
                ),
            },
            "variant_keys": ["base", "lora"],
            "comparison": {
                "overall_delta": (
                    lora_metrics["overall"] - base_metrics["overall"]
                ),
                "outputs_identical": (
                    str(base.get("raw_output", ""))
                    == str(lora.get("raw_output", ""))
                ),
                "lora_better": (
                    lora_metrics["overall"] > base_metrics["overall"]
                ),
            },
            "device_checks": {
                "gt": _check_device_rows(gt_rows),
                "base": _check_device_rows(base.get("pred_rows", [])),
                "lora": _check_device_rows(lora.get("pred_rows", [])),
            },
        }

    if layout == "legacy":
        path = EVAL_DIR / "per_sample" / f"{sample_id}.json"
        data = _load_json(path)
        if not isinstance(data, dict):
            raise HTTPException(404, f"Sample {sample_id} not found")
        return data

    raise HTTPException(404, "Evaluation data is not configured")


@app.get("/api/eval/samples/{sample_id}/device_check")
async def device_check(sample_id: str):
    detail = await get_sample(sample_id)
    if detail.get("layout") == "paired_vllm":
        return detail.get("device_checks", {})

    result = {"gt": _check_device_rows(detail.get("gt_rows", []))}
    variants = detail.get("variants")
    variants = variants if isinstance(variants, dict) else {}
    for key, value in variants.items():
        rows = (
            value.get("rows", value.get("pred_rows", []))
            if isinstance(value, dict)
            else value
        )
        result[key] = _check_device_rows(rows)
    return result


@app.post("/api/infer/qwen3")
async def infer_qwen3(_: InferReq):
    return {
        "success": False,
        "error": "外部 Qwen API 已禁用：当前服务采用严格离线模式",
    }


@app.post("/api/export/xlsx")
async def export_xlsx(req: ExportWorkbookReq):
    """下载一个模型结果；不在服务器保存用户编辑后的副本。"""

    if req.variant not in {"base", "lora"}:
        raise HTTPException(400, "variant 只能是 base 或 lora")
    if not req.final_rows:
        raise HTTPException(400, "没有可导出的结构化测试用例")
    if len(req.final_rows) > 1000:
        raise HTTPException(400, "单次最多导出 1000 行")
    if len(req.original_rows) != len(req.final_rows):
        raise HTTPException(400, "原始结果与最终结果行数不一致")
    if len(req.modifications) > len(req.final_rows):
        raise HTTPException(400, "修改审计数量不能超过结果行数")
    if len(req.ao_text) > 200_000:
        raise HTTPException(400, "AO 原文过长，无法导出")

    modified_indexes: set[int] = set()
    for modification in req.modifications:
        row_index = modification.get("row_index")
        if (
            isinstance(row_index, bool)
            or not isinstance(row_index, int)
            or row_index < 0
            or row_index >= len(req.final_rows)
        ):
            raise HTTPException(400, "修改审计中存在非法 row_index")
        if row_index in modified_indexes:
            raise HTTPException(400, "同一输出行不能出现重复修改审计")
        modified_indexes.add(row_index)

    try:
        workbook = await asyncio.to_thread(
            build_ao_workbook,
            variant=req.variant,
            model_name=req.model_name,
            ao_text=req.ao_text,
            prompt_mode=req.prompt_mode,
            original_rows=req.original_rows,
            final_rows=req.final_rows,
            modifications=req.modifications,
            generation=req.generation,
        )
    except Exception as exc:
        raise HTTPException(
            500,
            f"Excel 生成失败: {type(exc).__name__}: {str(exc)[:400]}",
        ) from exc

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"ao_qwen35_{req.variant}_{timestamp}.xlsx"
    return Response(
        content=workbook,
        media_type=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.post("/api/infer/base")
async def infer_base(req: InferReq):
    ao_text = _validate_ao_text(req.ao_text)
    prompt_mode = _normalize_prompt_mode(req.variant)
    return await _infer_one(
        model_name=VLLM_BASE_MODEL,
        ao_text=ao_text,
        prompt_mode=prompt_mode,
        system_prompt=_system_prompt(prompt_mode),
    )


@app.post("/api/infer/lora")
async def infer_lora(req: InferReq):
    ao_text = _validate_ao_text(req.ao_text)
    prompt_mode = _normalize_prompt_mode(req.variant)
    return await _infer_one(
        model_name=VLLM_LORA_MODEL,
        ao_text=ao_text,
        prompt_mode=prompt_mode,
        system_prompt=_system_prompt(prompt_mode),
    )


@app.post("/api/infer/compare")
async def infer_compare(req: InferReq):
    """用完全相同的上下文并行请求 Base 与 Base+LoRA。"""

    ao_text = _validate_ao_text(req.ao_text)
    prompt_mode = _normalize_prompt_mode(req.variant)
    system_prompt = _system_prompt(prompt_mode)
    started = time.perf_counter()

    base_result, lora_result = await asyncio.gather(
        _infer_one(
            model_name=VLLM_BASE_MODEL,
            ao_text=ao_text,
            prompt_mode=prompt_mode,
            system_prompt=system_prompt,
        ),
        _infer_one(
            model_name=VLLM_LORA_MODEL,
            ao_text=ao_text,
            prompt_mode=prompt_mode,
            system_prompt=system_prompt,
        ),
    )

    both_succeeded = bool(
        base_result.get("success") and lora_result.get("success")
    )
    return {
        "success": both_succeeded,
        "prompt_mode": prompt_mode,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        "base": base_result,
        "lora": lora_result,
        "comparison": {
            "same_prompt": True,
            "same_generation_parameters": True,
            "outputs_identical": (
                both_succeeded
                and base_result.get("raw_output")
                == lora_result.get("raw_output")
            ),
            "row_count_delta": (
                int(lora_result.get("row_count", 0) or 0)
                - int(base_result.get("row_count", 0) or 0)
                if both_succeeded
                else None
            ),
        },
    }


async def _available_vllm_models() -> tuple[bool, list[str], str | None]:
    try:
        response = await _get_client().get(f"{VLLM_URL}/models")
        if response.status_code != 200:
            return (
                False,
                [],
                f"HTTP {response.status_code}: {response.text[:300]}",
            )
        body = response.json()
        models = [
            str(item.get("id"))
            for item in body.get("data", [])
            if isinstance(item, dict) and item.get("id")
        ]
        return True, models, None
    except Exception as exc:
        return False, [], str(exc)[:300]


@app.get("/api/health")
async def health():
    vllm_reachable, available_models, vllm_error = (
        await _available_vllm_models()
    )
    base_available = VLLM_BASE_MODEL in available_models
    lora_available = VLLM_LORA_MODEL in available_models
    comparison_ready = vllm_reachable and base_available and lora_available

    layout = _eval_layout()
    sample_count = 0
    if layout == "paired_vllm":
        summary = _load_json(EVAL_DIR / "vllm_compare_summary.json") or {}
        sample_count = int(summary.get("num_samples", 0) or 0)
    elif layout == "legacy":
        summary = _load_json(EVAL_DIR / "did_batch_summary.json") or {}
        sample_count = int(summary.get("num_samples", 0) or 0)

    retrieval = _device_retrieval_status()
    service_ready = comparison_ready and (
        not retrieval["enabled"] or retrieval["ready"]
    )
    return {
        "status": "ok" if service_ready else "degraded",
        "eval_data": layout != "missing",
        "eval_samples": sample_count > 0,
        "eval_layout": layout,
        "eval_dir": str(EVAL_DIR),
        "eval_sample_count": sample_count,
        "retrieval_db": DB_PATH.is_file(),
        "retrieval_db_path": str(DB_PATH),
        "retrieval_enabled": retrieval["enabled"],
        "retrieval_ready": retrieval["ready"],
        "retrieval_mode": retrieval["mode"],
        "retrieval_data_kind": retrieval["data_kind"],
        "retrieval_data_label": retrieval["data_label"],
        "retrieval_corpus_records": retrieval["corpus_records"],
        "retrieval_index_dir": retrieval["index_dir"],
        "retrieval_bge_model": retrieval["bge_model"],
        "retrieval_device": retrieval["device"],
        "retrieval_error": retrieval["error"],
        "retrieval_warning": retrieval["warning"],
        "device_retrieval": retrieval,
        "qwen_api": False,
        "qwen_api_status": "offline_disabled",
        "vllm": vllm_reachable and lora_available,
        "vllm_reachable": vllm_reachable,
        "comparison_ready": comparison_ready,
        "base_model": VLLM_BASE_MODEL,
        "base_model_available": base_available,
        "lora_model": VLLM_LORA_MODEL,
        "lora_model_available": lora_available,
        "available_models": available_models,
        "vllm_error": vllm_error,
        "prompt_mode": DEFAULT_PROMPT_MODE,
        "generation": {
            "temperature": INFER_TEMPERATURE,
            "top_p": INFER_TOP_P,
            "max_tokens": INFER_MAX_TOKENS,
            "enable_thinking": False,
        },
        "xlsx_export": True,
    }


@app.post("/api/vllm/reconnect")
async def vllm_reconnect():
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
    _http_client = None

    vllm_reachable, available_models, error = await _available_vllm_models()
    base_available = VLLM_BASE_MODEL in available_models
    lora_available = VLLM_LORA_MODEL in available_models
    comparison_ready = vllm_reachable and base_available and lora_available
    return {
        "success": comparison_ready,
        "vllm": vllm_reachable and lora_available,
        "vllm_reachable": vllm_reachable,
        "comparison_ready": comparison_ready,
        "base_model": VLLM_BASE_MODEL,
        "base_model_available": base_available,
        "lora_model": VLLM_LORA_MODEL,
        "lora_model_available": lora_available,
        "available_models": available_models,
        "error": error,
    }


@app.on_event("startup")
async def initialize_device_retrieval():
    await asyncio.to_thread(_initialize_device_retrieval_sync)
    status = _device_retrieval_status()
    if status["enabled"] and status["ready"]:
        print(
            "[retrieval] ready: "
            f"{status['data_label']}, records={status['corpus_records']}, "
            "mode=annotate (no replacement)"
        )
    elif status["enabled"]:
        print(f"[retrieval] unavailable: {status['error']}")
    else:
        print("[retrieval] disabled (AO_DEVICE_RETRIEVAL_MODE=off)")


@app.on_event("shutdown")
async def close_http_client():
    global _http_client
    if _http_client is not None and not _http_client.is_closed:
        await _http_client.aclose()
    _http_client = None


static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Offline AO Test Case R&D Platform"
    )
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", 8081)),
    )
    parser.add_argument("--vllm-url", default=VLLM_URL)
    parser.add_argument("--base-model", default=VLLM_BASE_MODEL)
    parser.add_argument(
        "--lora-model",
        "--vllm-model",
        dest="lora_model",
        default=VLLM_LORA_MODEL,
        help="LoRA 服务名；--vllm-model 作为旧参数名继续兼容",
    )
    parser.add_argument(
        "--eval-dir",
        default=str(EVAL_DIR),
        help="包含 vllm_compare_summary.json 和两份 predictions.jsonl 的目录",
    )
    parser.add_argument(
        "--retrieval-db",
        default=str(DB_PATH),
        help="设备指令语料 devices.jsonl；也用于原有的指令号精确命中校验",
    )
    parser.add_argument(
        "--device-retrieval-mode",
        choices=("off", "annotate"),
        default=DEVICE_RETRIEVAL_MODE,
        help="默认 off；annotate 仅返回 BM25+BGE 候选，不改写模型输出",
    )
    parser.add_argument(
        "--retrieval-index-dir",
        default=str(RETRIEVAL_INDEX_DIR or ""),
        help="包含 BM25 与 BGE 索引文件的本地目录",
    )
    parser.add_argument(
        "--retrieval-bge-model",
        default=str(RETRIEVAL_BGE_MODEL or ""),
        help="本地 BGE 模型目录；开启 annotate 时必填",
    )
    parser.add_argument(
        "--retrieval-device",
        default=RETRIEVAL_DEVICE,
        help="BGE 运行设备；建议 server.py 使用 cpu，避免占用 vLLM 显存",
    )
    parser.add_argument(
        "--retrieval-data-kind",
        choices=("example", "production"),
        default=RETRIEVAL_DATA_KIND,
        help="设备语料性质；示例数据必须保持 example",
    )
    parser.add_argument(
        "--retrieval-data-label",
        default=os.environ.get("AO_RETRIEVAL_DATA_LABEL", "").strip(),
        help="网页展示的设备语料名称；留空时按 data-kind 自动生成",
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=RETRIEVAL_TOP_K,
    )
    parser.add_argument(
        "--retrieval-candidate-pool",
        type=int,
        default=RETRIEVAL_CANDIDATE_POOL,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=INFER_TEMPERATURE,
    )
    parser.add_argument("--top-p", type=float, default=INFER_TOP_P)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=INFER_MAX_TOKENS,
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=VLLM_REQUEST_TIMEOUT,
        help="调用本地 vLLM 的总超时秒数",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("compressed", "full"),
        default=DEFAULT_PROMPT_MODE,
        help="请求未指定 variant 时使用的默认 System Prompt",
    )
    parser.add_argument(
        "--system-prompt",
        default=SYSTEM_PROMPT_OVERRIDE,
        help="可选的本地 Prompt 文件；设置后覆盖 compressed/full 模式",
    )
    args = parser.parse_args()

    VLLM_URL = args.vllm_url.rstrip("/")
    VLLM_BASE_MODEL = args.base_model.strip()
    VLLM_LORA_MODEL = args.lora_model.strip()
    EVAL_DIR = _resolve_project_path(args.eval_dir)
    DB_PATH = _resolve_project_path(args.retrieval_db)
    DEVICE_RETRIEVAL_MODE = args.device_retrieval_mode
    RETRIEVAL_INDEX_DIR = _resolve_optional_project_path(
        args.retrieval_index_dir
    )
    RETRIEVAL_BGE_MODEL = _resolve_optional_project_path(
        args.retrieval_bge_model
    )
    RETRIEVAL_DEVICE = args.retrieval_device.strip()
    RETRIEVAL_DATA_KIND = args.retrieval_data_kind
    RETRIEVAL_DATA_LABEL = args.retrieval_data_label.strip() or (
        "示例设备库（2 条，仅研发验证）"
        if RETRIEVAL_DATA_KIND == "example"
        else "生产设备库"
    )
    RETRIEVAL_TOP_K = args.retrieval_top_k
    RETRIEVAL_CANDIDATE_POOL = args.retrieval_candidate_pool
    INFER_TEMPERATURE = args.temperature
    INFER_TOP_P = args.top_p
    INFER_MAX_TOKENS = args.max_tokens
    VLLM_REQUEST_TIMEOUT = args.request_timeout
    DEFAULT_PROMPT_MODE = args.prompt_mode
    SYSTEM_PROMPT_OVERRIDE = args.system_prompt.strip()
    _validate_runtime_config()

    print("=" * 72)
    print(f"AO R&D Platform: http://{args.host}:{args.port}")
    print(f"vLLM endpoint: {VLLM_URL}")
    print(f"Base model: {VLLM_BASE_MODEL}")
    print(f"LoRA model: {VLLM_LORA_MODEL}")
    print(f"Evaluation: {EVAL_DIR} (layout={_eval_layout()})")
    print(f"Retrieval DB: {DB_PATH} (exists={DB_PATH.is_file()})")
    print(
        "Device retrieval: "
        f"mode={DEVICE_RETRIEVAL_MODE}, "
        f"data_kind={RETRIEVAL_DATA_KIND}, "
        f"label={RETRIEVAL_DATA_LABEL}"
    )
    print(f"Retrieval index: {RETRIEVAL_INDEX_DIR}")
    print(f"Retrieval BGE model: {RETRIEVAL_BGE_MODEL}")
    print(
        "Retrieval runtime: "
        f"device={RETRIEVAL_DEVICE}, "
        f"top_k={RETRIEVAL_TOP_K}, "
        f"candidate_pool={RETRIEVAL_CANDIDATE_POOL}"
    )
    print(
        "Generation: "
        f"temperature={INFER_TEMPERATURE}, "
        f"top_p={INFER_TOP_P}, "
        f"max_tokens={INFER_MAX_TOKENS}, "
        "enable_thinking=False"
    )
    print(f"Default prompt mode: {DEFAULT_PROMPT_MODE}")
    print(f"CORS origins: {CORS_ORIGINS}")
    print("=" * 72)
    uvicorn.run(app, host=args.host, port=args.port)
