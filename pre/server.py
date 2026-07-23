"""Minimal offline AO test-case platform server.

This replacement keeps the API shape expected by static/index.html while
using only a local OpenAI-compatible vLLM endpoint for inference.
"""

import argparse
import json
import os
import re
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


app = FastAPI(title="AO Test Case Platform")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / "train"
EVAL_DIR = Path(
    os.environ.get(
        "AO_EVAL_DIR",
        str(TRAIN_DIR / "eval_results_did_v4_v2" / "batch"),
    )
)
DB_PATH = Path(
    os.environ.get(
        "AO_RETRIEVAL_DB",
        str(PROJECT_ROOT / "retrieval" / "device_corpus.jsonl"),
    )
)

VLLM_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "qwen25-7b-lora")
STEP_LIST_KEY = "\u6b65\u9aa4\u5217\u8868"

_http_client = None
_cache = {}


def _get_client():
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=180)
    return _http_client


def _load_json(path: Path):
    key = str(path)
    if key not in _cache:
        if path.exists():
            with path.open(encoding="utf-8") as f:
                _cache[key] = json.load(f)
        else:
            _cache[key] = None
    return _cache[key]


def _parse_json(raw):
    """Parse a row list from raw JSON, fenced JSON, or Qwen thinking output."""
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
    text = re.sub(r"^\s*\x60\x60\x60(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\x60\x60\x60\s*$", "", text).strip()

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


def _load_device_codes():
    codes = set()
    if not DB_PATH.exists():
        return codes
    with DB_PATH.open(encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            code = str(item.get("设备指令号", "")).strip()
            if code and code not in ("[]", "null"):
                codes.add(code)
    return codes


def _check_device_rows(rows):
    if not isinstance(rows, list):
        return None
    codes = _load_device_codes()
    details = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        if row.get("步骤层级") != "执行步骤":
            continue
        code = str(row.get("设备指令号", "")).strip()
        if not code or code in ("[]", "null"):
            continue
        details.append({
            "row_idx": index,
            "code": code,
            "type": row.get("设备类型", ""),
            "unit": row.get("设备单元号", ""),
            "in_db": code in codes if codes else None,
        })
    if not details:
        return None
    known = [item for item in details if item["in_db"] is not None]
    found = sum(1 for item in known if item["in_db"])
    return {
        "total": len(details),
        "found": found,
        "not_found": len(known) - found,
        "details": details,
    }


def _system_prompt():
    prompt_path = TRAIN_DIR / "system_prompt_v4_full.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return (
        "将用户提供的 AO 指令转换为 JSON 数组。只输出 JSON，"
        "每一项是一个测试用例表格行。"
    )


class InferReq(BaseModel):
    ao_text: str
    variant: str = "lora"


@app.get("/api/eval/summary")
async def eval_summary():
    data = _load_json(EVAL_DIR / "did_batch_summary.json")
    if not data:
        return {"num_samples": 0, "table": []}
    table = []
    for variant, values in data.get("aggregated", {}).items():
        values = values or {}
        def average(name):
            value = values.get(name, 0)
            return value.get("avg", 0) if isinstance(value, dict) else value
        table.append({
            "variant": variant,
            "label": variant,
            "overall": average("avg_overall"),
            "structure": average("avg_structure"),
            "content": average("avg_content"),
        })
    return {"num_samples": data.get("num_samples", 0), "table": table}


@app.get("/api/eval/field_scores")
async def field_scores():
    data = _load_json(EVAL_DIR / "field_aggregate.json") or {}
    variants = {}
    fields = set()
    for variant, values in data.items():
        variants[variant] = {}
        for field, value in values.items():
            variants[variant][field] = (
                value.get("avg", value) if isinstance(value, dict) else value
            )
            fields.add(field)
    return {
        "variants": variants,
        "fields": sorted(fields),
        "variant_labels": {key: key for key in variants},
    }


@app.get("/api/eval/samples")
async def list_samples(search: str = "", page: int = 0, page_size: int = 200):
    directory = EVAL_DIR / "per_sample"
    if not directory.exists():
        return {"total": 0, "samples": []}
    samples = []
    for path in directory.glob("*.json"):
        data = _load_json(path)
        if not data or (search and search not in str(data.get("id", path.stem))):
            continue
        scores = []
        for metric in data.get("metrics", {}).values():
            value = metric.get("overall", metric.get("avg_overall", 0))
            if isinstance(value, dict):
                value = value.get("avg", 0)
            scores.append(value or 0)
        samples.append({
            "id": data.get("id", path.stem),
            "source": data.get("source", ""),
            "best_score": max(scores, default=0),
            "gt_rows_count": len(data.get("gt_rows", [])),
        })
    samples.sort(key=lambda item: -item["best_score"])
    start = page * page_size
    return {
        "total": len(samples),
        "samples": samples[start:start + page_size],
    }


@app.get("/api/eval/samples/{sid}")
async def get_sample(sid: str):
    path = EVAL_DIR / "per_sample" / f"{sid}.json"
    data = _load_json(path)
    if not data:
        raise HTTPException(404, f"Sample {sid} not found")
    variants = {}
    for key, value in data.get("variants", {}).items():
        rows = value.get("rows", value.get("pred_rows", [])) if isinstance(value, dict) else value
        metric = data.get("metrics", {}).get(key, {})
        variants[key] = {
            "label": key,
            "rows": rows if isinstance(rows, list) else [],
            "metrics": {
                "overall": metric.get("overall", metric.get("avg_overall", 0)),
                "structure": metric.get("structure", metric.get("avg_structure", 0)),
                "content": metric.get("content", metric.get("avg_content", 0)),
            },
        }
    return {
        "id": data.get("id", sid),
        "source": data.get("source", ""),
        "gt_rows": data.get("gt_rows", []),
        "gt_rows_count": len(data.get("gt_rows", [])),
        "variants": variants,
        "variant_keys": sorted(variants),
    }


@app.get("/api/eval/samples/{sid}/device_check")
async def device_check(sid: str):
    path = EVAL_DIR / "per_sample" / f"{sid}.json"
    data = _load_json(path)
    if not data:
        raise HTTPException(404)
    result = {"gt": _check_device_rows(data.get("gt_rows", []))}
    for key, value in data.get("variants", {}).items():
        rows = value.get("rows", value.get("pred_rows", [])) if isinstance(value, dict) else value
        result[key] = _check_device_rows(rows)
    return result


@app.post("/api/infer/qwen3")
async def infer_qwen3(req: InferReq):
    return {
        "success": False,
        "error": "Qwen API disabled: this server is offline",
    }


@app.post("/api/infer/lora")
async def infer_lora(req: InferReq):
    ao = req.ao_text.strip()
    if len(ao) < 20:
        raise HTTPException(400, "AO text must contain at least 20 characters")
    payload = {
        "model": VLLM_MODEL,
        "messages": [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": ao},
        ],
        "temperature": 0.1,
        "max_tokens": 4096,
    }
    try:
        response = await _get_client().post(f"{VLLM_URL}/chat/completions", json=payload)
        if response.status_code != 200:
            return {
                "success": False,
                "error": f"vLLM {response.status_code}: {response.text[:300]}",
            }
        data = response.json()
        table = _parse_json(data)
        return {
            "success": True,
            "table": table,
            "row_count": len(table),
            "device_check": _check_device_rows(table),
            "raw": json.dumps(data, ensure_ascii=False)[:1000],
        }
    except Exception as exc:
        return {"success": False, "error": f"vLLM: {str(exc)[:300]}"}


@app.get("/api/health")
async def health():
    vllm_ok = False
    try:
        response = await _get_client().get(f"{VLLM_URL}/models")
        vllm_ok = response.status_code == 200
    except Exception:
        pass
    return {
        "status": "ok",
        "eval_data": EVAL_DIR.exists(),
        "eval_samples": (EVAL_DIR / "per_sample").exists(),
        "retrieval_db": DB_PATH.exists(),
        "qwen_api": False,
        "vllm": vllm_ok,
        "vllm_model": VLLM_MODEL,
    }


@app.post("/api/vllm/reconnect")
async def vllm_reconnect():
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
    _http_client = None
    try:
        response = await _get_client().get(f"{VLLM_URL}/models")
        return {"success": True, "vllm": response.status_code == 200}
    except Exception as exc:
        return {"success": False, "vllm": False, "error": str(exc)[:300]}


static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline AO Test Case Platform")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8081)))
    parser.add_argument("--vllm-url", default=VLLM_URL)
    parser.add_argument("--vllm-model", default=VLLM_MODEL)
    args = parser.parse_args()
    VLLM_URL = args.vllm_url
    VLLM_MODEL = args.vllm_model
    print(f"AO Platform on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

