#!/usr/bin/env python3
"""Build a BM25 index and optional offline BGE embedding matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.production import BGE_INDEX_FORMAT, DEFAULT_BGE_QUERY_INSTRUCTION



def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_devices(path: Path) -> list[dict[str, Any]]:
    devices = []
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not value.get("设备指令主键") or not value.get("设备指令号"):
                raise ValueError(f"{path}:{line_number}: missing device command identifier")
            devices.append(value)
    if not devices:
        raise ValueError("Device database is empty")
    return devices


def refuse_existing(paths: list[Path], force: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not force:
        raise FileExistsError(f"Outputs already exist; use --force: {existing}")


def build_bm25(devices: list[dict[str, Any]], output: Path) -> None:
    from retrieval.bm25_index import BM25

    index = BM25()
    index.build_doc_text = lambda doc: str(doc.get("_text", ""))
    index.fit(devices)
    index.save(output)


def build_bge(
    devices: list[dict[str, Any]],
    model_path: Path,
    vectors_path: Path,
    meta_path: Path,
    corpus_hash: str,
    batch_size: int,
    device: str,
    query_instruction: str,
) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import numpy as np
    from sentence_transformers import SentenceTransformer

    if not model_path.is_dir():
        raise FileNotFoundError(f"Local BGE model directory not found: {model_path}")
    model = SentenceTransformer(
        str(model_path.resolve()),
        device=None if device == "auto" else device,
        local_files_only=True,
    )
    texts = [str(value["_text"]) for value in devices]
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    np.save(vectors_path, vectors, allow_pickle=False)
    metadata = {
        "format": BGE_INDEX_FORMAT,
        "corpus_sha256": corpus_hash,
        "model_path": str(model_path.resolve()),
        "records": len(devices),
        "dimensions": int(vectors.shape[1]),
        "record_ids": [value["设备指令主键"] for value in devices],
        "document_instruction": "",
        "query_instruction": query_instruction,
        "normalize_embeddings": True,
        "similarity": "cosine_via_normalized_dot_product",
    }
    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build production device retrieval indexes.")
    parser.add_argument("--devices", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bge-model", type=Path, help="Offline SentenceTransformer directory")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    parser.add_argument(
        "--bge-query-instruction",
        default=DEFAULT_BGE_QUERY_INSTRUCTION,
        help=(
            "Instruction prepended to BGE queries only. "
            "Pass an empty string to disable it."
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bm25_path = args.output_dir / "bm25_devices.pkl"
    bge_vectors = args.output_dir / "bge_vectors.npy"
    bge_meta = args.output_dir / "bge_meta.json"
    build_meta = args.output_dir / "index_manifest.json"
    outputs = [bm25_path, build_meta]
    if args.bge_model:
        outputs.extend([bge_vectors, bge_meta])
    refuse_existing(outputs, args.force)

    devices = load_devices(args.devices)
    corpus_hash = sha256(args.devices)
    build_bm25(devices, bm25_path)
    if args.bge_model:
        build_bge(
            devices, args.bge_model, bge_vectors, bge_meta, corpus_hash,
            args.batch_size, args.device, args.bge_query_instruction,
        )
    manifest = {
        "format": "device-index-manifest-v2",
        "devices": str(args.devices.resolve()),
        "corpus_sha256": corpus_hash,
        "records": len(devices),
        "bm25": str(bm25_path.resolve()),
        "bge_enabled": bool(args.bge_model),
        "bge_model": str(args.bge_model.resolve()) if args.bge_model else None,
        "bge_query_instruction": (
            args.bge_query_instruction if args.bge_model else None
        ),
    }
    build_meta.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[ok] BM25: {bm25_path} ({len(devices)} records)")
    if args.bge_model:
        print(f"[ok] BGE:   {bge_vectors}")
    print(f"[ok] manifest: {build_meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
