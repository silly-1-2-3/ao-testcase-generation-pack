#!/usr/bin/env python3
"""Load production indexes and retrieve device commands."""

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

from retrieval.production import BGE_INDEX_FORMAT



def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DeviceRetriever:
    def __init__(
        self,
        devices_path: Path,
        index_dir: Path,
        bge_model: Path | None = None,
        device: str = "auto",
    ) -> None:
        from retrieval.bm25_index import BM25

        self.devices_path = devices_path
        self.devices = [
            json.loads(line)
            for line in devices_path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        self.by_primary_key = {value["设备指令主键"]: value for value in self.devices}
        self.by_command_code = {value["设备指令号"]: value for value in self.devices}
        self.bm25 = BM25()
        self.bm25.load(index_dir / "bm25_devices.pkl")
        manifest = json.loads((index_dir / "index_manifest.json").read_text(encoding="utf-8"))
        actual_hash = sha256(devices_path)
        if manifest["corpus_sha256"] != actual_hash:
            raise ValueError("Index corpus hash does not match devices JSONL; rebuild indexes")

        self.bge_model = None
        self.bge_vectors = None
        self.bge_query_instruction = ""
        self.bge_position: dict[str, int] = {}
        if bge_model:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            import numpy as np
            from sentence_transformers import SentenceTransformer

            metadata = json.loads((index_dir / "bge_meta.json").read_text(encoding="utf-8"))
            if metadata.get("format") != BGE_INDEX_FORMAT:
                raise ValueError(
                    "Unsupported BGE metadata format; rebuild indexes with the "
                    "current build_device_indexes.py"
                )
            if metadata["corpus_sha256"] != actual_hash:
                raise ValueError("BGE corpus hash does not match devices JSONL")
            query_instruction = metadata.get("query_instruction")
            if not isinstance(query_instruction, str):
                raise ValueError("BGE metadata is missing a string query_instruction")
            self.bge_query_instruction = query_instruction
            self.bge_vectors = np.load(index_dir / "bge_vectors.npy", allow_pickle=False)
            self.bge_position = {
                record_id: index for index, record_id in enumerate(metadata["record_ids"])
            }
            self.bge_model = SentenceTransformer(
                str(bge_model.resolve()),
                device=None if device == "auto" else device,
                local_files_only=True,
            )

    def is_known(self, identifier: str) -> bool:
        value = str(identifier or "").strip()
        return value in self.by_primary_key or value in self.by_command_code

    def search(self, query: str, top_k: int = 5, candidate_pool: int = 50) -> list[dict[str, Any]]:
        query = str(query).strip()
        if not query:
            return []
        bm25_results = self.bm25.search(query, max(top_k, candidate_pool))
        candidates = []
        for rank, (record, score) in enumerate(bm25_results, 1):
            candidates.append({
                "record": record,
                "primary_key": record["设备指令主键"],
                "bm25": float(score),
                "bm25_rank": rank,
                "bge": None,
                "bge_rank": None,
                "bge_margin_to_second": None,
            })
        if self.bge_model is not None and candidates:
            import numpy as np

            encoded_query = f"{self.bge_query_instruction}{query}"
            query_vector = self.bge_model.encode(
                [encoded_query],
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
            scores = []
            for candidate in candidates:
                position = self.bge_position[candidate["primary_key"]]
                score = float(np.dot(self.bge_vectors[position], query_vector))
                scores.append(score)
                candidate["bge"] = score
            order = sorted(range(len(candidates)), key=lambda index: -scores[index])
            for rank, index in enumerate(order, 1):
                candidates[index]["bge_rank"] = rank
            if len(order) >= 2:
                best_index, second_index = order[:2]
                candidates[best_index]["bge_margin_to_second"] = (
                    scores[best_index] - scores[second_index]
                )

        for item in candidates:
            item["rrf"] = 1.0 / (60 + item["bm25_rank"])
            if item["bge_rank"] is not None:
                item["rrf"] += 1.0 / (60 + item["bge_rank"])
        candidates.sort(key=lambda value: -value["rrf"])
        output = []
        for rank, item in enumerate(candidates[:top_k], 1):
            record = item["record"]
            output.append({
                "rank": rank,
                "设备指令主键": record["设备指令主键"],
                "设备指令号": record["设备指令号"],
                "设备类型": record["设备类型"],
                "设备类别主键": record["设备类别主键"],
                "设备指令功能说明": record["设备指令功能说明"],
                "bm25_score": round(item["bm25"], 6),
                "bm25_rank": item["bm25_rank"],
                "bge_score": round(item["bge"], 6) if item["bge"] is not None else None,
                "bge_rank": item["bge_rank"],
                "bge_margin_to_second": (
                    round(item["bge_margin_to_second"], 6)
                    if item["bge_margin_to_second"] is not None
                    else None
                ),
                "rrf_score": round(item["rrf"], 8),
            })
        return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Query the device command database.")
    parser.add_argument("--devices", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--bge-model", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-pool", type=int, default=50)
    args = parser.parse_args()
    retriever = DeviceRetriever(args.devices, args.index_dir, args.bge_model, args.device)
    results = retriever.search(args.query, args.top_k, args.candidate_pool)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
