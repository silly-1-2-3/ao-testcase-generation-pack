#!/usr/bin/env python3
"""Regression tests for real criterion-to-execution-step row selection."""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from retrieval.production import apply_device_retrieval as base
from retrieval.production import BGE_INDEX_FORMAT, DEFAULT_BGE_QUERY_INSTRUCTION
from retrieval.production.apply_device_retrieval import build_query
from retrieval.production.build_device_indexes import build_bge
from retrieval.production.device_retrieval_engine import DeviceRetriever

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "cases.jsonl"



class DeviceRowSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        line = next(value for value in FIXTURE.read_text(encoding="utf-8").splitlines() if value)
        cls.rows = json.loads(line)["rows"]

    def test_only_execution_steps_are_selected(self) -> None:
        selected = [
            index
            for index, row in enumerate(self.rows)
            if base.should_search(row, all_execution_steps=False)
        ]
        self.assertEqual(selected, [3, 6])

    def test_sibling_execution_context_is_isolated(self) -> None:
        voltage_query = build_query(self.rows, 3)
        torque_query = build_query(self.rows, 6)

        self.assertIn("测试点A直流电压", voltage_query)
        self.assertNotIn("扭矩扳手", voltage_query)
        self.assertIn("扭矩扳手", torque_query)
        self.assertNotIn("万用表", torque_query)


class KnownIdentifierDecisionTests(unittest.TestCase):
    class StubRetriever:
        @staticmethod
        def is_known(identifier: str) -> bool:
            return identifier in {
                "CMD-001",
                "CMD-002",
                "measure_dc_voltage",
                "apply_torque_cw",
            }

    @staticmethod
    def candidate(primary_key: str, command_code: str) -> dict[str, str]:
        return {
            "设备指令主键": primary_key,
            "设备指令号": command_code,
        }

    def decide(
        self,
        identifier: str,
        candidates: list[dict[str, str]],
    ) -> tuple[str, bool]:
        return base.replacement_decision(
            identifier,
            candidates,
            self.StubRetriever(),
            mode="annotate",
            bge_threshold=0.68,
            bge_margin=0.05,
        )

    def test_known_top1_is_kept(self) -> None:
        candidates = [
            self.candidate("CMD-002", "apply_torque_cw"),
            self.candidate("CMD-001", "measure_dc_voltage"),
        ]
        self.assertEqual(
            self.decide("apply_torque_cw", candidates),
            ("kept_known_identifier_top1", False),
        )

    def test_known_but_wrong_top1_requires_review(self) -> None:
        candidates = [
            self.candidate("CMD-001", "measure_dc_voltage"),
            self.candidate("CMD-002", "apply_torque_cw"),
        ]
        self.assertEqual(
            self.decide("apply_torque_cw", candidates),
            ("review_known_identifier_not_top1", False),
        )

    def test_known_but_not_retrieved_requires_review(self) -> None:
        candidates = [
            self.candidate("CMD-001", "measure_dc_voltage"),
        ]
        self.assertEqual(
            self.decide("apply_torque_cw", candidates),
            ("review_known_identifier_not_retrieved", False),
        )

    def test_primary_key_can_match_candidate(self) -> None:
        candidates = [
            self.candidate("CMD-002", "apply_torque_cw"),
        ]
        self.assertEqual(
            self.decide("CMD-002", candidates),
            ("kept_known_identifier_top1", False),
        )


class UnknownIdentifierReplacementTests(unittest.TestCase):
    class StubRetriever:
        @staticmethod
        def is_known(identifier: str) -> bool:
            return False

    @staticmethod
    def candidate(
        *,
        bge_score: float | None,
        bge_rank: int | None,
        bge_margin_to_second: float | None,
    ) -> dict[str, object]:
        return {
            "设备指令主键": "CMD-001",
            "设备指令号": "measure_dc_voltage",
            "bge_score": bge_score,
            "bge_rank": bge_rank,
            "bge_margin_to_second": bge_margin_to_second,
        }

    def decide(self, candidate: dict[str, object]) -> tuple[str, bool]:
        return base.replacement_decision(
            "hallucinated_voltage",
            [candidate],
            self.StubRetriever(),
            mode="replace-invalid",
            bge_threshold=0.68,
            bge_margin=0.05,
        )

    def test_bm25_only_never_auto_replaces(self) -> None:
        self.assertEqual(
            self.decide(
                self.candidate(
                    bge_score=None,
                    bge_rank=None,
                    bge_margin_to_second=None,
                )
            ),
            ("review_bm25_only_not_auto_replaced", False),
        )

    def test_rrf_top_must_also_be_bge_top1(self) -> None:
        self.assertEqual(
            self.decide(
                self.candidate(
                    bge_score=0.82,
                    bge_rank=2,
                    bge_margin_to_second=None,
                )
            ),
            ("review_rrf_top_not_bge_top1", False),
        )

    def test_bge_score_must_reach_threshold(self) -> None:
        self.assertEqual(
            self.decide(
                self.candidate(
                    bge_score=0.67,
                    bge_rank=1,
                    bge_margin_to_second=0.20,
                )
            ),
            ("review_bge_below_threshold", False),
        )

    def test_bge_margin_must_be_available(self) -> None:
        self.assertEqual(
            self.decide(
                self.candidate(
                    bge_score=0.82,
                    bge_rank=1,
                    bge_margin_to_second=None,
                )
            ),
            ("review_bge_margin_unavailable", False),
        )

    def test_bge_margin_must_reach_threshold(self) -> None:
        self.assertEqual(
            self.decide(
                self.candidate(
                    bge_score=0.82,
                    bge_rank=1,
                    bge_margin_to_second=0.049,
                )
            ),
            ("review_bge_margin_too_small", False),
        )

    def test_consensus_candidate_can_replace_invalid_identifier(self) -> None:
        self.assertEqual(
            self.decide(
                self.candidate(
                    bge_score=0.82,
                    bge_rank=1,
                    bge_margin_to_second=0.12,
                )
            ),
            ("replaced_unknown_identifier", True),
        )


class OutputPathSafetyTests(unittest.TestCase):
    def test_input_output_and_audit_must_be_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "same.jsonl"
            with self.assertRaises(ValueError):
                base.validate_io_paths(path, path, path, force=False)

    def test_existing_outputs_require_force(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.jsonl"
            output_path = root / "output.jsonl"
            audit_path = root / "audit.jsonl"
            output_path.touch()

            with self.assertRaises(FileExistsError):
                base.validate_io_paths(
                    input_path,
                    output_path,
                    audit_path,
                    force=False,
                )

            base.validate_io_paths(
                input_path,
                output_path,
                audit_path,
                force=True,
            )


class BgeEncodingPolicyTests(unittest.TestCase):
    def test_build_records_query_policy_without_prefixing_documents(self) -> None:
        encoded_texts = []

        class FakeSentenceTransformer:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def encode(self, texts, **kwargs):
                encoded_texts.extend(texts)
                return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        fake_module = types.ModuleType("sentence_transformers")
        fake_module.SentenceTransformer = FakeSentenceTransformer
        devices = [
            {"设备指令主键": "CMD-001", "_text": "直流电压"},
            {"设备指令主键": "CMD-002", "_text": "目标力矩"},
        ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / "model"
            model_dir.mkdir()
            vectors_path = root / "bge_vectors.npy"
            meta_path = root / "bge_meta.json"
            with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
                build_bge(
                    devices,
                    model_dir,
                    vectors_path,
                    meta_path,
                    "corpus-hash",
                    batch_size=2,
                    device="cpu",
                    query_instruction=DEFAULT_BGE_QUERY_INSTRUCTION,
                )

            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(encoded_texts, ["直流电压", "目标力矩"])
            self.assertEqual(metadata["format"], BGE_INDEX_FORMAT)
            self.assertEqual(
                metadata["query_instruction"],
                DEFAULT_BGE_QUERY_INSTRUCTION,
            )
            self.assertEqual(metadata["document_instruction"], "")
            self.assertTrue(metadata["normalize_embeddings"])

    def test_search_prepends_metadata_query_instruction(self) -> None:
        record = {
            "设备指令主键": "CMD-001",
            "设备指令号": "measure_dc_voltage",
            "设备类型": "数字万用表",
            "设备类别主键": "CAT-001",
            "设备指令功能说明": "测量直流电压",
        }

        class StubBM25:
            @staticmethod
            def search(query, limit):
                return [(record, 10.0)]

        class StubBgeModel:
            def __init__(self) -> None:
                self.inputs = []

            def encode(self, texts, **kwargs):
                self.inputs.extend(texts)
                return np.asarray([[1.0, 0.0]], dtype=np.float32)

        retriever = DeviceRetriever.__new__(DeviceRetriever)
        retriever.bm25 = StubBM25()
        retriever.bge_model = StubBgeModel()
        retriever.bge_vectors = np.asarray([[1.0, 0.0]], dtype=np.float32)
        retriever.bge_position = {"CMD-001": 0}
        retriever.bge_query_instruction = DEFAULT_BGE_QUERY_INSTRUCTION

        retriever.search("测量测试点电压", top_k=1, candidate_pool=1)

        self.assertEqual(
            retriever.bge_model.inputs,
            [DEFAULT_BGE_QUERY_INSTRUCTION + "测量测试点电压"],
        )


if __name__ == "__main__":
    unittest.main()
