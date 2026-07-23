#!/usr/bin/env python3
"""Deterministic metrics for AO-to-table generation.

This module is intentionally independent from ``metrics.py``.  It keeps the
old metric available for reproducing previous experiments while providing:

* one-to-one, monotonic dynamic-programming row alignment;
* character chrF-style similarity instead of set-only bigram Jaccard;
* separate row coverage/precision signals suitable for reward shaping; and
* an optional contextual similarity callback for offline evaluation.

Rows are only eligible to match when their ``步骤层级`` values are equal.
Within one level, the default alignment preserves row order.  This is the
right inductive bias for this project: a generated table is a sequence, and
the 8-versus-7 case has eight monotonic deletion choices rather than 8!
unordered permutations.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


SKIP_FIELDS = {"设备类型", "设备单元号", "设备指令号", "设备参数"}

LEVEL_FIELDS = {
    "用例": ["步骤层级", "说明", "注意事项"],
    "子用例": ["步骤层级", "说明", "注意事项"],
    "步骤": [
        "步骤层级", "说明", "注意事项", "操作内容", "操作对象", "操作目的",
        "是否同时发送", "多判据组合条件",
    ],
    "判据": [
        "步骤层级", "是否使用设备", "操作类型", "判据类型", "判据范围",
        "判据描述", "左值", "右值", "单位",
    ],
    "执行步骤": [
        "步骤层级", "设备类型", "设备单元号", "设备指令号", "设备参数",
        "判据关联标志",
    ],
}

FIELD_WEIGHTS = {
    "说明": 1.5,
    "操作内容": 1.3,
    "操作对象": 1.3,
    "注意事项": 1.2,
    "判据描述": 1.2,
    "操作目的": 1.1,
    "步骤层级": 1.0,
    "是否同时发送": 1.0,
    "多判据组合条件": 1.0,
    "是否使用设备": 1.0,
    "操作类型": 1.0,
    "判据类型": 1.0,
    "判据范围": 1.0,
    "左值": 1.0,
    "右值": 1.0,
    "判据关联标志": 1.0,
    "单位": 0.8,
}

REFERENCE_FIELDS = {"左值", "右值", "判据关联标志"}
NUMERIC_FIELDS = {"左值", "右值", "判据范围"}
LEVEL_ORDER = ("用例", "子用例", "步骤", "判据", "执行步骤")


def normalize_reference_expr(value: str) -> str:
    """Normalize reference identifiers while retaining expression structure."""
    if not isinstance(value, str) or not value.strip():
        return value
    value = value.strip()
    if "@" not in value:
        return value
    pattern = re.compile(r"@[A-Za-z0-9_\u4e00-\u9fff]+")
    value = pattern.sub("@ID", value)
    value = re.sub(r"\s*([*+\-/])\s*", r" \1 ", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_numeric(value: str) -> str:
    """Normalize scalar or semicolon-separated numeric values."""
    parts = value.split(";") if ";" in value else [value]
    result = []
    for part in parts:
        part = part.strip()
        try:
            number = float(part)
        except ValueError:
            result.append(part)
            continue
        result.append(str(int(number)) if number.is_integer() else f"{number:.6g}")
    return ";".join(result)


def normalize_value_for_comparison(value: Any, field_name: str) -> str:
    if value is None:
        return ""
    value = unicodedata.normalize("NFKC", str(value)).strip()
    if not value or value in {"null", "None", "[]"}:
        return ""
    if field_name in REFERENCE_FIELDS:
        return normalize_reference_expr(value)
    if field_name in {"是否同时发送", "是否使用设备"}:
        return value.casefold()
    if field_name in NUMERIC_FIELDS:
        return normalize_numeric(value)
    if field_name == "多判据组合条件":
        value = value.replace("全部成功", "all").replace("任一成功", "any")
    return value


def _text_surface(value: str) -> str:
    """Remove formatting-only differences before character metrics."""
    return re.sub(r"\s+", "", value.casefold())


def _ngrams(value: str, n: int) -> Counter:
    if not value:
        return Counter()
    if len(value) < n:
        return Counter({value: 1})
    return Counter(value[i : i + n] for i in range(len(value) - n + 1))


def chrf_similarity(a: str, b: str, max_order: int = 6, beta: float = 2.0) -> float:
    """Character n-gram F-score, a chrF-style deterministic text metric.

    Unlike set Jaccard, this uses n-gram multiplicity and precision/recall.
    It is useful for Chinese fields, where word segmentation is unavailable
    or unstable.  ``beta=2`` gives recall a little more weight, matching the
    usual chrF convention.
    """
    a, b = _text_surface(a), _text_surface(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    scores = []
    for order in range(1, min(max_order, len(a), len(b)) + 1):
        grams_a, grams_b = _ngrams(a, order), _ngrams(b, order)
        overlap = sum((grams_a & grams_b).values())
        precision = overlap / sum(grams_b.values())
        recall = overlap / sum(grams_a.values())
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append((1 + beta * beta) * precision * recall /
                          (beta * beta * precision + recall))
    return sum(scores) / len(scores) if scores else 0.0


def lcs_length(a: str, b: str) -> int:
    """Return LCS length using O(min(len(a), len(b))) memory."""
    if len(a) < len(b):
        a, b = b, a
    previous = [0] * (len(b) + 1)
    for char_a in a:
        current = [0]
        for j, char_b in enumerate(b, start=1):
            current.append(previous[j - 1] + 1 if char_a == char_b else
                           max(previous[j], current[j - 1]))
        previous = current
    return previous[-1]


def lcs_ratio(a: str, b: str) -> float:
    """Symmetric LCS-over-longest-sequence ratio."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return lcs_length(a, b) / max(len(a), len(b))


def _numeric_similarity(a: str, b: str) -> Optional[float]:
    """Compare numeric scalar/range strings, or return None for text."""
    try:
        left = [float(x.strip()) for x in a.split(";")]
        right = [float(x.strip()) for x in b.split(";")]
    except ValueError:
        return None
    if len(left) != len(right):
        return 0.0
    errors = []
    for x, y in zip(left, right):
        denominator = max(abs(x), abs(y), 1e-9)
        errors.append(abs(x - y) / denominator)
    error = sum(errors) / len(errors)
    return 1.0 if error <= 0.01 else max(0.0, math.exp(-5.0 * error))


def field_similarity(gt_val: Any, pred_val: Any, field_name: str) -> float:
    """Return a deterministic field score in [0, 1]."""
    gt = normalize_value_for_comparison(gt_val, field_name)
    pred = normalize_value_for_comparison(pred_val, field_name)
    if not gt and not pred:
        return 1.0
    if not gt or not pred:
        return 0.0
    if gt == pred:
        return 1.0
    if field_name in NUMERIC_FIELDS:
        numeric_score = _numeric_similarity(gt, pred)
        if numeric_score is not None:
            return numeric_score
    gt, pred = _text_surface(gt), _text_surface(pred)
    # chrF captures overlap with precision/recall; LCS preserves phrase order.
    return 0.7 * chrf_similarity(gt, pred) + 0.3 * lcs_ratio(gt, pred)


def _level(row: dict) -> str:
    value = row.get("步骤层级", "")
    return str(value).strip() if value is not None else ""


def _fields_for_row(row: dict) -> List[str]:
    level = _level(row)
    if level in LEVEL_FIELDS:
        return LEVEL_FIELDS[level]
    return [key for key in row if key != "步骤层级"]


def row_similarity(
    gt_row: dict,
    pred_row: dict,
    skip_device_fields: bool = True,
    semantic_scorer: Optional[Callable[[str, str, str], float]] = None,
) -> float:
    """Weighted content score for one eligible pair of rows."""
    fields = _fields_for_row(gt_row)
    weighted_sum = 0.0
    total_weight = 0.0
    for field in fields:
        if skip_device_fields and field in SKIP_FIELDS:
            continue
        weight = FIELD_WEIGHTS.get(field, 1.0)
        gt_value = normalize_value_for_comparison(gt_row.get(field, ""), field)
        pred_value = normalize_value_for_comparison(pred_row.get(field, ""), field)
        if semantic_scorer is not None and field not in NUMERIC_FIELDS:
            score = semantic_scorer(gt_value, pred_value, field)
            score = max(0.0, min(1.0, float(score)))
        else:
            score = field_similarity(gt_value, pred_value, field)
        weighted_sum += weight * score
        total_weight += weight
    return weighted_sum / total_weight if total_weight else 0.0


def _better_state(
    candidate: Tuple[float, int],
    current: Optional[Tuple[float, int]],
    tolerance: float = 1e-12,
) -> bool:
    """Compare DP states: maximize score, then maximize matched rows."""
    if current is None:
        return True
    if candidate[0] > current[0] + tolerance:
        return True
    return abs(candidate[0] - current[0]) <= tolerance and candidate[1] > current[1]


def _align_monotonic(
    gt_indices: Sequence[int],
    pred_indices: Sequence[int],
    gt_rows: Sequence[dict],
    pred_rows: Sequence[dict],
    row_scorer: Callable[[dict, dict], float],
) -> List[Tuple[int, int]]:
    """Maximum-weight order-preserving one-to-one alignment for one level."""
    m, n = len(gt_indices), len(pred_indices)
    dp: List[List[Tuple[float, int]]] = [[(0.0, 0) for _ in range(n + 1)]
                                         for _ in range(m + 1)]
    action: List[List[str]] = [["" for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(1, m + 1):
        action[i][0] = "skip_gt"
    for j in range(1, n + 1):
        action[0][j] = "skip_pred"

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            options = [
                (dp[i - 1][j], "skip_gt"),
                (dp[i][j - 1], "skip_pred"),
            ]
            score = row_scorer(gt_rows[gt_indices[i - 1]], pred_rows[pred_indices[j - 1]])
            options.append(((dp[i - 1][j - 1][0] + score,
                             dp[i - 1][j - 1][1] + 1), "match"))

            best_state, best_action = options[0]
            for state, candidate_action in options[1:]:
                if _better_state(state, best_state):
                    best_state, best_action = state, candidate_action
                elif (state == best_state and candidate_action == "match" and
                      best_action != "match"):
                    best_state, best_action = state, candidate_action
            dp[i][j] = best_state
            action[i][j] = best_action

    pairs: List[Tuple[int, int]] = []
    i, j = m, n
    while i > 0 and j > 0:
        current_action = action[i][j]
        if current_action == "match":
            pairs.append((gt_indices[i - 1], pred_indices[j - 1]))
            i -= 1
            j -= 1
        elif current_action == "skip_gt":
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def align_rows(
    gt_rows: List[dict],
    pred_rows: List[dict],
    skip_device_fields: bool = True,
    semantic_scorer: Optional[Callable[[str, str, str], float]] = None,
) -> List[Tuple[int, int]]:
    """Align rows by level using global maximum-weight monotonic matching."""
    gt_by_level: Dict[str, List[int]] = defaultdict(list)
    pred_by_level: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(gt_rows):
        gt_by_level[_level(row)].append(index)
    for index, row in enumerate(pred_rows):
        pred_by_level[_level(row)].append(index)

    levels = list(LEVEL_ORDER)
    for row in gt_rows + pred_rows:
        level = _level(row)
        if level not in levels:
            levels.append(level)

    scorer = lambda gt, pred: row_similarity(
        gt, pred, skip_device_fields=skip_device_fields,
        semantic_scorer=semantic_scorer,
    )
    alignments: List[Tuple[int, int]] = []
    for level in levels:
        alignments.extend(_align_monotonic(
            gt_by_level.get(level, []), pred_by_level.get(level, []),
            gt_rows, pred_rows, scorer,
        ))
    return sorted(alignments)


def _lcs_sequence_length(a: Sequence[str], b: Sequence[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = [0] * (len(b) + 1)
    for value_a in a:
        current = [0]
        for j, value_b in enumerate(b, start=1):
            current.append(previous[j - 1] + 1 if value_a == value_b else
                           max(previous[j], current[j - 1]))
        previous = current
    return previous[-1]


def _multiset_f1(a: Sequence[str], b: Sequence[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    overlap = sum((Counter(a) & Counter(b)).values())
    precision = overlap / len(b)
    recall = overlap / len(a)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


class UseCaseTableMetricV2:
    """Structure/content metric and reward components for AO-to-table output."""

    def __init__(
        self,
        alpha: float = 0.4,
        skip_device_fields: bool = True,
        semantic_scorer: Optional[Callable[[str, str, str], float]] = None,
        verbose: bool = False,
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = alpha
        self.skip_device_fields = skip_device_fields
        self.semantic_scorer = semantic_scorer
        self.verbose = verbose

    def _structure_score(self, gt_rows: List[dict], pred_rows: List[dict]) -> dict:
        gt_levels = [_level(row) for row in gt_rows]
        pred_levels = [_level(row) for row in pred_rows]
        if not gt_levels and not pred_levels:
            return {"row_count": 1.0, "level_lcs": 1.0, "level_multiset_f1": 1.0,
                    "score": 1.0}
        if not gt_levels or not pred_levels:
            return {"row_count": 0.0, "level_lcs": 0.0, "level_multiset_f1": 0.0,
                    "score": 0.0}
        row_count = min(len(gt_levels), len(pred_levels)) / max(len(gt_levels), len(pred_levels))
        level_lcs = _lcs_sequence_length(gt_levels, pred_levels) / max(len(gt_levels), len(pred_levels))
        level_multiset_f1 = _multiset_f1(gt_levels, pred_levels)
        # LCS remains dominant; count composition catches substitutions that
        # can look deceptively acceptable under sequence-only matching.
        score = 0.35 * row_count + 0.50 * level_lcs + 0.15 * level_multiset_f1
        return {
            "row_count": row_count,
            "level_lcs": level_lcs,
            "level_multiset_f1": level_multiset_f1,
            "score": score,
        }

    def compute(self, gt_rows: List[dict], pred_rows: List[dict]) -> dict:
        n_gt, n_pred = len(gt_rows), len(pred_rows)
        if n_gt == 0 and n_pred == 0:
            result = {
                "overall_score": 1.0,
                "structure_score": 1.0,
                "content_score": 1.0,
                "matched_content_score": 1.0,
                "row_coverage": 1.0,
                "row_precision": 1.0,
                "row_match_f1": 1.0,
                "row_count_match": True,
                "gt_rows": 0,
                "pred_rows": 0,
                "matched_rows": 0,
                "unmatched_gt": 0,
                "unmatched_pred": 0,
                "field_scores": {},
                "row_count_score": 1.0,
                "level_lcs_score": 1.0,
                "level_multiset_f1": 1.0,
            }
            if self.verbose:
                result["details"] = []
            return result
        structure = self._structure_score(gt_rows, pred_rows)
        alignments = align_rows(
            gt_rows, pred_rows,
            skip_device_fields=self.skip_device_fields,
            semantic_scorer=self.semantic_scorer,
        )
        row_scores: List[float] = []
        field_values: Dict[str, List[float]] = defaultdict(list)
        details = []
        for gt_index, pred_index in alignments:
            gt_row, pred_row = gt_rows[gt_index], pred_rows[pred_index]
            fields = _fields_for_row(gt_row)
            row_detail = {
                "gt_idx": gt_index,
                "pred_idx": pred_index,
                "level": _level(gt_row),
                "fields": {},
            }
            row_scores.append(row_similarity(
                gt_row, pred_row,
                skip_device_fields=self.skip_device_fields,
                semantic_scorer=self.semantic_scorer,
            ))
            for field in fields:
                if self.skip_device_fields and field in SKIP_FIELDS:
                    continue
                gt_value, pred_value = gt_row.get(field, ""), pred_row.get(field, "")
                if self.semantic_scorer is not None and field not in NUMERIC_FIELDS:
                    score = max(0.0, min(1.0, float(self.semantic_scorer(
                        normalize_value_for_comparison(gt_value, field),
                        normalize_value_for_comparison(pred_value, field), field,
                    ))))
                else:
                    score = field_similarity(gt_value, pred_value, field)
                field_values[field].append(score)
                row_detail["fields"][field] = {
                    "gt": str(gt_value)[:100],
                    "pred": str(pred_value)[:100],
                    "similarity": round(score, 4),
                }
            row_detail["row_similarity"] = round(row_scores[-1], 4)
            details.append(row_detail)

        matched = len(alignments)
        coverage = matched / n_gt if n_gt else (1.0 if n_pred == 0 else 0.0)
        precision = matched / n_pred if n_pred else (1.0 if n_gt == 0 else 0.0)
        match_f1 = (2 * coverage * precision / (coverage + precision)
                    if coverage + precision else 0.0)
        matched_content = sum(row_scores) / matched if matched else 0.0
        # F1 makes missing and hallucinated rows symmetric and is less gameable
        # as a reward than the old fixed 30% unmatched-row discount.
        content = matched_content * match_f1
        overall = self.alpha * structure["score"] + (1 - self.alpha) * content

        result = {
            "overall_score": round(max(0.0, min(1.0, overall)), 4),
            "structure_score": round(structure["score"], 4),
            "content_score": round(content, 4),
            "matched_content_score": round(matched_content, 4),
            "row_coverage": round(coverage, 4),
            "row_precision": round(precision, 4),
            "row_match_f1": round(match_f1, 4),
            "row_count_match": n_gt == n_pred,
            "gt_rows": n_gt,
            "pred_rows": n_pred,
            "matched_rows": matched,
            "unmatched_gt": n_gt - matched,
            "unmatched_pred": n_pred - matched,
            "field_scores": {
                field: round(sum(scores) / len(scores), 4)
                for field, scores in field_values.items()
            },
            "row_count_score": round(structure["row_count"], 4),
            "level_lcs_score": round(structure["level_lcs"], 4),
            "level_multiset_f1": round(structure["level_multiset_f1"], 4),
        }
        if self.verbose:
            result["details"] = details
        return result

    def reward(self, gt_rows: List[dict], pred_rows: List[dict]) -> float:
        """Return the scalar score used for RL or reranking experiments."""
        n_gt, n_pred = len(gt_rows), len(pred_rows)
        if n_gt == 0 and n_pred == 0:
            return 1.0
        return float(self.compute(gt_rows, pred_rows)["overall_score"])


# A short compatibility alias for experiments that prefer the new class name.
UseCaseTableMetric = UseCaseTableMetricV2


if __name__ == "__main__":
    gt = [{"步骤层级": "步骤", "说明": "检查液压系统", "操作内容": "检查"}]
    pred = [{"步骤层级": "步骤", "说明": "检查液压系统", "操作内容": "检查"}]
    print(UseCaseTableMetricV2(verbose=True).compute(gt, pred))


