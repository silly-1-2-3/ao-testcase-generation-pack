#!/usr/bin/env python3
"""
metrics.py —— 结构化用例表格评价指标

参考论文/方法：
    - TEDS (Tree Edit Distance based Similarity) [IBM, 2019]  
      https://arxiv.org/abs/1911.10683
    - SCORE: Structural and COntent Robust Evaluation [2025]
      https://arxiv.org/abs/2511.20227
    - TABREX: Tabular Referenceless eXplainable Evaluation [2025]
      https://arxiv.org/abs/2512.15907
    - TabXEval: eXhaustive Rubric for Table Evaluation [2025]
      https://arxiv.org/abs/2505.22176
    - Benchmarking Table Extraction from Heterogeneous Scientific Documents [2025]
      https://arxiv.org/abs/2511.16134

本指标专为「波音737 AMM → 结构化用例表格」场景设计，核心特点：

1. 双层级评分：结构分 + 内容分
    - 结构分（Structure Score）：行对齐、步骤层级匹配、行列数一致性
    - 内容分（Content Score）：逐字段的语义相似度

2. 字段分区（与 try_reasoning.py 的 HEADERS 一致）：
    - 用例/子用例字段：步骤层级, 说明, 注意事项
    - 步骤字段：步骤层级, 说明, 注意事项, 操作内容, 操作对象, 操作目的, 是否同时发送, 多判据组合条件
    - 判据字段：步骤层级, 是否使用设备, 操作类型, 判据类型, 判据范围, 判据描述, 左值, 右值, 单位
    - 执行步骤字段：步骤层级, 设备类型, 设备单元号, 设备指令号, 设备参数, 判据关联标志

3. 设备字段暂不参与评价（tasks 8）
    排除字段：设备类型, 设备单元号, 设备指令号, 设备参数

4. 判据关联（判据关联标志）归一化（task 7）
    - 变量归一化：@ID 占位符替换为统一 token（如 @REF_N），再比较
    - 算式归一化：a * @ID, b * @ID 等表达式做标准化（去空格、统一运算符）

5. 文本语义相似度：bigram Jaccard + LCS 组合
    - 替换纯编辑距离（Levenshtein），对中文描述性字段（说明、注意事项等）更合理
    - bigram Jaccard 捕捉共享子串，容忍词序变化
    - LCS 比率保留序列结构敏感度

6. 字段加权：不同字段按操作重要性分配权重
    - 高权重（1.3-1.5）：说明、操作内容、操作对象
    - 中权重（1.1-1.2）：注意事项、判据描述、操作目的
    - 标准权重（1.0）：布尔/枚举字段
    - 低权重（0.8）：单位

7. 最终分数 = α * StructureScore + (1-α) * ContentScore
   默认 α = 0.4（结构略轻于内容）

使用：
    from metrics import UseCaseTableMetric
    metric = UseCaseTableMetric(alpha=0.4, skip_device_fields=True)
    score = metric.compute(gt_rows, pred_rows)
    # score 是 0~1 的浮点数，1 表示完美匹配

    # 批量评测
    results = metric.batch_evaluate(list_of_samples)
"""
import re
import json
import math
from typing import List, Dict, Tuple, Set, Optional, Any
from collections import defaultdict


# ============================================================
# 字段定义
# ============================================================

# 不参与评价的设备字段 (task 8)
SKIP_FIELDS = {"设备类型", "设备单元号", "设备指令号", "设备参数"}

# 按步骤层级区分的必选字段
LEVEL_FIELDS = {
    "用例": ["步骤层级", "说明", "注意事项"],
    "子用例": ["步骤层级", "说明", "注意事项"],
    "步骤": ["步骤层级", "说明", "注意事项", "操作内容", "操作对象",
             "操作目的", "是否同时发送", "多判据组合条件"],
    "判据": ["步骤层级", "是否使用设备", "操作类型", "判据类型",
             "判据范围", "判据描述", "左值", "右值", "单位"],
    "执行步骤": ["步骤层级", "设备类型", "设备单元号", "设备指令号",
                  "设备参数", "判据关联标志"],
}

# 判据关联相关的字段


# 字段权重：用于内容分加权平均，反映不同字段对操作正确性的重要性
# 权重 > 1.0 的字段对最终分数影响更大，< 1.0 的影响更小
FIELD_WEIGHTS = {
    # 核心信息字段 —— 操作人员最关注的
    "说明": 1.5,        # 步骤名称，最关键
    "操作内容": 1.3,    # 做什么
    "操作对象": 1.3,    # 对什么操作
    # 安全与判断字段
    "注意事项": 1.2,    # 安全提醒重要但措辞灵活
    "判据描述": 1.2,    # 如何判断
    "操作目的": 1.1,    # 为什么做
    # 标准权重
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
    # 辅助字段
    "单位": 0.8,        # 单位信息辅助性强
}

REFERENCE_FIELDS = {"左值", "右值", "判据关联标志"}


# ============================================================
# 判据关联归一化 (task 7)
# ============================================================

def normalize_reference_expr(value: str) -> str:
    """
    对包含 @ID 的表达式做归一化。
    例如：
        "a * @ID_1"  -> "@REF_a_*_@ID"
        "2.5 * @ABC" -> "@REF_2.5_*_@ID"
        "@ID_1 + 3"  -> "@REF_@ID_+_3"
        "39.0;41.0"  -> 保持不变（判据范围，不含 @ID）
    """
    if not isinstance(value, str) or not value.strip():
        return value

    v = value.strip()
    # 如果不包含 @，直接返回原值（trim 空格）
    if "@" not in v:
        return v

    # 找到所有 @ID 标识符
    ref_pattern = re.compile(r'@[A-Za-z0-9_\u4e00-\u9fff]+')
    refs_found = ref_pattern.findall(v)

    # 替换所有 @ID 为 @ID (统一占位符)
    normalized = ref_pattern.sub("@ID", v)

    # 标准化空格和运算符
    normalized = re.sub(r'\s*\*\s*', ' * ', normalized)  # 乘号
    normalized = re.sub(r'\s*\+\s*', ' + ', normalized)  # 加号
    normalized = re.sub(r'\s*\-\s*', ' - ', normalized)  # 减号
    normalized = re.sub(r'\s*/\s*', ' / ', normalized)   # 除号
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    return normalized


def normalize_value_for_comparison(value: Any, field_name: str) -> str:
    """
    通用值归一化：
    - 空值/None -> ""
    - 判据关联字段 -> 调用 normalize_reference_expr
    - 布尔字段（是/否）-> 小写统一
    - 数值字段 -> 统一格式
    - 其他 -> strip
    """
    if value is None:
        return ""

    s = str(value).strip()
    if not s or s in ("null", "None", "[]"):
        return ""

    # 判据关联字段做归一化
    if field_name in REFERENCE_FIELDS:
        return normalize_reference_expr(s)

    # 布尔字段
    if field_name in ("是否同时发送", "是否使用设备"):
        return s.lower()

    # 数值字段：尝试统一数值格式（如 "39.0" 和 "39" 应该匹配）
    if field_name in ("左值", "右值", "判据范围"):
        # 如果是纯数值，做格式化
        return normalize_numeric(s)

    # 多判据组合条件
    if field_name == "多判据组合条件":
        return s.replace("全部成功", "all").replace("任一成功", "any").strip()

    return s


def normalize_numeric(s: str) -> str:
    """
    数值归一化：
    "39.0" -> "39"
    "39.0;41.0" -> "39;41"
    如果无法解析为数值，保持原样
    """
    # 先处理分号分隔的范围格式（判据范围）
    if ";" in s:
        parts = s.split(";")
        normalized_parts = []
        for p in parts:
            p = p.strip()
            try:
                num = float(p)
                if num == int(num):
                    normalized_parts.append(str(int(num)))
                else:
                    normalized_parts.append(f"{num:.6g}")
            except ValueError:
                normalized_parts.append(p)
        return ";".join(normalized_parts)

    # 单个数值
    try:
        num = float(s)
        if num == int(num):
            return str(int(num))
        return f"{num:.6g}"
    except ValueError:
        return s


# ============================================================
# 结构匹配
# ============================================================

def align_rows(gt_rows: List[dict], pred_rows: List[dict]) -> List[Tuple[int, int]]:
    """
    对 GT 和预测的行做贪心对齐。按 步骤层级 分组后逐组匹配。
    返回 [(gt_idx, pred_idx), ...] 的对齐对列表。
    """
    # 按步骤层级分组
    gt_by_level = defaultdict(list)
    for i, row in enumerate(gt_rows):
        lvl = row.get("步骤层级", "").strip()
        gt_by_level[lvl].append(i)

    pred_by_level = defaultdict(list)
    for j, row in enumerate(pred_rows):
        lvl = row.get("步骤层级", "").strip()
        pred_by_level[lvl].append(j)

    alignments = []

    # 对每个层级，贪心对齐
    for lvl in ["用例", "子用例", "步骤", "判据", "执行步骤"]:
        gt_indices = gt_by_level.get(lvl, [])
        pred_indices = pred_by_level.get(lvl, [])

        # 贪心匹配：按顺序一对一，多余的 unmatched
        for k in range(min(len(gt_indices), len(pred_indices))):
            alignments.append((gt_indices[k], pred_indices[k]))

    # 注意：GT 和 pred 中的行可能已有不同顺序
    # 更高级的做法是用匈牙利算法做最优匹配，这里先用贪心保证简单可用
    return alignments


# ============================================================
# 内容相似度
# ============================================================

def field_similarity(gt_val: Any, pred_val: Any, field_name: str) -> float:
    """
    计算单个字段的相似度，返回 0~1。

    策略：
    - 两者都为空 -> 1.0 (正确)
    - 一个为空一个非空 -> 0.0
    - 精确匹配 -> 1.0
    - 数值近似匹配 -> 1.0 if diff < 1% else 指数衰减
    - 文本语义匹配 -> 用 bigram Jaccard + LCS 组合（语义相似度）
    """
    gt_norm = normalize_value_for_comparison(gt_val, field_name)
    pred_norm = normalize_value_for_comparison(pred_val, field_name)

    # both empty
    if not gt_norm and not pred_norm:
        return 1.0
    # one empty
    if not gt_norm or not pred_norm:
        return 0.0
    # exact match after normalization
    if gt_norm == pred_norm:
        return 1.0

    # 数值近似匹配
    if field_name in ("左值", "右值"):
        try:
            gv = float(gt_norm)
            pv = float(pred_norm)
            if gv == 0 and pv == 0:
                return 1.0
            denom = max(abs(gv), abs(pv), 1e-9)
            rel_err = abs(gv - pv) / denom
            if rel_err < 0.01:  # 1% 以内
                return 1.0
            # 指数衰减
            return max(0.0, math.exp(-rel_err * 5))
        except (ValueError, TypeError):
            pass

    # 语义文本相似度（bigram Jaccard + LCS，替代纯编辑距离）
    # 对"说明"、"注意事项"、"判据描述"等中文描述性字段，
    # 即使措辞不同，只要共享足够多的子串/关键词就能得高分
    return semantic_text_similarity(gt_norm, pred_norm)


def bigram_jaccard(a: str, b: str) -> float:
    """字符二元组 Jaccard 相似度，0~1。
    对中文文本特别有效：共享的子串越多，分数越高，容忍词序调整。
    参考：字符 n-gram 在工业界文本去重/相似度检测中的广泛应用。
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    # 提取字符 bigrams
    bigrams_a = {a[i:i+2] for i in range(len(a)-1)} if len(a) >= 2 else {a}
    bigrams_b = {b[i:i+2] for i in range(len(b)-1)} if len(b) >= 2 else {b}
    if not bigrams_a and not bigrams_b:
        return 1.0
    intersection = bigrams_a & bigrams_b
    union = bigrams_a | bigrams_b
    return len(intersection) / len(union) if union else 0.0


def lcs_ratio(a: str, b: str) -> float:
    """最长公共子序列 (LCS) 占比，0~1。
    衡量两个文本的序列结构相似度，对保留原文顺序的改写容忍度高。
    参考：ROUGE-L 使用的 LCS-based F-measure。
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    m, n = len(a), len(b)
    # DP LCS with O(min(m,n)) space
    if m < n:
        a, b = b, a
        m, n = n, m
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev = curr
    lcs_len = prev[-1]
    return lcs_len / max(m, n)


def bigram_containment(a: str, b: str) -> float:
    """短文本 bigram 被长文本 bigram 覆盖的比例，0~1。
    衡量较短文本中的关键字符组合在较长文本中出现的程度。
    对于摘要式等价判断特别有效：如果短文本的所有 bigram 都出现在长文本中，得 1.0。
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    bigrams_a = {a[i:i+2] for i in range(len(a)-1)} if len(a) >= 2 else {a}
    bigrams_b = {b[i:i+2] for i in range(len(b)-1)} if len(b) >= 2 else {b}
    if not bigrams_a or not bigrams_b:
        return 0.0
    intersection = bigrams_a & bigrams_b
    return len(intersection) / min(len(bigrams_a), len(bigrams_b))


def semantic_text_similarity(a: str, b: str) -> float:
    """语义文本相似度，0~1。
    组合三种度量：
    - bigram Jaccard (0.3)：共享子串比例，捕捉关键词/短语重叠
    - 短文本包含度 containment (0.3)：较短文本的 bigram 被较长文本覆盖的比例，
      使摘要式改写（如 "确认无异常状况" vs "确认无损伤、腐蚀、渗漏等异常状况"）得到合理高分
    - LCS 比率 (0.4)：最长公共子序列占比，保持对序列结构的敏感度
    相比纯编辑距离，对「含义相同但措辞不同」的中文文本评分显著提高。
    参考：ROUGE-L (Lin, 2004), 字符 n-gram Jaccard + 包含度在工业文本去重中的实践。
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    jac = bigram_jaccard(a, b)
    cont = bigram_containment(a, b)
    lcs = lcs_ratio(a, b)
    return 0.3 * jac + 0.3 * cont + 0.4 * lcs


# ============================================================
# 主评价类
# ============================================================

class UseCaseTableMetric:
    """
    结构化用例表格评价指标。

    参数：
        alpha: 结构分权重 (0~1)，默认 0.4
        skip_device_fields: 是否跳过设备字段，默认 True
        verbose: 是否输出详细日志
    """

    def __init__(
        self,
        alpha: float = 0.4,
        skip_device_fields: bool = True,
        verbose: bool = False,
    ):
        self.alpha = alpha
        self.skip_device_fields = skip_device_fields
        self.verbose = verbose

    def compute(self, gt_rows: List[dict], pred_rows: List[dict]) -> dict:
        """
        计算单条样本的评分。

        Args:
            gt_rows: Ground Truth 行列表
            pred_rows: 预测行列表

        Returns:
            dict with keys:
                - overall_score (0~1)
                - structure_score (0~1)
                - content_score (0~1)
                - row_count_match: bool
                - field_scores: dict of field_name -> avg_score
                - details: list of per-alignment details
        """
        n_gt = len(gt_rows)
        n_pred = len(pred_rows)

        # === 结构分 ===
        # 1. 行数匹配
        row_count_score = 1.0 if n_gt == n_pred else (
            min(n_gt, n_pred) / max(n_gt, n_pred, 1)
        )

        # 2. 层级序列匹配
        gt_levels = [r.get("步骤层级", "").strip() for r in gt_rows]
        pred_levels = [r.get("步骤层级", "").strip() for r in pred_rows]
        level_seq_score = self._sequence_similarity(gt_levels, pred_levels)

        structure_score = 0.5 * row_count_score + 0.5 * level_seq_score

        # === 内容分 ===
        alignments = align_rows(gt_rows, pred_rows)
        field_scores = defaultdict(list)
        detail_list = []

        matched_count = len(alignments)
        unmatched_gt = n_gt - matched_count
        unmatched_pred = n_pred - matched_count

        for gt_idx, pred_idx in alignments:
            gt_row = gt_rows[gt_idx]
            pred_row = pred_rows[pred_idx]
            lvl = gt_row.get("步骤层级", "").strip()
            relevant_fields = LEVEL_FIELDS.get(lvl, [])

            row_detail = {"gt_idx": gt_idx, "pred_idx": pred_idx, "level": lvl, "fields": {}}

            for field in relevant_fields:
                if self.skip_device_fields and field in SKIP_FIELDS:
                    continue

                gt_val = gt_row.get(field, "")
                pred_val = pred_row.get(field, "")
                sim = field_similarity(gt_val, pred_val, field)
                field_scores[field].append(sim)
                row_detail["fields"][field] = {
                    "gt": str(gt_val)[:100],
                    "pred": str(pred_val)[:100],
                    "similarity": round(sim, 4),
                }

            detail_list.append(row_detail)

        # 计算每个字段的平均分
        field_avg = {}
        for field, scores in field_scores.items():
            field_avg[field] = sum(scores) / len(scores) if scores else 0.0

        # 内容总分 = 字段分数的加权平均（关键字段权重更高）
        weighted_sum = 0.0
        weight_total = 0.0
        for field, scores in field_scores.items():
            w = FIELD_WEIGHTS.get(field, 1.0)
            for s in scores:
                weighted_sum += w * s
                weight_total += w
        content_score = weighted_sum / weight_total if weight_total > 0 else 0.0

        # 未匹配行惩罚
        total_possible = max(n_gt, n_pred, 1)
        unmatched_penalty = (unmatched_gt + unmatched_pred) / total_possible
        content_score = content_score * (1.0 - 0.3 * unmatched_penalty)  # 最多扣 30%

        # === 综合分 ===
        overall = self.alpha * structure_score + (1 - self.alpha) * content_score
        overall = max(0.0, min(1.0, overall))

        result = {
            "overall_score": round(overall, 4),
            "structure_score": round(structure_score, 4),
            "content_score": round(content_score, 4),
            "row_count_match": n_gt == n_pred,
            "gt_rows": n_gt,
            "pred_rows": n_pred,
            "matched_rows": matched_count,
            "field_scores": {k: round(v, 4) for k, v in field_avg.items()},
            "row_count_penalty": round(row_count_score, 4),
            "level_seq_penalty": round(level_seq_score, 4),
        }

        if self.verbose:
            result["details"] = detail_list

        return result

    def _sequence_similarity(self, seq_a: List[str], seq_b: List[str]) -> float:
        """
        计算两个层级序列的相似度（基于最长公共子序列 LCS）。
        """
        m, n = len(seq_a), len(seq_b)
        if m == 0 and n == 0:
            return 1.0
        if m == 0 or n == 0:
            return 0.0

        # DP LCS
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq_a[i - 1] == seq_b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs_len = dp[m][n]
        return lcs_len / max(m, n)

    def batch_evaluate(self, samples: List[dict]) -> dict:
        """
        批量评测。

        Args:
            samples: [{"id": ..., "gt_rows": [...], "pred_rows": [...]}, ...]

        Returns:
            {
                "avg_overall": float,
                "avg_structure": float,
                "avg_content": float,
                "per_sample": [dict, ...],
                "field_avg": dict,
            }
        """
        results = []
        all_field_scores = defaultdict(list)

        for sample in samples:
            r = self.compute(
                sample.get("gt_rows", []),
                sample.get("pred_rows", []),
            )
            r["id"] = sample.get("id")
            r["source"] = sample.get("source", "")
            results.append(r)

            for field, score in r.get("field_scores", {}).items():
                all_field_scores[field].append(score)

        n = len(results)
        overall_avg = sum(r["overall_score"] for r in results) / n if n else 0
        structure_avg = sum(r["structure_score"] for r in results) / n if n else 0
        content_avg = sum(r["content_score"] for r in results) / n if n else 0

        field_avg = {}
        for field, scores in all_field_scores.items():
            field_avg[field] = round(sum(scores) / len(scores), 4)

        return {
            "num_samples": n,
            "avg_overall": round(overall_avg, 4),
            "avg_structure": round(structure_avg, 4),
            "avg_content": round(content_avg, 4),
            "field_avg": field_avg,
            "per_sample": results,
        }


# ============================================================
# 示例
# ============================================================
def demo():
    """演示 metrics 使用"""
    gt = [
        {"步骤层级": "用例", "说明": "左机翼干舱外部区域检查（GV）", "注意事项": ""},
        {"步骤层级": "步骤", "说明": "对左机翼干舱进行一般目视检查",
         "注意事项": "", "操作内容": "检查", "操作对象": "左机翼干舱",
         "操作目的": "确认无损伤、腐蚀、渗漏等异常状况",
         "是否同时发送": "否", "多判据组合条件": "全部成功"},
        {"步骤层级": "判据", "是否使用设备": "是", "操作类型": "仅操作",
         "判据类型": "其他", "判据范围": "", "判据描述": "", "左值": "", "右值": "", "单位": ""},
    ]

    pred = [
        {"步骤层级": "用例", "说明": "左机翼干舱外部检查（GV）", "注意事项": ""},
        {"步骤层级": "步骤", "说明": "对左机翼干舱进行目视检查",
         "注意事项": "", "操作内容": "检查", "操作对象": "左机翼干舱",
         "操作目的": "确认无异常状况",
         "是否同时发送": "否", "多判据组合条件": "全部成功"},
        {"步骤层级": "判据", "是否使用设备": "是", "操作类型": "仅操作",
         "判据类型": "其他", "判据范围": "", "判据描述": "", "左值": "", "右值": "", "单位": ""},
    ]

    m = UseCaseTableMetric(verbose=True)
    result = m.compute(gt, pred)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    demo()