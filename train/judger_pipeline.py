#!/usr/bin/env python3
"""AO generation pipeline with cross-case judging and retrieval.

The pipeline is intentionally stateless between generation calls:

    1. Generate one table for every AO.
    2. Ask a small, structured judger whether the AO needs another case.
    3. Retrieve relevant first-pass cases for positive judgements.
    4. Generate again with a fresh request containing the normal system prompt,
       the retrieval instructions, and the retrieved context.

The vLLM server must expose an OpenAI-compatible ``/v1`` endpoint. For
Qwen3.5, start vLLM with ``--reasoning-parser qwen3``. The final answer is
read from ``message.content``; ``reasoning_content`` is retained only in the
result log and is never sent to the JSON parser.
"""

import argparse
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


STEP_LIST_KEY = "步骤列表"
THINK_PREFIX = "<think>\n"

JUDGER_PROMPT = """你是航空维修测试用例的依赖关系审查器。
请判断当前 AO 是否需要从其他测试用例检索判据关联信息，再重新生成当前测试用例表格。

只有下列情况才应将 needs_retrieval 设为 true：
1. 当前 AO 明确引用另一个 TASK/测试用例；
2. 当前 AO 使用了由其他测试用例测量、记录或定义的 @ID 变量；
3. 当前 AO 的判据表达式依赖其他用例的结果，例如 a*@P1*0.75。
仅仅在当前 AO 内定义一个新变量，或普通地出现 TASK 编码，不足以触发检索。

请只返回一个 JSON 对象，不要 Markdown，不要解释：
{
  "needs_retrieval": true 或 false,
  "referenced_ids": ["P1"],
  "referenced_tasks": ["TASK 49-32-00-200-801"],
  "reason": "一句话说明"
}
如果没有明确的外部依赖，两个数组必须为空。"""

RETRIEVAL_HINT_PROMPT = """你正在第二次生成一个 AO 测试用例表格。下面的检索结果来自其他测试用例，
只把它们当作判据关联的事实来源，不要把其他用例的步骤机械复制到当前用例。

要求：
1. 仍然严格执行默认系统提示词中的 JSON 格式、字段名、行顺序和字段类型要求；
2. 仅在当前 AO 确实引用外部结果时使用检索到的 @ID、TASK 和数值关系；
3. 被引用的外部测量值应在当前表格中表达为判据关联关系，不能凭空填写一个固定测量值；
4. 检索结果不足以证明某个字段时，保留空值或按默认规则生成，不要臆造；
5. 只输出最终 JSON，不输出思考过程、检索说明或 Markdown。"""


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARN] skip invalid JSONL line {line_no}: {exc}", flush=True)
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def ao_text(record: Dict[str, Any]) -> str:
    """Accept the current AO JSONL shape and the older messages shape."""
    for key in ("content", "ao", "text", "instruction"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    messages = record.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                value = message.get("content")
                if isinstance(value, str):
                    return value.strip()
    return ""


def case_id(record: Dict[str, Any], index: int) -> str:
    value = record.get("id", record.get("source", f"case_{index:04d}"))
    return str(value)


def post_chat(base_url: str, model: str, messages: List[Dict[str, str]],
              max_tokens: int, temperature: float, timeout: int,
              retries: int = 2) -> Dict[str, Any]:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            request = Request(
                endpoint,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"vLLM request failed after {retries + 1} attempts: {last_error}")


def response_parts(response: Dict[str, Any]) -> Tuple[str, str]:
    """Return final content and reasoning separately for OpenAI/vLLM responses."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"vLLM response has no choices: {str(response)[:500]}")
    message = choices[0].get("message", {})
    if not isinstance(message, dict):
        return "", ""
    content = message.get("content")
    reasoning = message.get("reasoning_content", "")
    # A few older OpenAI-compatible servers return content as a list of blocks.
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
    return str(content or ""), str(reasoning or "")


def final_answer_text(text: str) -> str:
    """Strip fallback inline thinking tags without touching reasoning_content."""
    text = text.strip()
    closing = re.search(r"</think>", text, flags=re.IGNORECASE)
    if closing:
        text = text[closing.end():]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def json_values(text: str) -> Iterable[Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        yield value


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    for value in json_values(final_answer_text(text)):
        if isinstance(value, dict):
            return value
    return None


def parse_table(text: str) -> List[Dict[str, Any]]:
    """Parse the model's table without treating reasoning JSON as the answer."""
    for value in json_values(final_answer_text(text)):
        if isinstance(value, dict):
            candidate = value.get(STEP_LIST_KEY)
            if candidate is None:
                candidate = value.get("rows", value.get("steps", value.get("table")))
            if isinstance(candidate, list):
                return [row for row in candidate if isinstance(row, dict)]
        elif isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    return []


def normalize_judgement(value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    value = value if isinstance(value, dict) else {}
    needs = value.get("needs_retrieval", False)
    if isinstance(needs, str):
        needs = needs.strip().lower() in {"true", "yes", "是", "1"}
    def as_strings(item: Any) -> List[str]:
        if isinstance(item, str):
            return [item]
        if isinstance(item, list):
            return [str(x) for x in item if x is not None]
        return []

    return {
        "needs_retrieval": bool(needs),
        "referenced_ids": as_strings(value.get("referenced_ids", [])),
        "referenced_tasks": as_strings(value.get("referenced_tasks", [])),
        "reason": str(value.get("reason", "")),
        "parse_ok": bool(value),
    }


def referenced_signals(text: str) -> Tuple[List[str], List[str]]:
    ids = sorted(set(re.findall(r"(?<![A-Za-z0-9])@([A-Za-z][A-Za-z0-9_-]*)", text)))
    tasks = sorted(set(re.findall(r"TASK\s*[0-9]{2}(?:-[0-9]{2,3}){3,5}", text, re.I)))
    return ids, tasks


def heuristic_needs_retrieval(text: str, table: Sequence[Dict[str, Any]]) -> bool:
    """Conservative fallback: missing an explicit dependency is worse than an extra retrieval."""
    ids, tasks = referenced_signals(text + " " + json.dumps(table, ensure_ascii=False))
    external_task = len(set(tasks)) > 1
    cross_words = re.search(r"引用|其他测试用例|前序用例|来自.*用例|跨用例|另一个TASK", text, re.I)
    expression_ref = bool(re.search(r"[ab]\s*\*\s*@[_A-Za-z0-9-]+", text))
    return bool(external_task or cross_words or expression_ref or (ids and cross_words))


def tokens(text: str) -> List[str]:
    # Keep TASK codes, @IDs, Latin words, numbers, and CJK characters searchable.
    pieces = re.findall(r"[A-Za-z]+[A-Za-z0-9_-]*|[0-9]+(?:\.[0-9]+)?|@[A-Za-z0-9_-]+|[\u4e00-\u9fff]", text.lower())
    return [piece for piece in pieces if piece not in {"的", "了", "和", "是", "在", "将", "与", "或"}]


class CrossCaseRetriever:
    """Offline lexical retrieval with optional local BGE reranking."""

    def __init__(self, documents: List[Dict[str, Any]], method: str = "bm25",
                 bge_model: Optional[str] = None):
        self.documents = documents
        self.method = method
        self.term_freqs: List[Counter] = []
        self.doc_freq: Counter = Counter()
        self.doc_lengths: List[int] = []
        self.bge = None
        self.bge_vectors = None
        self._fit_bm25()
        if method in {"bge", "both"}:
            try:
                from sentence_transformers import SentenceTransformer
                if not bge_model:
                    raise ValueError("--bge-model is required for bge retrieval")
                import numpy as np
                self.bge = SentenceTransformer(bge_model)
                self.bge_vectors = self.bge.encode(
                    [doc["search_text"] for doc in documents],
                    normalize_embeddings=True, show_progress_bar=True,
                ).astype(np.float32)
            except Exception as exc:
                print(f"[WARN] BGE retrieval unavailable, using BM25: {exc}", flush=True)
                self.method = "bm25"

    def _fit_bm25(self) -> None:
        for doc in self.documents:
            tf = Counter(tokens(doc["search_text"]))
            self.term_freqs.append(tf)
            self.doc_lengths.append(sum(tf.values()))
            self.doc_freq.update(tf.keys())
        self.avgdl = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)

    def _bm25(self, query: str, excluded_id: str) -> List[Tuple[int, float]]:
        qtokens = set(tokens(query))
        n_docs = len(self.documents)
        scored: List[Tuple[int, float]] = []
        for index, (doc, tf) in enumerate(zip(self.documents, self.term_freqs)):
            if doc["id"] == excluded_id:
                continue
            score = 0.0
            length = self.doc_lengths[index]
            for term in qtokens:
                if term not in tf:
                    continue
                df = self.doc_freq[term]
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                numerator = tf[term] * 2.5
                denominator = tf[term] + 1.5 * (0.25 + 0.75 * length / max(self.avgdl, 1.0))
                score += idf * numerator / denominator
            scored.append((index, score))
        return sorted(scored, key=lambda item: item[1], reverse=True)

    def search(self, query: str, excluded_id: str, top_k: int,
               target_ids: Sequence[str] = ()) -> List[Dict[str, Any]]:
        lexical = self._bm25(query, excluded_id)
        lexical_rank = {index: rank for rank, (index, _) in enumerate(lexical)}
        scores: Dict[int, float] = defaultdict(float)
        for index, _ in lexical:
            scores[index] += 1.0 / (60 + lexical_rank[index] + 1)

        if self.method in {"bge", "both"} and self.bge is not None:
            import numpy as np
            query_vector = self.bge.encode([query], normalize_embeddings=True)[0]
            similarities = np.dot(self.bge_vectors, query_vector)
            dense = [
                (index, float(similarities[index]))
                for index, doc in enumerate(self.documents)
                if doc["id"] != excluded_id
            ]
            dense.sort(key=lambda item: item[1], reverse=True)
            for rank, (index, _) in enumerate(dense):
                scores[index] += 1.0 / (60 + rank + 1)

        targets = {target.upper().lstrip("@").strip() for target in target_ids}
        ranked = sorted(scores, key=lambda index: scores[index], reverse=True)
        # Prefer a document that defines a requested @ID over one that merely uses it.
        if targets:
            ranked.sort(
                key=lambda index: (
                    not bool(targets.intersection(self.documents[index]["defined_ids"])),
                    -scores[index],
                )
            )
        results = []
        for index in ranked[:top_k]:
            doc = self.documents[index]
            results.append({
                "id": doc["id"],
                "source": doc["source"],
                "score": round(scores[index], 6),
                "defined_ids": sorted(doc["defined_ids"]),
                "ao": doc["ao_text"][:6000],
                "table": doc["table"],
            })
        return results


def make_documents(records: Sequence[Dict[str, Any]], tables: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    documents = []
    for index, record in enumerate(records):
        cid = case_id(record, index)
        text = ao_text(record)
        table = tables.get(cid, [])
        combined = text + "\n" + json.dumps(table, ensure_ascii=False)
        defined: set[str] = set()
        # In the AO text, a variable is normally defined after "记录/定义/保存".
        for match in re.finditer(
            r"(?:记录|定义|保存|测量|计算)[^@\n]{0,80}@([A-Za-z0-9_-]+)",
            text,
        ):
            defined.add(match.group(1))
        for row in table:
            marker = row.get("判据关联标志", "")
            if isinstance(marker, str):
                defined.update(re.findall(r"@([A-Za-z0-9_-]+)", marker))
        documents.append({
            "id": cid,
            "source": str(record.get("source", "")),
            "ao_text": text,
            "table": table,
            "defined_ids": {item.upper() for item in defined},
            "search_text": combined,
        })
    return documents


def retrieval_context(results: Sequence[Dict[str, Any]]) -> str:
    if not results:
        return "[RETRIEVAL_RESULTS]\n未检索到可用的其他测试用例。"
    blocks = ["[RETRIEVAL_RESULTS]"]
    for rank, result in enumerate(results, 1):
        blocks.append(
            f"--- 参考用例 {rank}: {result['id']} (score={result['score']}) ---\n"
            f"AO:\n{result['ao']}\n"
            f"结构化表格:\n{json.dumps(result['table'], ensure_ascii=False)}"
        )
    return "\n".join(blocks)


def generate_one(args: argparse.Namespace, system_prompt: str, text: str,
                 extra_system: str = "", max_tokens: Optional[int] = None
                 ) -> Tuple[str, str, List[Dict[str, Any]]]:
    system = system_prompt.rstrip()
    if extra_system:
        system += "\n\n" + extra_system.strip()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": THINK_PREFIX + text},
    ]
    response = post_chat(
        args.base_url, args.model, messages, max_tokens or args.max_new_tokens,
        args.temperature, args.timeout, args.retries,
    )
    content, reasoning = response_parts(response)
    return content, reasoning, parse_table(content)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    system_prompt = Path(args.system_prompt).read_text(encoding="utf-8")
    records = read_jsonl(Path(args.input))
    if args.max_samples is not None:
        records = records[:args.max_samples]
    if not records:
        raise ValueError("input JSONL contains no usable records")

    print(f"Loaded {len(records)} AO records", flush=True)
    phase1: Dict[str, List[Dict[str, Any]]] = {}
    phase1_raw: Dict[str, str] = {}
    reasoning_log: Dict[str, str] = {}
    for index, record in enumerate(records):
        cid = case_id(record, index)
        text = ao_text(record)
        if not text:
            print(f"[WARN] {cid}: empty AO text", flush=True)
        raw, reasoning, table = generate_one(args, system_prompt, text)
        phase1[cid] = table
        phase1_raw[cid] = raw
        reasoning_log[f"phase1:{cid}"] = reasoning
        print(f"[Phase1] {cid}: {len(table)} rows", flush=True)

    documents = make_documents(records, phase1)
    retriever = CrossCaseRetriever(documents, args.retrieval_method, args.bge_model)
    judgements: Dict[str, Dict[str, Any]] = {}
    final_tables: Dict[str, List[Dict[str, Any]]] = {}
    final_raw: Dict[str, str] = {}
    retrievals: Dict[str, List[Dict[str, Any]]] = {}

    for index, record in enumerate(records):
        cid = case_id(record, index)
        text = ao_text(record)
        judge_input = (
            JUDGER_PROMPT + "\n\n当前 AO:\n" + text +
            "\n\n第一次生成的表格:\n" + json.dumps(phase1[cid], ensure_ascii=False)
        )
        judge_raw, judge_reasoning, _ = generate_one(
            args, "你是一个严格的结构化审查器。", judge_input,
            max_tokens=args.judge_max_new_tokens,
        )
        parsed = normalize_judgement(parse_json_object(judge_raw))
        ids, tasks = referenced_signals(text + " " + json.dumps(phase1[cid], ensure_ascii=False))
        heuristic = heuristic_needs_retrieval(text, phase1[cid])
        if heuristic and not parsed["needs_retrieval"]:
            parsed["needs_retrieval"] = True
            parsed["fallback_override"] = True
        parsed["heuristic_needs_retrieval"] = heuristic
        parsed["observed_ids"] = ids
        parsed["observed_tasks"] = tasks
        judgements[cid] = parsed
        reasoning_log[f"judge:{cid}"] = judge_reasoning
        print(f"[Judger] {cid}: needs_retrieval={parsed['needs_retrieval']} parse_ok={parsed['parse_ok']}", flush=True)

        if not parsed["needs_retrieval"]:
            final_tables[cid] = phase1[cid]
            final_raw[cid] = phase1_raw[cid]
            retrievals[cid] = []
            continue

        query = "\n".join([
            text,
            "referenced ids: " + " ".join(parsed.get("referenced_ids", [])),
            "referenced tasks: " + " ".join(parsed.get("referenced_tasks", [])),
            "observed ids: " + " ".join(ids),
            "observed tasks: " + " ".join(tasks),
        ])
        targets = parsed.get("referenced_ids", []) or ids
        hits = retriever.search(query, cid, args.top_k, targets)
        retrievals[cid] = hits
        context = retrieval_context(hits)
        second_input = text + "\n\n" + context
        raw, reasoning, table = generate_one(
            args, system_prompt, second_input, RETRIEVAL_HINT_PROMPT,
        )
        final_raw[cid] = raw
        final_tables[cid] = table
        reasoning_log[f"phase2:{cid}"] = reasoning
        print(f"[Phase2] {cid}: retrieved={len(hits)} final_rows={len(table)}", flush=True)

    result = {
        "model": args.model,
        "base_url": args.base_url,
        "retrieval_method": retriever.method,
        "system_prompt": str(Path(args.system_prompt)),
        "phase1_tables": phase1,
        "judgements": judgements,
        "retrievals": retrievals,
        "final_tables": final_tables,
        "raw_outputs": {"phase1": phase1_raw, "final": final_raw},
        "reasoning_content": reasoning_log,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {output}", flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AO judger + cross-case retrieval pipeline")
    parser.add_argument("--input", required=True, help="AO JSONL; supports content or messages format")
    parser.add_argument("--output", default="./judger_results.json")
    parser.add_argument("--system-prompt", default="./train/system_prompt_v4_full.txt")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True, help="vLLM model id")
    parser.add_argument("--retrieval-method", choices=["bm25", "bge", "both"], default="bm25")
    parser.add_argument("--bge-model", default=None, help="local BGE path; required for bge/both")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--judge-max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--retries", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
