# -*- coding: utf-8 -*-
"""
AO 指令 -> 操作建议图片检索示例。

这个脚本参考仓库根目录的 visdomrag.py 中 ColQwen/ColPali 的视觉检索逻辑，
但把场景改成了毕业设计里的需求：

1. 离线阶段：把 chatgpt_image 目录下的图片编码成 ColQwen 多向量表示；
2. 在线阶段：输入一条 AO 指令，把 AO 指令编码成查询向量；
3. 检索阶段：用 ColQwenProcessor.score_multi_vector 对 AO 指令和图片逐一打分；
4. 输出阶段：返回最相关的 top-k 图片，作为结构化测试用例的“操作建议图片”字段。

单条 AO 指令检索：
    python chatgpt_image/ao_colqwen_image_retrieval.py ^
        --query "检查液压系统压力表读数是否处于正常范围" ^
        --top-k 3

批量 AO 指令检索：
    python chatgpt_image/ao_colqwen_image_retrieval.py ^
        --ao-csv data/ao_cases.csv ^
        --query-col ao_instruction ^
        --id-col ao_id ^
        --output-csv chatgpt_image/ao_image_retrieval_results.csv

依赖：
    pip install torch pillow tqdm colpali-engine transformers accelerate
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from tqdm import tqdm


LOGGER = logging.getLogger("ao_colqwen_image_retrieval")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_MODEL_NAME = "vidore/colqwen2-v0.1"
DEFAULT_QUERY_COLUMNS = ("ao_instruction", "instruction", "query", "question", "AO指令", "指令")
DEFAULT_ID_COLUMNS = ("ao_id", "id", "q_id", "case_id", "AO编号", "编号")


@dataclass
class ImageRecord:
    """图片索引中的一条记录。"""

    image_id: str
    image_path: str
    embedding_path: str
    file_size: int
    mtime_ns: int
    source_ao_id: Optional[str] = None


@dataclass
class RetrievalResult:
    """一次检索返回的一条 top-k 结果。"""

    rank: int
    image_id: str
    image_path: str
    score: float
    source_ao_id: Optional[str]


class ColQwenImageRetriever:
    """封装 ColQwen 模型加载、图片编码、文本编码和多向量打分。"""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device_map: Optional[str] = None,
        dtype_name: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.device_map = device_map or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = self._resolve_torch_dtype(dtype_name)

        try:
            from colpali_engine.models import ColQwen2, ColQwen2Processor
        except ImportError as exc:
            raise ImportError(
                "缺少 colpali_engine。请先安装：pip install colpali-engine"
            ) from exc

        LOGGER.info("Loading ColQwen model: %s", model_name)
        self.model = ColQwen2.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
        ).eval()
        self.processor = ColQwen2Processor.from_pretrained(model_name)
        self.device = self._infer_model_device()

    @staticmethod
    def _resolve_torch_dtype(dtype_name: str) -> torch.dtype:
        """根据运行环境选择模型精度；CPU 上默认用 float32 更稳。"""

        if dtype_name == "auto":
            return torch.bfloat16 if torch.cuda.is_available() else torch.float32
        dtype_map = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if dtype_name not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype_name}")
        return dtype_map[dtype_name]

    def _infer_model_device(self) -> torch.device:
        """拿到模型所在设备，用于把 processor 输出的 tensor 移过去。"""

        if hasattr(self.model, "device"):
            return torch.device(self.model.device)
        return next(self.model.parameters()).device

    def _move_batch_to_model_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """processor 输出是一个 dict，需要逐个 tensor 移到模型设备。"""

        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }

    def embed_image(self, image_path: Path) -> torch.Tensor:
        """把一张图片编码成 ColQwen 多向量表示。"""

        # convert("RGB") 可以避免 PNG 透明通道或灰度图导致 processor 报错。
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            inputs = self.processor.process_images([image])

        inputs = self._move_batch_to_model_device(inputs)
        with torch.inference_mode():
            embedding = self.model(**inputs)
        return embedding.detach().cpu()

    def embed_query(self, ao_instruction: str) -> torch.Tensor:
        """把 AO 指令文本编码成查询侧多向量表示。"""

        inputs = self.processor.process_queries([ao_instruction])
        inputs = self._move_batch_to_model_device(inputs)
        with torch.inference_mode():
            embedding = self.model(**inputs)
        return embedding.detach().cpu()

    @staticmethod
    def _as_2d_multivector(embedding: torch.Tensor) -> torch.Tensor:
        """
        ColQwen 的输出通常是 [1, seq_len, hidden_dim]。
        score_multi_vector 需要的是每个样本一个 [seq_len, hidden_dim] tensor。
        """

        if embedding.dim() == 3 and embedding.size(0) == 1:
            embedding = embedding.squeeze(0)
        if embedding.dim() != 2:
            raise ValueError(f"Expected a 2D multi-vector embedding, got shape {tuple(embedding.shape)}")
        return embedding.float()

    def score(self, query_embedding: torch.Tensor, image_embeddings: Sequence[torch.Tensor]) -> List[float]:
        """计算一条 AO 指令和多张图片的相似度分数。"""

        query_vectors = [self._as_2d_multivector(query_embedding)]
        image_vectors = [self._as_2d_multivector(embedding) for embedding in image_embeddings]

        # 这里复用了 visdomrag.py 中的核心方法：
        # ColQwen/ColPali 是多向量检索，不是普通的单向量余弦相似度。
        scores = self.processor.score_multi_vector(query_vectors, image_vectors)
        return torch.as_tensor(scores).flatten().detach().cpu().float().tolist()


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def iter_image_paths(image_dir: Path, recursive: bool = False) -> List[Path]:
    """扫描图片目录，默认只扫一层；如果有子目录，可以加 --recursive。"""

    pattern = "**/*" if recursive else "*"
    image_paths = [
        path
        for path in image_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(image_paths)


def parse_source_ao_id(image_path: Path) -> Optional[str]:
    """
    从图片文件名中解析原始 AO 编号。

    例如 prompt_01_id131.png 会得到 131。
    这个字段只用于结果展示或离线评估，不参与模型打分，避免“用文件名作弊”。
    """

    match = re.search(r"_id([^_.-]+)", image_path.stem)
    return match.group(1) if match else None


def safe_embedding_name(image_path: Path) -> str:
    """为图片生成稳定且不冲突的 embedding 文件名。"""

    resolved = str(image_path.resolve())
    digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:12]
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", image_path.stem)
    return f"{safe_stem}_{digest}.pt"


def load_previous_metadata(metadata_path: Path) -> Dict[str, ImageRecord]:
    """读取上一次建索引时保存的元数据，用来判断缓存是否还能复用。"""

    if not metadata_path.exists():
        return {}

    records: Dict[str, ImageRecord] = {}
    with metadata_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            record = ImageRecord(**item)
            records[record.image_path] = record
    return records


def save_metadata(metadata_path: Path, records: Iterable[ImageRecord]) -> None:
    """保存图片索引元数据，方便后续直接复用图片 embedding。"""

    with metadata_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def load_tensor(path: Path) -> torch.Tensor:
    """兼容不同 PyTorch 版本的 tensor 加载。"""

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_or_load_image_index(
    retriever: ColQwenImageRetriever,
    image_dir: Path,
    cache_dir: Path,
    force_reindex: bool = False,
    recursive: bool = False,
) -> List[ImageRecord]:
    """
    构建或加载图片侧索引。

    图片是静态知识库，embedding 可以提前算好并缓存。
    AO 指令每次变化时，只需要重新算查询 embedding，不需要重复编码全部图片。
    """

    image_dir = image_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    embedding_dir = cache_dir / "embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / "image_index.jsonl"

    previous = load_previous_metadata(metadata_path)
    image_paths = iter_image_paths(image_dir, recursive=recursive)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    records: List[ImageRecord] = []
    for image_path in tqdm(image_paths, desc="Indexing images"):
        stat = image_path.stat()
        embedding_path = embedding_dir / safe_embedding_name(image_path)
        image_path_text = str(image_path.resolve())

        record = ImageRecord(
            image_id=image_path.stem,
            image_path=image_path_text,
            embedding_path=str(embedding_path.resolve()),
            file_size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            source_ao_id=parse_source_ao_id(image_path),
        )

        old_record = previous.get(image_path_text)
        cache_is_valid = (
            old_record is not None
            and old_record.file_size == record.file_size
            and old_record.mtime_ns == record.mtime_ns
            and Path(old_record.embedding_path).exists()
        )

        if force_reindex or not embedding_path.exists() or not cache_is_valid:
            embedding = retriever.embed_image(image_path)
            torch.save(embedding, embedding_path)

        records.append(record)

    save_metadata(metadata_path, records)
    LOGGER.info("Image index metadata saved to %s", metadata_path)
    return records


def search_images(
    retriever: ColQwenImageRetriever,
    ao_instruction: str,
    records: Sequence[ImageRecord],
    top_k: int = 3,
) -> List[RetrievalResult]:
    """对单条 AO 指令执行图片检索。"""

    if not ao_instruction.strip():
        raise ValueError("AO instruction is empty")
    if not records:
        raise ValueError("Image index is empty")

    query_embedding = retriever.embed_query(ao_instruction)
    image_embeddings = [load_tensor(Path(record.embedding_path)) for record in records]
    scores = retriever.score(query_embedding, image_embeddings)

    ranked = sorted(
        zip(records, scores),
        key=lambda item: item[1],
        reverse=True,
    )[:top_k]

    return [
        RetrievalResult(
            rank=rank,
            image_id=record.image_id,
            image_path=record.image_path,
            score=float(score),
            source_ao_id=record.source_ao_id,
        )
        for rank, (record, score) in enumerate(ranked, start=1)
    ]


def choose_column(fieldnames: Sequence[str], requested: Optional[str], candidates: Sequence[str], kind: str) -> str:
    """CSV 字段名可以手动指定；不指定时从常见字段名中自动选择。"""

    if requested:
        if requested not in fieldnames:
            raise ValueError(f"{kind} column '{requested}' not found. Available columns: {list(fieldnames)}")
        return requested

    for candidate in candidates:
        if candidate in fieldnames:
            return candidate

    raise ValueError(
        f"Cannot infer {kind} column. Please pass it explicitly. "
        f"Available columns: {list(fieldnames)}"
    )


def read_ao_csv(csv_path: Path, query_col: Optional[str], id_col: Optional[str]) -> List[Tuple[str, str]]:
    """读取批量 AO 指令 CSV，返回 (ao_id, ao_instruction)。"""

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")

        selected_query_col = choose_column(reader.fieldnames, query_col, DEFAULT_QUERY_COLUMNS, "AO instruction")
        selected_id_col: Optional[str] = None
        if id_col:
            selected_id_col = choose_column(reader.fieldnames, id_col, DEFAULT_ID_COLUMNS, "AO id")
        else:
            for candidate in DEFAULT_ID_COLUMNS:
                if candidate in reader.fieldnames:
                    selected_id_col = candidate
                    break

        rows: List[Tuple[str, str]] = []
        for row_index, row in enumerate(reader, start=1):
            ao_instruction = (row.get(selected_query_col) or "").strip()
            if not ao_instruction:
                continue
            ao_id = (row.get(selected_id_col) or "").strip() if selected_id_col else str(row_index)
            rows.append((ao_id or str(row_index), ao_instruction))

    return rows


def write_batch_results(output_csv: Path, rows: Sequence[Dict[str, object]]) -> None:
    """把批量检索结果保存为 CSV，后续可以和结构化测试用例按 ao_id 合并。"""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ao_id",
        "ao_instruction",
        "rank",
        "image_id",
        "image_path",
        "score",
        "source_ao_id",
    ]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Retrieve operation images for AO instructions with ColQwen.")
    parser.add_argument("--image-dir", type=Path, default=script_dir, help="图片库目录，默认是当前 chatgpt_image 目录。")
    parser.add_argument("--cache-dir", type=Path, default=None, help="embedding 缓存目录，默认是 image-dir/.colqwen_cache。")
    parser.add_argument("--recursive", action="store_true", help="递归扫描 image-dir 下的子目录。")
    parser.add_argument("--force-reindex", action="store_true", help="强制重新编码全部图片。")

    parser.add_argument("--query", type=str, default=None, help="单条 AO 指令文本。")
    parser.add_argument("--ao-csv", type=Path, default=None, help="批量 AO 指令 CSV。")
    parser.add_argument("--query-col", type=str, default=None, help="CSV 中 AO 指令所在列名。")
    parser.add_argument("--id-col", type=str, default=None, help="CSV 中 AO 编号所在列名。")
    parser.add_argument("--output-csv", type=Path, default=script_dir / "ao_image_retrieval_results.csv")
    parser.add_argument("--top-k", type=int, default=3, help="每条 AO 指令返回的图片数量。")

    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="HuggingFace 模型名或本地模型路径。")
    parser.add_argument("--device-map", type=str, default=None, help="例如 cuda、cpu 或 auto；默认有 GPU 用 cuda，否则用 cpu。")
    parser.add_argument("--dtype", type=str, default="auto", help="auto、float32、float16 或 bfloat16。")
    parser.add_argument("--hf-endpoint", type=str, default=None, help="可选，例如 https://hf-mirror.com。")
    parser.add_argument("--verbose", action="store_true", help="输出更详细的日志。")

    args = parser.parse_args()
    if not args.query and not args.ao_csv:
        parser.error("Please provide --query for single retrieval or --ao-csv for batch retrieval.")
    if args.top_k <= 0:
        parser.error("--top-k must be greater than 0.")
    return args


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    cache_dir = args.cache_dir or (args.image_dir / ".colqwen_cache")

    retriever = ColQwenImageRetriever(
        model_name=args.model_name,
        device_map=args.device_map,
        dtype_name=args.dtype,
    )
    records = build_or_load_image_index(
        retriever=retriever,
        image_dir=args.image_dir,
        cache_dir=cache_dir,
        force_reindex=args.force_reindex,
        recursive=args.recursive,
    )

    # 单条检索：适合调试和演示。
    if args.query:
        results = search_images(retriever, args.query, records, top_k=args.top_k)
        print(f"\nAO instruction: {args.query}")
        print(f"Top-{len(results)} retrieved images:")
        for result in results:
            print(
                f"{result.rank}. score={result.score:.4f} "
                f"image_id={result.image_id} "
                f"source_ao_id={result.source_ao_id or ''} "
                f"path={result.image_path}"
            )

    # 批量检索：适合和 SFT 输出的结构化测试用例做后处理融合。
    if args.ao_csv:
        ao_rows = read_ao_csv(args.ao_csv, args.query_col, args.id_col)
        output_rows: List[Dict[str, object]] = []
        for ao_id, ao_instruction in tqdm(ao_rows, desc="Retrieving AO images"):
            results = search_images(retriever, ao_instruction, records, top_k=args.top_k)
            for result in results:
                output_rows.append(
                    {
                        "ao_id": ao_id,
                        "ao_instruction": ao_instruction,
                        "rank": result.rank,
                        "image_id": result.image_id,
                        "image_path": result.image_path,
                        "score": result.score,
                        "source_ao_id": result.source_ao_id,
                    }
                )

        write_batch_results(args.output_csv, output_rows)
        LOGGER.info("Batch retrieval results saved to %s", args.output_csv)


if __name__ == "__main__":
    main()
