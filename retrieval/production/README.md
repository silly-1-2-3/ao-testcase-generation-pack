# 设备指令检索：BM25 + 离线 BGE

本目录实现 AO 结构化测试用例中的设备指令检索，解决模型生成的
`设备类型`、`设备指令号` 可能不在真实设备库中的问题。

当前实现的安全边界：

- `devices.jsonl` 是设备语料的唯一事实来源。
- BM25 和 BGE 索引都是可根据 `devices.jsonl` 重建的派生产物。
- Web 服务只允许 `annotate`，只返回候选和审计信息，不自动修改模型输出。
- 用户可以在网页副本中显式应用或撤销候选；原始 Base/LoRA 结果始终保留。
- 批处理脚本保留 `replace-invalid`，但只能在离线审核、阈值完成标定后使用。
- BGE 模型只从本地目录加载，运行时不访问 Hugging Face。

## 1. 文件职责

| 文件 | 职责 |
|---|---|
| `csv_to_devices_jsonl.py` | 将设备指令 CSV 和设备类别 CSV 转换为规范 `devices.jsonl` |
| `build_device_indexes.py` | 构建 BM25 索引，以及可选的离线 BGE 向量索引 |
| `device_retrieval_engine.py` | 加载索引并执行 BM25 召回、BGE 精排和 RRF 融合 |
| `apply_device_retrieval.py` | 对 JSONL 测试用例进行检索标注或受控替换 |
| `CSV_SCHEMA_MAPPING.md` | 现场两份设备 CSV 的字段映射说明 |
| `tests/test_row_selection.py` | 查询隔离、行筛选和安全决策的回归测试 |

父目录中的 `retrieval/bm25_index.py` 是当前 BM25 实现，建库和运行时都会
导入，不能作为旧脚本删除。

## 2. 数据流

```text
device_commands.csv + device_categories.csv
                     │
                     ▼
                devices.jsonl
                     │
                     ▼
        bm25_devices.pkl + index_manifest.json
                     │
                     ├── bge_vectors.npy
                     └── bge_meta.json
                     │
                     ▼
       BM25 粗召回 → BGE 精排 → RRF 融合
                     │
                     ▼
             候选列表 + 审计信息
```

修改 `devices.jsonl` 后必须重新构建索引。检索引擎会校验语料 SHA-256；
语料与索引不一致时会拒绝启动。

## 3. 环境与目录

所有命令都从项目根目录执行：

```bash
conda activate ao-retrieval

export AO_PROJECT_ROOT=/root/autodl-tmp/XIFEI_Agent/task1/ao-testcase-generation
export RETRIEVAL_RUN_DIR="$AO_PROJECT_ROOT/retrieval_runtime/smoke_bm25_v1"
export BGE_MODEL_DIR=/root/autodl-tmp/XIFEI_Agent/task1/retrieval_models/bge-small-zh-v1.5
export BGE_INDEX_DIR="$RETRIEVAL_RUN_DIR/index_hybrid_bge_v1"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$AO_PROJECT_ROOT"
mkdir -p "$RETRIEVAL_RUN_DIR"
```

主要依赖：

- Python 3.11
- NumPy
- jieba
- Transformers
- Sentence Transformers
- PyTorch

`jieba` 缺失时 BM25 会退回字符级中文切分，能够运行，但检索效果可能变化。

## 4. CSV 转换为规范设备语料

### 4.1 使用当前测试夹具

```bash
python retrieval/production/csv_to_devices_jsonl.py \
    --commands retrieval/production/tests/fixtures/device_commands.csv \
    --categories retrieval/production/tests/fixtures/device_categories.csv \
    --output "$RETRIEVAL_RUN_DIR/devices.jsonl"
```

### 4.2 使用现场真实数据

```bash
python retrieval/production/csv_to_devices_jsonl.py \
    --commands /data/source/device_commands.csv \
    --categories /data/source/device_categories.csv \
    --output /data/device_retrieval/devices.jsonl
```

默认会同时生成：

```text
devices.jsonl
devices.conversion_report.json
devices.rejected.jsonl
```

转换规则：

- 自动尝试 `utf-8-sig`、`utf-8` 和 `gb18030`。
- 自动识别逗号、Tab 或分号分隔符。
- 默认过滤 `deleted` 为真值的类别和指令。
- 指令表通过 `dev_cat_id` 关联类别表。
- `device_command_id` 是设备指令主键和去重依据。
- `command_code` 是业务设备指令号；为空时回退到主键，并在报告中计数。
- 主键重复、主键缺失或关联不到类别的指令会写入 rejected 文件。

字段含义和 `_text` 拼接顺序见
[`CSV_SCHEMA_MAPPING.md`](./CSV_SCHEMA_MAPPING.md)。

转换完成后至少检查：

```bash
python -m json.tool "$RETRIEVAL_RUN_DIR/devices.conversion_report.json"
wc -l \
    "$RETRIEVAL_RUN_DIR/devices.jsonl" \
    "$RETRIEVAL_RUN_DIR/devices.rejected.jsonl"
```

测试夹具只有两条设备指令，只能用于验证流程，不能标记为生产设备库。

## 5. 构建 BM25 + BGE 索引

网页端设备检索要求索引目录同时包含 BM25 和 BGE 文件，因此正式流程直接构建
混合索引：

```bash
python retrieval/production/build_device_indexes.py \
    --devices "$RETRIEVAL_RUN_DIR/devices.jsonl" \
    --output-dir "$BGE_INDEX_DIR" \
    --bge-model "$BGE_MODEL_DIR" \
    --device cpu
```

生成：

```text
index_hybrid_bge_v1/
├── bm25_devices.pkl
├── index_manifest.json
├── bge_vectors.npy
└── bge_meta.json
```

说明：

- `bm25_devices.pkl` 保存 BM25 词项和语料。
- `index_manifest.json` 保存语料哈希、记录数和建库配置。
- `bge_vectors.npy` 保存已归一化的设备文档向量。
- `bge_meta.json` 保存向量维度、记录顺序和 Query instruction。
- BGE 建库时只编码设备 `_text`，不为文档添加 instruction。
- 默认只在 Query 前添加：

```text
为这个句子生成表示以用于检索相关文章：
```

Query instruction 会写入索引元数据，查询时从元数据读取，防止建库和查询策略
不一致。

如果目标目录已有索引，脚本默认拒绝覆盖。确认需要重建时显式添加：

```bash
--force
```

`--device` 可以使用 `cpu`、`cuda`、`cuda:0` 或 `auto`。建库和在线 Query
编码可以使用不同计算设备，只要使用同一模型、同一 instruction 和同一归一化
策略即可。当前 Web 服务建议使用 CPU，避免与 vLLM 争用 GPU 显存。

### 5.1 仅构建 BM25

以下方式适合定位分词和关键词召回问题：

```bash
python retrieval/production/build_device_indexes.py \
    --devices "$RETRIEVAL_RUN_DIR/devices.jsonl" \
    --output-dir "$RETRIEVAL_RUN_DIR/index_bm25"
```

BM25-only 索引不能用于当前 Web `annotate` 模式，因为 `server.py` 会检查四个
混合索引文件是否齐全；它可以用于命令行诊断和批处理只标注。

## 6. 命令行检索验证

```bash
python retrieval/production/device_retrieval_engine.py \
    --devices "$RETRIEVAL_RUN_DIR/devices.jsonl" \
    --index-dir "$BGE_INDEX_DIR" \
    --bge-model "$BGE_MODEL_DIR" \
    --device cpu \
    --query "将万用表连接到测试点，测量直流电压并记录" \
    --top-k 5 \
    --candidate-pool 50
```

检索过程：

1. BM25 在整个设备库中粗召回 `candidate_pool` 个候选。
2. BGE 只编码当前 Query，并对 BM25 候选进行向量相似度计算。
3. BM25 rank 和 BGE rank 使用 RRF 融合。
4. 返回最终 `top_k` 个候选及其两路分数、排名和 BGE Top-1 margin。

BM25 和最终 NumPy 点积运行在 CPU。BGE Query 编码运行在 `--device` 指定的
设备上。

## 7. Query 构造与上下文隔离

只处理 `步骤层级 == "执行步骤"` 的行。默认要求该行至少有一个设备字段为
非空；`--all-execution-steps` 可以强制处理所有执行步骤。

每个执行步骤的 Query 由以下内容组成：

1. 当前行之前最多三行中的非执行步骤上下文。
2. 跳过前面其他 `执行步骤`，防止相邻设备指令互相污染。
3. 加入当前执行步骤自身的检索字段。
4. 去除空值和重复片段。

使用的字段：

```text
说明
操作内容
操作对象
操作目的
判据描述
操作类型
设备类型
设备单元号
设备参数
```

已有的 `设备指令号` 不进入 Query，避免模型生成的错误或幻觉指令号主导语义
检索。不同 AO 记录之间也不会共享上下文。

## 8. 批处理测试用例 JSONL

### 8.1 推荐：只标注，不替换

```bash
python retrieval/production/apply_device_retrieval.py \
    --input retrieval/production/tests/fixtures/cases.jsonl \
    --output "$RETRIEVAL_RUN_DIR/cases.annotated.jsonl" \
    --audit "$RETRIEVAL_RUN_DIR/audit.annotated.jsonl" \
    --devices "$RETRIEVAL_RUN_DIR/devices.jsonl" \
    --index-dir "$BGE_INDEX_DIR" \
    --bge-model "$BGE_MODEL_DIR" \
    --device cpu \
    --mode annotate
```

输出包括：

- `cases.annotated.jsonl`：原测试用例行保持不变，在执行步骤中增加
  `_device_retrieval`。
- `audit.annotated.jsonl`：每个被处理执行步骤一条独立审计记录。

审计内容包括：

- 记录 ID 和行号；
- 实际检索 Query；
- BGE Query instruction；
- 原始四个设备字段；
- 决策代码；
- 是否替换；
- BM25+BGE 候选及排名、分数和 margin。

输入、输出和审计必须是三个不同文件。输出文件已存在时脚本默认拒绝覆盖；
只有确认覆盖时才使用 `--force`。

### 8.2 受控自动替换

该模式仅用于离线批处理，不用于 Web：

```bash
python retrieval/production/apply_device_retrieval.py \
    --input /data/ao_work/ao_tables.jsonl \
    --output /data/ao_work/ao_tables.replaced.jsonl \
    --audit /data/ao_work/device_retrieval_audit.jsonl \
    --devices /data/device_retrieval/devices.jsonl \
    --index-dir /data/device_retrieval/index_hybrid \
    --bge-model /data/models/bge-small-zh-v1.5 \
    --device cpu \
    --mode replace-invalid \
    --bge-threshold 0.68 \
    --bge-margin 0.05
```

当前安全决策：

- 原设备指令号已存在且是 Top-1：保留。
- 原设备指令号已存在但不是 Top-1：只标记复核，不自动改写。
- 原设备指令号不存在且没有候选：不替换。
- BM25-only：不自动替换。
- RRF Top-1 必须同时是 BGE Top-1。
- BGE 分数必须达到 threshold。
- BGE Top-1 与 Top-2 的 margin 必须存在并达到设定值。
- 满足全部条件时，只替换 `设备类型` 和 `设备指令号`。
- `设备单元号` 和测试用例里的具体 `设备参数` 始终保留。

`0.68/0.05` 只是当前保守起点，必须使用现场真实标注集重新标定，不能直接
视为生产阈值。

## 9. 集成到 Web 服务

Web 服务固定使用只读 `annotate`，禁止 `replace-invalid`。先单独启动 vLLM，
再在 `ao-retrieval` 环境启动：

```bash
conda activate ao-retrieval
cd "$AO_PROJECT_ROOT"

python pre/server.py \
    --host 127.0.0.1 \
    --port 8081 \
    --vllm-url http://127.0.0.1:8000/v1 \
    --base-model qwen35-base \
    --lora-model qwen35-lora \
    --device-retrieval-mode annotate \
    --retrieval-db "$RETRIEVAL_RUN_DIR/devices.jsonl" \
    --retrieval-index-dir "$BGE_INDEX_DIR" \
    --retrieval-bge-model "$BGE_MODEL_DIR" \
    --retrieval-device cpu \
    --retrieval-data-kind example \
    --retrieval-data-label "示例设备库（2 条，仅研发验证）" \
    --retrieval-top-k 5 \
    --retrieval-candidate-pool 50
```

使用现场真实设备语料时必须改为：

```text
--retrieval-data-kind production
--retrieval-data-label "生产设备指令库"
```

网页端行为：

- Base 和 Base+LoRA 分别执行检索审计。
- 服务端不会修改两侧的原始结构化结果。
- 用户可以在页面副本中显式应用候选或撤销。
- 人工应用候选只修改 `设备类型` 和 `设备指令号`。
- 页面刷新或重新推理会清空尚未导出的人工修改。
- Excel 分别导出最终结果、原始模型结果和人工修改审计。

检查状态：

```bash
curl -s http://127.0.0.1:8081/api/health \
    | python -m json.tool
```

重点确认：

```text
retrieval_enabled: true
retrieval_ready: true
retrieval_device: cpu
retrieval_data_kind: example 或 production
```

## 10. 回归测试

```bash
conda activate ao-retrieval
cd "$AO_PROJECT_ROOT"

python -m unittest \
    retrieval.production.tests.test_row_selection \
    -v
```


当前测试覆盖：

- 只选择执行步骤；
- 相邻执行步骤上下文隔离；
- 已知指令 Top-1 保留；
- 已知但与 Top-1 不一致时要求复核；
- BM25-only 不自动替换；
- RRF/BGE Top-1 一致性；
- BGE 分数和 margin 阈值；
- 输出覆盖保护；
- BGE 建库与查询 instruction 一致性。

修改 Query、候选融合、替换决策或索引格式后必须重新运行这些测试。

## 11. 离线服务器需要携带的内容

运行 Web 检索至少需要：

```text
retrieval/
├── __init__.py
├── bm25_index.py
└── production/
    ├── __init__.py
    ├── apply_device_retrieval.py
    └── device_retrieval_engine.py

devices.jsonl
index_hybrid/
├── bm25_devices.pkl
├── index_manifest.json
├── bge_vectors.npy
└── bge_meta.json

bge-small-zh-v1.5/
└── 完整本地 SentenceTransformer 模型文件
```

如果离线服务器还需要重新转换数据或重建索引，还应携带：

```text
csv_to_devices_jsonl.py
build_device_indexes.py
CSV_SCHEMA_MAPPING.md
原始设备 CSV
```
