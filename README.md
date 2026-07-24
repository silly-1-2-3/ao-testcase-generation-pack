# AO 指令测试用例生成

这是 AO 指令到结构化测试用例表格流程的可部署代码子集，包含数据准备、LoRA SFT、指标评估、本地 vLLM 推理、网页展示、设备检索和图像检索。

仓库不包含真实业务数据、旧实验结果、模型权重、LoRA 权重、向量缓存或外部 API 密钥。

## 目录

```text
pack/
├── data/                     # 新数据入口，不包含旧数据
│   ├── ao_instructions/
│   └── testcase_tables/
├── train/                    # 数据切分、SFT、指标和评估
├── vllm/                     # vLLM 启动和命令行测试
├── pre/                      # FastAPI 后端和 static/index.html
├── retrieval/                # 设备指令检索
├── img_retrieval/            # ColQwen 图像检索
└── scripts/                  # XML/XLSX 解析工具
```

## 数据格式

当前代码以 JSONL 为基础格式，每行至少包含：

```json
{"id":"example-001","source":"manual","content":"AO 指令文本","rows":[]}
```

`content` 是成型的 AO 指令，`rows` 是由表格行对象组成的列表。新版数据格式尚未完全冻结，使用新数据时请检查 `train/split_dataset.py` 的字段映射。

数据目录只保留 `.gitkeep`，请不要把真实数据提交到公开仓库。

## SFT 流程

从 `pack` 根目录执行：

```bash
python train/split_dataset.py \
  --output-all data/testcase_tables/all.jsonl \
  --cleaned-ao data/ao_instructions/ao.jsonl \
  --out-dir data

python train/prepare_sft_data.py --input-dir data --output-dir data

python train/train_sft_final.py \
  --model_dir /models/Qwen3.5-9B \
  --train_file data/train_sft.jsonl \
  --eval_file data/eval_sft.jsonl \
  --output_dir outputs/qwen35_lora
```

模型权重不在仓库中，必须把 `--model_dir` 改为服务器上的 Qwen3.5-9B 路径。由于当前 GT 只有最终 JSON，训练和网页推理默认关闭 thinking；训练传入 `--enable_thinking`、网页后端传入 `--enable-thinking` 可显式开启。
LoRA 目标层默认使用 `--lora_target_modules auto`：代码会读取实际模型结构。Qwen3.5 会同时覆盖全注意力、线性注意力、FFN 和受 PEFT 支持的 `conv1d` 模块；Qwen2.5 自动回退到原有目标层。只有排查兼容性时才需要显式传入逗号分隔的目标层。

## vLLM 和网页服务

建议使用 vLLM `>=0.9.0`。在线 OpenAI-compatible server 使用 Qwen3 reasoning parser：

```bash
vllm serve /models/Qwen3.5-9B \
  --served-model-name qwen35-lora \
  --enable-lora \
  --lora-modules qwen35-lora=/models/adapters/qwen35_lora \
  --reasoning-parser qwen3 \
  --host 127.0.0.1 \
  --port 8000
```

该 parser 将响应拆成 `reasoning_content` 和最终的 `content`。后端使用最终 `content`，同时兼容原始 `<think>...</think>` 文本。

启动网页后端：

```bash
python pre/server.py \
  --host 0.0.0.0 \
  --port 8081 \
  --vllm-url http://127.0.0.1:8000/v1 \
  --vllm-model qwen35-lora
```

访问 `http://服务器地址:8081`。`pre/server.py` 和 `pre/static/index.html` 必须一起保留。

后端只连接本地 vLLM，不调用外网 Qwen API；judger 不属于默认 SFT 或推理流程。历史评估面板只有在设置 `AO_EVAL_DIR` 并提供对应评估 JSON 后才可用。

命令行测试：

```bash
python vllm/ask_deployed_model.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen35-lora \
  --question "请输入一条 AO 指令"
```

`vllm/start_vllm_server.py` 是启动模板，其中保留了历史 Qwen2.5 默认路径，使用时请显式传入模型路径和 served model name。

## 评估与指标

```bash
python train/eval_model_vllm.py \
  --base_model /models/Qwen3.5-9B \
  --adapter outputs/qwen35_lora \
  --test_file data/test_sft.jsonl \
  --output_dir eval_results
```

这个脚本使用 vLLM 本地 Python 接口，不使用 server 的 `--reasoning-parser` 参数；它会自行处理原始 `<think>...</think>` 输出。

- `train/metrics.py`：当前评估脚本默认使用的兼容指标。
- `train/metrics_v2.py`：新的独立指标模块，包含改进后的字段匹配和结构评分，可继续用于 RL 奖励函数。
- `test_metrics_v2.py`：开发验证文件，不包含在发布包中。

`train/system_prompt_v4.txt` 是当前 SFT 默认提示词。`train/system_prompt_v4_full.txt` 是从原工作区 `mock_data/try_reasoning.py` 的 `build_system_instructions()` 提取的长版提示词，供完整推理流程参考。

## 设备指令检索

```bash
python retrieval/retrieval_pipeline.py \
  --query "一个执行步骤的完整文本" \
  --data /path/to/device_corpus.jsonl \
  --method both \
  --top_k 10
```

流程是 BM25、可选 BGE 语义重排和 RRF 融合。BM25 使用仓库内实现，中文分词可安装 `jieba`；BGE 默认模型为 `BAAI/bge-small-zh-v1.5`，由 `sentence-transformers` 加载。设备语料库不随仓库提供，`step_to_device.py` 可对表格中的设备步骤执行检索。

HyDE 查询改写默认关闭，只有明确提供 OpenAI-compatible 地址和密钥时才启用，不属于离线主流程。

## 图像检索

入口：`img_retrieval/ao_colqwen_image_retrieval.py`

默认模型：`vidore/colqwen2-v0.1`。需要安装 `colpali-engine`，并在有网络的机器上下载或缓存模型。图片、向量和实验结果不放进仓库。

```bash
python img_retrieval/ao_colqwen_image_retrieval.py --help
```

## scripts 和依赖

`scripts/xlsx_parser.py`、`scripts/xml_parser.py` 是为未来数据格式准备的工具，不是当前 SFT 主流程的必需依赖。

```bash
pip install -r requirements-train.txt
pip install -r requirements-server.txt
pip install -r requirements-retrieval.txt
pip install -r requirements-image.txt
```

PyTorch 和 vLLM 与 CUDA 强相关，请按服务器的 CUDA、驱动和 Python 版本选择兼容版本。

## 不包含的内容

- `mock_data` 和旧版翻译数据
- 旧实验数据、日志、HTML 报告和评估结果
- 旧设备语料库
- 模型、LoRA checkpoint 和向量缓存
- judger 实验代码
- 临时调试脚本、备份脚本和短命实验文件
