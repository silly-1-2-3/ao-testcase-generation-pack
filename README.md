# Qwen3.5-9B AO 指令结构化测试用例生成

本文档对应当前仓库代码，目标是指导使用者在**无 Docker、无互联网的保密
Linux 服务器**上完成以下工作：

1. 使用本地 Qwen3.5-9B 进行 LoRA 训练。
2. 使用本地 vLLM 对 Base 与 Base+LoRA 进行成对评测。
3. 启动 vLLM OpenAI-compatible 服务。
4. 启动 FastAPI 网页，实时比较 Base 与 Base+LoRA。

所有示例命令均从项目根目录 `ao-testcase-generation` 执行。
---

## 1. 当前系统边界

主流程采用以下本地链路：

```text
浏览器
  ↓ 127.0.0.1:8081
pre/server.py
  ↓ 127.0.0.1:8000/v1
Conda 环境中的 vLLM 0.24.0
  ├─ qwen35-base：本地 Qwen3.5-9B Base
  └─ qwen35-lora：同一个 Base + 本地 LoRA Adapter
```

安全约束：

- Base 模型、Tokenizer、LoRA、数据集和评测结果全部从本地路径读取。
- `train/train_sft_final.py` 强制 Hugging Face、Datasets、Xet、W&B 和遥测
  处于离线/禁用状态。
- `train/eval_model_vllm.py` 强制 Hugging Face 和 vLLM 使用统计处于禁用状态。
- `pre/server.py` 不加载模型，也不访问外部 Qwen API，只请求本机 vLLM。
- 每条 AO 是独立请求，只包含 System Prompt 和本次 AO，不携带上一条 AO 的
  对话历史。
- 默认只监听回环地址；可以在服务器本机访问，也可以在安全策略允许时通过
  SSH 本地端口转发访问，不需要将服务监听到外部网卡。

主流程不需要以下内容：

- Docker。
- Hugging Face Token。
- ModelScope Token。
- W&B。
- 外部大模型 API。
- 项目内的旧 vLLM 启动脚本。

---

## 2. 三种运行模式

| 目标 | Conda 环境 | 是否先运行 `vllm serve` |
|---|---|---|
| 直接部署已有 LoRA 和网页 | `ao-qwen35-vllm` | 是 |
| Base/LoRA 离线批量评测 | `ao-qwen35-vllm` | 否 |
| 从数据开始重新训练 | `ao-qwen35-train` | 否 |

不要混淆两种 vLLM 用法：

- `train/eval_model_vllm.py` 使用 vLLM 的本地 Python API，自行创建引擎，不访问
  8000 端口。
- `pre/server.py` 使用 OpenAI-compatible HTTP API，因此必须先启动
  `vllm serve`。

同一张 GPU 上进行离线评测或训练前，应先停止正在运行的 vLLM 服务，释放显存。

---

## 3. 目录说明

```text
ao-testcase-generation/
├── README.md
├── data/
│   ├── ao_instructions/               # AO 原文输入
│   ├── testcase_tables/               # Ground Truth 输入
│   ├── train.jsonl                    # 切分后的训练集
│   ├── eval.jsonl                     # 切分后的验证集
│   ├── test.jsonl                     # 切分后的最终测试集
│   ├── train_sft.jsonl                # SFT 训练数据
│   ├── eval_sft.jsonl                 # SFT 验证数据
│   └── test_sft.jsonl                 # SFT 最终测试数据
├── train/
│   ├── split_dataset.py               # 合并并切分数据
│   ├── prepare_sft_data.py            # 生成 messages 格式
│   ├── prepare_training/
│   │   └── count_dataset.py           # 真实 Token 长度统计
│   ├── train_sft_final.py             # assistant-only LoRA SFT
│   ├── eval_model_vllm.py             # Base/LoRA 成对评测
│   ├── metrics.py
│   ├── metrics_v2.py                  # 当前默认指标
│   ├── system_prompt_v4.txt            # 训练同款压缩 Prompt
│   ├── system_prompt_v4_full.txt       # 完整 Prompt
│   └── environment-train-a800.yml
├── outputs/
│   ├── qwen35_lora_full_v1/            # 原始 PEFT Adapter
│   └── qwen35_lora_full_v1_vllm_prefixfix/
│                                         # vLLM 专用 Adapter
├── eval_results/
│   └── qwen35_lora_prefixfix_full/     # 成对评测结果
├── pre/
│   ├── server.py                       # 离线 FastAPI 后端
│   └── static/index.html               # 网页前端
├── retrieval/                          # 可选设备指令检索
├── img_retrieval/                      # 可选图像检索
└── scripts/                            # XLSX/XML 辅助解析
```

原始 PEFT Adapter 与 vLLM 前缀转换版用途不同，不能互相替代：

- 原始目录用于审计、Transformers/PEFT 加载和断点续训。
- `*_vllm_prefixfix` 目录只用于 vLLM 评测和部署。

---

## 4. 已验证运行基线

当前训练清单和评测清单记录的主要版本为：

| 项目 | 已验证值 |
|---|---|
| 操作系统 | Linux x86_64 |
| Python | 3.11 |
| GPU | NVIDIA A800 80GB PCIe |
| Compute Capability | 8.0 / SM80 |
| PyTorch | 2.11.0 |
| PyTorch CUDA runtime | 12.9 |
| Transformers | 5.14.1 |
| Datasets | 5.0.0 |
| PEFT | 0.19.1 |
| Accelerate | 1.14.0 |
| vLLM | 0.24.0+cu129 |

训练与 vLLM 使用两个独立环境：

- `ao-qwen35-train`：Transformers、PEFT、Datasets，用于训练。
- `ao-qwen35-vllm`：vLLM、Triton、FastAPI、Uvicorn、HTTPX，用于评测和网页
  部署。

不要在同一个环境中混装两套 PyTorch/CUDA 依赖。目标服务器仍需提供兼容的
NVIDIA 驱动；Conda 压缩包不会包含宿主机驱动和 Linux 内核。

---

## 5. 传入保密服务器的离线材料

### 5.1 只部署网页和已有模型

最少需要：

```text
ao-testcase-generation/
  pre/server.py
  pre/static/index.html
  train/system_prompt_v4.txt
  train/system_prompt_v4_full.txt
  outputs/qwen35_lora_full_v1_vllm_prefixfix/
    adapter_model.safetensors
    adapter_config.json

qwen3_5_9b_deploy/models/Qwen3.5-9B/
  config.json
  configuration.json
  chat_template.jinja
  tokenizer.json
  tokenizer_config.json
  model.safetensors.index.json
  model.safetensors-00001-of-00004.safetensors
  model.safetensors-00002-of-00004.safetensors
  model.safetensors-00003-of-00004.safetensors
  model.safetensors-00004-of-00004.safetensors

ao-qwen35-vllm.tar.gz
```

如果网页还要展示历史评测结果，额外传入：

```text
eval_results/qwen35_lora_prefixfix_full/
  evaluation_manifest.json
  base_predictions.jsonl
  base_metrics.json
  lora_predictions.jsonl
  lora_metrics.json
  vllm_compare_summary.json
```

### 5.2 需要重新训练

除上述 Base 模型外，还需要：

```text
ao-qwen35-train.tar.gz
data/ao_instructions/ao.jsonl
data/testcase_tables/all.jsonl
train/ 下的全部训练脚本和 Prompt
```

如果已经生成六个 `train/eval/test` 文件，可以直接传输六个文件而不重新切分。

---

## 6. 设置本地路径

根据离线服务器真实目录修改以下变量：

```bash
export AO_PROJECT_ROOT=/root/autodl-tmp/XIFEI_Agent/task1/ao-testcase-generation
export AO_BASE_MODEL_DIR=/root/autodl-tmp/XIFEI_Agent/task1/qwen3_5_9b_deploy/models/Qwen3.5-9B
export AO_PEFT_ADAPTER_DIR="$AO_PROJECT_ROOT/outputs/qwen35_lora_full_v1"
export AO_VLLM_ADAPTER_DIR="$AO_PROJECT_ROOT/outputs/qwen35_lora_full_v1_vllm_prefixfix"
export AO_EVAL_DIR="$AO_PROJECT_ROOT/eval_results/qwen35_lora_prefixfix_full"
cd "$AO_PROJECT_ROOT"
```

这些变量只在当前终端有效。每个新终端都需要重新设置。

部署前检查核心文件：

```bash
test -f "$AO_BASE_MODEL_DIR/config.json"
test -f "$AO_BASE_MODEL_DIR/model.safetensors.index.json"
test -f "$AO_VLLM_ADAPTER_DIR/adapter_model.safetensors"
test -f "$AO_VLLM_ADAPTER_DIR/adapter_config.json"
test -f "$AO_PROJECT_ROOT/train/system_prompt_v4.txt"
test -f "$AO_PROJECT_ROOT/pre/static/index.html"
echo "核心文件检查通过"
```

---

## 7. 直接部署 Base、Base+LoRA 和网页

这是已经训练完成后最常用的运行方式，需要两个终端。

### 7.1 终端一：启动 vLLM

激活 vLLM 环境并设置严格离线变量：

```bash
激活 vllm 环境

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_UPDATE_CHECK=1
export HF_HUB_DISABLE_XET=1
export VLLM_DO_NOT_TRACK=1
export VLLM_NO_USAGE_STATS=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

unset HF_TOKEN HUGGING_FACE_HUB_TOKEN WANDB_API_KEY
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

cd "$AO_PROJECT_ROOT"
```

启动一个 vLLM 引擎，同时注册 Base 和 LoRA 两个模型名：

```bash
vllm serve "$AO_BASE_MODEL_DIR" \
  --served-model-name qwen35-base \
  --enable-lora \
  --lora-modules "qwen35-lora=$AO_VLLM_ADAPTER_DIR" \
  --max-lora-rank 32 \
  --max-loras 1 \
  --language-model-only \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  --generation-config vllm \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --host 127.0.0.1 \
  --port 8000
```

说明：

- `qwen35-base` 只使用 Base 权重。
- `qwen35-lora` 使用同一个 Base 加载转换后的 LoRA。
- 一张 GPU 使用 `--tensor-parallel-size 1`。
- Adapter 的 rank 为 32，因此必须设置 `--max-lora-rank 32`。
- `--language-model-only` 让 Qwen3.5 以纯文本模式运行，不使用视觉输入。
- `VLLM_USE_FLASHINFER_SAMPLER=0` 允许当前 Conda 构建在没有
  `flashinfer` Python 包时使用原生采样器。
- 第一次启动需要加载权重、torch.compile、Triton JIT 和 CUDA Graph
  预热，数分钟内 8000 端口不可用属于正常现象。

不要将原始 `outputs/qwen35_lora_full_v1` 传给 `--lora-modules`，否则可能出现
服务不报错但 LoRA 实际未生效的情况。

### 7.2 检查 vLLM

在另一个终端执行：

```bash
curl -s http://127.0.0.1:8000/v1/models | python -m json.tool
```

返回的模型列表应包含：

```text
qwen35-base
qwen35-lora
```

只出现 Base 时，网页无法进行成对比较，应检查 `--lora-modules` 路径、Adapter
前缀和启动日志。

### 7.3 终端二：启动网页后端

网页后端本身不加载模型，可以继续使用 `ao-qwen35-vllm` 环境：

```bash
export AO_PROJECT_ROOT=/root/autodl-tmp/XIFEI_Agent/task1/ao-testcase-generation
export AO_BASE_MODEL_DIR=/root/autodl-tmp/XIFEI_Agent/task1/qwen3_5_9b_deploy/models/Qwen3.5-9B
export AO_PEFT_ADAPTER_DIR="$AO_PROJECT_ROOT/outputs/qwen35_lora_full_v1"
export AO_VLLM_ADAPTER_DIR="$AO_PROJECT_ROOT/outputs/qwen35_lora_full_v1_vllm_prefixfix"
export AO_EVAL_DIR="$AO_PROJECT_ROOT/eval_results/qwen35_lora_prefixfix_full"
cd "$AO_PROJECT_ROOT"

激活 vllm 环境

export OMP_NUM_THREADS=1
cd "$AO_PROJECT_ROOT"

python pre/server.py \
  --host 127.0.0.1 \
  --port 8081 \
  --vllm-url http://127.0.0.1:8000/v1 \
  --base-model qwen35-base \
  --lora-model qwen35-lora \
  --eval-dir "$AO_EVAL_DIR" \
  --prompt-mode compressed \
  --temperature 0 \
  --top-p 1 \
  --max-tokens 8192 \
  --request-timeout 600
```

评测目录不是网页实时推理的必要条件：

- 存在完整评测目录时，网页会展示汇总指标、字段指标和逐样本结果。
- 不存在评测目录时，历史评测面板显示不可用，但实时 Base/LoRA 推理仍可运行。

### 7.4 检查网页服务

```bash
curl -s http://127.0.0.1:8081/api/health | python -m json.tool
```

实时对比就绪时，关键字段应类似：

```json
{
  "comparison_ready": true,
  "base_model_available": true,
  "lora_model_available": true,
  "qwen_api_status": "offline_disabled"
}
```

`qwen_api_status=offline_disabled` 是预期状态，不是故障。

### 7.5 访问网页的两种方式

无论使用哪种方式，都保持 `pre/server.py` 使用 `--host 127.0.0.1`，不要改成
`0.0.0.0`。

#### 方式一：在部署服务器本机访问

在**运行于部署服务器上的浏览器**中打开：

```text
http://127.0.0.1:8081
```

如果通过远程桌面操作服务器桌面，只要浏览器进程运行在服务器上，也属于服务器
本机访问。这是保密服务器环境中的优先方式。

#### 方式二：通过 SSH 本地端口转发访问

该方式适合在租用的实验服务器上调试，但必须得到服务器安全策略允许。

首先确认服务器端的 vLLM 和 `pre/server.py` 都在运行。然后在**自己的电脑**
终端执行下面的一行命令，不要在远程服务器终端中执行：

```text
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:8081:127.0.0.1:8081 -p <SSH端口> <用户名>@<SSH服务器地址>
```

其中：

- `<SSH端口>` 是租用平台控制台提供的 SSH 登录端口，不是网页端口 8081。
- `<用户名>` 通常是 `root`，以租用平台提供的信息为准。
- `<SSH服务器地址>` 是平台提供的 SSH 主机地址。
- 如果平台要求私钥，在命令中增加 `-i <私钥文件路径>`。

隧道建立成功后，保持该 SSH 终端窗口运行，在自己电脑的浏览器中访问：

```text
http://127.0.0.1:8081
```

该流量路径为：

```text
本地浏览器 127.0.0.1:8081
  → SSH 加密隧道
  → 服务器 127.0.0.1:8081
  → pre/server.py
```

关闭本地 SSH 终端或按 `Ctrl+C` 即可关闭隧道，不会停止服务器上的 vLLM 和
网页后端。保密服务器如果禁止端口转发，只能使用方式一或管理员批准的访问方式。

#### 如果打开后仍然是旧版网页

第 7.5 节只改变访问路径，不会把新版前端自动上传到服务器。新版网页依赖服务器
上的以下两个文件：

```text
pre/server.py
pre/static/index.html
```

先在部署服务器上进入实际项目目录，确认文件是新版：

```bash
cd "$AO_PROJECT_ROOT"

grep -n "api/infer/compare" pre/server.py
grep -n "Base / LoRA 成对评测" pre/static/index.html
grep -n "同一 AO 实时对比" pre/static/index.html
```

三条命令都应找到内容。如果没有输出，说明服务器上仍是旧代码，需要先按照保密
服务器允许的文件传输方式更新整个 `pre/` 目录。

然后检查当前占用 8081 端口的进程：

```bash
ss -lntp | grep ':8081'
ps -ef | grep '[p]re/server.py'
```

如果存在旧的 `pre/server.py`，优先回到启动它的终端按 `Ctrl+C`。找不到原终端
时，根据上一步显示的准确 PID 结束对应进程：

```bash
kill <PID>
```

不要使用不带目标确认的批量结束命令。确认 8081 已释放后，严格按照第 7.3 节从
`$AO_PROJECT_ROOT` 重新启动 `pre/server.py`。

重启后先在服务器本机验证实际返回的 HTML：

```bash
curl -s http://127.0.0.1:8081/ \
  | grep -E "Base / LoRA 成对评测|同一 AO 实时对比"
```

能够看到这两个新版标志，说明服务器已经正确返回新版网页。如果服务器端检查
通过，但浏览器仍显示旧页面，可使用 `Ctrl+F5` 强制刷新，或者访问：

```text
http://127.0.0.1:8081/?v=2
```

如果使用 SSH 隧道，还要确认隧道连接的是当前这台租用服务器及其正确 SSH
端口；必要时关闭旧隧道并重新建立。

---

## 8. 从数据开始重新训练

训练前激活 `ao-qwen35-train`：

```bash
激活 train 环境

export OMP_NUM_THREADS=1
cd "$AO_PROJECT_ROOT"
```

### 8.1 切分数据

输入：

- `all.jsonl`：包含 `id`、`source` 和 Ground Truth `rows`。
- `ao.jsonl`：包含相同 `id` 对应的 AO 原文 `content`。

不要依赖脚本中的历史默认路径，应显式传参：

```bash
python train/split_dataset.py \
  --output-all data/testcase_tables/all.jsonl \
  --cleaned-ao data/ao_instructions/ao.jsonl \
  --out-dir data \
  --train-ratio 0.80 \
  --eval-ratio 0.10 \
  --test-ratio 0.10 \
  --seed 42 \
  --min-rows 1
```

生成：

```text
data/train.jsonl
data/eval.jsonl
data/test.jsonl
```

`id` 是业务样本 ID，不是 JSONL 行号。分层切分和打乱后，ID 不会按数字顺序
排列。

### 8.2 转换为 SFT messages

```bash
python train/prepare_sft_data.py \
  --input-dir data \
  --output-dir data
```

生成：

```text
data/train_sft.jsonl
data/eval_sft.jsonl
data/test_sft.jsonl
```

每条数据包含 `system/user/assistant` 三条消息：

- `system` 来自 `train/system_prompt_v4.txt`。
- `user` 是 AO 原文。
- `assistant` 是 Ground Truth `rows` 的紧凑 JSON 数组。

训练只计算 assistant JSON 和结束标记的因果语言模型交叉熵；system、user 和
padding token 的 label 为 `-100`，不参与损失。

### 8.3 统计真实 Token 长度

更换数据或 Prompt 后必须重新统计：

```bash
python train/prepare_training/count_dataset.py \
  --model-dir "$AO_BASE_MODEL_DIR" \
  --files data/train_sft.jsonl data/eval_sft.jsonl \
  --training-max-length 8192 \
  --thresholds 4096 6144 8192 12288 16384 \
  --top-n 20 \
  --output-json train/prepare_training/token_length_report.json
```

该脚本只加载本地 Tokenizer，不加载 9B 权重，也不会修改数据。当前数据已知最大
总长度为 7780 token，因此训练使用 `--max_seq_length 8192`。训练脚本遇到
超长样本会直接报错，不会静默截断 assistant 答案。

### 8.4 一步冒烟训练

```bash
python train/train_sft_final.py \
  --model_dir "$AO_BASE_MODEL_DIR" \
  --train_file data/train_sft.jsonl \
  --eval_file data/eval_sft.jsonl \
  --output_dir outputs/qwen35_lora_smoke \
  --max_train_samples 2 \
  --max_eval_samples 2 \
  --max_steps 1 \
  --gradient_accumulation_steps 1 \
  --warmup_ratio 0 \
  --logging_steps 1 \
  --save_steps 1 \
  --eval_steps 1 \
  --run_name qwen35_lora_smoke
```

只有在模型类别、纯语言模型检查、LoRA 目标层、assistant-only labels、一步训练、
一次验证和 Adapter 保存都成功后，才能开始正式训练。

### 8.5 正式训练

当前在 A800 80GB 上验证过的命令：

```bash
python train/train_sft_final.py \
  --model_dir "$AO_BASE_MODEL_DIR" \
  --train_file data/train_sft.jsonl \
  --eval_file data/eval_sft.jsonl \
  --output_dir outputs/qwen35_lora_full_v1 \
  --num_train_epochs 2 \
  --max_seq_length 8192 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.03 \
  --logging_steps 10 \
  --save_steps 200 \
  --eval_steps 200 \
  --run_name qwen35_lora_full_v1
```

验证集每 200 个优化器更新步参与一次 `eval_loss` 计算，训练结束后按最低
`eval_loss` 恢复最佳 checkpoint。测试集不得传入 `--eval_file`。

训练日志位于：

```text
outputs/qwen35_lora_full_v1/train_logs/
  training_manifest.json
  train_status.jsonl
  train_summary.json
```

原始 Adapter 核心文件：

```text
outputs/qwen35_lora_full_v1/
  adapter_model.safetensors
  adapter_config.json
  trainer_state.json
  train_results.json
```

32GB 显卡即使 batch size 为 1，也可能在长样本反向传播时显存不足。梯度累积
不会同时放入 8 条样本，但单条长序列的激活值、梯度、LoRA 参数和优化器状态仍然
占用显存。

---

## 9. 训练后必须转换 Adapter 前缀

当前 PEFT 输出使用：

```text
base_model.model.model.layers.
```

当前 vLLM 0.24.0 对 Qwen3.5 动态 LoRA 使用：

```text
base_model.model.language_model.model.layers.
```

直接使用原始 Adapter 时，vLLM 可能不报错，但 Base 与 LoRA 输出完全相同。
必须保留原始目录，并生成新的 vLLM 专用目录。

执行转换：

```bash
python - <<'PY'
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

source_dir = Path("outputs/qwen35_lora_full_v1")
target_dir = Path("outputs/qwen35_lora_full_v1_vllm_prefixfix")
source_weights = source_dir / "adapter_model.safetensors"
source_config = source_dir / "adapter_config.json"

if not source_weights.is_file() or not source_config.is_file():
    raise FileNotFoundError("原始 Adapter 权重或配置不存在")
if target_dir.exists():
    raise FileExistsError(f"目标目录已存在，请人工确认后使用新目录名：{target_dir}")

old_prefix = "base_model.model.model.layers."
new_prefix = "base_model.model.language_model.model.layers."

with safe_open(str(source_weights), framework="pt", device="cpu") as handle:
    metadata = handle.metadata()
    original = {
        key: handle.get_tensor(key)
        for key in handle.keys()
    }

converted = {}
converted_count = 0
for key, tensor in original.items():
    if key.startswith(old_prefix):
        key = new_prefix + key[len(old_prefix):]
        converted_count += 1
    if key in converted:
        raise KeyError(f"转换后出现重复键：{key}")
    converted[key] = tensor

if converted_count == 0:
    raise RuntimeError("没有发现旧前缀，请检查 PEFT/vLLM 版本和 Adapter")

target_dir.mkdir(parents=True)
save_file(
    converted,
    str(target_dir / "adapter_model.safetensors"),
    metadata=metadata or {"format": "pt"},
)
shutil.copy2(source_config, target_dir / "adapter_config.json")

print("总张量:", len(original))
print("转换张量:", converted_count)
print("输出目录:", target_dir.resolve())
PY
```

校验转换结果：

```bash
python - <<'PY'
from pathlib import Path
from safetensors import safe_open

path = Path(
    "outputs/qwen35_lora_full_v1_vllm_prefixfix/"
    "adapter_model.safetensors"
)
old_prefix = "base_model.model.model.layers."
new_prefix = "base_model.model.language_model.model.layers."

with safe_open(str(path), framework="pt", device="cpu") as handle:
    keys = list(handle.keys())
    old_count = sum(key.startswith(old_prefix) for key in keys)
    new_count = sum(key.startswith(new_prefix) for key in keys)
    b_norms = [
        handle.get_tensor(key).float().norm().item()
        for key in keys
        if ".lora_B." in key
    ]

print("权重张量:", len(keys))
print("旧前缀:", old_count)
print("vLLM 前缀:", new_count)
print("全零 LoRA-B:", sum(value == 0 for value in b_norms))

assert old_count == 0
assert new_count > 0
assert b_norms and any(value > 0 for value in b_norms)
PY
```

当前 rank-32 训练结果应有 496 个权重张量，其中 248 个 LoRA-B 张量，且
LoRA-B 不应全部为零。

---

## 10. 使用 vLLM 进行离线成对评测

评测前先停止 `vllm serve`，然后激活 `ao-qwen35-vllm`。评测脚本自己创建
vLLM 引擎，不需要启动 HTTP 服务：

```bash
激活 vllm 环境

export OMP_NUM_THREADS=1
cd "$AO_PROJECT_ROOT"
```

### 10.1 先评测 8 条

```bash
python train/eval_model_vllm.py \
  --base_model "$AO_BASE_MODEL_DIR" \
  --adapter "$AO_VLLM_ADAPTER_DIR" \
  --data_file data/eval_sft.jsonl \
  --output_dir eval_results/qwen35_lora_prefixfix_8 \
  --max_samples 8
```

默认使用相同的 tokenized prompt 和生成参数依次评测 Base 与 LoRA：

```text
enable_thinking=False
temperature=0
top_p=1
max_new_tokens=8192
max_model_len=16384
dtype=bfloat16
metric_version=v2
```

必须检查 Base 与 LoRA 原始输出不是全部相同。全部相同时停止后续评测，检查是否
误用了原始 PEFT Adapter。

### 10.2 完整验证集

不传 `--max_samples`：

```bash
python train/eval_model_vllm.py \
  --base_model "$AO_BASE_MODEL_DIR" \
  --adapter "$AO_VLLM_ADAPTER_DIR" \
  --data_file data/eval_sft.jsonl \
  --output_dir eval_results/qwen35_lora_prefixfix_full
```

### 10.3 最终测试集

只有模型、Prompt、解码参数、checkpoint 和指标全部冻结后，才执行一次：

```bash
python train/eval_model_vllm.py \
  --base_model "$AO_BASE_MODEL_DIR" \
  --adapter "$AO_VLLM_ADAPTER_DIR" \
  --data_file data/test_sft.jsonl \
  --output_dir eval_results/qwen35_lora_prefixfix_test_full
```


### 10.4 评测输出

每次成功评测生成：

| 文件 | 作用 |
|---|---|
| `evaluation_manifest.json` | 模型、Adapter、数据、版本、GPU、哈希和参数 |
| `base_predictions.jsonl` | Base 逐样本输出、解析状态和评分 |
| `lora_predictions.jsonl` | LoRA 逐样本输出、解析状态和评分 |
| `base_metrics.json` | Base 汇总指标 |
| `lora_metrics.json` | LoRA 汇总指标 |
| `vllm_compare_summary.json` | Base/LoRA 汇总及成对差值 |

当前 `eval_sft.jsonl` 的 615 条参考结果为：

```text
Base overall：      0.5123
LoRA overall：      0.9014
绝对提升：          +0.3891
相对提升：          +75.95%
LoRA 更好：         602
LoRA 更差：         13
LoRA schema 合法：  615/615
LoRA 长度截断：     0
```

这是验证集结果，不是最终独立测试集结果。

---
