# Qwen3.5-9B AO 测试用例生成：离线训练与评测指南

本文档对应当前仓库中的以下脚本：

| 阶段 | 脚本 | 作用 |
|---|---|---|
| 数据切分 | `train/split_dataset.py` | 关联 AO 原文与结构化 Ground Truth，按 `source` 分层切分 |
| SFT 数据准备 | `train/prepare_sft_data.py` | 生成 `system/user/assistant` messages 格式 |
| Token 统计 | `train/prepare_training/count_dataset.py` | 使用本地 Qwen3.5 tokenizer 统计真实长度 |
| LoRA 训练 | `train/train_sft_final.py` | 使用 Transformers + PEFT 进行 assistant-only LoRA SFT |
| Base/LoRA 评测 | `train/eval_model_vllm.py` | 使用一个本地 vLLM 引擎进行 Base 与 LoRA 成对评测 |
| 指标 | `train/metrics_v2.py` | 当前默认使用的结构与内容综合指标 |

所有命令都建议从项目根目录 `ao-testcase-generation` 执行。

> 重要：当前 `train_sft_final.py` 和 `eval_model_vllm.py` 都显式设置
> `enable_thinking=False`，评测默认使用 `metrics_v2.py`。如果仓库根目录旧版
> README 与本文档不一致，以当前代码和本文档为准。

---

## 1. 完整流程

推荐顺序如下：

```text
准备离线环境和本地模型
        ↓
split_dataset.py
        ↓
prepare_sft_data.py
        ↓
count_dataset.py
        ↓
1-step 冒烟训练
        ↓
正式 LoRA 训练
        ↓
保留原始 PEFT adapter
        ↓
转换 LoRA 权重前缀，生成 vLLM adapter
        ↓
8 条 Base/LoRA 冒烟评测并确认输出不同
        ↓
完整 eval 评测
        ↓
冻结参数后执行一次最终 test 评测
```

训练和评测使用两个独立 Conda 环境：

- `ao-qwen35-train`：Transformers、PEFT、Datasets，用于训练。
- `ao-qwen35-vllm`：vLLM，用于 Base/LoRA 离线批量推理和评测。

不要在同一个环境中混装训练版 PyTorch 与 vLLM 的 CUDA 依赖。

---

## 2. 无网络服务器部署准备

### 2.1 必须传入离线服务器的内容

至少需要：

```text
ao-testcase-generation/                 # 本项目代码
qwen3_5_9b_deploy/models/Qwen3.5-9B/   # 完整 Base 模型
ao-qwen35-train.tar.gz                  # 已验证的训练 Conda 环境
ao-qwen35-vllm.tar.gz                   # 已验证的 vLLM Conda 环境
```

Base 模型目录至少应包含：

```text
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
```

代码只读取本地模型路径，不会自动从 Hugging Face 或 ModelScope 下载缺失文件。

### 2.2 在有网络、且硬件兼容的服务器上打包 Conda 环境

训练环境可以根据 `train/environment-train-a800.yml` 创建：

```bash
conda env create -f train/environment-train-a800.yml
conda activate ao-qwen35-train
python -c "import torch, transformers, datasets, peft, accelerate; print(torch.__version__, torch.version.cuda); print(transformers.__version__)"
conda pack -n ao-qwen35-train -o ao-qwen35-train.tar.gz
```

对已经验证可运行的 vLLM 环境执行：

```bash
conda activate ao-qwen35-vllm
python -c "import torch, triton, vllm; print('GPU:', torch.cuda.get_device_name(0)); print('PyTorch CUDA:', torch.version.cuda); print('vLLM:', vllm.__version__); print('Triton:', triton.runtime.driver.active.get_current_target())"
conda pack -n ao-qwen35-vllm -o ao-qwen35-vllm.tar.gz
```

当前已经验证的主要版本为：

```text
Python        3.11
PyTorch       2.11.0
CUDA runtime  12.9
Transformers  5.14.1
Datasets      5.0.0
PEFT          0.19.1
Accelerate    1.14.0
vLLM          0.24.0
```

`conda-pack` 应在 Linux x86_64 环境生成，并用于相同平台。宿主机 NVIDIA
Driver 和 Linux 内核不会被打入压缩包，目标服务器仍须提供兼容的 NVIDIA
驱动。

如果在没有 GPU 的机器上创建 `pytorch-gpu` 环境，Conda 可能报告缺少
`__cuda`。这不是模型问题，而是求解器没有检测到 CUDA 驱动。最稳妥的方式是
在与离线目标机相同或兼容的 GPU 服务器上创建、验证并打包环境。

### 2.3 在无网络服务器上解压环境

以下路径仅为示例，可按实际磁盘位置修改：

```bash
mkdir -p /root/autodl-tmp/conda-envs/ao-qwen35-train
tar -xzf ao-qwen35-train.tar.gz \
  -C /root/autodl-tmp/conda-envs/ao-qwen35-train
source /root/autodl-tmp/conda-envs/ao-qwen35-train/bin/activate
conda-unpack
```

vLLM 环境：

```bash
mkdir -p /root/autodl-tmp/conda-envs/ao-qwen35-vllm
tar -xzf ao-qwen35-vllm.tar.gz \
  -C /root/autodl-tmp/conda-envs/ao-qwen35-vllm
source /root/autodl-tmp/conda-envs/ao-qwen35-vllm/bin/activate
conda-unpack
```

建议在有网服务器生成传输清单：

```bash
sha256sum ao-qwen35-train.tar.gz ao-qwen35-vllm.tar.gz > conda_envs.sha256
sha256sum -c conda_envs.sha256
```

在离线服务器再次执行 `sha256sum -c`，确认传输文件没有损坏。

---

## 3. 设置项目路径

下面以当前 AutoDL 目录结构为例：

```bash
export AO_PROJECT_ROOT=/root/autodl-tmp/XIFEI_Agent/task1/ao-testcase-generation
export AO_BASE_MODEL_DIR=/root/autodl-tmp/XIFEI_Agent/task1/qwen3_5_9b_deploy/models/Qwen3.5-9B
cd "$AO_PROJECT_ROOT"
```

这些变量只在当前终端有效。重新登录服务器后需要重新设置。

---

## 4. 训练前的数据准备

如果已经有以下六个文件，可以直接跳到第 4.3 节：

```text
data/train.jsonl
data/eval.jsonl
data/test.jsonl
data/train_sft.jsonl
data/eval_sft.jsonl
data/test_sft.jsonl
```

### 4.1 切分 train/eval/test

输入文件的职责：

- `all.jsonl`：提供样本 `id`、`source` 和 Ground Truth `rows`。
- `ao.jsonl`：提供相同 `id` 对应的 AO 原文 `content`。

执行：

```bash
python train/split_dataset.py \
  --output-all /path/to/all.jsonl \
  --cleaned-ao /path/to/ao.jsonl \
  --out-dir data \
  --train-ratio 0.80 \
  --eval-ratio 0.10 \
  --test-ratio 0.10 \
  --seed 42 \
  --min-rows 1
```

输出：

```text
data/train.jsonl
data/eval.jsonl
data/test.jsonl
```

切分逻辑是：

1. 以 `all.jsonl` 中的 Ground Truth 为基准。
2. 使用 `id` 在 `ao.jsonl` 中寻找 AO 原文。
3. 缺少 AO 原文或 `rows` 少于 `min_rows` 的样本被过滤。
4. 按 `source` 分层，并使用固定随机种子切分。

必须检查终端输出中的：

- 两个输入文件各自成功加载的数量。
- “跳过（无 AO）”数量。
- 合并后的有效样本数量。
- train/eval/test 的最终数量。

数据文件中的 `id` 是业务样本 ID，不是 JSONL 行号；切分和打乱后，ID 不会按
数值顺序排列。

### 4.2 转换为 SFT messages 格式

```bash
python train/prepare_sft_data.py \
  --input-dir data \
  --output-dir data
```

输出：

```text
data/train_sft.jsonl
data/eval_sft.jsonl
data/test_sft.jsonl
```

每条 SFT 数据结构为：

```json
{
  "id": "样本ID",
  "source": "来源章节",
  "messages": [
    {"role": "system", "content": "train/system_prompt_v4.txt 的完整内容"},
    {"role": "user", "content": "AO 原文"},
    {"role": "assistant", "content": "Ground Truth rows 的紧凑 JSON 数组"}
  ]
}
```

训练只计算最终 `assistant` JSON 的损失，`system` 和 `user` token 的 label
会被设置为 `-100`。

### 4.3 统计真实 Token 长度

该脚本只读取 tokenizer 和数据，不加载 9B 模型权重，也不会修改或截断数据：

```bash
python train/prepare_training/count_dataset.py \
  --model-dir "$AO_BASE_MODEL_DIR" \
  --files data/train_sft.jsonl data/eval_sft.jsonl \
  --training-max-length 8192 \
  --thresholds 4096 6144 8192 12288 16384 \
  --top-n 20 \
  --output-json train/prepare_training/token_length_report.json
```

当前数据的已知最大长度为：

```text
train_sft.jsonl 最大总长度：7780 token
eval_sft.jsonl  最大总长度：5952 token
```

因此正式训练使用 `--max_seq_length 8192`，当前数据不会被截断。

`train_sft_final.py` 对超长样本采用“立即报错”策略，不会静默截断
assistant 答案。如果更换了数据或 prompt，必须重新运行 Token 统计。

---

## 5. LoRA 训练

### 5.1 训练前检查

激活训练环境：

```bash
conda activate ao-qwen35-train
cd "$AO_PROJECT_ROOT"
```

检查 GPU、CUDA 和本地模型：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.version.cuda)"
python -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('$AO_BASE_MODEL_DIR', local_files_only=True); print(type(t).__name__, t.model_max_length)"
```

注意：

- 当前正式训练参数已经在 A800 80GB 上验证。
- 32GB 显卡可能在长样本反向传播时显存不足。
- `gradient_accumulation_steps=8` 不会同时把 8 条样本放进显存，但单条长序列的
  激活值、反向梯度、LoRA 参数和优化器状态仍然需要显存。
- 缺少 `causal_conv1d` 或 `flash_linear_attention` 时，Transformers 会回退到
  PyTorch 实现，通常可以训练，但速度会变慢。

训练脚本在导入 Hugging Face 组件前会强制设置严格离线环境变量，并且
`from_pretrained(..., local_files_only=True)`，因此运行期不会尝试连接
Hugging Face Hub、W&B 或 Xet。

### 5.2 先执行 1-step 冒烟训练

每次冒烟使用新的输出目录：

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

冒烟测试通过应同时满足：

- 正确识别 `Qwen3_5ForCausalLM` / `qwen3_5_text`。
- 输出“纯语言模型检查通过”。
- LoRA 目标层均有匹配。
- 可训练参数全部属于 LoRA。
- assistant-only tokenization 成功。
- 完成 1 个训练 step 和 1 次验证。
- 输出目录中存在 `adapter_model.safetensors` 和 `adapter_config.json`。

### 5.3 正式训练

当前已经验证的正式命令：

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

关键参数：

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `num_train_epochs` | 2 | 完整遍历训练集两次 |
| `max_seq_length` | 8192 | 单条完整对话最大 Token 数 |
| `per_device_train_batch_size` | 1 | 每次前向/反向实际处理一条样本 |
| `gradient_accumulation_steps` | 8 | 累积 8 次梯度后更新一次参数 |
| 有效 batch size | 8 | 单卡时为 `1 × 8` |
| `learning_rate` | `1e-4` | LoRA 学习率 |
| `warmup_ratio` | 0.03 | 前 3% 更新步逐渐升高学习率 |
| `eval_steps` | 200 | 每 200 个优化器更新步验证一次 |
| `save_steps` | 200 | 每 200 个优化器更新步保存一次 |
| `lora_r` | 32 | LoRA 秩，代码默认值 |
| `lora_alpha` | 16 | LoRA 缩放参数，代码默认值 |
| `lora_dropout` | 0.05 | LoRA dropout，代码默认值 |

训练期间：

- 使用因果语言模型交叉熵。
- prompt token 和 padding token 使用 `-100`，不参与损失。
- 只有 assistant JSON 与对话结束标记参与损失。
- 每 `eval_steps` 使用 `eval_sft.jsonl` 计算 `eval_loss`。
- 根据最低 `eval_loss` 选择最佳 checkpoint。
- 最终在输出目录根部保存最佳 LoRA adapter 和 tokenizer。
- 训练日志保存在 `OUTPUT_DIR/train_logs/`。

不要把 `test_sft.jsonl` 传给 `--eval_file`。测试集不应参与 checkpoint 或
超参数选择。

### 5.4 断点续训

断点续训必须使用原始 PEFT 训练目录，而不是后面转换过前缀的 vLLM 目录：

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
  --run_name qwen35_lora_full_v1 \
  --resume_from_checkpoint outputs/qwen35_lora_full_v1/checkpoint-1200
```

`num_train_epochs=2` 表示训练的总目标仍为 2 轮，不是从 checkpoint 再追加
2 轮。不要同时使用 `--resume_from_checkpoint` 和 `--overwrite_output_dir`。

---

## 6. 训练后检查

正式训练结束后至少检查：

```bash
ls -lh outputs/qwen35_lora_full_v1/adapter_model.safetensors
ls -lh outputs/qwen35_lora_full_v1/adapter_config.json
ls -lh outputs/qwen35_lora_full_v1/trainer_state.json
ls -lh outputs/qwen35_lora_full_v1/train_logs/
```

核心文件含义：

- `adapter_model.safetensors`：LoRA A/B 权重，不包含完整 9B Base 权重。
- `adapter_config.json`：LoRA 秩、alpha、dropout、目标层和任务类型。
- `trainer_state.json`：step、epoch、训练/验证历史、最佳 checkpoint。
- `train_results.json`：最终训练性能与 loss。
- `train_logs/training_manifest.json`：数据、模型、版本、参数和损失配置清单。
- `train_logs/train_status.jsonl`：逐 step 训练日志。

检查 LoRA-B 不是全零：

```bash
python - <<'PY'
from pathlib import Path
from safetensors import safe_open

path = Path("outputs/qwen35_lora_full_v1/adapter_model.safetensors")
with safe_open(str(path), framework="pt", device="cpu") as handle:
    keys = list(handle.keys())
    norms = [
        handle.get_tensor(key).float().norm().item()
        for key in keys
        if ".lora_B." in key
    ]

print("权重张量数:", len(keys))
print("LoRA-B 张量数:", len(norms))
print("全零 LoRA-B:", sum(value == 0 for value in norms))
print("LoRA-B norm:", min(norms), "->", max(norms))
assert norms and any(value > 0 for value in norms), "LoRA-B 全零，训练结果无效"
PY
```

---

## 7. 必做：将 PEFT LoRA 权重转换为 vLLM 前缀

### 7.1 为什么必须转换

当前 PEFT 训练输出的权重键前缀为：

```text
base_model.model.model.layers.
```

当前 vLLM 0.24.0 加载 Qwen3.5 时需要：

```text
base_model.model.language_model.model.layers.
```

如果直接把原始 adapter 交给 vLLM，vLLM 可能不报错，但 LoRA 实际没有作用，
最终 Base 与 LoRA 输出会完全相同。因此不能只根据“程序成功运行”判断 LoRA
已经加载。

必须保留两份目录：

```text
outputs/qwen35_lora_full_v1/                 # 原始 PEFT 权重，用于审计和续训
outputs/qwen35_lora_full_v1_vllm_prefixfix/  # 转换后权重，只用于 vLLM
```

不要覆盖原始权重，也不要转换 `checkpoint-*` 目录。

### 7.2 执行转换

如果目标目录已经存在，请先换一个新的目录名；脚本会拒绝覆盖已有目录：

```bash
python - <<'PY'
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

source_dir = Path("outputs/qwen35_lora_full_v1")
target_dir = Path("outputs/qwen35_lora_full_v1_vllm_prefixfix")
source_weights = source_dir / "adapter_model.safetensors"
source_config = source_dir / "adapter_config.json"

if not source_weights.is_file() or not source_config.is_file():
    raise FileNotFoundError("原始 adapter 权重或配置不存在")
if target_dir.exists():
    raise FileExistsError(f"目标目录已存在，请更换目录名或人工确认后处理：{target_dir}")

old_prefix = "base_model.model.model.layers."
new_prefix = "base_model.model.language_model.model.layers."

with safe_open(str(source_weights), framework="pt", device="cpu") as handle:
    metadata = handle.metadata()
    original = {key: handle.get_tensor(key) for key in handle.keys()}

converted = {}
converted_count = 0
for key, tensor in original.items():
    if key.startswith(old_prefix):
        new_key = new_prefix + key[len(old_prefix):]
        converted_count += 1
    else:
        new_key = key
    if new_key in converted:
        raise KeyError(f"转换后出现重复权重键：{new_key}")
    converted[new_key] = tensor

if converted_count == 0:
    raise RuntimeError("没有发现需要转换的旧前缀，请检查 adapter 或 PEFT/vLLM 版本")

target_dir.mkdir(parents=True)
save_file(
    converted,
    str(target_dir / "adapter_model.safetensors"),
    metadata=metadata or {"format": "pt"},
)
shutil.copy2(source_config, target_dir / "adapter_config.json")

print("总权重张量:", len(original))
print("已转换张量:", converted_count)
print("vLLM adapter:", target_dir.resolve())
PY
```

### 7.3 校验转换结果

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

print("权重张量数:", len(keys))
print("旧前缀数量:", old_count)
print("vLLM 前缀数量:", new_count)
print("全零 LoRA-B:", sum(value == 0 for value in b_norms))
assert old_count == 0, "仍然存在旧前缀"
assert new_count > 0, "没有发现 vLLM 前缀"
assert b_norms and any(value > 0 for value in b_norms), "LoRA-B 权重无效"
PY
```

对于当前训练配置，正常结果是：

```text
权重张量数：496
旧前缀数量：0
vLLM 前缀数量：496
LoRA-B 张量数：248，且不应全部为零
```

---

## 8. 使用 vLLM 评测 Base 与 LoRA

### 8.1 不需要启动 vLLM API 服务

`train/eval_model_vllm.py` 使用 vLLM 的本地 Python API：

1. 初始化一个 Base 模型引擎。
2. 对选中的同一批 prompt 运行 Base。
3. 使用动态 `LoRARequest` 挂载转换后的 adapter。
4. 使用相同 prompt、相同生成参数运行 LoRA。
5. 计算成对指标并保存逐样本证据。

因此运行该评测脚本前，不需要执行 `vllm serve`，也不需要监听 8000 端口。

### 8.2 评测环境检查

```bash
conda activate ao-qwen35-vllm
cd "$AO_PROJECT_ROOT"
nvidia-smi
python -c "import torch, triton, vllm; print('GPU:', torch.cuda.get_device_name(0)); print('CUDA:', torch.version.cuda); print('vLLM:', vllm.__version__); print('Triton target:', triton.runtime.driver.active.get_current_target())"
```

必须能看到类似：

```text
GPU: NVIDIA A800 80GB PCIe
CUDA: 12.9
vLLM: 0.24.0
Triton target: GPUTarget(backend='cuda', arch=80, warp_size=32)
```

如果 Triton 检查报告 `cuda.h: No such file or directory`，说明 vLLM 环境或
CUDA 开发头文件没有完整迁移。应先修复/重新打包环境，不能把它误认为模型结构
错误。

当前评测脚本设置 `VLLM_USE_FLASHINFER_SAMPLER=0`，所以 Conda 环境没有
`flashinfer` Python 包时会使用 vLLM 原生采样回退，不需要联网安装
FlashInfer。

### 8.3 先评测 8 条

```bash
python train/eval_model_vllm.py \
  --base_model "$AO_BASE_MODEL_DIR" \
  --adapter outputs/qwen35_lora_full_v1_vllm_prefixfix \
  --data_file data/eval_sft.jsonl \
  --output_dir eval_results/qwen35_lora_prefixfix_8 \
  --max_samples 8
```

默认生成配置：

```text
enable_thinking=False
temperature=0
top_p=1
max_new_tokens=8192
max_model_len=16384
seed=42
dtype=bfloat16
tensor_parallel_size=1
gpu_memory_utilization=0.85
guided/structured decoding=False
metric_version=v2
```

第一次启动会进行 torch.compile、Triton JIT 和 CUDA Graph 捕获，耗时较长属于
正常现象。第二次通常会复用编译缓存。

### 8.4 强制确认 LoRA 确实生效

8 条评测完成后执行：

```bash
python - <<'PY'
import json
from pathlib import Path

result_dir = Path("eval_results/qwen35_lora_prefixfix_8")

def load_jsonl(name):
    result = {}
    with (result_dir / name).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                result[str(row["id"])] = row
    return result

base = load_jsonl("base_predictions.jsonl")
lora = load_jsonl("lora_predictions.jsonl")
assert base.keys() == lora.keys(), "Base 与 LoRA 样本 ID 不一致"

different = [
    sample_id
    for sample_id in base
    if base[sample_id]["raw_output"] != lora[sample_id]["raw_output"]
]

print("样本数:", len(base))
print("原始输出不同:", f"{len(different)}/{len(base)}")
print("不同的 ID:", different)
assert different, "Base 与 LoRA 输出全部相同：停止全量评测并检查 adapter 前缀"
PY
```

当前已训练 adapter 的已知结果是 8/8 原始输出不同。如果结果为 0/8，不要开始
全量评测，优先检查是否误用了原始 `qwen35_lora_full_v1` 目录。

### 8.5 可选：32 条中等规模验证

```bash
python train/eval_model_vllm.py \
  --base_model "$AO_BASE_MODEL_DIR" \
  --adapter outputs/qwen35_lora_full_v1_vllm_prefixfix \
  --data_file data/eval_sft.jsonl \
  --output_dir eval_results/qwen35_lora_prefixfix_32 \
  --max_samples 32
```

32 条用于在全量运行前检查：

- Base/LoRA 输出是否确实不同。
- LoRA JSON 和 schema 合法率是否提高。
- 是否出现 8192 token 截断。
- vLLM 显存和运行时间是否正常。

### 8.6 完整验证集评测

不传 `--max_samples` 即评测全部样本：

```bash
python train/eval_model_vllm.py \
  --base_model "$AO_BASE_MODEL_DIR" \
  --adapter outputs/qwen35_lora_full_v1_vllm_prefixfix \
  --data_file data/eval_sft.jsonl \
  --output_dir eval_results/qwen35_lora_prefixfix_eval_full
```

验证集可以用于分析模型、选择 checkpoint 或调整超参数，但这会使其不再是完全
独立的最终测试结果。

### 8.7 最终测试集评测

模型、checkpoint、prompt、解码参数和评测指标全部冻结后，只执行一次：

```bash
python train/eval_model_vllm.py \
  --base_model "$AO_BASE_MODEL_DIR" \
  --adapter outputs/qwen35_lora_full_v1_vllm_prefixfix \
  --data_file data/test_sft.jsonl \
  --output_dir eval_results/qwen35_lora_prefixfix_test_full
```

不要根据 test 结果继续挑选 checkpoint 或调整参数；否则 test 集也会产生信息
泄漏。

---

## 9. 评测输出说明

每次成功评测生成六个文件：

| 文件 | 内容 |
|---|---|
| `evaluation_manifest.json` | 模型、adapter、数据、GPU、版本、哈希和生成参数 |
| `base_predictions.jsonl` | Base 逐样本原始输出、解析结果、token 和评分 |
| `lora_predictions.jsonl` | LoRA 逐样本原始输出、解析结果、token 和评分 |
| `base_metrics.json` | Base 汇总指标 |
| `lora_metrics.json` | LoRA 汇总指标 |
| `vllm_compare_summary.json` | Base/LoRA 汇总及成对差值 |

默认 v2 总分由结构分和内容分组成：

```text
overall = 0.4 × structure + 0.6 × content
```

逐样本 JSONL 必须保留。它们可以用于人工抽查、错误分析和重新计算指标，而不必
再次运行模型。

当前 `eval_sft.jsonl` 上已经得到的参考结果：

```text
样本数：615

Base overall：      0.5123
LoRA overall：      0.9014
绝对提升：          +0.3891
相对提升：          +75.95%

LoRA 更好：         602
LoRA 更差：         13
LoRA schema 合法：  615/615
LoRA 长度截断：     0
```

该结果是验证集结果，不应冒充最终独立测试集结果。

---

## 10. 常见错误

### 10.1 `Connection refused`

只有调用 OpenAI-compatible API 客户端时才需要提前启动 vLLM 服务。
`train/eval_model_vllm.py` 不访问 8000 端口，不需要服务进程。

### 10.2 Base 与 LoRA 输出完全相同

最常见原因是直接把原始 PEFT adapter 交给了 vLLM。检查：

```text
评测参数 --adapter 是否指向 *_vllm_prefixfix
旧前缀数量是否为 0
新前缀数量是否大于 0
LoRA-B 是否非零
```

### 10.3 `ModuleNotFoundError: flashinfer`

确认使用当前 `eval_model_vllm.py`。脚本会在导入 vLLM 前设置：

```text
VLLM_USE_FLASHINFER_SAMPLER=0
```

### 10.4 `cuda.h: No such file or directory`

这是 Triton/vLLM 环境缺少 CUDA 开发头文件，不是 Qwen3.5 权重错误。应重新检查
Conda 环境打包内容及目标服务器的 CUDA/编译工具链。

### 10.5 CUDA Out of Memory

训练时依次尝试：

1. 保持 `per_device_train_batch_size=1`。
2. 确认启用了默认 gradient checkpointing。
3. 降低 `max_seq_length`，但必须先处理超过该长度的数据，不能截断答案。
4. 使用显存更大的 GPU。

评测时可以适当降低：

```text
--gpu_memory 0.80
--max_model_len
--max_new_tokens
```

但必须保证 `max_model_len` 足以容纳最长 prompt 与 `max_new_tokens`。

### 10.6 输出目录已经存在

训练和评测默认拒绝覆盖已有结果。推荐使用新的版本目录，例如：

```text
outputs/qwen35_lora_full_v2
eval_results/qwen35_lora_v2_eval_full
```

只有确认不需要旧结果时才使用 `--overwrite_output_dir`。

---

## 11. 应归档的核心文件

训练结束且不再续训时建议归档：

```text
outputs/qwen35_lora_full_v1/
  adapter_model.safetensors
  adapter_config.json
  trainer_state.json
  train_results.json
  training_args.bin
  tokenizer.json
  tokenizer_config.json
  chat_template.jinja
  train_logs/

outputs/qwen35_lora_full_v1_vllm_prefixfix/
  adapter_model.safetensors
  adapter_config.json

eval_results/qwen35_lora_prefixfix_test_full/
  evaluation_manifest.json
  base_predictions.jsonl
  base_metrics.json
  lora_predictions.jsonl
  lora_metrics.json
  vllm_compare_summary.json
```

只有需要精确断点续训时才保留最终 `checkpoint-*`。原始 PEFT adapter 和转换后
的 vLLM adapter 用途不同，不能只保留其中一个。
