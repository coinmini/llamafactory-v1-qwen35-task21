# LlamaFactory v1 + Qwen3.5 工具调用 SFT 操作手册

> 用 LlamaFactory **v1** 训练 Qwen3.5 系列模型的 LoRA，数据是带工具调用的 ShareGPT 风格 jsonl。
> 在远端 H100 80GB 单卡上从 0 到 1 跑通过的完整流程。

**预计耗时**：环境 30 分钟 + 训练 80 分钟（4B 模型 / 321 样本 / 3 epoch）。

**目录**：

- [Part A — 准备：一次性的环境搭建（10 步）](#part-a--准备一次性的环境搭建)
- [Part B — 训练：跑你自己的工具调用数据（5 步）](#part-b--训练跑你自己的工具调用数据)
- [Part C — 验证 & 后处理（3 步）](#part-c--验证--后处理)
- [附录 1 — 背景：为什么走 v1 而不是 v0](#附录-1--背景为什么走-v1-而不是-v0)
- [附录 2 — 改动清单（这次往仓库里加了哪些文件）](#附录-2--改动清单这次往仓库里加了哪些文件)
- [附录 3 — 常见报错速查表](#附录-3--常见报错速查表)

---

## Part A — 准备：一次性的环境搭建

> 整个 Part A 只需要在一台机器上做一次。

### Step A1 — clone 仓库

```bash
git clone https://github.com/coinmini/llamafactory-v1-qwen35-task21.git
cd llamafactory-v1-qwen35-task21
```

后续所有命令都假定你在仓库根目录里。

### Step A2 — 配置 pip 国内镜像（强烈建议）

国内服务器走默认 PyPI 可能慢到 50 KB/s，配镜像后速度差几十倍：

```bash
mkdir -p ~/.pip
cat > ~/.pip/pip.conf <<'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn

[install]
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF
pip config list      # 验证
```

清华慢的话备选：阿里 `https://mirrors.aliyun.com/pypi/simple/`、中科大 `https://pypi.mirrors.ustc.edu.cn/simple/`、腾讯云 `https://mirrors.cloud.tencent.com/pypi/simple/`。

### Step A3 — 创建 Python 3.12 conda 环境

LlamaFactory 仓库 `requires-python = ">=3.11.0"`，且 v1 用了 `NotRequired`/`StrEnum`（3.11+ 特性），所以 **3.10 不行**。

```bash
conda create -n llf_v1 python=3.12 -y
conda activate llf_v1
```

> 如果机器上已有 conda 环境（比如 `lf`），**别想当然假设它就够用**。先验证（见 Step A6 末尾的诊断指令）。本机就有过 `lf` 环境装的 LlamaFactory 是 `/LLaMA-Factory/...` 旧路径，clone 仓库改的代码根本没生效的事故。

### Step A4 — 装 LlamaFactory（可编辑模式）

```bash
pip install -e .
```

5–10 分钟。`pyproject.toml` 没有定义 extras，单纯 `-e .` 即可。

> **如果中途 Ctrl-C 或网络断了**：setuptools 可能处于半装状态，下次跑会报莫名其妙的错。修复：
> ```bash
> pip install --force-reinstall setuptools
> pip install -e .
> ```

### Step A5 — 检查 / 修复 torch CUDA 版本

如果 `import torch` 后 `torch.cuda.is_available()` 是 False，或运行训练时报 `NVIDIA driver too old` —— 是 torch wheel 跟 GPU 驱动不匹配。

```bash
# 1) 看 GPU 驱动支持的最高 CUDA 版本（顶部那个 "CUDA Version: x.y"）
nvidia-smi | head -3

# 2) 看 torch 当前编译版本
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

**两个 CUDA 版本号别混淆**：

- `nvidia-smi` 顶部的 `CUDA Version: 12.8` —— 驱动支持的最高 CUDA 版本
- `torch.version.cuda` —— torch wheel 编译时绑定的 CUDA 版本

**如果 torch 比驱动支持的更新**（比如这次实测：H100 驱动只到 cu128，但默认装的是 cu126 wheel），重装匹配的：

```bash
pip uninstall -y torch torchvision torchaudio

# 装 cu128 build（H100 实测）
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

# 国内连不上官方 index 时换镜像
pip install torch torchvision torchaudio \
  --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu128

# 复验
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# 期望：True
```

驱动只支持 cu121 或更老的话用 `--index-url https://download.pytorch.org/whl/cu121`。

### Step A6 — 验证 Python 路径指向 clone 的代码

**这一步很关键** — 防止 conda 环境装错了 LlamaFactory：

```bash
python -c "import llamafactory, os; print(os.path.dirname(llamafactory.__file__))"
```

期望输出：`/<your-clone-path>/src/llamafactory`

如果输出是 `/LLaMA-Factory/...` 或别的路径 —— 说明环境装的是另一份代码，clone 仓库里加的 qwen3_5 模板**不会生效**。修复：

```bash
pip uninstall llamafactory -y      # 跑两次更彻底
pip uninstall llamafactory -y
cd /path/to/llamafactory-v1-qwen35-task21
pip install -e .
# 再验证一次
python -c "import llamafactory, os; print(os.path.dirname(llamafactory.__file__))"
```

### Step A7 — 验证 qwen3_5 模板能注册

```bash
python -c "
import sys; sys.path.insert(0,'src')
from llamafactory.v1.plugins.model_plugins.rendering import RenderingPlugin
for n in ('qwen3_5','qwen3_5_nothink'):
    p = RenderingPlugin(n); p._ensure_template_imported()
    print(n, p['render_messages'].__name__, p['parse_message'].__name__)
print('OK')
"
```

期望输出：

```
qwen3_5 render_qwen3_5_messages parse_qwen3_5_message
qwen3_5_nothink render_qwen3_5_nothink_messages parse_qwen3_5_nothink_message
OK
```

### Step A8 — 用 modelscope 下载 Qwen3.5-4B 模型

国内 H100 服务器对 HuggingFace **完全不通**（`hf-mirror.com` 也常常不通），**用 modelscope** 是最稳的：

```bash
pip install modelscope        # pyproject 已经声明了，多半已装

python <<'PY'
from modelscope import snapshot_download
path = snapshot_download('Qwen/Qwen3.5-4B')   # ← 注意命名见下方说明
print(f'\n>>> MODEL_PATH: {path}')
PY
```

输出会是：`>>> MODEL_PATH: /root/.cache/modelscope/hub/models/Qwen/Qwen3___5-4B`

> ⚠️ **modelscope 把路径里的 `.` 转成 `___`**（三个下划线）—— 所以 `Qwen3.5-4B` 在文件系统里变成 `Qwen3___5-4B`。后面写 yaml 路径时记得用这个变形过的路径。
>
> ⚠️ **Qwen3.5 没有 `-Instruct` 后缀** — `Qwen/Qwen3.5-4B` 本身就是带 chat / 工具调用能力的版本，不像 Qwen2.5 系列分 base / Instruct。早期 yaml 里写的 `Qwen/Qwen3.5-4B-Instruct` 在 modelscope / HF 上**都不存在**，记得用 `Qwen/Qwen3.5-4B`。

### Step A9 — 验证下载的模型带工具调用 chat template

```bash
MODEL_PATH=/root/.cache/modelscope/hub/models/Qwen/Qwen3___5-4B

python -c "
import json
with open('$MODEL_PATH/tokenizer_config.json') as f:
    cfg = json.load(f)
tmpl = cfg.get('chat_template', '')
print('Has chat_template:', bool(tmpl))
print('Has <tool_call>:', '<tool_call>' in tmpl)
print('Has <function=:', '<function=' in tmpl)
"
```

期望全部 `True`。这证明模型原生支持 qwen3.5 的 XML 风格工具调用，跟仓库里 `qwen3_5_nothink` 模板对齐。

### Step A10 — 配置 transformers offline 模式

不然 transformers 每次加载模型都会去 HF HEAD 一次（即使本地有 cache），离线服务器会卡 `timeout`：

```bash
echo 'export HF_HUB_OFFLINE=1' >> ~/.bashrc
echo 'export TRANSFORMERS_OFFLINE=1' >> ~/.bashrc
source ~/.bashrc
echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE  TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"
```

> 即使配 `HF_ENDPOINT=https://hf-mirror.com` 也不够 —— `HF_HUB_OFFLINE=1` 才是彻底跳过网络检查的开关。

✅ **Part A 完成**。后面所有操作都基于这个环境。

---

## Part B — 训练：跑你自己的工具调用数据

> 这一部分如果只是想复现仓库里 task21 那次训练，**Step B1 / B2 都可以跳过**，仓库里 `data/task21_train.jsonl` 和训练 yaml 已经准备好。
>
> 想换自己的数据再做 Step B1 / B2。

### Step B1 — 转换数据格式（如果用自己的数据）

LlamaFactory v1 sharegpt converter 期望的格式：每行 jsonl，字段如下：

```json
{
  "conversations": [
    {"from": "human",         "value": "用户输入文本"},
    {"from": "function_call", "value": "{\"name\": ..., \"arguments\": {...}}"},
    {"from": "observation",   "value": "工具返回值文本"},
    {"from": "gpt",           "value": "助手回答文本"}
  ],
  "tools": "[{\"name\": \"exec\", \"description\": ..., \"parameters\": {...}}, ...]"
}
```

**关键 tag 规则**：

| tag (`from` 字段值) | 角色 | loss_weight |
|---|---|---|
| `system` | 系统提示 | 0.0（mask） |
| `human` | 用户 | 0.0（mask） |
| `observation` | 工具返回 | 0.0（mask） |
| `gpt` | 助手文本 | **1.0**（计 loss） |
| `function_call` | 助手的工具调用 | **1.0**（计 loss） |

仓库里有现成的转换脚本 [scripts/convert_task21_to_sharegpt.py](scripts/convert_task21_to_sharegpt.py)，把 OpenAI 风格 (`role/content`) 的 jsonl 转成 ShareGPT 风格 (`from/value`)：

```python
ROLE_MAP = {
    "user": "human",
    "assistant": "gpt",
    "function_call": "function_call",
    "observation": "observation",
    "system": "system",
}
```

把 `task21_TEO.jsonl` / `task21_eval_final.json` 替换成你自己的输入路径，然后跑：

```bash
python scripts/convert_task21_to_sharegpt.py
```

### Step B2 — 写 dataset yaml（如果用自己的数据）

LlamaFactory v1 的 dataset yaml 格式跟 v0 的 `dataset_info.json` 完全不同。仓库里两个例子可以直接抄：

[data/task21_dataset.yaml](data/task21_dataset.yaml)：

```yaml
task21_train:
  path: data/task21_train.jsonl
  source: local
  converter: sharegpt
```

[data/task21_eval.yaml](data/task21_eval.yaml)：

```yaml
task21_eval:
  path: data/task21_eval.jsonl
  source: local
  converter: sharegpt
```

每个 key 是一个 dataset 的名字，下面三个核心字段：`path` 数据文件路径、`source: local` 本地 loader、`converter: sharegpt` 用 sharegpt converter 插件。

### Step B3 — 改训练 yaml 的模型路径

打开 [examples/v1/train_lora/train_lora_task21_qwen35.yaml](examples/v1/train_lora/train_lora_task21_qwen35.yaml)，把 `model:` 字段改成 Step A8 下载的路径：

```bash
MODEL_PATH=/root/.cache/modelscope/hub/models/Qwen/Qwen3___5-4B
sed -i "s|^model:.*|model: $MODEL_PATH|" examples/v1/train_lora/train_lora_task21_qwen35.yaml
grep "^model:" examples/v1/train_lora/train_lora_task21_qwen35.yaml
```

完整 yaml 应该长这样：

```yaml
model: /root/.cache/modelscope/hub/models/Qwen/Qwen3___5-4B
model_class: llm

template: qwen3_5_nothink           # 或 qwen3_5（思考模式，需要数据带 <think> 内容）

peft_config:
  name: lora
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: all

kernel_config:
  name: auto
  include_kernels: auto

dist_config:                        # 多卡才用，单卡见 Step B4
  name: fsdp2
  dcp_path: null

train_dataset: data/task21_dataset.yaml
eval_dataset: data/task21_eval.yaml

output_dir: ./outputs/task21_qwen35_lora
micro_batch_size: 1
cutoff_len: 4096                    # tools schema 较大，2048 可能不够
learning_rate: 1.0e-4
num_train_epochs: 3
bf16: true                          # H100/A100 sweet spot

sample_backend: hf
max_new_tokens: 512
```

> ⚠️ **不要加** `warmup_ratio / logging_steps / save_steps / save_total_limit` —— 这些在旧版 v1 的 `TrainingArguments` 里不存在，会让 `HfArgumentParser` 抛 `Some keys are not used` 异常。warmup 在新版 v1 是通过 `lr_scheduler_config:` plugin 配置的，不是顶层字段。

### Step B4 — 单卡训练时注释掉 dist_config

`dist_config: fsdp2` 是为多卡设计的；单卡（如 1 张 H100）必须把这块注释掉，否则启动 FSDP 失败：

```bash
sed -i '/^dist_config:/,/^  dcp_path: null/s/^/# /' \
  examples/v1/train_lora/train_lora_task21_qwen35.yaml
grep -n dist_config examples/v1/train_lora/train_lora_task21_qwen35.yaml
# 期望：3 行都被 # 开头
```

> 多卡：保持 `dist_config:` 不动即可，[v1/launcher.py](src/llamafactory/v1/launcher.py) 会自动 `torchrun --nproc-per-node=<gpu_count>`。

### Step B5 — 启动训练

```bash
USE_V1=1 llamafactory-cli sft examples/v1/train_lora/train_lora_task21_qwen35.yaml
```

> **`USE_V1=1` 是必需的** — [src/llamafactory/cli.py:19-22](src/llamafactory/cli.py#L19-L22) 通过这个环境变量切到 `v1.launcher`，否则会进 v0。

期望开头看到这样的 log：

```text
[INFO] DistributedInterface initialized: ... is_distributed=False, current_device=cuda:0,
       rank=0, world_size=1
Loading weights: 100%|██| 723/723 [00:03<00:00, 213.10it/s]
[INFO] llamafactory.v1.plugins.model_plugins.peft: Fine-tuning method: LoRA
[INFO] LoRA target modules: ['in_proj_a','gate_proj','q_proj','k_proj','v_proj',
       'o_proj','linear_fc1','linear_fc2','up_proj','down_proj','qkv', ...]
trainable params: 39,034,880 || all params: 4,578,300,416 || trainable%: 0.8526
[INFO] Init unified data loader with global batch size 1, micro batch size 1,
       cutoff len 4096, batching strategy normal.
[INFO] epoch: 0, step: 1, loss: 2.0204, grad_norm: 4.7957, lr: 0.0001, total_steps: 963
[INFO] epoch: 0, step: 2, loss: 0.5614, grad_norm: 7.2953, lr: 0.0001, total_steps: 963
[INFO] epoch: 0, step: 3, loss: 0.5755, grad_norm: 3.2949, lr: 0.0001, total_steps: 963
```

**训练健康度检查**（参考 H100 + 4B + LoRA 的实测值）：

| 指标 | 期望值 | 备注 |
|---|---|---|
| Step 1 loss | 1.5 ~ 3.0 | 看到新 tool 格式的初始 loss |
| Step 2-10 loss | 应该明显下降到 < 1.0 | 不降说明数据 / 模板对不上 |
| `grad_norm` | < 10 | > 100 是要炸 |
| Step 间隔 | H100 上 ~5 秒 | 比这慢得多看 GPU 利用率 |
| `total_steps` | 应等于 数据条数 × epoch / batch_size | task21 = 321 × 3 = 963 |
| `trainable%` | LoRA 0.5%-2% | r=16 时大概 0.85% |

监控显存：另开一个 terminal 跑 `nvidia-smi`，4B + LoRA + bf16 + cutoff_len=4096 单卡大概 30-40GB。

✅ **训练完成后**，输出在 `./outputs/task21_qwen35_lora/`，包含：

- `adapter_model.safetensors` — LoRA 权重
- `adapter_config.json` — LoRA 配置
- `tokenizer.json` / `tokenizer_config.json` / `chat_template.jinja` — tokenizer 一套
- `trainer_log.jsonl` — 训练 log（每 step 一行 JSON）

---

## Part C — 验证 & 后处理

### Step C1 — 画训练曲线图

v1 默认**不输出** loss 曲线图（不像 v0 的 `training_loss.png`），但 `trainer_log.jsonl` 里有所有数据。下面脚本生成 `training_curves.png`：

```bash
python <<'PY'
import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOG = './outputs/task21_qwen35_lora/trainer_log.jsonl'
OUT = './outputs/task21_qwen35_lora/training_curves.png'

records = [json.loads(l) for l in open(LOG) if l.strip()]
steps = [r['step'] for r in records]
loss  = [r['loss'] for r in records]
gnorm = [r['grad_norm'] for r in records]
lr    = [r['learning_rate'] for r in records]

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
axes[0].plot(steps, loss, color='tab:red');    axes[0].set_ylabel('loss');  axes[0].grid(True, alpha=0.3)
axes[1].plot(steps, gnorm, color='tab:blue');  axes[1].set_ylabel('grad_norm'); axes[1].grid(True, alpha=0.3)
axes[2].plot(steps, lr,    color='tab:green'); axes[2].set_ylabel('learning_rate'); axes[2].set_xlabel('step'); axes[2].grid(True, alpha=0.3)
fig.suptitle(f'task21 qwen3.5-4B LoRA SFT  ({len(records)} log entries)')
fig.tight_layout()
fig.savefig(OUT, dpi=120)
print('Saved:', OUT)
PY
```

### Step C2 — 快速本地验证 LoRA 能加载并生成

不要直接跑大型评测脚本 —— 先用一个最小的 python 脚本验证 LoRA 推理 OK：

```bash
python <<'PY'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "/root/.cache/modelscope/hub/models/Qwen/Qwen3___5-4B"
LORA = "./outputs/task21_qwen35_lora"

print("Loading base...")
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base, LORA)
model.eval()

prompt = "Run 'df -h' and explain the disk usage"
inputs = tok.apply_chat_template(
    [{"role": "user", "content": prompt}],
    add_generation_prompt=True, return_tensors="pt"
).to("cuda")

print("Generating...")
out = model.generate(inputs, max_new_tokens=200, do_sample=False)
print("=" * 60)
print(tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=False))
PY
```

期望输出包含 XML 风格的 tool call：

```text
<tool_call>
<function=exec>
<parameter=command>
df -h
</parameter>
</function>
</tool_call><|im_end|>
```

→ 看到这个就说明 LoRA 训练成功 + 模板对齐 + 推理工作。

### Step C3 — 跑你的评测脚本

到这一步 LoRA 已经验证可用，套你自己的评测框架。需要 4 个路径作为输入：

| 参数 | 值 |
|---|---|
| LoRA adapter | `./outputs/task21_qwen35_lora`（含 `adapter_model.safetensors`） |
| base 模型 | `/root/.cache/modelscope/hub/models/Qwen/Qwen3___5-4B` |
| 评测数据 | `data/task21_eval.jsonl`（ShareGPT 风格）或自己的 OpenAI 风格数据 |
| `--is-lora` 标志 | 别忘加 |

> 如果评测脚本是 `uv run` 启动的，但你的环境是 conda —— 要把 `uv run` 换成普通 `python`，并先 `conda activate llf_v1`，否则 uv 会启另一个临时 venv 找不到依赖。
>
> 评测脚本若需要调用外部 LLM API（`--eval-with-llm`），先用 `curl -m 5 -sI https://api.deepseek.com` 之类的命令确认这台机器能调到。离线服务器经常调不通，这种情况只能去掉 `--eval-with-llm` 跑纯本地推理。

---

## 附录 1 — 背景：为什么走 v1 而不是 v0

### v0 的 odd/even 配对问题

`ROLE_BASED_LOSS_MASK.md` 里描述的问题：v0 用 **odd/even position** 决定哪条 message 是 prompt（mask）哪条是 response（计 loss），要求严格交替（user→assistant→user→assistant…）。数据中只要出现：

- 连续 `function_call`（并行工具调用）
- 连续 `observation`（并行工具返回）
- 任何打破交替的 agent 流程

**整条样本会被静默丢弃**。

v0 当前代码这三处问题都在：

- [src/llamafactory/data/converter.py:144-146](src/llamafactory/data/converter.py#L144-L146)、[L173-174](src/llamafactory/data/converter.py#L173-L174) — odd/even tag 校验和消息总数奇偶校验
- [src/llamafactory/data/template.py:85](src/llamafactory/data/template.py#L85)、[L461](src/llamafactory/data/template.py#L461) — `encode_multiturn` 固定 stride-2 配对
- [src/llamafactory/data/processor/supervised.py:110](src/llamafactory/data/processor/supervised.py#L110)、[L153](src/llamafactory/data/processor/supervised.py#L153) — prompt 长度奇偶校验

### v1 的设计

**逐条 message 标 `loss_weight`**（per-message，而非 per-pair）：

- **converter** ([v1/plugins/data_plugins/converter.py:115-152](src/llamafactory/v1/plugins/data_plugins/converter.py#L115-L152))：`gpt`/`function_call`→`1.0`，`human`/`system`/`observation`→`0.0`，没有奇偶校验
- **rendering** ([v1/core/utils/rendering.py:53-60](src/llamafactory/v1/core/utils/rendering.py#L53-L60))：每条消息按自己的 `loss_weight` 决定 labels/loss_weights
- **没有** v0 `supervised.py` 那种 prompt 长度校验

→ v1 天然支持连续 `function_call` / `observation`，**不需要打 patch**。

### v1 缺的：qwen3.5 模板（已补上）

v1 [templates/](src/llamafactory/v1/plugins/model_plugins/templates/) 原本只有 `qwen3.py` 和 `qwen3_nothink.py`。Qwen3.5 的工具调用格式跟 Qwen3 完全不同：

| 项 | Qwen3 | Qwen3.5 |
|---|---|---|
| tool_call 格式 | `<tool_call>{"name":..., "arguments":...}</tool_call>`（JSON） | `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>`（XML） |
| system 工具提示 | `QWEN_TOOL_PROMPT` | `QWEN35_TOOL_PROMPT`（带 `<IMPORTANT>` 块） |

**本次工作的核心**：给 v1 补上 `qwen3_5` / `qwen3_5_nothink` 模板，**对仓库已有源码零侵入**（用插件注册机制和数据 yaml 注入）。

---

## 附录 2 — 改动清单（这次往仓库里加了哪些文件）

| 文件 | 类型 | 用途 |
|---|---|---|
| `src/llamafactory/v1/plugins/model_plugins/templates/qwen3_5.py` | 新增 | qwen3_5 thinking 模板（XML 风格 tool call） |
| `src/llamafactory/v1/plugins/model_plugins/templates/qwen3_5_nothink.py` | 新增 | qwen3_5_nothink 模板（无思考） |
| `scripts/convert_task21_to_sharegpt.py` | 新增 | OpenAI 风格 → ShareGPT 风格转换脚本 |
| `data/task21_train.jsonl` | 新增（脚本生成） | 训练集（321 条） |
| `data/task21_eval.jsonl` | 新增（脚本生成） | 验证集（30 条） |
| `data/task21_dataset.yaml` | 新增 | v1 训练集 dataset yaml |
| `data/task21_eval.yaml` | 新增 | v1 验证集 dataset yaml |
| `examples/v1/train_lora/train_lora_task21_qwen35.yaml` | 新增 | LoRA SFT 训练配置 |

### v1 模板自动注册机制

不需要改 `__init__.py` —— [v1 RenderingPlugin](src/llamafactory/v1/plugins/model_plugins/rendering.py#L32) 用懒加载：

```python
full_module_name = f"{__package__}.templates.{self.name}"
importlib.import_module(full_module_name)
```

只要文件名是 `qwen3_5.py` / `qwen3_5_nothink.py`，会被自动 import。

### Qwen3.5 模板要点

`qwen3_5.py` 关键函数：

1. **`QWEN35_TOOL_PROMPT`**：从 v0 的 `tool_utils.py` 移植过来的工具说明 prompt（带 `<IMPORTANT>` 块）
2. **`_format_qwen35_tool_call(tool_call)`**：把 `{"name": ..., "arguments": {...}}` JSON 渲染成 XML 风格
3. **`render_qwen3_5_messages`**：复用 qwen3 模板的 thinking 模式逻辑（`<think>` + `_get_last_query_index`），但工具调用部分调用 `_format_qwen35_tool_call`
4. **`parse_qwen3_5_message`**：用正则反向解析 XML 风格 tool_call，转回 `{"name":..., "arguments":...}` 给下游

`qwen3_5_nothink.py` 复用 `qwen3_5.py` 的 `QWEN35_TOOL_PROMPT` 和 `_format_qwen35_tool_call`，渲染逻辑跟 v1 已有的 `qwen3_nothink.py` 一致。

---

## 附录 3 — 常见报错速查表

| 报错 | 在哪一步 | 解法 |
|---|---|---|
| `ModuleNotFoundError: No module named 'X'` | A4 | `pip install X`，依赖装漏了 |
| `pip install` 卡住 / 50 KB/s | A4 | 见 Step A2 配镜像 |
| setuptools 半装报错 | A4 | `pip install --force-reinstall setuptools` 后重试 |
| `NotRequired` / `StrEnum` import 失败 | A6/A7 | Python 版本 < 3.11，重建 3.12 环境 |
| `NVIDIA driver too old (found version 12080)` | B5 | 见 Step A5 装匹配的 torch wheel |
| `torch.cuda.is_available() == False` | A5 | 见 Step A5 |
| `import llamafactory` 路径不对 | A6 | 见 Step A6 修复 |
| `qwen3_5 模板 not registered` | A7 | 模板文件没 import；查路径是不是 clone 的代码 |
| `'[Errno 101] Network is unreachable'` 访问 HF | A8/B5 | 见 Step A8 走 modelscope + Step A10 offline |
| `Some keys are not used by the HfArgumentParser` | B5 | yaml 里别加 `warmup_ratio / logging_steps / save_steps / save_total_limit`，见 Step B3 |
| `Qwen/Qwen3.5-4B-Instruct does not exist` | A8 | 改用 `Qwen/Qwen3.5-4B`（没有 -Instruct 后缀），见 Step A8 |
| FSDP 启动失败 / 挂起 | B5 | 单卡要注释 `dist_config`，见 Step B4 |
| Step 1 loss 正常但后面不降 | B5 | 数据 / 模板对不上；用 Step C2 单条样本验证 |
| `git pull` 拒绝 merge（有本地改动） | 多次 sed 后 | `git checkout -- <file> && git pull` 抛弃本地，重新跑 sed |

---

## 附录 4 — 后续工作建议

- 想看 eval loss 曲线：训练 yaml 已经声明 `eval_dataset`，但需要新版 v1 才会真在循环里跑 eval — 先确认你的 v1 build 支不支持
- 想加 logging 频率 / save checkpoint：[src/llamafactory/v1/config/training_args.py](src/llamafactory/v1/config/training_args.py) 里能找到 `logging_steps / save_steps` 字段就支持
- 想跑 thinking 模式：把 `template: qwen3_5_nothink` 改成 `qwen3_5`，前提是数据里有 `<think>` reasoning 内容
- 想训更大模型（如 Qwen3.5-32B）：单张 H100 80GB 可能装不下，需要 `dist_config: fsdp2` + 多卡

---

## 仓库

- **URL**：https://github.com/coinmini/llamafactory-v1-qwen35-task21
- **可见性**：Public
- **默认分支**：`task21-v1-qwen35`（基于 hiyouga/LlamaFactory 上游 main + 本次改动）

### 与上游同步

```bash
git fetch upstream
git rebase upstream/main           # 或 git merge upstream/main
git push origin task21-v1-qwen35
```

### 后续提交新工作

```bash
git checkout -b <feature-name>
# ... 改动 + commit ...
git push -u origin <feature-name>
```
