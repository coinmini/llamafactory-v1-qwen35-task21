# LlamaFactory v1 + Qwen3.5 + task21 工具调用 SFT 实施文档

> 目标：在 LlamaFactory **v1** 上跑通 Qwen3.5 系列模型的 LoRA SFT，训练数据是带工具调用的 task21 数据集。
>
> 范围：本文档只覆盖 v1 流程，不涉及 v0。

---

## 1. 背景：为什么走 v1

### 1.1 v0 的 odd/even 配对问题

`ROLE_BASED_LOSS_MASK.md` 里描述的问题：v0 用 **odd/even position** 决定哪条 message 是 prompt（mask）哪条是 response（计 loss），这要求严格交替（user→assistant→user→assistant…），数据中只要出现：

- 连续 `function_call`（并行工具调用）
- 连续 `observation`（并行工具返回）
- 任何打破交替的 agent 流程

**整条样本会被静默丢弃**。

确认 v0 当前代码仍然有这三处问题：

- [src/llamafactory/data/converter.py:144-146](src/llamafactory/data/converter.py#L144-L146)、[L173-174](src/llamafactory/data/converter.py#L173-L174) — odd/even tag 校验和消息总数奇偶校验
- [src/llamafactory/data/template.py:85](src/llamafactory/data/template.py#L85)、[L461](src/llamafactory/data/template.py#L461) — `encode_multiturn` 固定 stride-2 配对
- [src/llamafactory/data/processor/supervised.py:110](src/llamafactory/data/processor/supervised.py#L110)、[L153](src/llamafactory/data/processor/supervised.py#L153) — prompt 长度奇偶校验

### 1.2 v1 没有这个问题

v1 的设计是**逐条 message 标 `loss_weight`**（per-message，而非 per-pair）。

- **converter** ([src/llamafactory/v1/plugins/data_plugins/converter.py:115-152](src/llamafactory/v1/plugins/data_plugins/converter.py#L115-L152))：`gpt`/`function_call`→`loss_weight=1.0`，`human`/`system`/`observation`→`0.0`，没有奇偶校验
- **rendering** ([src/llamafactory/v1/core/utils/rendering.py:53-60](src/llamafactory/v1/core/utils/rendering.py#L53-L60))：每条消息按自己的 `loss_weight` 决定 labels/loss_weights
- **没有** v0 `supervised.py` 那种 prompt 长度校验

→ v1 天然支持连续 `function_call` / `observation`，不需要打 patch。

### 1.3 v1 缺的：qwen3.5 模板

v1 的 [templates/](src/llamafactory/v1/plugins/model_plugins/templates/) 目录原本只有 `qwen3.py` 和 `qwen3_nothink.py`。**Qwen3.5 的工具调用格式跟 Qwen3 完全不同**：

| 项 | Qwen3 | Qwen3.5 |
|---|---|---|
| tool_call 格式 | `<tool_call>{"name":..., "arguments":...}</tool_call>`（JSON） | `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>`（XML） |
| system 里的 tool 提示 | `QWEN_TOOL_PROMPT` | `QWEN35_TOOL_PROMPT`（带 `<IMPORTANT>` 提示块） |

直接用 `template: qwen3` 训 Qwen3.5 会让训出来的 tool_call 格式跟模型原生格式不一致。所以本次工作的核心是 **给 v1 补上 `qwen3_5` / `qwen3_5_nothink` 模板**。

---

## 2. 改动清单

### 2.1 新增 v1 qwen3.5 模板（两个文件）

#### `src/llamafactory/v1/plugins/model_plugins/templates/qwen3_5.py`

带 thinking 模式的 Qwen3.5 模板。关键点：

1. **`QWEN35_TOOL_PROMPT`**：从 v0 的 `src/llamafactory/data/tool_utils.py` 移植过来的工具说明 prompt（带 `<IMPORTANT>` 块）
2. **`_format_qwen35_tool_call(tool_call)`**：把 v1 内部的 `{"name": ..., "arguments": {...}}` JSON 渲染成 XML 风格
   ```
   <tool_call>
   <function=name>
   <parameter=key>
   value
   </parameter>
   </function>
   </tool_call>
   ```
3. **`render_qwen3_5_messages`**：复用 qwen3 模板的 thinking 模式逻辑（`<think>` 标签 + `_get_last_query_index`），但工具调用部分调用 `_format_qwen35_tool_call`
4. **`parse_qwen3_5_message`**：用正则反向解析模型输出的 XML 风格 tool_call，提取 `function` 名和 `parameter` 列表，转回 `{"name":..., "arguments":...}` 给下游

#### `src/llamafactory/v1/plugins/model_plugins/templates/qwen3_5_nothink.py`

不带 thinking 的 Qwen3.5 模板（对应 Instruct/Chat 系列）。复用 `qwen3_5.py` 里的 `QWEN35_TOOL_PROMPT` 和 `_format_qwen35_tool_call`，渲染逻辑跟 v1 已有的 `qwen3_nothink.py` 一致，差别只在工具调用渲染/解析。

#### 自动注册机制（无需改动）

v1 的 `RenderingPlugin` 用懒加载（[rendering.py:32](src/llamafactory/v1/plugins/model_plugins/rendering.py#L32)）：
```python
full_module_name = f"{__package__}.templates.{self.name}"
importlib.import_module(full_module_name)
```
所以**只要文件名是 `qwen3_5.py` / `qwen3_5_nothink.py`，会被自动 import**，不需要改 `__init__.py`。

### 2.2 数据转换脚本

#### `scripts/convert_task21_to_sharegpt.py`

task21 数据特点：

- ShareGPT 风格的 `conversations` + `tools` 字段
- **role 名是 OpenAI 风格的 `user / assistant / function_call / observation`**
- v1 的 sharegpt converter ([converter.py:115-121](src/llamafactory/v1/plugins/data_plugins/converter.py#L115-L121)) 期望 ShareGPT 原生 tag（写在 `from` 字段，值为 `human / gpt / function_call / observation / system`）

不转换会被全部 warning 跳过。脚本做的事：

1. `role` → `from`，`content` → `value`
2. role 重命名：`user→human`，`assistant→gpt`（其余保持）
3. `tools` 字段直接透传

输入输出：
- `task21_TEO.jsonl` (321 条) → `data/task21_train.jsonl`
- `task21_eval_final.json` (30 条，实际是 jsonl 格式) → `data/task21_eval.jsonl`

### 2.3 数据集 yaml

#### `data/task21_dataset.yaml`

```yaml
task21_train:
  path: data/task21_train.jsonl
  source: local
  converter: sharegpt
```

#### `data/task21_eval.yaml`

```yaml
task21_eval:
  path: data/task21_eval.jsonl
  source: local
  converter: sharegpt
```

> v1 的 dataset yaml 格式跟 v0 的 `dataset_info.json` 不同。每个 key 是 dataset 名字，下面三个核心字段：`path`（数据文件）、`source: local`（走本地 loader）、`converter: sharegpt`（用哪个 converter 插件）。

### 2.4 训练 yaml

#### `examples/v1/train_lora/train_lora_task21_qwen35.yaml`

```yaml
model: Qwen/Qwen3.5-4B-Instruct       # ← 改成你实际的 checkpoint
model_class: llm

template: qwen3_5_nothink              # 或 qwen3_5（thinking 模式）

peft_config:
  name: lora
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: all

kernel_config:
  name: auto
  include_kernels: auto

dist_config:                           # 单卡可注释掉
  name: fsdp2
  dcp_path: null

train_dataset: data/task21_dataset.yaml
eval_dataset: data/task21_eval.yaml

output_dir: ./outputs/task21_qwen35_lora
micro_batch_size: 1
cutoff_len: 4096                       # tools schema 较大，2048 可能不够
learning_rate: 1.0e-4
num_train_epochs: 3
warmup_ratio: 0.03
logging_steps: 5
save_steps: 100
save_total_limit: 3

sample_backend: hf
max_new_tokens: 512
```

---

## 3. 环境

### 3.1 Python 版本

LlamaFactory 仓库 `requires-python = ">=3.11.0"`，且 v1 代码用了 `NotRequired`（typing，3.11+）和 `StrEnum`（enum，3.11+）。

旧 conda 环境 `develop` 是 **Python 3.10**，无法直接用。

### 3.2 新建 llf_v1 环境

```bash
conda create -n llf_v1 python=3.12 -y
conda activate llf_v1

cd /Users/bolin/Documents/GitHub/linbo/llamafactory_official/LlamaFactory
pip install -e .
```

> `pyproject.toml` 没有定义 extras，所以不需要 `pip install -e ".[xxx]"`，单纯 `-e .` 即可。

### 3.3 装机过程踩到的坑

第一次 `pip install -e .` 在 setuptools 升级时被中断，导致 setuptools 处于半装状态（81.0.0 不完整）。修复办法：

```bash
pip install --force-reinstall setuptools
pip install -e .
```

---

## 4. 端到端验证

在 `llf_v1` 环境里跑了两个 smoke test：

### 4.1 模板注册

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

输出：
```
qwen3_5 render_qwen3_5_messages parse_qwen3_5_message
qwen3_5_nothink render_qwen3_5_nothink_messages parse_qwen3_5_nothink_message
OK
```

### 4.2 真实样本端到端 render

用 task21 训练集第一条样本（`user → function_call`，无 assistant 文本）走完整 pipeline：sharegpt converter → qwen3_5_nothink 渲染。

**结果验证**：

| 检查项 | 预期 | 实际 |
|---|---|---|
| user 消息 | `loss_weight=0.0`（mask） | ✅ `0.0` |
| function_call 消息（映射成 assistant） | `loss_weight=1.0`（计 loss） | ✅ `1.0` |
| system prompt | 包含 `QWEN35_TOOL_PROMPT` 的 `<IMPORTANT>` 块 | ✅ |
| tool_call 渲染格式 | XML（`<function=...>`/`<parameter=...>`） | ✅ |
| 总 token / 计 loss token | mask 比例合理 | 1102 总 / 28 计 loss / 1074 mask |

渲染出的 tool_call 片段：
```
<|im_start|>assistant
<tool_call>
<function=exec>
<parameter=command>
df -h
</parameter>
</function>
</tool_call><|im_end|>
```

跟 Qwen3.5 原生 chat template 一致。

---

## 5. 启动训练

```bash
conda activate llf_v1
cd /Users/bolin/Documents/GitHub/linbo/llamafactory_official/LlamaFactory

USE_V1=1 llamafactory-cli sft examples/v1/train_lora/train_lora_task21_qwen35.yaml
```

**`USE_V1=1` 是必需的**：[src/llamafactory/cli.py:19-22](src/llamafactory/cli.py#L19-L22) 通过这个环境变量切到 `v1.launcher`，否则会进 v0。

### 5.1 多 GPU

`launcher.py` 检测到 GPU 数 > 1 自动 `torchrun --nproc-per-node=<n>`（[src/llamafactory/v1/launcher.py:48-119](src/llamafactory/v1/launcher.py#L48-L119)）。强制开启：`FORCE_TORCHRUN=1`。

### 5.2 多机

```bash
USE_V1=1 NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
  llamafactory-cli sft examples/v1/train_lora/train_lora_task21_qwen35.yaml
```

每个节点设对应的 `NODE_RANK`。Elastic：再设 `RDZV_ID` / `MIN_NNODES` / `MAX_NNODES`。

### 5.3 本机 mps（仅 dry-run）

如果想在 Mac 上验证 pipeline 通畅，把 yaml 里：

- `model` 改成 `Qwen/Qwen2.5-0.5B-Instruct`（chatml，验证 pipeline 用）
- 注释掉 `dist_config:` 整块
- `template` 留 `qwen3_5_nothink`（虽然小模型不是真正的 qwen3.5，但模板渲染逻辑能跑通）

> 真正训练 4B+ 模型还是建议在 NVIDIA GPU 机器上做。

---

## 6. 文件清单

新增/修改的文件：

| 文件 | 类型 | 用途 |
|---|---|---|
| `src/llamafactory/v1/plugins/model_plugins/templates/qwen3_5.py` | 新增 | qwen3_5 thinking 模板 |
| `src/llamafactory/v1/plugins/model_plugins/templates/qwen3_5_nothink.py` | 新增 | qwen3_5_nothink 模板 |
| `scripts/convert_task21_to_sharegpt.py` | 新增 | 数据格式转换脚本 |
| `data/task21_train.jsonl` | 新增（脚本生成） | 训练集（321 条） |
| `data/task21_eval.jsonl` | 新增（脚本生成） | 验证集（30 条） |
| `data/task21_dataset.yaml` | 新增 | v1 训练集 dataset yaml |
| `data/task21_eval.yaml` | 新增 | v1 验证集 dataset yaml |
| `examples/v1/train_lora/train_lora_task21_qwen35.yaml` | 新增 | LoRA SFT 训练配置 |

**没有修改**仓库已有源码 — 所有 v1 改动都通过插件注册机制（templates 目录）和数据 yaml 注入，对原有代码零侵入。

---

## 7. 后续可选优化

- **跑完看 loss 曲线和 eval 指标**：v1 是否真的在训练循环里跑 eval（`eval_dataset` 字段虽然存在，但具体在 sft_trainer 里的执行逻辑没验证），需要跑完看 logging 输出确认
- **如果效果不好**：检查 `cutoff_len=4096` 是否够（tools schema 一行就上千 token），不够要调大或者裁剪 tools 字段
- **如果要训 thinking 模式**：把 yaml 里 `template: qwen3_5_nothink` 改成 `qwen3_5`，并确保数据里有 `<think>` reasoning 内容
- **如果要训 Qwen3.5-VL**：`model_class` 从 `llm` 改成 `vl`，并确认数据中视觉 content 的格式（task21 目前是纯文本，无视觉部分）
