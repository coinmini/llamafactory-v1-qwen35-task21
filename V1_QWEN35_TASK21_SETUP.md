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
bf16: true                             # H100/A100 sweet spot

sample_backend: hf
max_new_tokens: 512
```

> **历史变更（v1 字段兼容性踩坑）**：早先版本的 yaml 里有 `warmup_ratio / logging_steps / save_steps / save_total_limit`，但远端某些 v1 build 的 `TrainingArguments` 不认这些字段（HfArgumentParser 会抛 `Some keys are not used by the HfArgumentParser` 异常）。
>
> 当前 yaml 只保留**自 v1 第一版起就存在的最小字段集**（output_dir / micro_batch_size / cutoff_len / learning_rate / num_train_epochs / bf16），最大化跨版本可移植性。如果你跑的 v1 是新版（在 [src/llamafactory/v1/config/training_args.py](src/llamafactory/v1/config/training_args.py) 里能找到 `logging_steps` 等字段），可以加回来。
>
> warmup 在新版 v1 是通过 `lr_scheduler_config:` plugin 配置的，不是顶层字段。

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

完整的踩坑实录见 **[§8.4 远端环境踩坑实录](#84-远端环境踩坑实录重要)**，覆盖：

- §8.4.1 `pip install -e .` 中途被中断导致 setuptools 半装
- §8.4.2 已存在的 conda 环境可能装了"另一个" LlamaFactory 路径
- §8.4.3 yaml 字段在旧版 v1 不被认（`HfArgumentParser` 报错）
- §8.4.4 单卡必须注释 `dist_config`
- §8.4.5 `git pull` 因为本地 yaml 改动被拒
- §8.4.6 国内服务器 pip / HuggingFace 慢，配镜像
- §8.4.7 torch CUDA build 跟驱动版本不匹配（`NVIDIA driver too old`）
- §8.4.8 离线服务器 / HF 完全不通：用 modelscope 下载 + offline 模式
- §8.4.9 Qwen3.5 命名约定（**没有 `-Instruct` 后缀**，base id 本身就是 chat 版）

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
cd <repo-root>          # 本地为 /Users/bolin/Documents/GitHub/linbo/llamafactory_official/LlamaFactory
                        # 远端 clone 后为 ./llamafactory-v1-qwen35-task21

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

---

## 8. 远端部署：GitHub 仓库

### 8.1 仓库信息

- **URL**：https://github.com/coinmini/llamafactory-v1-qwen35-task21
- **可见性**：Public
- **默认分支**：`task21-v1-qwen35`（基于 `hiyouga/LlamaFactory` 上游 main + 本次 task21/v1 改动）
- **Remote 配置（本地）**：
  - `origin` → `https://github.com/coinmini/llamafactory-v1-qwen35-task21.git`（你的部署仓库）
  - `upstream` → `https://github.com/hiyouga/LlamaFactory.git`（hiyouga 上游，仅用于 fetch）

### 8.2 仓库内的 commit

| Commit | 说明 |
|---|---|
| `feat: add v1 qwen3.5 templates and task21 SFT pipeline` | qwen3_5/qwen3_5_nothink 模板、转换脚本、3 个 yaml 配置 |
| `docs: add task21 setup guide, v0 loss-mask analysis, and dataset` | 本文档第 1-7 节、`ROLE_BASED_LOSS_MASK.md`、原始数据 + 转换后数据 |
| `docs: add GitHub repo deployment section to setup guide` | 本文档第 8 节（远端部署、上游同步、新分支工作流） |
| `fix(yaml): drop newer-only training fields for older v1 compat` | 训练 yaml 移除 `warmup_ratio / logging_steps / save_steps / save_total_limit`，加 `bf16: true`，使其在旧版 v1 也能解析 |

### 8.3 远端机器一条龙部署

```bash
git clone https://github.com/coinmini/llamafactory-v1-qwen35-task21.git
cd llamafactory-v1-qwen35-task21

conda create -n llf_v1 python=3.12 -y
conda activate llf_v1
pip install -e .

# 按需修改 yaml 中的 model 字段（模型 ID / 本地路径）
# 单卡：注释掉 dist_config 整块；多卡：保留即可

USE_V1=1 llamafactory-cli sft examples/v1/train_lora/train_lora_task21_qwen35.yaml
```

数据已包含在仓库里（`data/task21_train.jsonl` / `data/task21_eval.jsonl`），无需额外传输。

### 8.4 远端环境踩坑实录（重要）

实际在远端 H100 机器上首次跑训练，按上面的步骤还会遇到几个隐形问题，这里记录一下避免下次重复踩坑：

#### 8.4.1 `pip install -e .` 在 setuptools 升级时被中断

现象：第一次 `pip install -e .` 中途 Ctrl-C 或网络断开，setuptools 处于半装状态（如 81.0.0 不完整），后续任何 pip 操作都会报奇怪的错。

修复：
```bash
pip install --force-reinstall setuptools
pip install -e .
```

#### 8.4.2 lf 等已存在的 conda 环境可能指向"另一个" LlamaFactory

如果机器上已经有一个名为 `lf`（或别的）的 conda 环境，**别想当然假设它装的就是你 clone 的那份代码**。这次实测：

```bash
# 远端 lf 环境查到的真实路径
$ python -c "import llamafactory, os; print(os.path.dirname(llamafactory.__file__))"
/LLaMA-Factory/src/llamafactory                 # ← 不是 ~/llamafactory-v1-qwen35-task21!
$ ls /LLaMA-Factory/src/llamafactory/v1/plugins/model_plugins/templates/
                                                 # ← 空目录，连 qwen3.py 都没有，是更老的 v1
```

也就是说：报错栈里的 `File "/LLaMA-Factory/..."` 跟你 `cd` 到的 clone 目录**完全不是同一份代码**，clone 仓库里加的 `qwen3_5.py` 模板根本没生效。

诊断指令：
```bash
python -c "import llamafactory, os; print(os.path.dirname(llamafactory.__file__))"
```
- 输出 `/<your-clone>/src/llamafactory` → ✅ 正确
- 输出 `/LLaMA-Factory/...` 或别的路径 → ❌ 装错了

修复（强制把 conda 环境切到 clone 仓库）：
```bash
pip uninstall llamafactory -y      # 跑两次更彻底
pip uninstall llamafactory -y
cd <clone-dir>
pip install -e .

# 复验
python -c "import llamafactory, os; print(os.path.dirname(llamafactory.__file__))"
```

#### 8.4.3 yaml 字段被旧版 v1 拒绝

现象：
```
ValueError: Some keys are not used by the HfArgumentParser:
['logging_steps', 'save_steps', 'save_total_limit', 'warmup_ratio']
```

原因和修复：见 [§2.4](#24-训练-yaml) 末尾的「历史变更」说明，本仓库 yaml 已经只保留兼容字段。

#### 8.4.4 单卡跑要注释 `dist_config`

`dist_config: fsdp2` 是为多卡设计的；单卡（如单张 H100）跑必须把这块注释掉，否则会启动 FSDP 失败或挂起。一行 sed 搞定：
```bash
sed -i '/^dist_config:/,/^  dcp_path: null/s/^/# /' \
  examples/v1/train_lora/train_lora_task21_qwen35.yaml
grep -n dist_config examples/v1/train_lora/train_lora_task21_qwen35.yaml   # 确认都被 # 开头
```

#### 8.4.5 `git pull` 因为本地 yaml 改动被拒

如果你按 8.4.4 用 `sed` 改了 yaml，下次 `git pull` 会报：
```
error: Your local changes to the following files would be overwritten by merge:
        examples/v1/train_lora/train_lora_task21_qwen35.yaml
Please commit your changes or stash them before you merge.
```

最干净的处理方式（这种本地改动不需要保留，注释 `dist_config` 这步反正会重做）：
```bash
git checkout -- examples/v1/train_lora/train_lora_task21_qwen35.yaml
git pull origin task21-v1-qwen35
# 然后重新跑 8.4.4 的 sed
```

如果本地有别的改动想保留：
```bash
git stash
git pull origin task21-v1-qwen35
git stash pop                       # 必要时手动解冲突
```

#### 8.4.6 国内服务器 pip / HuggingFace 下载慢

`pip install -e .` 走默认 PyPI 在国内可能只有几十 KB/s。配镜像后 1-2 分钟搞定：

```bash
# pip 永久镜像（清华 TUNA）
mkdir -p ~/.pip
cat > ~/.pip/pip.conf <<'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn

[install]
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF

# HuggingFace 镜像（下载 Qwen3.5 等模型用）
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc

# 验证
pip config list
echo "HF_ENDPOINT=$HF_ENDPOINT"
```

备选 pip 镜像（按速度依次）：
| 镜像 | URL |
|---|---|
| 清华 TUNA | `https://pypi.tuna.tsinghua.edu.cn/simple` |
| 阿里云 | `https://mirrors.aliyun.com/pypi/simple/` |
| 中科大 | `https://pypi.mirrors.ustc.edu.cn/simple/` |
| 腾讯云 | `https://mirrors.cloud.tencent.com/pypi/simple/` |

> 中途 Ctrl-C 重新跑 `pip install -e .` 不会从头下载，已下到 cache 的包会复用，所以临时切镜像不亏。

#### 8.4.7 torch CUDA build 跟驱动版本不匹配

现象（在驱动 = CUDA 12.8 的机器上跑训练）：
```
UserWarning: CUDA initialization: The NVIDIA driver on your system is too old
(found version 12080). Please update your GPU driver ...
```

**两个 CUDA 版本号别混淆**：
- `nvidia-smi` 顶部显示的 `CUDA Version: 12.8` —— 是**驱动支持的最高 CUDA 版本**
- `torch.version.cuda` —— 是 **torch wheel 编译时绑定的 CUDA 版本**（如 cu126、cu128）

报错的真正原因是 torch wheel 编译用的 CUDA toolkit 比驱动支持的某些 ABI 新。修法是**装匹配驱动 CUDA 版本的 torch wheel**：

```bash
# 先确认机器实际支持的 CUDA 上限
nvidia-smi | head -3

# 卸载现有 torch
pip uninstall -y torch torchvision torchaudio

# 装 CUDA 12.8 build（PyTorch 官方）
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

# 国内连不通官方 index 时换镜像
pip install torch torchvision torchaudio \
  --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu128

# 验证
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

实测案例：远端 H100 机器驱动支持 CUDA 12.8，原 lf 环境装的是 cu126 build → 改装 cu128 build 后 `torch.cuda.is_available()` 由 False 变 True，训练正常启动。

> 同理，如果驱动很老（比如只支持 CUDA 12.1），用 `--index-url https://download.pytorch.org/whl/cu121` 装 cu121 build。`torch==2.5.x` 系列对老驱动友好。

#### 8.4.8 离线服务器 / HuggingFace 完全不通：modelscope + offline

很多国内云服务器对 HuggingFace 是**完全不通**的（不仅是慢），连 `hf-mirror.com` 也访问不了。现象：

```
'[Errno 101] Network is unreachable' thrown while requesting HEAD https://huggingface.co/...
'timed out' thrown while requesting HEAD https://hf-mirror.com/...
```

解决方案：用 **modelscope** 把模型下到本地，yaml 直接用本地路径，并启用 transformers offline 模式跳过所有网络检查。

```bash
# 1) 装 modelscope（pyproject 已经声明了，通常已装）
pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2) 下载模型到本地（modelscope 国内速度极快，几十秒到几分钟）
python <<'PY'
from modelscope import snapshot_download
path = snapshot_download('Qwen/Qwen3.5-4B')   # 注意命名见 §8.4.9
print(f'\n>>> MODEL_PATH: {path}')
PY
# 输出形如：>>> MODEL_PATH: /root/.cache/modelscope/hub/models/Qwen/Qwen3___5-4B
# 注意 modelscope 把点号 `.` 转成三连下划线 `___`

# 3) 把 yaml 的 model 字段改成本地路径
MODEL_PATH=/root/.cache/modelscope/hub/models/Qwen/Qwen3___5-4B
sed -i "s|^model:.*|model: $MODEL_PATH|" examples/v1/train_lora/train_lora_task21_qwen35.yaml
grep "^model:" examples/v1/train_lora/train_lora_task21_qwen35.yaml

# 4) 关掉 transformers 的网络检查（不然它还会去 HEAD 一次 HF 看有没有更新）
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo 'export HF_HUB_OFFLINE=1' >> ~/.bashrc
echo 'export TRANSFORMERS_OFFLINE=1' >> ~/.bashrc

# 5) 跑训练
USE_V1=1 llamafactory-cli sft examples/v1/train_lora/train_lora_task21_qwen35.yaml
```

> 即使配了 `HF_ENDPOINT=https://hf-mirror.com`，transformers 仍然会发请求；只有 `HF_HUB_OFFLINE=1` 才能让它彻底走本地。

#### 8.4.9 Qwen3.5 模型命名约定

`Qwen/Qwen3.5-4B` 在 modelscope（以及 HuggingFace）上**本身就是带 chat / tool-calling 能力的版本**，不像 Qwen2.5 系列分 base / Instruct。验证方法：

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

预期输出：
```
Has chat_template: True
Has <tool_call>: True
Has <function=: True
```

→ 这台模型已经懂 qwen3_5 的 XML 工具调用格式，跟本仓库的 `qwen3_5_nothink` 模板对齐。

⚠️ 历史 yaml 里写的 `Qwen/Qwen3.5-4B-Instruct` 在 modelscope 上**不存在**（HuggingFace 上同样找不到），如果你还看到这个 id，记得改成 `Qwen/Qwen3.5-4B`。

> 顺便：Qwen3.5 默认 build 是 omni/VL（带视觉模块）。加载日志里会看到 `model.visual.pos_embed.weight` 这种 weight。**纯文本工具调用数据照样能训**，VL 部分自动跳过、不参与梯度。

### 8.5 后续与上游同步

如果想拉 hiyouga 上游的新 commit：

```bash
git fetch upstream
git rebase upstream/main           # 或 git merge upstream/main
git push origin task21-v1-qwen35
```

### 8.6 后续往这个仓库提交新工作

```bash
# 不要直推默认分支；新建 feature 分支 → push → 在 GitHub 上 merge
git checkout -b <feature-name>
# ... 改动 + commit ...
git push -u origin <feature-name>
```

---

## 9. 训练启动实测（H100 80GB，单卡）

### 9.1 启动 log

```text
[INFO] DistributedInterface initialized: ... is_distributed=False, current_device=cuda:0,
       rank=0, world_size=1
Loading weights: 100%|██| 723/723 [00:03<00:00, 213.10it/s,
                  Materializing param=model.visual.pos_embed.weight]
[INFO] llamafactory.v1.plugins.model_plugins.peft: Fine-tuning method: LoRA
[INFO] LoRA target modules:
       ['in_proj_a','gate_proj','in_proj_b','up_proj','linear_fc2','q_proj','k_proj',
        'o_proj','in_proj_qkv','out_proj','proj','linear_fc1','in_proj_z','v_proj',
        'down_proj','qkv']
trainable params: 39,034,880 || all params: 4,578,300,416 || trainable%: 0.8526
[INFO] Init unified data loader with global batch size 1, micro batch size 1,
       num micro batch 1, cutoff len 4096, batching workers 16, batching strategy normal.
[INFO] epoch: 0, step: 1, loss: 2.0204, grad_norm: 4.7957, lr: 0.0001, total_steps: 963
[INFO] epoch: 0, step: 2, loss: 0.5614, grad_norm: 7.2953, lr: 0.0001, total_steps: 963
[INFO] epoch: 0, step: 3, loss: 0.5755, grad_norm: 3.2949, lr: 0.0001, total_steps: 963
```

### 9.2 关键指标

| 指标 | 值 | 解读 |
|---|---|---|
| 模型加载 | 723 个 weight tensor / 3 秒 | 本地 modelscope cache 加速明显 |
| LoRA target modules | 16 个 | `target_modules: all` 自动展开 |
| Trainable params | 39M / 4.58B = **0.85%** | LoRA 典型比例 |
| `total_steps` | **963** = 321 样本 × 3 epoch / batch=1 | |
| Step 1 loss | 2.02 | 见到新数据格式的正常初始 loss |
| Step 2-3 loss | 0.56 / 0.58 | 立刻降下来 → 数据 + 模板对齐成功 |
| `grad_norm` | 4.8 / 7.3 / 3.3 | 没炸 |
| Step 间隔 | ~5 秒 | H100 + 4B + bf16 + 4096 length 正常 |
| 总耗时预估 | 963 × 5s ≈ **80 分钟** | |

### 9.3 后续可选改进

- 想看 eval loss：训练 yaml 加 `eval_dataset` 已经声明，但需要在新版 v1 才会真正在循环里跑 eval
- 想加 logging 频率 / save checkpoint：取决于你的 v1 build 是否支持 `logging_steps / save_steps`（见 §2.4 末尾说明）
- 想跑 thinking 模式：把 `template: qwen3_5_nothink` 改成 `qwen3_5`，前提是你的训练数据带 `<think>` reasoning 内容
- 想训更大模型（如 Qwen3.5-32B）：单张 H100 80GB 可能装不下 LoRA + bf16，需要打开 `dist_config: fsdp2` + 多卡
