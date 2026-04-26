# 构建基于 Qwen3.5-4B 的 OpenClaw 工具调用大模型微调实战

本文深入解析了一个面向 agent 工具调用领域的 LLM 微调实战案例 —— 利用 Qwen3.5-4B 模型为 OpenClaw（一个 agent 工具调用框架）构建专属的工具调用能力。文章全景式复盘了从数据准备、框架选型到落地评测的完整流程，重点攻克了 v0 框架对 agent 数据的位置交替限制、qwen3.5 XML 风格 tool call 模板缺失、离线服务器无法访问 HuggingFace 等真实工程难题，并通过 LlamaFactory v1 + LoRA 高效完成微调，最终用 flock_validator 三模型 LLM judge 评测得到 **Normalized score 0.7566 / 1.0** 的成绩。无论你是想了解如何为 v1 自定义模板，还是想看 agent 风格数据集应该怎么处理，相信都能从本文找到可直接复用的工程经验。

模型微调前后，对话效果对比如下所示：

> 微调前的 Qwen3.5-4B 虽然原生支持 XML 风格的 `<tool_call><function=...>` 格式，但在面对 OpenClaw 的具体 tools schema（cron / exec / read / write / browser 等）时，**对工具的选择和参数填写都不够准确**。微调后的模型在评测中（见后文「模型评估」一节）展示了对 OpenClaw 工具集合的稳定调用能力，能正确识别工具名、填写必需 parameter，并按 OpenClaw 的实际行为模式作多步推理。

微调后的模型对 OpenClaw 工具调用片段示例（来自评测过程）：

![微调后模型对话示例](pic/验证过程2.png)

## 前提条件

- 已注册 LLaMA-Factory Online 平台账号，余额充足，能够在 [实例空间] 启动 H800 / H100 80GB 实例。
- （可选）有自己的 GPU 机器：单卡 80GB 显存可跑 4B + LoRA + cutoff_len=4096。
- 项目代码与数据已托管：<https://github.com/coinmini/llamafactory-v1-qwen35-task21>。

## 操作步骤

### 配置概览

| 配置项 | 值 | 是否预置 | 说明 |
|---|---|---|---|
| 模型 | Qwen3.5-4B | 否，需 modelscope 下载 | 原生支持 XML 风格 tool_call，自带 chat_template |
| 数据集 | task21（OpenClaw distill） | 否 | OpenClaw agent 工具调用合成数据 |
| 训练 / 验证规模 | 321 / 30 条 | - | ShareGPT 风格 jsonl |
| GPU | H800 / H100 80GB × 1（推荐） | 是 | LLaMA-Factory Online 实例空间预置 |
| 框架 | LlamaFactory v1（不是 v0） | - | 见下文「数据集制作」一节解释为什么走 v1 |
| 微调方法 | LoRA r=16 / alpha=32 / dropout=0.05 | - | target_modules: all |
| 训练时长 | 约 80 分钟（H800 单卡，3 epoch） | - | total_steps = 963 |

### 资源消耗预览

- **模型微调时长**：H800 单卡约 1h 20min，平均 ~5 秒 / step。
- **微调后模型 Validate 时长**：约 7 分钟（30 条样本 × 3 个 LLM judge × 3 次）。
- **数据准备时长**：< 1 分钟（脚本一次性转换格式）。

## 数据集制作

### 确认和理解项目的要求

OpenClaw 是一个本地化的 agent 工具调用框架，模型需要学会用 XML 风格的 `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>` 调用平台提供的工具集（cron 调度任务、exec 执行 shell、read/write/edit 文件、browser 控制浏览器、web_search 联网搜索、sessions_spawn 派生子 agent 等）。项目要求归纳如下：

- **Context Length**：8,192
- **Max number of parameters**：6,000,000,000
- 选型：参数量上限 6B，结合 Qwen 系列原生 chat template 的对齐情况，**选 Qwen3.5-4B 最合适**（~4.5B params，自带 XML 风格 tool_call chat_template）。

> tips：Qwen3.5 在 modelscope 和 HF 上的命名都是 `Qwen/Qwen3.5-4B`，**没有 `-Instruct` 后缀**（与 Qwen2.5 系列不同）。base id 本身就是 chat 版。可以用 `tokenizer_config.json` 里 `chat_template` 是否包含 `<tool_call>` 验证。

### 阅读数据

至少要阅读 10 条以上数据，理解项目方对工具调用的真实预期。task21 数据是 ShareGPT 风格的 `conversations` + `tools`，但 role 名沿用 OpenAI 风格的 `user / assistant / function_call / observation`。原始数据示例（从 `task21_TEO.jsonl` 摘抄一条）：

```json
{
  "conversations": [
    {"role": "user", "content": "Run 'df -h' and explain the disk usage"},
    {"role": "function_call", "content": "{\"name\":\"exec\",\"arguments\":{\"command\":\"df -h\"}}"}
  ],
  "tools": "[{\"name\":\"exec\",\"description\":\"Run shell commands ...\",\"parameters\":{...}}, {\"name\":\"cron\",...}, ...]"
}
```

注意三个特征：

1. 一条样本可能只有 `user → function_call` 两轮（直接调工具，没有 assistant 文本回答）。
2. 多步 agent 流程里会出现**连续的 `function_call`**（并行调用）或**连续的 `observation`**（并行返回）。
3. `tools` 字段是个字符串化的 JSON 数组，schema 较大（一行就上千 token）。

### 扩展数据 — 选 v1 而不是 v0（核心决策）

321 条训练数据对工具调用任务是偏少的，但**这次先不扩展数据，先验证 pipeline 能跑通**。在动手之前，遇到了第一个关键工程决策：

> **重大坑**：v0 LlamaFactory 用 **odd/even position** 来配对 prompt（mask）和 response（计 loss），这要求严格交替（user → assistant → user → assistant…）。但**真实 agent 数据中很容易出现连续的 `function_call` 或连续的 `observation`**（并行工具调用或多步执行），违反交替规则的样本会被静默丢弃。

v0 这三处都还有这个问题：

- [src/llamafactory/data/converter.py:144-146](src/llamafactory/data/converter.py#L144-L146)：odd/even tag 校验
- [src/llamafactory/data/template.py:85](src/llamafactory/data/template.py#L85)：`encode_multiturn` 固定 stride-2 配对
- [src/llamafactory/data/processor/supervised.py:110](src/llamafactory/data/processor/supervised.py#L110)：prompt 长度奇偶校验

**LlamaFactory v1 的设计天然解决了这个问题** —— v1 改成**逐条 message 标 `loss_weight`**（per-message 而非 per-pair）：

| 角色（converter 后） | loss_weight | 含义 |
|---|---|---|
| `system` / `human` / `observation` | 0.0 | masked，不算 loss |
| `gpt` / `function_call` | 1.0 | 计 loss，模型要学的部分 |

v1 的 converter ([v1/plugins/data_plugins/converter.py:115-152](src/llamafactory/v1/plugins/data_plugins/converter.py#L115-L152)) 不再校验奇偶位置，rendering 也按每条 message 自己的 `loss_weight` 决定 labels —— 任意 agent 流程都能完整保留。

> tips：如果你的工具调用数据每条样本都严格 `user → assistant → user → assistant` 交替，那 v0 / v1 行为完全一致。**只要数据有任何并行工具调用或多步执行，必须走 v1。**

#### 给 v1 补一个 qwen3.5 模板（XML 风格 tool call）

第二个关键工程决策：v1 [templates/](src/llamafactory/v1/plugins/model_plugins/templates/) 目录原本只有 `qwen3.py` 和 `qwen3_nothink.py`。Qwen3 和 Qwen3.5 的工具调用格式**完全不一样**：

| 项 | Qwen3 | Qwen3.5 |
|---|---|---|
| tool_call 格式 | `<tool_call>{"name":..., "arguments":...}</tool_call>`（JSON） | `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>`（XML） |
| system 工具提示 | `QWEN_TOOL_PROMPT` | `QWEN35_TOOL_PROMPT`（带 `<IMPORTANT>` 块） |

如果直接用 `template: qwen3` 训 Qwen3.5 模型，训练时 ground truth 的 tool_call 会被渲染成 JSON 风格，**和模型原生的 XML 风格不对齐**，效果会很差。

**解决方案**：仿照 v1 已有 `qwen3_nothink.py` 的结构，新增 `qwen3_5.py` / `qwen3_5_nothink.py` 两个模板，把工具调用渲染逻辑替换成 XML 风格。核心函数 `_format_qwen35_tool_call`：

```python
def _format_qwen35_tool_call(tool_call: ToolCall) -> str:
    name = tool_call["name"]
    arguments = tool_call.get("arguments", {})
    out = f"<tool_call>\n<function={name}>"
    for key, value in arguments.items():
        out += f"\n<parameter={key}>"
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False)
        out += f"\n{value}\n</parameter>"
    out += "\n</function>\n</tool_call>"
    return out
```

> tips：v1 的 `RenderingPlugin` 用懒加载（[rendering.py:32](src/llamafactory/v1/plugins/model_plugins/rendering.py#L32)），**只要文件名是 `qwen3_5.py`，就会被自动 import**，不需要改 `__init__.py`。整个改动对仓库已有源码**零侵入**。

### 数据格式转换

v1 sharegpt converter 期望 `from / value` 字段（不是 OpenAI 风格的 `role / content`）。需要做 role 重命名 + 字段重命名：

| 原始 role | v1 期望的 from |
|---|---|
| `user` | `human` |
| `assistant` | `gpt` |
| `function_call` | `function_call` |
| `observation` | `observation` |
| `system` | `system` |

转换脚本核心逻辑（[scripts/convert_task21_to_sharegpt.py](scripts/convert_task21_to_sharegpt.py)）：

```python
ROLE_MAP = {
    "user": "human",
    "assistant": "gpt",
    "function_call": "function_call",
    "observation": "observation",
    "system": "system",
}

def convert_file(in_path, out_path):
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            sample = json.loads(line)
            new_conv = [
                {"from": ROLE_MAP[m["role"]], "value": m["content"]}
                for m in sample["conversations"]
            ]
            out = {"conversations": new_conv}
            if "tools" in sample:
                out["tools"] = sample["tools"]
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
```

执行：

```bash
python scripts/convert_task21_to_sharegpt.py
# task21_TEO.jsonl       (321 条) -> data/task21_train.jsonl
# task21_eval_final.json (30 条)  -> data/task21_eval.jsonl
```

### 数据集 yaml 配置

LlamaFactory v1 的 dataset 注册方式跟 v0 的 `dataset_info.json` 完全不同。每个数据集是一个 yaml entry，三个核心字段：`path` / `source: local` / `converter: sharegpt`。

`data/task21_dataset.yaml`：

```yaml
task21_train:
  path: data/task21_train.jsonl
  source: local
  converter: sharegpt
```

`data/task21_eval.yaml`：

```yaml
task21_eval:
  path: data/task21_eval.jsonl
  source: local
  converter: sharegpt
```

## 参数配置

### 超参数设置

LoRA 微调对小数据集（几百到几千条）相对鲁棒，下面这套参数在 H800 单卡上验证有效：

```yaml
# examples/v1/train_lora/train_lora_task21_qwen35.yaml
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

# 单卡训练时把 dist_config 整块注释掉
# dist_config:
#   name: fsdp2
#   dcp_path: null

train_dataset: data/task21_dataset.yaml
eval_dataset: data/task21_eval.yaml

output_dir: ./outputs/task21_qwen35_lora
micro_batch_size: 1
cutoff_len: 4096                    # tools schema 较大，2048 不够
learning_rate: 1.0e-4
num_train_epochs: 3
bf16: true                          # H100 / H800 sweet spot

sample_backend: hf
max_new_tokens: 512
```

### 建议的范围

- **lora_rank** 一般 8 ~ 32；**lora_alpha** 经验上设为 lora_rank 的 2 倍。
- **数据量几百条**：`num_train_epochs = 2 ~ 3`；**数据量 2 万 ~ 10 万**：`num_train_epochs = 1` 即可。
- **cutoff_len**：tools schema 单行就上千 token，**2048 不够，4096 起步**；schema 特别长时考虑 8192。
- **避坑**：旧版 v1 的 `TrainingArguments` 不认 `warmup_ratio / logging_steps / save_steps / save_total_limit` 等字段，加了会让 `HfArgumentParser` 抛 `Some keys are not used` 异常。本配置只保留自 v1 第一版起就有的最小字段集。

> tips：modelscope 把模型路径里的 `.` 转成 `___`（三个下划线），所以 `Qwen3.5-4B` 在文件系统里是 `Qwen3___5-4B`，写 yaml 路径时记得用变形过的版本。

## 模型训练

LLaMA-Factory Online 提供「任务模式」（Web UI 一路点击）和「实例模式」（VSCode / JupyterLab Web）两种入口。本次实战走的是**实例模式 + VSCode Web**，最贴合开发者本地工作流，下面详细讲这条路径。

### 实例模式微调

**1. 启动 H800 实例**

进入 [实例空间] 选 H800（80GB 显存）× 1 卡，点击 [LlamaFactory 快速微调模型] 旁边的 [VSCode 处理专属数据] 入口。

![实例空间启动 - 选择微调入口](pic/登陆界面1.png)

进入 VSCode Web 后，右上角的实例信息卡片可以看到资源详情（H800（显存 80G）× 1、Storage 3.6%、对外服务地址、SSH 连接信息等）。

![VSCode Web 进入 + 实例信息](pic/登陆界面2.png)

**2. clone 仓库 + 配置 conda 环境**

在 VSCode 终端里：

```bash
git clone https://github.com/coinmini/llamafactory-v1-qwen35-task21.git
cd llamafactory-v1-qwen35-task21

# 配 pip 国内镜像（强烈建议，否则 50KB/s）
mkdir -p ~/.pip && cat > ~/.pip/pip.conf <<'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF

# 必须 Python 3.11+，v1 用了 NotRequired/StrEnum
conda create -n llf_v1 python=3.12 -y
conda activate llf_v1

pip install -e .
```

**3. 装匹配驱动的 torch wheel（关键坑）**

H800 / H100 的驱动通常支持到 CUDA 12.8，但默认装的 torch 是 cu126 build，启动训练时会报 `NVIDIA driver too old (found version 12080)`。修法：

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

# 复验
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# 期望：True
```

> tips：`nvidia-smi` 顶部的 `CUDA Version: 12.8` 是**驱动支持的最高版本**，跟 `torch.version.cuda`（wheel 编译版本）是两回事。一般装比驱动**等于或更老**的 wheel 才能跑。

**4. 用 modelscope 下载 Qwen3.5-4B（关键坑）**

国内 H100 / H800 服务器对 HuggingFace 通常**完全不通**，连 `hf-mirror.com` 也常常不通。走 modelscope：

```bash
pip install modelscope        # pyproject 已经声明，多半已装

python <<'PY'
from modelscope import snapshot_download
path = snapshot_download('Qwen/Qwen3.5-4B')
print(f'\n>>> MODEL_PATH: {path}')
PY
# 输出：>>> MODEL_PATH: /root/.cache/modelscope/hub/models/Qwen/Qwen3___5-4B
```

接下来必须**关掉 transformers 的网络检查**（不然每次加载都会去 HF HEAD 一次）：

```bash
echo 'export HF_HUB_OFFLINE=1' >> ~/.bashrc
echo 'export TRANSFORMERS_OFFLINE=1' >> ~/.bashrc
source ~/.bashrc
```

**5. 启动训练**

```bash
USE_V1=1 llamafactory-cli sft examples/v1/train_lora/train_lora_task21_qwen35.yaml
```

> tips：`USE_V1=1` 必需，[src/llamafactory/cli.py](src/llamafactory/cli.py) 通过它切到 `v1.launcher`，否则会进 v0。

启动后看到的健康日志大致是这样：

![训练启动 log（含 LoRA target / trainable% / step 1-3 loss）](pic/训练截图.png)

关键指标：

| 指标 | 实测值 | 解读 |
|---|---|---|
| 模型加载 | 723 个 weight tensor / 3 秒 | 本地 modelscope cache 加速明显 |
| LoRA target modules | 16 个（自动展开） | `target_modules: all` 起作用 |
| Trainable params | 39M / 4.58B = **0.85%** | LoRA 典型比例 |
| `total_steps` | **963** = 321 × 3 epoch / batch=1 | 数据规模符合预期 |
| Step 1 loss | 2.02 | 见到新 tools schema 的初始 loss |
| Step 2-3 loss | 0.56 / 0.58 | 立刻降下来 → 数据 + 模板对齐成功 |
| Step 间隔 | ~5 秒 | H800 + 4B + bf16 + cutoff=4096 正常水准 |

**6. 监控 GPU 显存**

另开一个 terminal 跑 `nvidia-smi`：

![训练中的 GPU 显存监控](pic/训练截图2.png)

H800 80GB 上，4B + LoRA + bf16 + cutoff_len=4096 单卡占用大约 30-40GB。

**7. 训练完成 — 检查产出**

训练大约 80 分钟后完成，输出在 `./outputs/task21_qwen35_lora/`。在 VSCode 的 EXPLORER 里展开就能看到：

![VSCode 中 outputs/task21_qwen35_lora/ 目录](pic/vs目录.png)

也可以在 LLaMA-Factory Online 的「文件管理」面板里查看：

![文件管理 — 项目顶层](pic/文件管理.png)

进入 `outputs/task21_qwen35_lora/` 里，看到关键产物：

![文件管理 — outputs 目录详情](pic/文件管理2.png)

产物清单：

| 文件 | 大小 | 用途 |
|---|---|---|
| `adapter_model.safetensors` | 150 MB | LoRA 权重（推理时叠加在 base 模型上） |
| `adapter_config.json` | 1.2 KB | LoRA 配置（r、alpha、target_modules 等） |
| `tokenizer.json` / `tokenizer_config.json` | 20 MB / 1.3 KB | tokenizer 一套，自动跟 base 对齐 |
| `chat_template.jinja` | 7.6 KB | Qwen3.5 原生 chat template |
| `trainer_log.jsonl` | 87 KB | 训练 log（每 logging step 一行 JSON） |

### 训练曲线

v1 默认**不输出** loss 曲线图（不像 v0 自动出 `training_loss.png`），但 `trainer_log.jsonl` 里有所有数据，用一段 matplotlib 脚本即可绘制 loss / grad_norm / learning_rate 三合一曲线：

![训练曲线 — loss / grad_norm / learning_rate（671 log entries）](pic/llamafactory-v1-qwen35-task21_outputs_task21_qwen35_lora_training_curves.png)

**曲线解读**：

- **loss**：从 2.0 在前 50 步内迅速降到 0.5 以下，后续稳定在 0.0 ~ 0.5 区间抖动，没有上升或剧烈震荡 → **已收敛**。
- **grad_norm**：稳定在 0 ~ 4 之间，step ~480 附近有一次尖刺到 12，但很快回落 → 没炸。
- **learning_rate**：恒定 1e-4（本配置没显式配 scheduler）。

> tips：训练 log 里 671 条记录对应 963 个 step（部分早期 logging 间隔不同），属于正常。

### 任务模式微调（备选）

LLaMA-Factory Online 同时提供 [模型微调] 任务模式 Web UI：选模型 → 选数据集 → 资源配置 → 任务中心查看日志 → 模型评估 → 模型对话，全程不用写代码。本文走的是实例模式 + VSCode 自定义代码路径，因此不重复展开任务模式（具体步骤可参考平台官方文档及同类实战教程）。

## 模型评估

### 验证脚本

LoRA 训完后，用 [flock_validator](https://github.com/FLock-io) 框架做评测。验证脚本会：

1. 加载 base 模型 + LoRA adapter
2. 对每条 eval 样本生成模型回答
3. 调用 3 个外部 LLM judge（kimi-k2.5 / gemini-3.1-pro-preview / deepseek-v3.2）打分
4. 综合得到 normalized score

完整命令：

```bash
conda activate llf_v1
cd /workspace/flock_validator_local_only_local_path

nohup python local_validate.py \
  --model-path /workspace/llamafactory-v1-qwen35-task21/outputs/task21_qwen35_lora \
  --validation-file /workspace/llamafactory-v1-qwen35-task21/task21_eval_final.jsonl \
  --base-model-path /root/.cache/modelscope/hub/models/Qwen/Qwen3___5-4B \
  --is-lora \
  --eval-with-llm \
  --context-length 8192 \
  --max-params 6000000000 \
  --eval-model-list "kimi-k2.5,gemini-3.1-pro-preview-low,deepseek-v3.2" \
  --prompt-id 3 \
  --eval-require 3 \
  --gen-require 1 \
  > task21.log 2>&1 &
```

### 验证过程

**单条样本生成 vs Reference 对比**（Conv 0，write 工具调用）：

![Conv 0 — write 工具调用，生成 vs Reference 完全对齐](pic/验证过程1.png)

可以看到模型对「Create /tmp/sample.py with a simple function」这个请求，生成了正确的 `<function=write>` 调用，参数 `content` 和 `path` 都填对了，跟 Reference 一致（仅有微小的换行差异）。

**单条样本（Conv 14）— cron 工具调用**：

![Conv 14 — cron 工具调用对比](pic/验证过程2.png)

模型在被问到「What cron jobs are currently scheduled?」时，正确调用了 `<function=cron>` 并填写 `<parameter=action>list</parameter>`，跟 Reference 完全一致。

**多条样本评估打分细节**：

![3 个 judge × 3 次评估每条样本的细节打分](pic/验证过程1-2.png)

每条样本被 3 个 LLM judge × 3 次评估 = 9 次打分，每次给出 Score（满分 10）+ Confidence（0-1）+ Reasoning（自然语言解释）。下面是另一组样本的总览：

![评估打分结果列表](pic/验证过程2-2.png)

评估过程的格式化输出：

![评估对比格式输出](pic/验证过程3.png)

### 验证分数

最终 final results：

![Final Results — 0.7566 normalized score](pic/验证分数.png)

**核心数据**：

| 指标 | 值 |
|---|---|
| Raw weighted avg score | **7.5660 / 10** |
| **Normalized score (0-1)** | **0.7566** |
| Total weighted scores | 262 |
| 评测次数 | 30 条 × 3 模型 × 3 次 = 270 次（少数 retry 后实际 262） |
| 单条均分（如 Conv 29） | Avg Score: 7.78 \| Avg Confidence: 0.88 |

### 评估结果解读

- **0.7566 / 1.0** 在 agent 工具调用场景下是个**有竞争力的成绩** —— 模型能在 8 成以上的样本里选对工具、填对必需参数、输出符合 OpenClaw XML 格式。
- 三个独立 LLM judge（kimi / gemini / deepseek）的打分**一致性较高**（Avg Confidence ~0.88），说明评分标准稳定，不是被某个 judge 偏好拉偏。
- 表现好的样本：**纯本地工具调用**（write / read / exec / cron list 等参数固定的请求）几乎是满分。
- 仍有改进空间的样本：**错误处理 / 多步推理**（比如工具返回 error 后接下来该调什么工具）的得分稍低，是后续扩展数据可以重点补强的方向。

## 总结

本次实战在 H800 单卡上用 80 分钟、321 条训练数据，把 Qwen3.5-4B 微调成了一个会用 OpenClaw 工具集合的 agent 模型，最终 **Normalized score 0.7566 / 1.0**。整个流程攻克了几个工程实战中常见的关键障碍：

1. **数据兼容性**：v0 的 odd/even 配对会丢掉真实 agent 数据中的并行工具调用，**走 v1 + per-message loss_weight** 一劳永逸。
2. **模板对齐**：Qwen3.5 的 XML 风格 `<tool_call><function=...>` 跟 Qwen3 的 JSON 风格完全不同，**给 v1 补 `qwen3_5` / `qwen3_5_nothink` 模板**，仓库零侵入。
3. **离线服务器**：HF 不通用 modelscope 替代，注意 `___` 三连下划线和 `HF_HUB_OFFLINE=1`。
4. **CUDA 匹配**：H100/H800 驱动到 cu128，记得装匹配的 torch wheel。
5. **yaml 字段兼容**：旧版 v1 不认 `warmup_ratio / logging_steps / save_steps`，先用最小字段集，确认 build 版本后再加。

对于开发者而言，**不要陷入"过度调参"的陷阱** —— LoRA r=16 / alpha=32 / lr=1e-4 / 3 epoch 对于工具调用任务已经足够，提升空间更多在**数据质量**和**工具 schema 设计**上。把更多精力投入到：

- 扩展数据时关注**错误处理样本**和**多步 agent 流程**的覆盖率；
- 用三个独立 LLM judge 评测可以避免单 judge 偏见，**confidence 比单纯 score 更值得参考**；
- 把训练 yaml、模板代码、转换脚本全部纳入版控（本仓库把这些做成开箱即用的模板）。

完整代码与数据：<https://github.com/coinmini/llamafactory-v1-qwen35-task21>

详细操作手册（每一步的可执行命令 + 故障排查）：[V1_QWEN35_TASK21_SETUP.md](V1_QWEN35_TASK21_SETUP.md)
