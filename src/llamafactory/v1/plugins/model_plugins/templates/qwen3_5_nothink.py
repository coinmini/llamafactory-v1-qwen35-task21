# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import re

from ....utils.constants import IGNORE_INDEX
from ....utils.helper import get_tokenizer
from ....utils.types import Message, ModelInput, Processor, ToolCall
from ..rendering import RenderingPlugin
from .qwen3_5 import QWEN35_TOOL_PROMPT, _format_qwen35_tool_call


def _update_model_input(
    processor: Processor,
    input_ids: list[int],
    labels: list[int],
    loss_weights: list[int],
    temp_str: str,
    temp_weight: float,
) -> str:
    if not temp_str:
        return ""

    tokenizer = get_tokenizer(processor)
    temp_ids = tokenizer.encode(temp_str, add_special_tokens=False)
    input_ids.extend(temp_ids)
    loss_weights.extend([temp_weight] * len(temp_ids))
    if temp_weight > 1e-6:
        labels.extend(temp_ids)
    else:
        labels.extend([IGNORE_INDEX] * len(temp_ids))

    return ""


def _concat_text_content(message: Message) -> str:
    message_text = ""
    for content in message["content"]:
        if content["type"] == "text":
            message_text += content["value"]
        else:
            raise ValueError(f"Unsupported content type: {content['type']}")

    return message_text


@RenderingPlugin("qwen3_5_nothink").register("render_messages")
def render_qwen3_5_nothink_messages(
    processor: Processor,
    messages: list[Message],
    tools: str | None = None,
    is_generate: bool = False,
    enable_thinking: bool = False,
) -> ModelInput:
    """Render messages in the Qwen3.5 nothink template format with XML-style tool calls."""
    input_ids, labels, loss_weights = [], [], []
    temp_str, temp_weight = "", 0.0

    if tools:
        temp_str += "<|im_start|>system\n"
        if messages[0]["role"] == "system":
            temp_str += _concat_text_content(messages[0])
            temp_weight = messages[0].get("loss_weight", 0.0)

        try:
            tools_list = json.loads(tools)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid tools format: {str(tools)}.")

        if not isinstance(tools_list, list):
            tools_list = [tools_list]

        tool_text = ""
        for tool in tools_list:
            tool = tool.get("function", tool) if isinstance(tool, dict) and tool.get("type") == "function" else tool
            tool_text += "\n" + json.dumps(tool, ensure_ascii=False)

        temp_str += QWEN35_TOOL_PROMPT.format(tool_text=tool_text) + "<|im_end|>\n"
    elif messages[0]["role"] == "system":
        temp_str += "<|im_start|>system\n" + _concat_text_content(messages[0]) + "<|im_end|>\n"
        temp_weight = messages[0].get("loss_weight", 0.0)

    temp_str = _update_model_input(processor, input_ids, labels, loss_weights, temp_str, temp_weight)

    for turn_idx, message in enumerate(messages):
        if message["role"] == "user" or (message["role"] == "system" and turn_idx != 0):
            temp_str += "<|im_start|>" + message["role"] + "\n" + _concat_text_content(message) + "<|im_end|>\n"
            temp_weight = message.get("loss_weight", 0.0)
        elif message["role"] == "assistant":
            temp_str += "<|im_start|>" + message["role"] + "\n"
            for val_idx, content in enumerate(message["content"]):
                if content["type"] == "text":
                    temp_str += content["value"]
                elif content["type"] == "reasoning":
                    # nothink: keep the reasoning marker out of special tokens
                    temp_str += "<thinking>\n" + content["value"] + "\n</thinking>\n\n"
                elif content["type"] == "tool_call":
                    if val_idx != 0 and message["content"][val_idx - 1]["type"] in ["text", "tool_call"]:
                        temp_str += "\n"

                    try:
                        tool_call: ToolCall = json.loads(content["value"])
                    except json.JSONDecodeError:
                        raise ValueError(f"Invalid tool call format: {content['value']}.")

                    temp_str += _format_qwen35_tool_call(tool_call)
                else:
                    raise ValueError(f"Unsupported content type: {content['type']}")

            temp_str += "<|im_end|>\n"
            temp_weight = message.get("loss_weight", 1.0)
        elif message["role"] == "tool":
            if turn_idx == 0 or messages[turn_idx - 1]["role"] != "tool":
                temp_str += "<|im_start|>user"

            temp_str += "\n<tool_response>\n" + _concat_text_content(message) + "\n</tool_response>"
            if turn_idx == len(messages) - 1 or messages[turn_idx + 1]["role"] != "tool":
                temp_str += "<|im_end|>\n"

            temp_weight = message.get("loss_weight", 0.0)

        temp_str = _update_model_input(processor, input_ids, labels, loss_weights, temp_str, temp_weight)

    if is_generate:
        temp_str += "<|im_start|>assistant\n"
        temp_weight = 0.0
        if enable_thinking:
            raise ValueError("The qwen3_5_nothink template does not support thinking mode.")

    temp_str = _update_model_input(processor, input_ids, labels, loss_weights, temp_str, temp_weight)

    attention_mask = [1] * len(input_ids)
    return ModelInput(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        loss_weights=loss_weights,
    )


@RenderingPlugin("qwen3_5_nothink").register("parse_message")
def parse_qwen3_5_nothink_message(generated_text: str) -> Message:
    """Parse a Qwen3.5 nothink message: text + interleaved <thinking> and XML-style tool calls."""
    content: list[dict] = []

    thinking_pattern = re.compile(r"<thinking>\s*(.*?)\s*</thinking>\s*", re.DOTALL)
    tool_call_pattern = re.compile(
        r"<tool_call>\s*<function=\s*([^\s<>]+)\s*(.*?)\s*</function>\s*</tool_call>\s*",
        re.DOTALL,
    )
    param_pattern = re.compile(r"<parameter=(.*?)>(.*?)</parameter>", re.DOTALL)

    combined = re.compile(
        r"(?:<thinking>\s*(?P<reasoning>.*?)\s*</thinking>)|"
        r"(?:<tool_call>\s*<function=\s*(?P<name>[^\s<>]+)\s*(?P<params>.*?)\s*</function>\s*</tool_call>)",
        re.DOTALL,
    )

    last_end = 0
    for match in combined.finditer(generated_text):
        if match.start() > last_end:
            leading = generated_text[last_end : match.start()].strip()
            if leading:
                content.append({"type": "text", "value": leading})

        if match.group("reasoning") is not None:
            reasoning = match.group("reasoning").strip()
            if reasoning:
                content.append({"type": "reasoning", "value": reasoning})
        else:
            func_name = match.group("name").strip()
            params_block = match.group("params").strip()
            args = {}
            for key, raw_value in param_pattern.findall(params_block):
                value = raw_value.strip()
                try:
                    args[key] = json.loads(value)
                except json.JSONDecodeError:
                    args[key] = value

            content.append(
                {
                    "type": "tool_call",
                    "value": json.dumps({"name": func_name, "arguments": args}, ensure_ascii=False),
                }
            )

        last_end = match.end()

    if last_end < len(generated_text):
        tail = generated_text[last_end:].strip()
        if tail:
            content.append({"type": "text", "value": tail})

    return Message(role="assistant", content=content)
