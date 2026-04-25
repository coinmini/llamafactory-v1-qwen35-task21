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


# Qwen3.5 uses an XML-style tool prompt and tool-call format that differs from Qwen3.
# Reference: src/llamafactory/data/tool_utils.py (QWEN35_TOOL_PROMPT, Qwen35ToolUtils)
QWEN35_TOOL_PROMPT = (
    "\n\n# Tools\n\nYou have access to the following functions:\n\n<tools>{tool_text}"
    "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
    "- Function calls MUST follow the specified format: "
    "an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language "
    "BEFORE the function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal with your current knowledge "
    "and do not tell the user about function calls\n</IMPORTANT>"
)


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


def _get_last_query_index(messages: list[Message]) -> int:
    last_query_index = len(messages) - 1
    for idx in range(len(messages) - 1, -1, -1):
        message = messages[idx]
        if message["role"] != "user":
            continue

        user_text = ""
        is_plain_text = True
        for content in message["content"]:
            if content["type"] != "text":
                is_plain_text = False
                break
            user_text += content["value"]

        if not is_plain_text:
            continue

        if not (user_text.startswith("<tool_response>") and user_text.endswith("</tool_response>")):
            last_query_index = idx
            break

    return last_query_index


def _split_assistant_content(message: Message) -> tuple[str, str, list[ToolCall]]:
    text_content = ""
    reasoning_content = ""
    tool_calls: list[ToolCall] = []

    for content in message["content"]:
        if content["type"] == "text":
            text_content += content["value"]
        elif content["type"] == "reasoning":
            reasoning_content += content["value"]
        elif content["type"] == "tool_call":
            try:
                tool_call: ToolCall = json.loads(content["value"])
            except json.JSONDecodeError:
                raise ValueError(f"Invalid tool call format: {content['value']}.")

            tool_calls.append(tool_call)
        else:
            raise ValueError(f"Unsupported content type: {content['type']}")

    return text_content, reasoning_content, tool_calls


def _format_qwen35_tool_call(tool_call: ToolCall) -> str:
    name = tool_call["name"]
    arguments = tool_call.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"_raw": arguments}

    out = f"<tool_call>\n<function={name}>"
    for key, value in arguments.items():
        out += f"\n<parameter={key}>"
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False)
        out += f"\n{value}\n</parameter>"
    out += "\n</function>\n</tool_call>"
    return out


@RenderingPlugin("qwen3_5").register("render_messages")
def render_qwen3_5_messages(
    processor: Processor,
    messages: list[Message],
    tools: str | None = None,
    is_generate: bool = False,
    enable_thinking: bool = False,
) -> ModelInput:
    """Render messages in the Qwen3.5 (thinking) template format with XML-style tool calls."""
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
    last_query_index = _get_last_query_index(messages)

    for turn_idx, message in enumerate(messages):
        if message["role"] == "user" or (message["role"] == "system" and turn_idx != 0):
            temp_str += "<|im_start|>" + message["role"] + "\n" + _concat_text_content(message) + "<|im_end|>\n"
            temp_weight = message.get("loss_weight", 0.0)
        elif message["role"] == "assistant":
            temp_str += "<|im_start|>" + message["role"] + "\n"

            text_content, reasoning_content, tool_calls = _split_assistant_content(message)
            if turn_idx > last_query_index and (turn_idx == len(messages) - 1 or reasoning_content):
                temp_str += "<think>\n" + reasoning_content.strip("\n") + "\n</think>\n\n" + text_content.lstrip("\n")
            else:
                temp_str += text_content

            for tool_call_idx, tool_call in enumerate(tool_calls):
                if (tool_call_idx == 0 and text_content) or tool_call_idx > 0:
                    temp_str += "\n"

                temp_str += _format_qwen35_tool_call(tool_call)

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
        if enable_thinking is False:
            temp_str += "<think>\n\n</think>\n\n"

    temp_str = _update_model_input(processor, input_ids, labels, loss_weights, temp_str, temp_weight)

    attention_mask = [1] * len(input_ids)
    return ModelInput(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        loss_weights=loss_weights,
    )


@RenderingPlugin("qwen3_5").register("parse_message")
def parse_qwen3_5_message(generated_text: str) -> Message:
    """Parse a message in the Qwen3.5 template format (think + XML-style tool calls)."""
    content: list[dict] = []

    think_pattern = re.compile(r"<think>\s*(.*?)\s*</think>\s*", re.DOTALL)
    tool_call_pattern = re.compile(
        r"<tool_call>\s*<function=\s*([^\s<>]+)\s*(.*?)\s*</function>\s*</tool_call>\s*",
        re.DOTALL,
    )
    param_pattern = re.compile(r"<parameter=(.*?)>(.*?)</parameter>", re.DOTALL)

    cursor = 0
    text = generated_text

    think_match = think_pattern.match(text)
    if think_match:
        reasoning = think_match.group(1).strip()
        if reasoning:
            content.append({"type": "reasoning", "value": reasoning})
        cursor = think_match.end()

    while cursor < len(text):
        tc_match = tool_call_pattern.search(text, cursor)
        if not tc_match:
            tail = text[cursor:].strip()
            if tail:
                content.append({"type": "text", "value": tail})
            break

        if tc_match.start() > cursor:
            leading = text[cursor : tc_match.start()].strip()
            if leading:
                content.append({"type": "text", "value": leading})

        func_name = tc_match.group(1).strip()
        params_block = tc_match.group(2).strip()
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
        cursor = tc_match.end()

    return Message(role="assistant", content=content)
