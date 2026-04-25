"""Convert task21 OpenAI-style ShareGPT data to v1 sharegpt converter format.

Input rows look like:
    {"conversations": [{"role": "user", "content": "..."},
                       {"role": "function_call", "content": "{...}"}, ...],
     "tools": "[ ... ]"}

The v1 sharegpt converter expects:
    {"conversations": [{"from": "human", "value": "..."},
                       {"from": "function_call", "value": "{...}"}, ...],
     "tools": "[ ... ]"}
"""

import json
import os
import sys


ROLE_MAP = {
    "user": "human",
    "assistant": "gpt",
    "function_call": "function_call",
    "observation": "observation",
    "system": "system",
}


def convert_file(in_path: str, out_path: str) -> int:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = 0
    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            new_conv = []
            for msg in sample["conversations"]:
                role = msg["role"]
                if role not in ROLE_MAP:
                    raise ValueError(f"unknown role {role!r} in line {n + 1}")
                new_conv.append({"from": ROLE_MAP[role], "value": msg["content"]})
            out = {"conversations": new_conv}
            if "tools" in sample:
                out["tools"] = sample["tools"]
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pairs = [
        (os.path.join(repo_root, "task21_TEO.jsonl"), os.path.join(repo_root, "data", "task21_train.jsonl")),
        (os.path.join(repo_root, "task21_eval_final.json"), os.path.join(repo_root, "data", "task21_eval.jsonl")),
    ]
    for src, dst in pairs:
        n = convert_file(src, dst)
        print(f"{src} -> {dst}: {n} samples", file=sys.stderr)


if __name__ == "__main__":
    main()
