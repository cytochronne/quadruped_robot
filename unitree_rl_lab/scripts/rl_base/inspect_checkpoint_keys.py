from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch


def summarize_prefixes(keys: list[str], depth: int = 2) -> Counter[str]:
    counter: Counter[str] = Counter()
    for key in keys:
        parts = key.split(".")
        prefix = ".".join(parts[:depth]) if len(parts) >= depth else key
        counter[prefix] += 1
    return counter


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect checkpoint keys and checkpoint layout.")
    parser.add_argument("checkpoint", type=Path, help="Path to checkpoint .pt file")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"checkpoint: {args.checkpoint}")
    print(f"type: {type(ckpt).__name__}")

    if not isinstance(ckpt, dict):
        raise TypeError(f"Expected checkpoint to be a dict, got {type(ckpt).__name__}")

    top_keys = list(ckpt.keys())
    print("\n[top-level keys]")
    for key in top_keys:
        print(key)

    state_dict = ckpt.get("model_state_dict", ckpt)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected model_state_dict to be a dict, got {type(state_dict).__name__}")

    model_keys = list(state_dict.keys())
    print(f"\n[model_state_dict] total keys: {len(model_keys)}")
    print(f"has 'teacher.' prefix: {any(k.startswith('teacher.') for k in model_keys)}")
    print(f"has 'student' prefix: {any(k.startswith('student') for k in model_keys)}")
    print(f"has 'memory_s.' prefix: {any(k.startswith('memory_s.') for k in model_keys)}")
    print(f"has raw 'critic.' prefix: {any(k.startswith('critic.') for k in model_keys)}")

    print("\n[prefix summary depth=1]")
    for prefix, count in sorted(summarize_prefixes(model_keys, depth=1).items()):
        print(f"{prefix}: {count}")

    print("\n[prefix summary depth=2]")
    for prefix, count in sorted(summarize_prefixes(model_keys, depth=2).items()):
        print(f"{prefix}: {count}")

    print("\n[all model_state_dict keys]")
    for key in model_keys:
        print(key)


if __name__ == "__main__":
    main()
