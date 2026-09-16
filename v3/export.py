"""Export Koda checkpoints into compact inference artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import GPT, GPTConfig


def main() -> None:
    p = argparse.ArgumentParser(description="Export a Koda checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dtype", choices=["fp32", "fp16"], default="fp16")
    p.add_argument("--format", choices=["pt", "safetensors"], default="pt")
    a = p.parse_args()

    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    if a.dtype == "fp16":
        state = {k: v.half() if torch.is_floating_point(v) else v for k, v in state.items()}

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "config": ckpt["config"], "tokenizer": ckpt.get("tokenizer", ""), "model": state}
    if a.format == "pt":
        torch.save(payload, out)
    else:
        try:
            from safetensors.torch import save_file
        except ImportError as exc:
            raise SystemExit("Install safetensors first: pip install safetensors") from exc
        save_file(state, out)
        out.with_suffix(".json").write_text(
            __import__("json").dumps({"config": ckpt["config"], "tokenizer": ckpt.get("tokenizer", "")}, indent=2),
            encoding="utf-8",
        )
    total = sum(t.numel() * t.element_size() for t in state.values())
    print(f"exported: {out}")
    print(f"model weights: {total / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
