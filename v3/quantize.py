"""Create a CPU-friendly dynamic INT8 Koda model."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from model import GPT, GPTConfig


def main() -> None:
    p = argparse.ArgumentParser(description="Dynamic INT8 quantization for Koda")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="v3/checkpoints/model.int8.pt")
    a = p.parse_args()
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    quantized = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "version": 1,
        "quantization": "dynamic-int8",
        "config": ckpt["config"],
        "tokenizer": ckpt.get("tokenizer", ""),
        "model": quantized,
    }, out)
    print(f"saved INT8 model: {out}")
    print("Note: dynamic INT8 inference is intended for CPU; MPS/CUDA inference should use the FP16/FP32 checkpoint.")


if __name__ == "__main__": main()
