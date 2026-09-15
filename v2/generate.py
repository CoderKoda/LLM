"""Generate text from a V2 checkpoint."""

from __future__ import annotations

import argparse

import torch

from model import GPT, GPTConfig
from tokenizer import BPETokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="v2/checkpoints/model.pt")
    p.add_argument("--prompt", default="")
    p.add_argument("--tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    a = p.parse_args()

    if a.device == "cuda":
        dev = torch.device("cuda")
    elif a.device == "mps":
        dev = torch.device("mps")
    elif a.device == "cpu":
        dev = torch.device("cpu")
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")

    ckpt = torch.load(a.checkpoint, map_location=dev, weights_only=False)
    model = GPT(GPTConfig(**ckpt["config"])).to(dev)
    model.load_state_dict(ckpt["model"])
    tokenizer = BPETokenizer.load(ckpt["tokenizer"])

    ids = torch.tensor([tokenizer.encode(a.prompt)], dtype=torch.long, device=dev)
    with torch.no_grad():
        out = model.generate(ids, a.tokens, a.temperature, a.top_k)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
