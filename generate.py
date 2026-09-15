"""Generate text from a trained checkpoint.

Example:
    python generate.py --checkpoint checkpoints/model.pt --prompt "Hello"
"""

from __future__ import annotations

import argparse

import torch

from model import GPT, GPTConfig
from tokenizer import ByteTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from the trained GPT")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = GPTConfig(**checkpoint["config"])
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    tokenizer = ByteTokenizer()
    prompt_tokens = tokenizer.encode(args.prompt)
    idx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        generated = model.generate(
            idx,
            max_new_tokens=args.tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
