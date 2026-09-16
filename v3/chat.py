"""Interactive terminal chat for a trained Koda LLM checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import GPT, GPTConfig
from tokenizer import BPETokenizer
from ui import header, kv


def parse_args():
    p = argparse.ArgumentParser(description="Chat with a locally trained Koda LLM")
    p.add_argument("--model", "--checkpoint", dest="model", default="v3/checkpoints/model.pt")
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--max-tokens", "--tokens", dest="max_tokens", type=int, default=120)
    return p.parse_args()


def get_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        return torch.device("cuda")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    a = parse_args()
    device = get_device(a.device)
    checkpoint_path = Path(a.model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    tokenizer_path = Path(a.tokenizer or checkpoint.get("tokenizer", ""))
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    tokenizer = BPETokenizer.load(tokenizer_path)
    config = GPTConfig(**checkpoint["config"])
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    header("KODA LLM CHAT")
    kv("model", checkpoint_path)
    kv("device", device)
    kv("parameters", f"{model.parameter_count():,}")
    kv("context", config.block_size)
    kv("temperature", a.temperature)
    kv("top-k", a.top_k)
    kv("max tokens", a.max_tokens)
    print("\nType /exit to quit, /clear to reset the context.\n")

    history_ids: list[int] = []
    while True:
        try:
            user = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return
        if user.lower() == "/exit":
            return
        if user.lower() == "/clear":
            history_ids = []
            print("Context cleared.")
            continue
        if not user:
            continue

        history_ids.extend(tokenizer.encode(f"User: {user}\nAssistant:"))
        context = history_ids[-config.block_size:]
        prompt_len = len(context)
        input_ids = torch.tensor([context], dtype=torch.long, device=device)
        with torch.inference_mode():
            output = model.generate(
                input_ids,
                max_new_tokens=a.max_tokens,
                temperature=a.temperature,
                top_k=a.top_k,
            )[0].tolist()

        new_ids = output[prompt_len:]
        response = tokenizer.decode(new_ids).split("\nUser:", 1)[0].strip()
        print(f"LLM > {response}\n")
        history_ids.extend(new_ids)
        history_ids = history_ids[-config.block_size:]


if __name__ == "__main__":
    main()
