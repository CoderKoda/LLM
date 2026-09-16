"""Separate local MLX chat interface for Qwen or another MLX model."""
from __future__ import annotations

import argparse
import time


def main() -> None:
    p = argparse.ArgumentParser(description="Chat with a local MLX model")
    p.add_argument("--model", default="models/qwen3-0.6b-base-mlx")
    p.add_argument("--tokens", type=int, default=160)
    p.add_argument("--temperature", type=float, default=0.7)
    a = p.parse_args()

    try:
        from mlx_lm import generate, load
    except ImportError as exc:
        raise SystemExit("Install MLX-LM first: pip install -U mlx-lm") from exc

    model, tokenizer = load(a.model)
    print("╭──────────────────────────────────────────╮")
    print("│              QWEN MLX CHAT               │")
    print("╰──────────────────────────────────────────╯")
    print(f"model: {a.model}")
    print("Commands: /exit  /clear  /settings")

    history = []
    while True:
        try:
            user = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if user.lower() == "/exit":
            return
        if user.lower() == "/clear":
            history = []
            print("Context cleared.")
            continue
        if user.lower() == "/settings":
            print(f"temperature={a.temperature} max tokens={a.tokens}")
            continue
        if not user:
            continue

        history.append({"role": "user", "content": user})
        try:
            prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = "".join(f"User: {m['content']}\n" for m in history) + "Assistant:"

        start = time.perf_counter()
        output = generate(model, tokenizer, prompt=prompt, max_tokens=a.tokens, temp=a.temperature, verbose=False)
        elapsed = max(1e-9, time.perf_counter() - start)
        print(f"Qwen > {output.strip()}\n[{a.tokens / elapsed:,.1f} tok/s target generation rate]")
        history.append({"role": "assistant", "content": output.strip()})


if __name__ == "__main__": main()
