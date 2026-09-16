"""Small deterministic benchmark for Koda checkpoints and optional MLX/Qwen models."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent

PROMPTS = [
    ("Arithmetic", "What is 17 + 25? Answer with only the number.", "42"),
    ("Arithmetic", "What is 12 * 8? Answer with only the number.", "96"),
    ("Knowledge", "What is the capital of France?", "Paris"),
    ("Knowledge", "What gas do humans need to breathe to stay alive?", "oxygen"),
    ("Logic", "If all bloops are razzies and all razzies are blue, are all bloops blue?", "yes"),
    ("Completion", "Complete this sentence: The sun rises in the", "east"),
]


def koda_run(checkpoint: str, prompts: list[str], tokens: int, temperature: float, top_k: int, device: str):
    import sys
    sys.path.insert(0, str(ROOT / "v3"))
    from model import GPT, GPTConfig
    from tokenizer import BPETokenizer

    if device == "mps": dev = torch.device("mps")
    elif device == "cuda": dev = torch.device("cuda")
    else: dev = torch.device("cpu")
    ckpt = torch.load(checkpoint, map_location=dev, weights_only=False)
    tok = BPETokenizer.load(ckpt["tokenizer"])
    model = GPT(GPTConfig(**ckpt["config"])).to(dev)
    model.load_state_dict(ckpt["model"]); model.eval()
    results = []
    for prompt in prompts:
        ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=dev)
        start = time.perf_counter()
        out = model.generate_cached(ids, tokens, temperature, top_k)[0].tolist()
        elapsed = max(1e-9, time.perf_counter() - start)
        text = tok.decode(out)[len(prompt):].strip()
        results.append({"prompt": prompt, "output": text, "tokens_per_sec": tokens / elapsed})
    return results


def main():
    p = argparse.ArgumentParser(description="Benchmark a Koda model")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps")
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--output", default=None)
    a = p.parse_args()

    prompts = [x[1] for x in PROMPTS]
    results = koda_run(a.checkpoint, prompts, a.tokens, a.temperature, a.top_k, a.device)
    for spec, result in zip(PROMPTS, results):
        expected = spec[2].lower()
        ok = expected in result["output"].lower()
        print(f"[{spec[0]:10}] {'PASS' if ok else 'FAIL':4} | {result['tokens_per_sec']:7.1f} tok/s | {result['output'][:120]!r}")
    if a.output:
        Path(a.output).write_text(json.dumps({"checkpoint": a.checkpoint, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__": main()
