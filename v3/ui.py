"""Small terminal UI helpers for Koda LLM training."""
from __future__ import annotations

import time


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def progress_bar(label: str, done: int, total: int, start: float, extra: str = "", width: int = 36) -> None:
    total = max(total, 1)
    ratio = min(max(done / total, 0.0), 1.0)
    filled = int(width * ratio)
    bar = "█" * filled + "░" * (width - filled)
    elapsed = time.perf_counter() - start
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else 0.0
    print(
        f"\r{label:<10} [{bar}] {ratio * 100:6.2f}% "
        f"{done:,}/{total:,} | {rate:9.1f}/s | "
        f"elapsed {format_seconds(elapsed):>9} | ETA {format_seconds(eta):>9}{extra}",
        end="",
        flush=True,
    )


def header(title: str, width: int = 68) -> None:
    print("╭" + "─" * width + "╮")
    print("│" + title.center(width) + "│")
    print("╰" + "─" * width + "╯")


def kv(label: str, value, width: int = 18) -> None:
    print(f"  {label:<{width}} {value}")
