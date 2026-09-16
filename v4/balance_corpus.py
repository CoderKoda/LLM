"""Build a source-balanced corpus from V4 raw documents."""
from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path


def collect(root: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for source_dir in root.iterdir() if root.exists() else []:
        if not source_dir.is_dir() or source_dir.name.startswith("."):
            continue
        files = sorted(source_dir.rglob("*.txt"))
        if files:
            groups[source_dir.name] = files
    return groups


def main() -> None:
    p = argparse.ArgumentParser(description="Balance V4 corpus sources")
    p.add_argument("--raw", default="v4/data/raw")
    p.add_argument("--out", default="v4/data/train.txt")
    p.add_argument("--total-docs", type=int, default=10000)
    p.add_argument("--seed", type=int, default=1337)
    a = p.parse_args()
    random.seed(a.seed)
    groups = collect(Path(a.raw))
    if not groups:
        raise SystemExit("no raw source directories found")

    names = sorted(groups)
    base, remainder = divmod(a.total_docs, len(names))
    selected: list[tuple[str, Path]] = []
    for i, name in enumerate(names):
        quota = base + (1 if i < remainder else 0)
        pool = groups[name][:]
        random.shuffle(pool)
        selected.extend((name, path) for path in pool[:quota])

    random.shuffle(selected)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    seen = set(); written = 0
    with out.open("w", encoding="utf-8") as f:
        for source, path in selected:
            try:
                text = path.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if len(text) < 200 or digest in seen:
                continue
            seen.add(digest)
            f.write(text + "\n\n")
            written += 1
    print(f"built balanced corpus: {written:,} documents")
    print("source quotas:", ", ".join(f"{n}={len([1 for s,_ in selected if s == n])}" for n in names))


if __name__ == "__main__": main()
