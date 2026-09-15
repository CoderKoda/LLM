"""Build a model-facing corpus from cleaned documents.

URLs, titles and source metadata stay outside the training text. The language
model sees only cleaned prose separated by blank lines.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

URL_RE = re.compile(r"https?://|www\\.", re.I)


def _quality_ok(text: str, min_chars: int) -> bool:
    if len(text) < min_chars:
        return False
    letters = sum(c.isalpha() for c in text)
    printable = sum(c.isprintable() or c in "\n\t" for c in text)
    if letters < 80:
        return False
    if printable / max(len(text), 1) < 0.90:
        return False
    if text.count("\n") > max(20, len(text) // 40) and letters / max(len(text), 1) < 0.30:
        return False
    urls = len(URL_RE.findall(text))
    if urls > max(3, len(text) // 500):
        return False
    return True


def build(
    input_dir: Path,
    output_file: Path,
    min_chars: int = 200,
    shuffle: bool = True,
    seed: int = 1337,
) -> tuple[int, int]:
    """Combine, deduplicate, quality-filter and deterministically shuffle documents."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    records: list[tuple[str, str]] = []
    seen: set[str] = set()

    for path in sorted(input_dir.rglob("*.txt")):
        if path.resolve() == output_file.resolve():
            continue

        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not _quality_ok(text, min_chars):
            continue

        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        records.append((str(path), text))

    if shuffle:
        random.Random(seed).shuffle(records)

    chars = 0
    with output_file.open("w", encoding="utf-8") as out:
        for _, text in records:
            out.write(text + "\n\n")
            chars += len(text)

    return len(records), chars


def write_manifest(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
