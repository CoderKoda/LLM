"""Build a model-facing corpus from cleaned documents.

URLs, titles and source metadata stay outside the training text.  The language
model sees only cleaned prose separated by blank lines.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def build(input_dir: Path, output_file: Path, min_chars: int = 200) -> tuple[int, int]:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    documents = 0
    chars = 0

    with output_file.open("w", encoding="utf-8") as out:
        for path in sorted(input_dir.glob("*.txt")):
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if len(text) < min_chars:
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            out.write(text + "\n\n")
            documents += 1
            chars += len(text)

    return documents, chars


def write_manifest(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
