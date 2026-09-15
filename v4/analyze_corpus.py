"""Dataset health analyzer for Koda LLM V4."""
from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path

URL_RE = re.compile(r"https?://|www\\.", re.I)
WORD_RE = re.compile(r"[A-Za-z]{2,}")


def iter_documents(corpus: Path):
    text = corpus.read_text(encoding="utf-8", errors="ignore")
    for chunk in re.split(r"\n\s*\n", text):
        chunk = chunk.strip()
        if chunk:
            yield chunk


def analyze(corpus: Path) -> dict:
    docs = list(iter_documents(corpus))
    chars = len(corpus.read_text(encoding="utf-8", errors="ignore"))
    words = sum(len(WORD_RE.findall(d)) for d in docs)
    avg = (sum(map(len, docs)) / len(docs)) if docs else 0
    largest = max(map(len, docs), default=0)
    tiny = sum(len(d) < 200 for d in docs)
    url_heavy = sum(bool(URL_RE.search(d)) for d in docs)

    first_words = Counter()
    for d in docs:
        first_words.update(w.lower() for w in WORD_RE.findall(d[:400]))

    return {
        "documents": len(docs),
        "characters": chars,
        "words_estimate": words,
        "avg_document_chars": round(avg, 1),
        "largest_document_chars": largest,
        "tiny_documents": tiny,
        "url_heavy_documents": url_heavy,
        "top_terms": first_words.most_common(20),
    }


def main():
    p = argparse.ArgumentParser(description="Analyze a Koda LLM training corpus")
    p.add_argument("corpus")
    args = p.parse_args()
    stats = analyze(Path(args.corpus))
    print("Koda LLM corpus health")
    print("────────────────────────────────")
    for key in (
        "documents", "characters", "words_estimate", "avg_document_chars",
        "largest_document_chars", "tiny_documents", "url_heavy_documents",
    ):
        print(f"{key:>24}: {stats[key]:,}")
    print("\nCommon early terms:")
    print(", ".join(f"{term} ({count})" for term, count in stats["top_terms"]))

    if stats["documents"] < 1000:
        print("\nWARNING: fewer than 1,000 documents. Topic skew and memorization are likely.")
    if stats["characters"] < 10_000_000:
        print("WARNING: under ~10 MB of text. A small model is preferable for experiments.")
    if stats["largest_document_chars"] > max(10_000, stats["characters"] * 0.10):
        share = stats["largest_document_chars"] / max(stats["characters"], 1)
        print(f"WARNING: largest document is {share:.1%} of the corpus; topic dominance is possible.")
    if stats["url_heavy_documents"]:
        print("WARNING: some corpus documents still contain URL-like text.")


if __name__ == "__main__":
    main()
