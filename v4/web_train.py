"""V4 web-data collector and self-training launcher for Koda LLM."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from build_corpus import build
from web_sources import WikipediaSource, save_document

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
CORPUS = DATA / "train.txt"
JOBS = DATA / "jobs"
METADATA = DATA / "sources.jsonl"
CHECKPOINT = ROOT / "checkpoints"
TOKENIZER = CHECKPOINT / "tokenizer.json"


def record_metadata(source: str, title: str, url: str, text_path: Path) -> None:
    METADATA.parent.mkdir(parents=True, exist_ok=True)
    with METADATA.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "source": source,
            "title": title,
            "url": url,
            "file": str(text_path),
        }, ensure_ascii=False) + "\n")


def collect_wikipedia(pages: int, delay: float) -> None:
    out = RAW / "wikipedia"
    state = DATA / "jobs" / "wikipedia_seen.json"
    source = WikipediaSource(rate_limit=delay, state_path=state)
    count = 0
    try:
        for doc in source.pages(limit=pages):
            path = save_document(doc, out)
            record_metadata(doc.source, doc.title, doc.url, path)
            count += 1
            print(f"[wiki] {count}/{pages}  {doc.title}", flush=True)
    except KeyboardInterrupt:
        print("\nStopped. Wikipedia documents and its seen-page state were saved.")


def collect_scrapy(source: str, pages: int) -> None:
    job = JOBS / source
    output = RAW / source
    job.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "run_spider.py"),
        "--source", source,
        "--output", str(output),
        "--pages", str(pages),
        "--jobdir", str(job),
    ]
    print("Starting crawler. Press Ctrl+C to stop safely; run the same command to resume.")
    try:
        subprocess.run(cmd, cwd=ROOT, check=False)
    except KeyboardInterrupt:
        print("\nCrawler stopped. The persistent job directory was preserved for resume.")


def train_existing(steps: int, device: str) -> None:
    if not CORPUS.exists():
        raise FileNotFoundError(f"Training corpus not found: {CORPUS}. Collect data first.")
    train = ROOT.parent / "v3" / "train.py"
    CHECKPOINT.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(train),
        "--data", str(CORPUS),
        "--out", str(CHECKPOINT / "model.pt"),
        "--tokenizer", str(TOKENIZER),
        "--steps", str(steps),
        "--device", device,
    ]
    print("\n[4/4] Training local model. Press Ctrl+C for a safe emergency checkpoint.")
    subprocess.run(cmd, cwd=train.parent, check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Koda LLM V4 web data collector")
    parser.add_argument("--source", choices=["wikipedia", "gutenberg", "arxiv"], default="wikipedia")
    parser.add_argument("--pages", type=int, default=100,
                        help="Number of new unique random pages/documents to collect")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Minimum delay between Wikipedia API batches")
    parser.add_argument("--only-collect", action="store_true",
                        help="Collect/clean data without starting model training")
    parser.add_argument("--only-train", action="store_true",
                        help="Skip collection and train from the existing v4/data/train.txt")
    parser.add_argument("--build-only", action="store_true",
                        help="Only rebuild v4/data/train.txt from already-collected documents")
    parser.add_argument("--rebuild-before-train", action="store_true",
                        help="Rebuild train.txt from raw documents before --only-train")
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.only_train:
        if args.rebuild_before_train:
            docs, chars = build(RAW, CORPUS)
            print(f"[3/4] Built {CORPUS} from {docs:,} documents ({chars:,} characters).")
        train_existing(args.train_steps, args.device)
        return

    if args.build_only:
        docs, chars = build(RAW, CORPUS)
        print(f"Built {CORPUS} from {docs:,} documents ({chars:,} characters).")
        return

    print("[1/4] Collecting source data...")
    if args.source == "wikipedia":
        collect_wikipedia(args.pages, args.delay)
    else:
        collect_scrapy(args.source, args.pages)

    print("[2/4] Building clean model-facing corpus...")
    docs, chars = build(RAW, CORPUS)
    print(f"Corpus: {docs:,} unique documents, {chars:,} characters.")

    if args.only_collect:
        print(f"Saved cleaned training text to: {CORPUS}")
        return

    if chars < 1000:
        print("Not enough clean text to begin training yet. Collect more data first.")
        return

    train_existing(args.train_steps, args.device)


if __name__ == "__main__":
    main()
