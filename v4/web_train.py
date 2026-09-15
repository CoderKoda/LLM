"""V4 web-data collector for Koda LLM.

Examples:
    python v4/web_train.py --source wikipedia --pages 500
    python v4/web_train.py --source gutenberg --pages 100 --only-collect
    python v4/web_train.py --build-only

Stop with Ctrl+C.  Scrapy's JOBDIR preserves crawl state so the same command
can be started again to continue instead of beginning from scratch.
"""

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
    source = WikipediaSource(rate_limit=delay)
    count = 0
    try:
        for doc in source.pages(limit=pages):
            path = save_document(doc, out)
            record_metadata(doc.source, doc.title, doc.url, path)
            count += 1
            print(f"[wiki] {count}/{pages}  {doc.title}", flush=True)
    except KeyboardInterrupt:
        print("\nStopped. Wikipedia documents already saved are safe to reuse.")


def collect_scrapy(source: str, pages: int) -> None:
    job = JOBS / source
    job.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "scrapy",
        "crawl",
        "training_spider",
        "-a",
        f"source={source}",
        "-a",
        f"output={RAW / source}",
        "-a",
        f"max_pages={pages}",
        "-s",
        f"JOBDIR={job}",
    ]
    print("Starting crawler. Press Ctrl+C to stop safely; run the same command to resume.")
    try:
        subprocess.run(cmd, cwd=ROOT, check=False)
    except KeyboardInterrupt:
        print("\nCrawler stopped. JOBDIR was preserved for resume.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Koda LLM V4 web data collector")
    parser.add_argument("--source", choices=["wikipedia", "gutenberg", "arxiv"], default="wikipedia")
    parser.add_argument("--pages", type=int, default=100, help="Target number of pages/documents for this run")
    parser.add_argument("--delay", type=float, default=1.0, help="Minimum delay between Wikipedia API batches")
    parser.add_argument("--only-collect", action="store_true", help="Collect/clean data without starting model training")
    parser.add_argument("--build-only", action="store_true", help="Only rebuild v4/data/train.txt from already-cleaned documents")
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.build_only:
        docs, chars = build(RAW, CORPUS)
        print(f"Built {CORPUS} from {docs} documents ({chars:,} characters).")
        return

    if args.source == "wikipedia":
        collect_wikipedia(args.pages, args.delay)
    else:
        collect_scrapy(args.source, args.pages)

    docs, chars = build(RAW, CORPUS)
    print(f"\nCorpus: {docs:,} unique documents, {chars:,} characters.")

    if args.only_collect:
        print(f"Saved cleaned training text to: {CORPUS}")
        return

    if chars < 1000:
        print("Not enough clean text to begin training yet. Collect more data first.")
        return

    train = ROOT.parent / "v3" / "train.py"
    checkpoint = ROOT / "checkpoints"
    cmd = [
        sys.executable,
        str(train),
        "--data", str(CORPUS),
        "--out", str(checkpoint),
        "--steps", str(args.train_steps),
        "--device", args.device,
    ]
    print("\nStarting local model training. Press Ctrl+C to stop; the latest checkpoint remains available.")
    subprocess.run(cmd, cwd=train.parent, check=False)


if __name__ == "__main__":
    main()
