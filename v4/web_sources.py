"""Source adapters for the V4 web-training pipeline.

The project deliberately separates source discovery from text cleaning. New
sources can be added without changing the tokenizer or model.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import quote
from urllib.request import Request, urlopen


USER_AGENT = "KodaLLM/4.0 (+https://github.com/CoderKoda/LLM) research trainer"


@dataclass(frozen=True)
class Document:
    source: str
    title: str
    url: str
    text: str
    page_id: str = ""


def _get_json(url: str, timeout: int = 30) -> dict:
    req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class WikipediaSource:
    """Fetch random, clean article extracts through the MediaWiki API."""

    name = "wikipedia"
    api = "https://en.wikipedia.org/w/api.php"

    def __init__(self, rate_limit: float = 1.0, state_path: Path | None = None):
        self.delay = max(0.05, rate_limit)
        self.state_path = state_path
        self.seen = self._load_seen()

    def _load_seen(self) -> set[str]:
        if not self.state_path or not self.state_path.exists():
            return set()
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return {str(x) for x in data}
            if isinstance(data, dict) and isinstance(data.get("seen_pageids"), list):
                return {str(x) for x in data["seen_pageids"]}
        except (OSError, json.JSONDecodeError):
            pass
        return set()

    def _save_seen(self) -> None:
        if not self.state_path:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.state_path.with_suffix(".tmp")
        temp.write_text(
            json.dumps({"seen_pageids": sorted(self.seen)}, separators=(",", ":")),
            encoding="utf-8",
        )
        temp.replace(self.state_path)

    def pages(self, limit: int = 1000) -> Iterable[Document]:
        """Yield `limit` random, non-repeating main-namespace Wikipedia articles.

        The seen-page state is persisted between runs, so another invocation
        continues sampling new articles instead of starting over.
        """
        fetched = 0
        attempts = 0
        max_attempts = max(limit * 20, 200)

        while fetched < limit and attempts < max_attempts:
            batch_size = min(20, max(1, limit - fetched))
            params = {
                "action": "query",
                "format": "json",
                "generator": "random",
                "grnnamespace": "0",
                "grnlimit": str(batch_size),
                "grnfilterredir": "nonredirects",
                "prop": "extracts|info",
                "explaintext": "1",
                "exsectionformat": "plain",
                "inprop": "url",
            }
            query = "&".join(f"{quote(str(k))}={quote(str(v))}" for k, v in params.items())
            data = _get_json(f"{self.api}?{query}")
            pages = data.get("query", {}).get("pages", {})
            if not pages:
                break

            new_found = False
            for page in pages.values():
                page_id = str(page.get("pageid", ""))
                if not page_id or page_id in self.seen:
                    continue

                self.seen.add(page_id)
                self._save_seen()
                new_found = True

                text = (page.get("extract") or "").strip()
                if not text:
                    continue

                yield Document(
                    source=self.name,
                    title=page.get("title", ""),
                    url=page.get("fullurl", ""),
                    text=text,
                    page_id=page_id,
                )
                fetched += 1
                if fetched >= limit:
                    break

            attempts += batch_size
            if new_found:
                time.sleep(self.delay)
            else:
                time.sleep(max(self.delay, 0.5))


class URLSeedSource:
    """Small, policy-aware adapter for explicitly allowed seed URLs.

    Trafilatura performs the actual extraction. Crawling is intentionally kept
    conservative here; robots.txt and per-domain throttling are handled by the
    crawler module.
    """

    def __init__(self, urls: list[str]):
        self.urls = urls

    def seeds(self) -> list[str]:
        return list(dict.fromkeys(self.urls))


DEFAULT_SOURCES = {
    "wikipedia": {
        "kind": "api",
        "description": "Random Wikipedia article extracts via MediaWiki API",
    },
    "gutenberg": {
        "kind": "web",
        "description": "Project Gutenberg public-domain texts",
        "seeds": ["https://www.gutenberg.org/"],
    },
    "arxiv": {
        "kind": "web",
        "description": "Open scientific papers and metadata",
        "seeds": ["https://arxiv.org/"],
    },
    "common_crawl": {
        "kind": "index",
        "description": "Open web corpus index; use only through its public index/terms",
    },
}


def save_document(doc: Document, root: Path) -> Path:
    """Store text plus source metadata without putting the training URL in the corpus."""
    root.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in doc.title[:100]).strip("_")
    path = root / f"{safe or 'document'}_{doc.page_id or 'document'}.txt"
    if not path.exists():
        path.write_text(doc.text.strip() + "\n", encoding="utf-8")
    return path
