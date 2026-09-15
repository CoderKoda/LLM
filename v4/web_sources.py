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


def _get_json(url: str, timeout: int = 30) -> dict:
    req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class WikipediaSource:
    """Fetch clean article extracts through the MediaWiki API, not HTML scraping."""

    name = "wikipedia"
    api = "https://en.wikipedia.org/w/api.php"

    def __init__(self, rate_limit: float = 1.0, state_path: Path | None = None):
        self.delay = max(0.05, rate_limit)
        self.state_path = state_path

    def _load_state(self) -> dict[str, str]:
        if not self.state_path or not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_state(self, state: dict[str, str]) -> None:
        if not self.state_path:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state), encoding="utf-8")

    def pages(self, limit: int = 1000, start: str | None = None) -> Iterable[Document]:
        fetched = 0
        cont: dict[str, str] = self._load_state()
        if start:
            cont["apfrom"] = start

        while fetched < limit:
            params = {
                "action": "query",
                "format": "json",
                "generator": "allpages",
                "gapnamespace": "0",
                "gaplimit": str(min(20, limit - fetched)),
                "prop": "extracts|info",
                "explaintext": "1",
                "exsectionformat": "plain",
                "inprop": "url",
            }
            params.update(cont)
            query = "&".join(f"{quote(str(k))}={quote(str(v))}" for k, v in params.items())
            data = _get_json(f"{self.api}?{query}")
            pages = data.get("query", {}).get("pages", {})
            if not pages:
                self._save_state({})
                break

            for page in sorted(pages.values(), key=lambda p: p.get("pageid", 0)):
                text = (page.get("extract") or "").strip()
                if text:
                    yield Document(
                        source=self.name,
                        title=page.get("title", ""),
                        url=page.get("fullurl", ""),
                        text=text,
                    )
                    fetched += 1
                    if fetched >= limit:
                        break

            next_cont = data.get("continue", {})
            if not next_cont:
                self._save_state({})
                break
            cont = {k: v for k, v in next_cont.items() if k != "continue" and isinstance(v, str)}
            self._save_state(cont)
            time.sleep(self.delay)


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
        "description": "Wikipedia article extracts via MediaWiki API",
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
    path = root / f"{safe or 'document'}.txt"
    if not path.exists():
        path.write_text(doc.text.strip() + "\n", encoding="utf-8")
    return path
