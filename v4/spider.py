"""Conservative Scrapy spider used by V4.

Scrapy handles scheduling, robots.txt, retries, throttling and persistent crawl
state. Trafilatura handles page-to-prose extraction in cleaner.py.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import scrapy
from scrapy.linkextractors import LinkExtractor

from cleaner import clean_text


SOURCE_RULES = {
    "gutenberg": {
        "allowed_domains": ["gutenberg.org", "www.gutenberg.org"],
        "start_urls": ["https://www.gutenberg.org/"],
    },
    "arxiv": {
        "allowed_domains": ["arxiv.org", "export.arxiv.org"],
        "start_urls": ["https://arxiv.org/"],
    },
}


class TrainingSpider(scrapy.Spider):
    name = "training_spider"

    custom_settings = {
        "ROBOTSTXT_OBEY": True,
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 1.0,
        "AUTOTHROTTLE_MAX_DELAY": 10.0,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 2,
        "DOWNLOAD_TIMEOUT": 30,
        "RETRY_TIMES": 2,
        "USER_AGENT": "KodaLLM/4.0 (+https://github.com/CoderKoda/LLM)",
        "LOG_LEVEL": "INFO",
    }

    def __init__(self, source: str, output: str = "v4/data/raw", max_pages: int = 100, metadata: str | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if source not in SOURCE_RULES:
            raise ValueError(f"Unknown crawl source: {source}")
        self.source = source
        self.max_pages = int(max_pages)
        self.seen_pages = 0
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.metadata = Path(metadata) if metadata else self.output.parent.parent / "sources.jsonl"
        self.metadata.parent.mkdir(parents=True, exist_ok=True)
        rule = SOURCE_RULES[source]
        self.allowed_domains = rule["allowed_domains"]
        self.start_urls = rule["start_urls"]
        self.link_extractor = LinkExtractor(allow_domains=self.allowed_domains)

    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(url, callback=self.parse, dont_filter=True)

    def parse(self, response: scrapy.http.Response):
        if self.seen_pages >= self.max_pages:
            return
        self.seen_pages += 1

        content_type = response.headers.get(b"Content-Type", b"").decode("latin-1").lower()
        if "html" not in content_type:
            return

        text = clean_text(response.text)
        if len(text) < 500:
            return

        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        path = self.output / f"{digest}.txt"
        if not path.exists():
            path.write_text(text + "\n", encoding="utf-8")

        title = " ".join(response.css("title::text").getall()).strip()
        with self.metadata.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "source": self.source,
                "title": title,
                "url": response.url,
                "file": str(path),
            }, ensure_ascii=False) + "\n")

        if self.seen_pages < self.max_pages:
            for link in self.link_extractor.extract_links(response):
                yield scrapy.Request(link.url, callback=self.parse)
