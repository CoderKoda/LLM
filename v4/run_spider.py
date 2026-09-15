"""Run the V4 Scrapy spider without requiring a scrapy.cfg project."""

from __future__ import annotations

import argparse

from scrapy.crawler import CrawlerProcess

from spider import TrainingSpider


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pages", type=int, default=100)
    parser.add_argument("--jobdir", required=True)
    args = parser.parse_args()

    settings = {
        "ROBOTSTXT_OBEY": True,
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 1.0,
        "AUTOTHROTTLE_MAX_DELAY": 10.0,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 2,
        "DOWNLOAD_TIMEOUT": 30,
        "RETRY_TIMES": 2,
        "USER_AGENT": "KodaLLM/4.0 (+https://github.com/CoderKoda/LLM)",
        "LOG_LEVEL": "INFO",
        "JOBDIR": args.jobdir,
    }

    process = CrawlerProcess(settings=settings)
    process.crawl(
        TrainingSpider,
        source=args.source,
        output=args.output,
        max_pages=args.pages,
    )
    process.start()


if __name__ == "__main__":
    main()
