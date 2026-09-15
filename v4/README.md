# V4 Web Training Pipeline

V4 turns the LLM project into a small, self-hosted web-data collection pipeline.

## What it does

```text
approved source
      ↓
robots.txt + domain rules + throttling
      ↓
Scrapy crawler / Wikimedia API
      ↓
Trafilatura extraction
      ↓
quality filtering + URL removal + deduplication
      ↓
clean training corpus
      ↓
V3 tokenizer + GPT training
```

The model-facing corpus contains cleaned prose only. URLs and source metadata are stored separately in `v4/data/sources.jsonl`, so the training text does not teach the model navigation URLs or scraper bookkeeping.

## Sources

V4 starts with three conservative source families:

- **Wikipedia** — retrieved through the MediaWiki API instead of scraping rendered HTML.
- **Project Gutenberg** — public-domain books exposed through a normal web crawl.
- **arXiv** — scientific material exposed through a normal web crawl.

Common Crawl is intentionally listed as a future index-backed source rather than pretending that downloading arbitrary web pages from it is the same thing as a normal polite crawl. Common Crawl is an open web research corpus with its own access and terms. See the project documentation before enabling it for large-scale collection.

## Rules the crawler follows

The normal Scrapy sources use:

- `ROBOTSTXT_OBEY=True`
- AutoThrottle with conservative delays
- a low per-domain concurrency limit
- request timeouts and limited retries
- explicit domain allowlists
- a descriptive user agent
- persistent Scrapy `JOBDIR` state for pause/resume

This is deliberately not a scraper designed to bypass site restrictions.

## Install

From the repository root:

```bash
pip install -r requirements.txt
```

## Collect Wikipedia

```bash
python v4/web_train.py --source wikipedia --pages 500 --only-collect
```

Press `Ctrl+C` at any time. Already-collected pages remain on disk.

Then rebuild the corpus:

```bash
python v4/web_train.py --build-only
```

To collect and immediately train:

```bash
python v4/web_train.py --source wikipedia --pages 500 --train-steps 2000
```

## Crawl Gutenberg or arXiv

```bash
python v4/web_train.py --source gutenberg --pages 100 --only-collect
python v4/web_train.py --source arxiv --pages 100 --only-collect
```

For these Scrapy sources, `v4/data/jobs/<source>/` stores the persistent crawl state. Start the same command again after stopping it to continue the crawl.

## Where the data goes

```text
v4/data/
├── raw/              # cleaned per-document text
├── jobs/             # Scrapy pause/resume state
├── train.txt         # final text seen by the tokenizer
└── sources.jsonl     # URL/title/source metadata kept OUT of training text
```

## Cleaning

Trafilatura strips common web boilerplate such as scripts, styles, navigation and footers, then extracts the main content. V4 additionally removes obvious URLs, filters short/UI-like fragments, and deduplicates documents before writing the training corpus.

## Important limitation

Web collection gives the model more training text; it does not automatically make the model ChatGPT-level. Data quality, scale, model size, training compute, tokenizer design and evaluation still matter enormously.
