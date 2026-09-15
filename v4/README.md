# V4 Web Training Pipeline

V4 turns the LLM project into a small, self-hosted web-data collection and training pipeline.

## Pipeline

```text
approved source
      ↓
robots.txt + domain rules + throttling
      ↓
Scrapy crawler / Wikimedia API
      ↓
main-content extraction
      ↓
quality filtering + URL removal + deduplication
      ↓
shuffle + corpus health analysis
      ↓
fast BPE tokenizer + cached encoding
      ↓
GPU training + live progress + checkpoints
```

The model-facing corpus contains cleaned prose only. URLs and source metadata are stored separately in `v4/data/sources.jsonl`, so the training text does not teach the model navigation URLs or scraper bookkeeping.

## Sources

V4 starts with three conservative source families:

- **Wikipedia** — retrieved through the MediaWiki API instead of scraping rendered HTML. Sampling is random and persisted by page ID, so repeated runs request new articles rather than returning to the same first pages.
- **Project Gutenberg** — public-domain books exposed through a normal web crawl.
- **arXiv** — scientific material exposed through a normal web crawl.

Common Crawl is intentionally kept as a future index-backed source rather than treating it like an ordinary polite crawl.

## Rules the crawler follows

The normal Scrapy sources use:

- `ROBOTSTXT_OBEY=True`
- AutoThrottle with conservative delays
- low per-domain concurrency
- timeouts and limited retries
- explicit domain allowlists
- a descriptive user agent
- persistent crawl state for pause/resume

This is deliberately not a scraper designed to bypass site restrictions.

## Install

From the repository root:

```bash
python3 -m pip install -r requirements.txt
```

## Collect Wikipedia

```bash
python3 v4/web_train.py --source wikipedia --pages 500 --only-collect
```

That requests **500 new random, non-repeating articles** relative to the saved Wikipedia page-ID state.

Press `Ctrl+C` at any time. Already-collected pages remain on disk.

## Analyze your corpus

```bash
python3 v4/analyze_corpus.py v4/data/train.txt
```

The analyzer reports document count, size, estimated words, largest-document share, URL-heavy documents, and common early terms. It warns when the corpus is small enough that topic skew or memorization is likely.

## Train existing data only

When you already have `v4/data/train.txt` and do not want to download anything else:

```bash
python3 v4/web_train.py --only-train --train-steps 1000 --device mps
```

V4 automatically chooses a model profile from corpus size:

```text
<10 MB    → tiny   4 layers / 4 heads / 256 embedding
10–100 MB → small  6 layers / 6 heads / 384 embedding
>100 MB   → base   8 layers / 8 heads / 512 embedding
```

Override it with `--profile tiny|small|base`.

Use `--rebuild-before-train` after adding raw documents.

## Collect and immediately train

```bash
python3 v4/web_train.py --source wikipedia --pages 500 --train-steps 2000 --device mps
```

## Gutenberg or arXiv

```bash
python3 v4/web_train.py --source gutenberg --pages 100 --only-collect
python3 v4/web_train.py --source arxiv --pages 100 --only-collect
```

For these Scrapy sources, `v4/data/jobs/<source>/` stores persistent crawl state. Start the same command again after stopping it to continue the crawl.

## Performance features

V3 now includes:

- heap-based BPE encoding
- bounded tokenizer-learning samples
- tokenizer compatibility checks
- encoded-token caching
- GPU-resident training/validation tensors
- vectorized random batches
- gradient accumulation
- warmup + cosine learning-rate decay
- true optimizer-state resume
- latest, best and interrupted checkpoints
- live progress with speed, loss, validation loss, LR and ETA
- deterministic seeds

`--tokenizer-bytes 0` uses the entire corpus for BPE learning. Otherwise, V4 can use a bounded sample for much faster tokenizer construction.

The project deliberately does **not** force `torch.compile` on MPS. Current PyTorch documentation supports MPS GPU training, but current PyTorch issue tracking still shows cases where compiled MPS training regresses or fails, so eager MPS is the safer default here.

## Data layout

```text
v4/data/
├── raw/              # cleaned per-document text
├── jobs/             # scraper/Wikipedia state
├── train.txt         # final model-facing corpus
└── sources.jsonl     # URL/title/source metadata kept OUT of training text
```

## Important limitation

More web data does not automatically make the model ChatGPT-level. Data quality, scale, model size, optimization, evaluation and instruction tuning still matter enormously.
