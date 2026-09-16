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
GPU training + live dashboard + checkpoints
      ↓
run metrics + graphs + local chat
```

The model-facing corpus contains cleaned prose only. URLs and source metadata are stored separately in `v4/data/sources.jsonl`.

## Sources

- **Wikipedia** — MediaWiki API, random non-repeating page IDs with persistent state.
- **Project Gutenberg** — public-domain books through a normal web crawl.
- **arXiv** — scientific material through a normal web crawl.

Common Crawl is intentionally kept as a future index-backed source rather than treating it like an ordinary polite crawl.

## Install

From the repository root:

```bash
python3 -m pip install -r requirements.txt
```

## Collect Wikipedia

```bash
python3 v4/web_train.py --source wikipedia --pages 500 --only-collect
```

Press `Ctrl+C` at any time. Already-collected pages remain on disk.

## Analyze your corpus

```bash
python3 v4/analyze_corpus.py v4/data/train.txt
```

The analyzer reports document count, size, estimated words, largest-document share and quality signals.

## Train existing data only

```bash
python3 v4/web_train.py --only-train --train-steps 1000 --device mps
```

V4 automatically chooses a profile from corpus size:

```text
<10 MB    → tiny   4 layers / 4 heads / 256 embedding
10–100 MB → small  6 layers / 6 heads / 384 embedding
>100 MB   → base   8 layers / 8 heads / 512 embedding
```

Override it with `--profile tiny|small|base`.

Useful training options:

```bash
--compile             try torch.compile where supported
--eval-batches 10     validation samples per evaluation
--early-stopping 5   stop after 5 evaluations without improvement
--no-plot             skip PNG graph generation
```

MPS intentionally stays on eager execution by default.

## Collect and immediately train

```bash
python3 v4/web_train.py --source wikipedia --pages 500 --train-steps 2000 --device mps
```

## Training artifacts

Each upgraded V3 run creates a directory similar to:

```text
v3/checkpoints/runs/2026-09-16_15-42-10/
├── config.json
├── metrics.jsonl
├── summary.json
├── loss.png
└── learning_rate.png
```

The trainer also maintains separate latest, best and interrupted checkpoints.

## Interactive local chat

After training:

```bash
python3 v3/chat.py --model v3/checkpoints/model.pt --device mps
```

Inside chat:

```text
You > What is 1 + 1?
LLM > ...

You > /clear
Context cleared.

You > /exit
```

Generation controls are available from the command line:

```bash
python3 v3/chat.py \
  --model v3/checkpoints/model.pt \
  --temperature 0.8 \
  --top-k 40 \
  --max-tokens 200
```

## Performance features

V3/V4 now include heap-based BPE encoding, tokenizer reuse, encoded-token caching, GPU-resident training tensors, vectorized batches, gradient accumulation, warmup + cosine LR decay, optimizer-state resume, gradient clipping, richer live progress, validation tracking, best-checkpoint fixes, optional compilation, early stopping and experiment metrics.

The project deliberately does not force `torch.compile` on MPS.

## Data layout

```text
v4/data/
├── raw/
├── jobs/
├── train.txt
└── sources.jsonl
```

Training data, checkpoints and caches are intended to stay local and are ignored by Git.

## Important limitation

More web data does not automatically make the model ChatGPT-level. Data quality, scale, model size, optimization, evaluation and instruction tuning still matter enormously.
