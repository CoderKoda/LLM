# Koda LLM V3

A self-trainable GPT-style language model built from scratch in Python + PyTorch.

**No API. No pretrained model required. No downloaded vocabulary.** You provide training text, the model learns its weights locally, and you control the whole pipeline.

V3 is the main version of Koda LLM. It contains the Transformer, trainable BPE tokenizer, fast training paths, local chat, fine-tuning, benchmarking, model export, quantization, model registration, and comparison tools.

## 1. First-time setup

Run these commands from the **root of the repository** (`LLM/`):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS with an Apple Silicon Mac, PyTorch can use the MPS GPU backend. Check it with:

```bash
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

You normally want this to print `MPS: True`.

### Where should commands be run?

If your terminal is inside the repository, your prompt should be somewhere like:

```text
.../LLM $
```

Then commands such as `python v3/train_ultra.py` will work exactly as written below.

If you are inside another directory, either `cd` back to the repository first or adjust the paths.

---

# 2. The basic workflow

The whole project is easiest to understand as this pipeline:

```text
raw text
   ↓
trainable BPE tokenizer
   ↓
token IDs
   ↓
GPT Transformer
   ↓
training / backpropagation
   ↓
checkpoint (.pt)
   ↓
chat / generation / benchmark / fine-tuning / export
```

You do **not** have to use every tool every time.

For a normal training run, the most important commands are:

```text
train_ultra.py → chat.py → benchmark.py
```

The other tools are for specific jobs.

---

# 3. Training the model

## Recommended: Ultra trainer

For your main V3 model, use:

```bash
python v3/train_ultra.py \
  --data v4/data/train.txt \
  --steps 20000 \
  --device mps \
  --batch-size 0 \
  --max-batch-size 32 \
  --block-size 512 \
  --n-layer 8 --n-head 8 --n-embd 512 \
  --vocab-size 4096 \
  --memory-map auto
```

### What this command means

`--data v4/data/train.txt`

The text file the model learns from.

`--steps 20000`

Run 20,000 optimizer steps. More steps generally means more training, although the useful amount depends on the size and quality of your dataset.

`--device mps`

Use Apple's Metal GPU backend on Apple Silicon. Other options are `cpu`, `cuda`, and `auto`.

`--batch-size 0`

Automatically find a batch size that fits in memory.

`--max-batch-size 32`

Do not let automatic tuning go above 32 samples per batch.

`--block-size 512`

The maximum context length used for each training example. A value of 512 means the model trains on sequences up to 512 tokens at a time.

`--n-layer 8`

Use 8 Transformer blocks.

`--n-head 8`

Use 8 attention heads.

`--n-embd 512`

Use a 512-dimensional hidden representation.

`--vocab-size 4096`

Train a tokenizer with up to 4,096 tokens.

`--memory-map auto`

Let the trainer decide whether token data should be memory-mapped from disk instead of loading the whole token array into RAM.

### What Ultra does for you

The Ultra trainer includes:

- automatic batch-size tuning
- memory-mapped token data
- automatic OOM recovery
- validation
- gradient accumulation
- gradient clipping
- learning-rate scheduling
- checkpoint saving
- resumable training
- training metrics
- progress information
- optional graphs

### A smaller test run

Before committing your Mac to a huge run, test the pipeline with something like:

```bash
python v3/train_ultra.py \
  --data v4/data/train.txt \
  --steps 100 \
  --device mps \
  --batch-size 0 \
  --max-batch-size 16
```

This is useful for checking that the dataset, tokenizer, model and checkpoint system all work before starting a long run.

---

# 4. Other training scripts

## `train.py` - original trainer

```bash
python v3/train.py --data v4/data/train.txt --steps 5000 --device mps
```

This is the simpler, original V3 training path.

Use it when you want a straightforward training loop without the larger Ultra feature set.

## `train_pro.py` - experiment-focused trainer

```bash
python v3/train_pro.py --data v4/data/train.txt --steps 5000 --device mps
```

The Pro trainer is designed for more structured experiments. It provides experiment tracking, metrics, graphs, safer best-checkpoint handling, early stopping, and resumable runs.

For serious long runs, Ultra is generally the more hardware-focused choice.

---

# 5. Pausing and resuming training

You can safely stop a training run with:

```text
Ctrl+C
```

The trainer saves an interrupted checkpoint so you can continue later.

A typical resume command is:

```bash
python v3/train_ultra.py \
  --data v4/data/train.txt \
  --resume v3/checkpoints/model.interrupted.pt \
  --steps 20000 \
  --device mps
```

### Important

When resuming, `--steps` is treated as additional training steps by the Ultra trainer. It is **not** necessarily the final absolute step number.

Always check the printed progress at startup so you know where the run is continuing from.

---

# 6. Chat with your trained model

After training, start the local chat interface:

```bash
python v3/chat.py --checkpoint v3/checkpoints/model.pt --device mps
```

### What it does

This loads your saved Koda model and lets you type prompts directly into the terminal.

The chat interface uses the model's attention KV cache to avoid recomputing earlier attention keys and values during normal generation.

### Chat commands

Inside chat:

```text
/clear
```

Clear the current conversation context.

```text
/stats
```

Show model and generation statistics.

```text
/settings
```

Show the current generation settings.

```text
/exit
```

Leave the chat program.

### Generation settings

You can also control generation from the command line. For example:

```bash
python v3/chat.py \
  --checkpoint v3/checkpoints/model.pt \
  --device mps \
  --temperature 0.8 \
  --top-k 40 \
  --max-tokens 200
```

`temperature` controls randomness. Lower values usually make output more conservative; higher values make it more varied.

`top-k` limits sampling to the most likely K tokens.

`max-tokens` controls how many new tokens can be generated for a response.

---

# 7. One-shot text generation

`v3/generate.py` is useful when you want a single generation rather than an interactive chat session.

Example:

```bash
python v3/generate.py \
  --checkpoint v3/checkpoints/model.pt \
  --prompt "Hello" \
  --tokens 200 \
  --temperature 0.8 \
  --top-k 40 \
  --device mps
```

Use this when you are testing prompts or scripting generation.

---

# 8. Instruction fine-tuning

Pretraining teaches the model general patterns from text. Fine-tuning can then teach it to respond to specific instruction/response examples.

Your data is JSONL: one JSON object per line.

Example:

```json
{"prompt":"What is 2 + 2?","response":"4"}
{"prompt":"Say hello.","response":"Hello!"}
```

Then run:

```bash
python v3/finetune.py \
  --checkpoint v3/checkpoints/model.pt \
  --data examples/instructions.jsonl \
  --out v3/checkpoints/instruct.pt \
  --steps 1000 \
  --device mps
```

The fine-tuner trains primarily on the response portion rather than treating the prompt and response as equally important targets.

### When to use this

Use fine-tuning **after** you have a pretrained-on-your-corpus checkpoint. It is not a replacement for the main pretraining run.

---

# 9. Benchmark your model

Run:

```bash
python benchmark.py \
  --checkpoint v3/checkpoints/model.pt \
  --device mps
```

The benchmark performs small tests involving areas such as:

- arithmetic
- factual/knowledge-style prompts
- logic
- text completion
- generation speed

It also gives you a repeatable way to measure inference performance.

### Why benchmark?

If you change the model, tokenizer, attention implementation, cache, quantization or generation code, benchmark before and after the change.

That lets you see whether an optimization actually made the model faster rather than just assuming it did.

---

# 10. Compare Koda with another local model

Koda can be compared with an MLX model such as Qwen.

Example:

```bash
python compare.py \
  --koda v3/checkpoints/model.pt \
  --other models/qwen3-0.6b-base-mlx \
  --device mps
```

This is for comparison only. It does **not** replace Koda's from-scratch model.

For the MLX side, install the runtime separately if required:

```bash
pip install -U mlx-lm
```

The external model remains a separate comparison model.

---

# 11. Chat with the Qwen MLX comparison model

To launch the MLX comparison model:

```bash
python qwen_chat.py --model models/qwen3-0.6b-base-mlx
```

This is useful when you want to compare the experience of a pretrained model against your own locally trained Koda model.

This command does **not** train Qwen and does **not** change Koda's weights.

---

# 12. Export a compact FP16 model

The export tool converts a checkpoint into a more compact inference artifact.

## FP16 PyTorch file

```bash
python v3/export.py \
  --checkpoint v3/checkpoints/model.pt \
  --output v3/checkpoints/model.fp16.pt \
  --dtype fp16 \
  --format pt
```

`fp16` means 16-bit floating point weights, which can reduce the storage size compared with FP32.

## SafeTensors

```bash
python v3/export.py \
  --checkpoint v3/checkpoints/model.pt \
  --output v3/checkpoints/model.safetensors \
  --dtype fp16 \
  --format safetensors
```

SafeTensors is a weight-file format designed for storing tensors safely and efficiently.

### Important

Exporting creates another representation of the model. It does not replace your original checkpoint unless you explicitly overwrite it.

---

# 13. CPU INT8 quantization

For CPU inference, you can create a dynamically quantized checkpoint:

```bash
python v3/quantize.py \
  --checkpoint v3/checkpoints/model.pt \
  --output v3/checkpoints/model.int8.pt
```

INT8 uses 8-bit integer representations for supported layers and is intended primarily for CPU inference.

For your Apple Silicon MPS workflow, the normal FP16/FP32 path is the relevant one. Do not assume the CPU INT8 artifact will be faster on MPS.

---

# 14. Local model registry

The registry gives your local models names so you do not have to remember every path.

## Add a Koda model

```bash
python v3/registry.py add koda-main v3/checkpoints/model.pt --kind koda
```

## Add the Qwen comparison model

```bash
python v3/registry.py add qwen-base models/qwen3-0.6b-base-mlx --kind mlx
```

## List registered models

```bash
python v3/registry.py list
```

## Inspect one model

```bash
python v3/registry.py info koda-main
```

Think of the registry as a small local catalogue of models and their file locations.

---

# 15. Tokenizer

V3 uses a **trainable byte-level BPE tokenizer**.

This is different from the earliest versions of the project, which used a much simpler byte/character-style tokenizer.

The tokenizer learns useful byte combinations from your training data and then turns text into integer token IDs for the Transformer.

Most users do not need to run the tokenizer manually because the training scripts handle the tokenizer workflow for you.

Useful tokenizer-related files and cached artifacts are normally kept alongside your local training outputs rather than committed to GitHub.

---

# 16. Build a balanced training corpus

If you collect several source families with V4, one source can otherwise dominate the final corpus simply because it contains more documents.

Use:

```bash
python v4/balance_corpus.py \
  --raw v4/data/raw \
  --out v4/data/train.txt \
  --total-docs 10000
```

### What it does

It samples from the different source directories so that the final training mix is more balanced.

For example:

```text
v4/data/raw/
├── wikipedia/
├── gutenberg/
└── arxiv/
```

Instead of taking almost everything from whichever folder is largest, the balancing step creates a more even training mixture.

---

# 17. Recommended workflow for Koda

A typical full workflow looks like this:

### Step 1 - Get your data

Create or collect your training corpus.

Example:

```text
v4/data/train.txt
```

### Step 2 - Train

Start with a small test run first:

```bash
python v3/train_ultra.py \
  --data v4/data/train.txt \
  --steps 100 \
  --device mps \
  --batch-size 0
```

Then start a serious run with your chosen architecture.

### Step 3 - Check the checkpoint

Your training output will produce checkpoints under the configured checkpoint/run directory.

### Step 4 - Chat

```bash
python v3/chat.py --checkpoint v3/checkpoints/model.pt --device mps
```

### Step 5 - Benchmark

```bash
python benchmark.py --checkpoint v3/checkpoints/model.pt --device mps
```

### Step 6 - Fine-tune if needed

Use `finetune.py` with instruction/response data.

### Step 7 - Export if needed

Use `export.py` for FP16 or SafeTensors, or `quantize.py` for CPU INT8.

---

# 18. Which command should I use?

| Goal | Command |
|---|---|
| Main training run | `python v3/train_ultra.py` |
| Simpler training | `python v3/train.py` |
| Experiment-heavy training | `python v3/train_pro.py` |
| Interactive Koda chat | `python v3/chat.py` |
| One-shot generation | `python v3/generate.py` |
| Instruction fine-tuning | `python v3/finetune.py` |
| Measure speed/capabilities | `python benchmark.py` |
| Compare with another model | `python compare.py` |
| Chat with Qwen MLX | `python qwen_chat.py` |
| FP16/SafeTensors export | `python v3/export.py` |
| CPU INT8 export | `python v3/quantize.py` |
| Manage local model names | `python v3/registry.py` |
| Balance collected datasets | `python v4/balance_corpus.py` |

---

# 19. Understanding the most important training options

These are the options you are most likely to change in `train_ultra.py`.

### `--steps`

How long to train.

```bash
--steps 20000
```

More steps means more optimizer updates.

### `--batch-size`

How many training sequences are processed together.

```bash
--batch-size 32
```

Or let Ultra choose:

```bash
--batch-size 0
```

### `--grad-accum`

Accumulate gradients over multiple batches before making an optimizer update.

This can give you a larger effective batch without requiring the entire batch to fit in memory at once.

### `--block-size`

Context length.

```bash
--block-size 512
```

Larger context uses more memory and computation.

### `--n-layer`

Number of Transformer blocks.

More layers generally make the model larger and more computationally expensive.

### `--n-head`

Number of attention heads.

### `--n-embd`

Hidden/embedding width.

Larger values make the model substantially larger.

### `--vocab-size`

Maximum tokenizer vocabulary size.

### `--lr`

Initial learning rate.

### `--min-lr`

Minimum learning rate used by the scheduler.

### `--warmup-steps`

Number of initial steps used to ramp the learning rate up gradually.

### `--weight-decay`

Regularization applied by the optimizer.

### `--eval-every`

How often validation is performed.

### `--early-stopping`

Stop when validation stops improving for the configured patience.

### `--resume`

Load a saved checkpoint and continue training.

### `--memory-map`

Controls whether large token arrays are memory-mapped instead of fully loaded into RAM.

### `--precision`

Choose automatic precision, FP32 or FP16 where supported by the training path.

---

# 20. Checkpoint files

Training can create several different checkpoint types.

Typical examples include:

```text
model.pt
model.best.pt
model.interrupted.pt
```

The exact files depend on which trainer and output/run directory you use.

**Do not delete your best checkpoint just because another checkpoint is newer.** A newer model is not automatically a better model; validation loss and benchmark results matter.

---

# 21. Local vs GitHub files

Large files are intentionally kept local.

That includes things such as:

- model weights
- checkpoints
- generated corpora
- token caches
- large dataset files
- local experiment outputs

This keeps GitHub practical and prevents enormous model artifacts from being committed accidentally.

Your source code and documentation belong in GitHub; your large training artifacts normally stay on your Mac.

---

# 22. Troubleshooting

## `No module named torch`

Activate the virtual environment and install requirements:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## `MPS: False`

Check that you are running a PyTorch build with MPS support on Apple Silicon.

## Out-of-memory errors

Try a smaller batch size, context length, or model:

```bash
--batch-size 8
--block-size 256
```

Or let Ultra auto-tune the batch size:

```bash
--batch-size 0
```

## Training is unexpectedly slow

Run the benchmark and watch the training progress metrics. Compare different batch sizes and model settings rather than changing many things at once.

## A long run needs to stop

Use:

```text
Ctrl+C
```

Then resume from the interrupted checkpoint rather than throwing away the run.

---

# 23. Project files

| File | Purpose |
|---|---|
| `model.py` | GPT decoder Transformer and optional KV-cache inference |
| `tokenizer.py` | Trainable byte-level BPE tokenizer |
| `train.py` | Original V3 trainer |
| `train_pro.py` | Experiment-focused trainer |
| `train_ultra.py` | Hardware-tuned trainer with memory mapping and OOM recovery |
| `finetune.py` | Instruction fine-tuning |
| `chat.py` | Interactive Koda chat |
| `generate.py` | One-shot generation |
| `export.py` | FP16 and SafeTensors export |
| `quantize.py` | CPU dynamic INT8 export |
| `registry.py` | Local model registry |
| `ui.py` | Shared terminal UI helpers |

Root-level tools such as `benchmark.py`, `compare.py`, and `qwen_chat.py` provide benchmarking and model-comparison functionality.

---

# 24. The important idea

Koda is meant to be understandable and modifiable.

You are not just downloading a chatbot. The pipeline is yours:

```text
YOUR DATA
   ↓
YOUR TOKENIZER
   ↓
YOUR MODEL ARCHITECTURE
   ↓
YOUR TRAINING RUN
   ↓
YOUR WEIGHTS
   ↓
YOUR INFERENCE
```

That is what makes this a self-trainable LLM project rather than simply a wrapper around somebody else's API.
