# Koda LLM V3

A self-trainable GPT-style language model built from scratch in Python + PyTorch.

**No API. No pretrained model required. No downloaded vocabulary.** You provide training text, the model learns its weights locally, and you control the whole pipeline.

V3 contains the Transformer, trainable byte-level BPE tokenizer, high-performance training paths, local chat, fine-tuning, benchmarking, model export, quantization, model registration, and comparison tools.

---

# 1. Setup

Run these commands from the **root of the repository** (`LLM/`):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The project now requires modern PyTorch (`torch>=2.14`) so current Apple-Silicon MPS optimizations are available.

Check your MPS backend:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('MPS:', torch.backends.mps.is_available())"
```

On an Apple Silicon Mac, you normally want `MPS: True`.

---

# 2. The Koda pipeline

```text
raw text
   ↓
trainable BPE tokenizer
   ↓
token IDs / cached dataset
   ↓
GPU-optimized training batches
   ↓
GPT Transformer
   ↓
backpropagation + AdamW
   ↓
checkpoint
   ↓
chat / benchmark / fine-tune / export
```

The normal workflow is simply:

```text
train_ultra.py → chat.py → benchmark.py
```

---

# 3. The super-optimized trainer

## Recommended command for an Apple Silicon Mac

```bash
python v3/train_ultra.py \
  --data v4/data/train.txt \
  --steps 20000 \
  --device mps \
  --batch-size 0 \
  --max-batch-size 128 \
  --block-size 512 \
  --n-layer 8 --n-head 8 --n-embd 512 \
  --vocab-size 4096 \
  --memory-map auto \
  --precision auto \
  --token-device auto
```

The important thing is **not** to keep the old `--max-batch-size 32` limit. The new tuner is allowed to test larger batches and chooses the batch size that actually gives the highest measured throughput on your hardware.

## Optional compiler mode

Once the normal run works, try compiler optimization too:

```bash
python v3/train_ultra.py \
  --data v4/data/train.txt \
  --steps 20000 \
  --device mps \
  --batch-size 0 \
  --max-batch-size 128 \
  --block-size 512 \
  --n-layer 8 --n-head 8 --n-embd 512 \
  --vocab-size 4096 \
  --memory-map auto \
  --precision auto \
  --token-device auto \
  --compile
```

`--compile` is optional because compiler performance is hardware/model dependent. The trainer catches compilation failures and falls back to normal execution instead of killing the training run.

---

# 4. What Ultra optimizes

The Ultra trainer attacks the whole training pipeline rather than only the Transformer.

### Batch construction

Ultra uses vectorized NumPy indexing for CPU/memory-mapped batches instead of repeatedly building each sequence with Python slice loops.

### Token placement

Ultra can keep the token dataset on the accelerator when it is small enough:

```bash
--token-device auto
```

For large datasets it can keep the tokens memory-mapped on the CPU instead:

```bash
--token-device cpu
```

You can control the automatic accelerator token-cache limit with:

```bash
--token-device-max-mb 768
```

### Batch-size autotuning

`--batch-size 0` does more than find a batch that fits. It tests multiple batch sizes and measures actual training throughput, then selects the fastest stable one.

Example output:

```text
Auto-tuning batch size for maximum throughput...
  batch   1      900 tok/s  | 2.10 GiB
  batch   2    1,650 tok/s  | 2.30 GiB
  batch   4    3,000 tok/s  | 2.55 GiB
  batch   8    5,400 tok/s  | 3.00 GiB
  batch  16    8,700 tok/s  | 3.90 GiB
  batch  32   10,200 tok/s  | 5.20 GiB
  batch  64   10,900 tok/s  | 8.10 GiB
Selected batch size: 64 (10,900 tok/s during tuning)
```

The exact numbers depend on your machine.

### FP16 autocast

With:

```bash
--precision auto
```

Ultra uses FP16 autocast on MPS/CUDA and keeps an explicit FP32 option for stability testing.

Use:

```bash
--precision fp32
```

when you specifically want FP32.

### Less validation overhead

The default evaluation interval is now 1,000 steps rather than 250, with only 4 evaluation batches by default.

That reduces time spent repeatedly stopping the main training loop for validation.

### Less checkpoint overhead

The main checkpoint is normally saved every 1,000 steps, while the best-validation checkpoint is updated only when validation actually improves.

Ctrl+C still writes an interrupt checkpoint immediately.

### Compiler optimization

```bash
--compile
```

asks PyTorch to try:

```python
torch.compile(model, backend="inductor", mode="max-autotune")
```

The trainer reports whether compilation was enabled or safely fell back to eager execution.

### Device-aware timing

The trainer synchronizes MPS/CUDA when it needs an accurate benchmark or validation result, rather than forcing a synchronization around every training operation.

---

# 5. Understanding the main training options

## `--data`

The training text:

```bash
--data v4/data/train.txt
```

## `--steps`

Number of optimizer updates:

```bash
--steps 20000
```

## `--batch-size`

```bash
--batch-size 0
```

`0` means automatic throughput tuning.

A fixed value such as `32` disables batch autotuning.

## `--max-batch-size`

The largest batch size the tuner is allowed to test:

```bash
--max-batch-size 128
```

## `--block-size`

Maximum context length:

```bash
--block-size 512
```

Larger contexts increase compute cost substantially, so keep 512 unless you actually need more context.

## `--n-layer`, `--n-head`, `--n-embd`

These control model size:

```bash
--n-layer 8
--n-head 8
--n-embd 512
```

Increasing them makes each step more expensive.

## `--vocab-size`

Tokenizer vocabulary size:

```bash
--vocab-size 4096
```

## `--precision`

Recommended:

```bash
--precision auto
```

Force FP16:

```bash
--precision fp16
```

Force FP32:

```bash
--precision fp32
```

## `--memory-map`

```bash
--memory-map auto
```

Recommended. Lets the trainer use the cached token representation when appropriate.

## `--token-device`

```bash
--token-device auto
```

Recommended. Smaller token datasets can stay on the accelerator; larger ones remain memory-mapped on CPU.

## `--token-device-max-mb`

Maximum token-cache size for automatic accelerator placement:

```bash
--token-device-max-mb 768
```

## `--eval-every`

Validation frequency:

```bash
--eval-every 1000
```

## `--save-every`

Main checkpoint frequency:

```bash
--save-every 1000
```

## `--grad-accum`

Simulate a larger effective batch without making one enormous batch fit in memory:

```bash
--batch-size 32 --grad-accum 2
```

The effective batch is 64 sequences.

## `--compile`

Try PyTorch compilation:

```bash
--compile
```

Benchmark it against the normal eager mode on your exact machine.

---

# 6. A quick speed test before a 20,000-step run

Do not immediately start the giant run.

First run:

```bash
python v3/train_ultra.py \
  --data v4/data/train.txt \
  --steps 20 \
  --device mps \
  --batch-size 0 \
  --max-batch-size 128 \
  --block-size 512 \
  --n-layer 8 --n-head 8 --n-embd 512 \
  --vocab-size 4096 \
  --memory-map auto \
  --precision auto \
  --token-device auto
```

Then repeat with:

```bash
--compile
```

Compare the reported `tok/s`. Keep the faster configuration for the full run.

---

# 7. Resuming training

Press:

```text
Ctrl+C
```

The trainer saves:

```text
v3/checkpoints/model.interrupted.pt
```

Resume with:

```bash
python v3/train_ultra.py \
  --data v4/data/train.txt \
  --resume v3/checkpoints/model.interrupted.pt \
  --steps 20000 \
  --device mps
```

Important: when using `--resume`, `--steps` means **additional training steps**.

---

# 8. Chat with Koda

```bash
python v3/chat.py \
  --checkpoint v3/checkpoints/model.pt \
  --device mps
```

Inside chat:

```text
/clear
/stats
/settings
/exit
```

Command-line generation settings can also be supplied:

```bash
python v3/chat.py \
  --checkpoint v3/checkpoints/model.pt \
  --device mps \
  --temperature 0.8 \
  --top-k 40 \
  --max-tokens 200
```

The normal cached generation path reuses the attention KV cache so the existing prefix does not need to be recomputed at every generated token.

---

# 9. One-shot generation

```bash
python v3/generate.py \
  --checkpoint v3/checkpoints/model.pt \
  --prompt "Hello" \
  --tokens 200 \
  --temperature 0.8 \
  --top-k 40 \
  --device mps
```

---

# 10. Instruction fine-tuning

JSONL format:

```json
{"prompt":"What is 2 + 2?","response":"4"}
{"prompt":"Say hello.","response":"Hello!"}
```

Then:

```bash
python v3/finetune.py \
  --checkpoint v3/checkpoints/model.pt \
  --data examples/instructions.jsonl \
  --out v3/checkpoints/instruct.pt \
  --steps 1000 \
  --device mps
```

Fine-tuning is for teaching response behavior after the main pretraining stage.

---

# 11. Benchmarking

```bash
python benchmark.py \
  --checkpoint v3/checkpoints/model.pt \
  --device mps
```

Use this whenever you change attention, caching, quantization, generation, or hardware settings.

For training optimization, compare the `tok/s` reported by short Ultra runs with identical settings.

---

# 12. Compare Koda with Qwen MLX

```bash
python compare.py \
  --koda v3/checkpoints/model.pt \
  --other models/qwen3-0.6b-base-mlx \
  --device mps
```

This is comparison only; the external model does not replace Koda's from-scratch model.

Install the MLX runtime when needed:

```bash
pip install -U mlx-lm
```

Chat with it separately:

```bash
python qwen_chat.py --model models/qwen3-0.6b-base-mlx
```

---

# 13. Export

FP16 PyTorch artifact:

```bash
python v3/export.py \
  --checkpoint v3/checkpoints/model.pt \
  --output v3/checkpoints/model.fp16.pt \
  --dtype fp16 \
  --format pt
```

SafeTensors:

```bash
python v3/export.py \
  --checkpoint v3/checkpoints/model.pt \
  --output v3/checkpoints/model.safetensors \
  --dtype fp16 \
  --format safetensors
```

---

# 14. CPU INT8

```bash
python v3/quantize.py \
  --checkpoint v3/checkpoints/model.pt \
  --output v3/checkpoints/model.int8.pt
```

This is intended for CPU inference, not the normal MPS training path.

---

# 15. Local model registry

Add a Koda model:

```bash
python v3/registry.py add koda-main v3/checkpoints/model.pt --kind koda
```

Add the Qwen comparison model:

```bash
python v3/registry.py add qwen-base models/qwen3-0.6b-base-mlx --kind mlx
```

List models:

```bash
python v3/registry.py list
```

Inspect one:

```bash
python v3/registry.py info koda-main
```

---

# 16. Tokenizer

V3 uses a trainable byte-level BPE tokenizer. It learns byte combinations from your training data and turns text into integer token IDs.

The training scripts handle tokenizer creation and caching automatically.

---

# 17. Balanced data

If several V4 source families are collected, balance them with:

```bash
python v4/balance_corpus.py \
  --raw v4/data/raw \
  --out v4/data/train.txt \
  --total-docs 10000
```

This samples from the source directories instead of letting the largest directory dominate the final corpus.

---

# 18. Other training scripts

### Original trainer

```bash
python v3/train.py --data v4/data/train.txt --steps 5000 --device mps
```

Use this for the simpler training path.

### Pro trainer

```bash
python v3/train_pro.py --data v4/data/train.txt --steps 5000 --device mps
```

Use this for experiment-focused metrics, plots, early stopping, and run management.

For maximum training throughput, use `train_ultra.py`.

---

# 19. Typical outputs

```text
v3/checkpoints/model.pt
v3/checkpoints/model.best.pt
v3/checkpoints/model.interrupted.pt
v3/checkpoints/tokenizer.json
v3/checkpoints/encoded.npy
v3/checkpoints/encoded.meta.json
v3/checkpoints/runs/<timestamp>/config.json
v3/checkpoints/runs/<timestamp>/metrics.jsonl
v3/checkpoints/runs/<timestamp>/loss.png
```

Local model weights, training data, and generated caches are intended to stay out of Git.

---

# 20. Recommended workflow

### Test the full pipeline

```bash
python v3/train_ultra.py \
  --data v4/data/train.txt \
  --steps 20 \
  --device mps \
  --batch-size 0 \
  --max-batch-size 128 \
  --block-size 512 \
  --n-layer 8 --n-head 8 --n-embd 512 \
  --vocab-size 4096 \
  --memory-map auto \
  --precision auto \
  --token-device auto
```

### Compare compiler mode

Run the same test again with:

```bash
--compile
```

Use the faster measured configuration for the long run.

### Full run

```bash
python v3/train_ultra.py \
  --data v4/data/train.txt \
  --steps 20000 \
  --device mps \
  --batch-size 0 \
  --max-batch-size 128 \
  --block-size 512 \
  --n-layer 8 --n-head 8 --n-embd 512 \
  --vocab-size 4096 \
  --memory-map auto \
  --precision auto \
  --token-device auto
```

Then:

```bash
python v3/chat.py --checkpoint v3/checkpoints/model.pt --device mps
python benchmark.py --checkpoint v3/checkpoints/model.pt --device mps
```

---

# 21. Which command should I use?

| Goal | Command |
|---|---|
| Maximum-throughput training | `python v3/train_ultra.py` |
| Simple training | `python v3/train.py` |
| Experiment-focused training | `python v3/train_pro.py` |
| Interactive Koda chat | `python v3/chat.py` |
| One-shot generation | `python v3/generate.py` |
| Instruction fine-tuning | `python v3/finetune.py` |
| Benchmark speed/capabilities | `python benchmark.py` |
| Compare Koda with another local model | `python compare.py` |
| Chat with Qwen MLX | `python qwen_chat.py` |
| FP16/SafeTensors export | `python v3/export.py` |
| CPU INT8 export | `python v3/quantize.py` |
| Manage model names | `python v3/registry.py` |
| Balance collected data | `python v4/balance_corpus.py` |

---

# 22. Performance mindset

The key metric for training optimization is:

```text
tok/s
```

Do not assume an optimization is faster because the code looks more sophisticated. Keep the dataset, model, context length, tokenizer, and training settings identical, then compare throughput.

Koda's goal is a local training stack that uses the available hardware efficiently while remaining understandable and hackable.
