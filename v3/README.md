# Koda LLM V3

A self-trainable GPT-style language model built from scratch in Python + PyTorch.

**No API. No pretrained model required. No downloaded vocabulary.** You supply the text and the model learns its weights locally.

## Main tools

```text
data → tokenizer → GPT training → checkpoints
                         ↓
              Pro / Ultra training
                         ↓
        chat / benchmark / export / fine-tune
                         ↓
              local model registry
```

### Train

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

The Ultra trainer supports automatic batch tuning, memory-mapped token data, OOM recovery, gradient accumulation, validation, checkpoints, experiment metrics, and graphs.

### Chat

```bash
python v3/chat.py --checkpoint v3/checkpoints/model.pt --device mps
```

Chat uses the GPT attention KV cache for generation and supports `/clear`, `/stats`, `/settings`, and `/exit`.

### Instruction fine-tuning

Prepare JSONL with one object per line:

```json
{"prompt":"What is 2 + 2?","response":"4"}
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

Only response tokens contribute to the fine-tuning loss.

### Benchmark

```bash
python benchmark.py --checkpoint v3/checkpoints/model.pt --device mps
```

The benchmark includes small arithmetic, knowledge, logic and completion checks plus generation speed.

### Compare Koda with an MLX model

```bash
python compare.py \
  --koda v3/checkpoints/model.pt \
  --other models/qwen3-0.6b-base-mlx \
  --device mps
```

Install the MLX runtime separately with `pip install -U mlx-lm` when using an MLX model.

### Qwen MLX chat

```bash
python qwen_chat.py --model models/qwen3-0.6b-base-mlx
```

### Export

Compact FP16 PyTorch artifact:

```bash
python v3/export.py \
  --checkpoint v3/checkpoints/model.pt \
  --output v3/checkpoints/model.fp16.pt \
  --dtype fp16 --format pt
```

SafeTensors export:

```bash
python v3/export.py \
  --checkpoint v3/checkpoints/model.pt \
  --output v3/checkpoints/model.safetensors \
  --dtype fp16 --format safetensors
```

### CPU INT8

```bash
python v3/quantize.py \
  --checkpoint v3/checkpoints/model.pt \
  --output v3/checkpoints/model.int8.pt
```

Dynamic INT8 is intended for CPU inference; FP16/FP32 is the path for MPS.

### Model registry

```bash
python v3/registry.py add koda-main v3/checkpoints/model.pt --kind koda
python v3/registry.py add qwen-base models/qwen3-0.6b-base-mlx --kind mlx
python v3/registry.py list
python v3/registry.py info koda-main
```

### Balanced multi-source corpus

After collecting several V4 source families:

```bash
python v4/balance_corpus.py \
  --raw v4/data/raw \
  --out v4/data/train.txt \
  --total-docs 10000
```

It samples each source directory evenly instead of letting the largest source dominate the training mix.

## Files

- `model.py` - GPT decoder Transformer with optional KV-cache inference.
- `tokenizer.py` - trainable byte-level BPE tokenizer.
- `train.py` - original training path.
- `train_pro.py` - experiment-focused trainer.
- `train_ultra.py` - hardware-tuned trainer with memory mapping.
- `finetune.py` - instruction fine-tuning.
- `chat.py` - Koda local chat interface.
- `export.py` - compact FP16/SafeTensors export.
- `quantize.py` - CPU dynamic INT8 export.
- `registry.py` - local model registry.

All generated model weights, training data and caches are intended to remain local rather than being committed to Git.
