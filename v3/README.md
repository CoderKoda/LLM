# Koda LLM V3

A self-trainable GPT-style language model built from scratch in Python + PyTorch.

**No API. No pretrained model. No downloaded vocabulary.** You supply the text and the model learns its weights locally.

## Pipeline

```text
your UTF-8 text
      ↓
trainable byte-level BPE tokenizer
      ↓
subword token IDs
      ↓
GPT decoder Transformer
      ↓
next-token prediction loss
      ↓
backpropagation + AdamW
      ↓
checkpoint
      ↓
generation / terminal chat
```

## Files

- `tokenizer.py` - learns BPE merges directly from your corpus and saves them as JSON.
- `model.py` - decoder-only Transformer with causal self-attention, residual connections, layer normalization, learned positions and tied embeddings.
- `train.py` - original trainer.
- `train_pro.py` - experiment-oriented trainer with richer progress, run history, plots, early stopping and resumable checkpoints.
- `train_ultra.py` - performance-oriented trainer with automatic batch-size tuning and disk-backed token datasets.
- `generate.py` - generate text from a checkpoint.
- `chat.py` - simple terminal conversation interface.

## 1. Install

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Give it your own data

Create `data.txt` in the repository root. Put UTF-8 text in it: writing, documentation, code, stories, conversations, or other data you have permission to use.

More clean text generally matters more than simply increasing model size.

## 3. Train normally

```bash
python v3/train.py --data data.txt --steps 500
```

## 4. Use the production trainer

```bash
python v3/train_pro.py --data data.txt --steps 5000
```

This keeps run metadata under the checkpoint directory and records training/validation metrics plus plots.

## 5. Use the ultra trainer

For larger datasets, use:

```bash
python v3/train_ultra.py --data data.txt --steps 5000 --device mps
```

With `--batch-size 0` (the default), the trainer probes increasing batch sizes and selects the largest size that completes a forward/backward pass. Set `--max-batch-size` to cap the search.

For large token datasets, `--memory-map auto` automatically switches to a disk-backed NumPy `.npy` token file at the configured threshold. You can force the behavior with `--memory-map always` or `--memory-map never`.

The ultra trainer also:

- reports device memory during batch probing and evaluation
- automatically retries with a smaller batch after an out-of-memory failure
- keeps token arrays on disk when memory mapping is enabled instead of copying the entire dataset to GPU memory
- preserves checkpoint resume, validation, cosine learning-rate decay, gradient clipping and best/interrupted checkpoints

On Apple Silicon, this is designed for PyTorch MPS. PyTorch's current MPS backend provides GPU execution on macOS, and exposes MPS memory accounting APIs that the trainer can use for diagnostics. citeturn541249search0turn541249search2

## 6. Continue training

```bash
python v3/train_pro.py --data data.txt --resume v3/checkpoints/model.pt --steps 10000
```

Or resume with the performance trainer:

```bash
python v3/train_ultra.py --data data.txt --resume v3/checkpoints/model.pt --steps 10000 --device mps
```

## 7. Generate

```bash
python v3/generate.py --checkpoint v3/checkpoints/model.pt --prompt "Hello" --tokens 200
```

Adjust creativity with `--temperature` and `--top-k`.

## 8. Chat

```bash
python v3/chat.py --checkpoint v3/checkpoints/model.pt
```

Type `/exit` to leave.

## What it is — and isn't

This is a real causal Transformer language model. It can genuinely learn patterns from your dataset, but a small model trained on a small corpus will not have ChatGPT-level knowledge or reasoning. Building something much stronger requires substantially more data, parameters, training tokens and compute.

The important part is that the whole learning path is yours: you can inspect, modify and retrain every component.
