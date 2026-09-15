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
- `train.py` - trains the tokenizer and model, evaluates validation loss, saves checkpoints, and can continue training from a checkpoint.
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

## 3. Train

Start small so you can see the entire system working:

```bash
python v3/train.py --data data.txt --steps 500
```

For a larger run:

```bash
python v3/train.py --data data.txt --steps 20000 --vocab-size 8192 --block-size 512 --n-layer 8 --n-head 8 --n-embd 512
```

The script automatically uses CUDA, then Apple MPS, then CPU when `--device auto` is selected.

## 4. Continue training

```bash
python v3/train.py --data data.txt --resume v3/checkpoints/model.pt --steps 10000
```

## 5. Generate

```bash
python v3/generate.py --checkpoint v3/checkpoints/model.pt --prompt "Hello" --tokens 200
```

Adjust creativity with `--temperature` and `--top-k`.

## 6. Chat

```bash
python v3/chat.py --checkpoint v3/checkpoints/model.pt
```

Type `/exit` to leave.

## What it is — and isn't

This is a real causal Transformer language model. It can genuinely learn patterns from your dataset, but a small model trained on a small corpus will not have ChatGPT-level knowledge or reasoning. Building something much stronger requires substantially more data, parameters, training tokens and compute.

The important part is that the whole learning path is yours: you can inspect, modify and retrain every component.
