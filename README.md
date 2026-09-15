# Koda LLM

A small GPT-style language model built from scratch in Python + PyTorch.

There is no API call and no pretrained model in this repository. You provide the training text, run the training loop, and the model learns its weights locally.

## What is included

- `tokenizer.py` - dependency-free UTF-8 byte tokenizer. Every byte is a token, so any text can be trained without downloading a vocabulary.
- `model.py` - decoder-only Transformer with causal self-attention, MLPs, residual connections, layer normalization, learned positional embeddings, and tied input/output embeddings.
- `train.py` - self-training loop with train/validation split, AdamW, gradient clipping, automatic CPU/MPS/CUDA selection, and checkpoints.
- `generate.py` - text generation from a saved checkpoint with temperature and top-k sampling.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\\Scripts\\activate`.

## Prepare your own data

Create a UTF-8 text file such as `data.txt`. The model learns to predict the next byte from the previous bytes, so the text can be books, notes, documentation, code, conversations, or a mixture of sources you have permission to use.

For a first test, even a small text file works. For useful language modeling, use much more data.

## Train

A good small-model starting point:

```bash
python train.py --data data.txt --steps 5000 --device auto
```

The default model is 6 Transformer blocks, 6 attention heads, and a 384-dimensional hidden state. On a capable machine you can scale it up, for example:

```bash
python train.py --data data.txt --steps 20000 --n-layer 8 --n-head 8 --n-embd 512 --block-size 512
```

The best validation checkpoint is written to `checkpoints/model.pt`.

## Generate text

```bash
python generate.py --checkpoint checkpoints/model.pt --prompt "Hello"
```

Try changing the sampling settings:

```bash
python generate.py --checkpoint checkpoints/model.pt --prompt "Once upon a time" --tokens 300 --temperature 0.7 --top-k 40
```

## What “self-trainable” means here

The repository owns the complete learning path: raw text -> tokenizer -> token batches -> Transformer -> next-token loss -> backpropagation -> updated weights -> checkpoint -> generation.

The current tokenizer is intentionally simple rather than a full BPE implementation. That makes the first model much easier to understand and modify. A later version can replace it with a learned BPE tokenizer without changing the Transformer itself.

## Important expectations

This is a real language model, but it is intentionally small. A few thousand steps on a small text file will not produce ChatGPT-level capabilities. Larger datasets, more parameters, more training, and stronger hardware are needed for substantially better results.

## License

Add the license you want to use for your project.
