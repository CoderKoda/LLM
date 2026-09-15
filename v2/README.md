# LLM V2

V2 keeps V1 intact and adds a better tokenizer and a stronger training setup.

## What changed

- **Trainable BPE tokenizer** in `tokenizer.py`. It learns byte-pair merges from your own corpus instead of treating every byte as a separate token. BPE/subword tokenization is a standard approach for language models. citeturn180042search3turn180042search1
- **Faster causal attention** using PyTorch's scaled dot-product attention when supported.
- Larger default model: 8 Transformer blocks, 8 heads, 512 hidden dimensions, 512-token context.
- Tokenizer and model are saved together with the checkpoint.
- `train.py` still gives you a single command to train everything from VS Code.

## Train

From the repository root:

```bash
python v2/train.py --data your_data.txt --steps 5000
```

For a small first test:

```bash
python v2/train.py --data examples/tiny.txt --steps 500
```

The training script will:

1. Read your UTF-8 dataset.
2. Train a BPE tokenizer on it.
3. Encode the dataset into learned subword tokens.
4. Train the Transformer to predict the next token.
5. Evaluate validation loss.
6. Save `v2/checkpoints/model.pt` and `v2/checkpoints/tokenizer.json`.

This next-token prediction objective is the basic causal language-modeling setup used by GPT-style models. citeturn180042search0

## Generate

```bash
python v2/generate.py --prompt "Hello" --tokens 200
```

## Important

V2 is still a learning-scale model, not a ChatGPT-scale model. The biggest improvement comes from feeding it a much larger, clean text corpus and training for substantially more tokens.
