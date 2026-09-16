---
license: apache-2.0
library_name: mlx
pipeline_tag: text-generation
tags:
- mlx
base_model: Qwen/Qwen3-0.6B-Base
---

# mlx-community/Qwen3-0.6B-Base

This model [mlx-community/Qwen3-0.6B-Base](https://huggingface.co/mlx-community/Qwen3-0.6B-Base) was
converted to MLX format from [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base)
using mlx-lm version **0.30.7**.

## Use with mlx

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen3-0.6B-Base")

prompt = "hello"

if tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_dict=False,
    )

response = generate(model, tokenizer, prompt=prompt, verbose=True)
```
