"""A small GPT-style decoder-only Transformer implemented from scratch."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 512
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0


class Attention(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        if c.n_embd % c.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        self.heads = c.n_head
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd)
        self.proj = nn.Linear(c.n_embd, c.n_embd)
        self.dropout = c.dropout

    def _split(self, x):
        b, t, c = x.shape
        d = c // self.heads
        return x.view(b, t, self.heads, d).transpose(1, 2)

    def forward(self, x, past_key_value=None, use_cache=False):
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = self._split(q), self._split(k), self._split(v)
        past_len = 0
        if past_key_value is not None:
            past_k, past_v = past_key_value
            past_len = past_k.size(-2)
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)

        if past_len == 0:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            scale = q.size(-1) ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            q_len, k_len = q.size(-2), k.size(-2)
            query_positions = torch.arange(past_len, past_len + q_len, device=x.device)
            key_positions = torch.arange(k_len, device=x.device)
            mask = key_positions[None, :] <= query_positions[:, None]
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            probs = F.softmax(scores, dim=-1)
            if self.dropout and self.training:
                probs = F.dropout(probs, p=self.dropout)
            y = torch.matmul(probs, v)

        y = self.proj(y.transpose(1, 2).contiguous().view(b, t, c))
        present = (k, v) if use_cache else None
        return y, present


class Block(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.n_embd)
        self.attn = Attention(c)
        self.ln2 = nn.LayerNorm(c.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(c.n_embd, 4 * c.n_embd),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * c.n_embd, c.n_embd),
            nn.Dropout(c.dropout),
        )

    def forward(self, x, past_key_value=None, use_cache=False):
        attn_out, present = self.attn(self.ln1(x), past_key_value, use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, present


class GPT(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.config = c
        self.tok = nn.Embedding(c.vocab_size, c.n_embd)
        self.pos = nn.Embedding(c.block_size, c.n_embd)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln = nn.LayerNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.head.weight = self.tok.weight
        self.register_buffer("position_ids", torch.arange(c.block_size), persistent=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None, past_key_values=None, use_cache=False):
        _, t = idx.shape
        past_len = 0
        if past_key_values is not None and past_key_values:
            past_len = past_key_values[0][0].size(-2)
        if past_len + t > self.config.block_size:
            raise ValueError("sequence exceeds block_size")
        p = self.position_ids[past_len:past_len + t]
        x = self.drop(self.tok(idx) + self.pos(p)[None])
        presents = []
        for i, block in enumerate(self.blocks):
            past = past_key_values[i] if past_key_values is not None else None
            x, present = block(x, past, use_cache)
            if use_cache:
                presents.append(present)
        logits = self.head(self.ln(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return (logits, loss, presents) if use_cache else (logits, loss)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.config.block_size:])
            logits = logits[:, -1] / max(temperature, 1e-5)
            if top_k:
                k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, k)
                logits[logits < values[:, -1, None]] = -float("inf")
            idx = torch.cat((idx, torch.multinomial(F.softmax(logits, dim=-1), 1)), dim=1)
        return idx

    @torch.no_grad()
    def generate_cached(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        """Generate with an attention KV cache to avoid recomputing the prefix."""
        self.eval()
        idx = idx[:, -self.config.block_size:]
        logits, _, cache = self(idx, use_cache=True)
        for _ in range(max_new_tokens):
            next_logits = logits[:, -1] / max(temperature, 1e-5)
            if top_k:
                k = min(top_k, next_logits.size(-1))
                values, _ = torch.topk(next_logits, k)
                next_logits[next_logits < values[:, -1, None]] = -float("inf")
            next_token = torch.multinomial(F.softmax(next_logits, dim=-1), 1)
            idx = torch.cat((idx, next_token), dim=1)
            if idx.size(1) > self.config.block_size:
                idx = idx[:, -self.config.block_size:]
                logits, _, cache = self(idx, use_cache=True)
            else:
                logits, _, cache = self(next_token, past_key_values=cache, use_cache=True)
        return idx

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
