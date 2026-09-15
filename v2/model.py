"""V2 decoder-only Transformer language model."""

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


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, channels = x.shape
        head_dim = channels // self.n_head
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, seq, self.n_head, head_dim).transpose(1, 2)
        k = k.view(batch, seq, self.n_head, head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.n_head, head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq, channels)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        hidden = 4 * config.n_embd
        self.fc = nn.Linear(config.n_embd, hidden)
        self.proj = nn.Linear(hidden, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        _, seq = idx.shape
        if seq > self.config.block_size:
            raise ValueError("sequence exceeds block_size")
        positions = torch.arange(seq, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(positions)[None])
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8, top_k: int = 40):
        self.eval()
        for _ in range(max_new_tokens):
            context = idx[:, -self.config.block_size:]
            logits, _ = self(context)
            logits = logits[:, -1] / max(temperature, 1e-5)
            if top_k:
                k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, k)
                logits[logits < values[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
