import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size, n_heads, ff_size, p_drop, ln_type):
        super().__init__()
        if ln_type == "pre":
            self.attention = PreAttention(hidden_size, n_heads, p_drop)
        elif ln_type == "post":
            self.attention = PostAttention(hidden_size, n_heads, p_drop)
        self.ffn = FeedForwardNorm(hidden_size, ff_size, p_drop, ln_type)

    def forward(self, query, ref=None, *, mask):
        x, score = self.attention(query, ref, mask)
        x = self.ffn(x)
        return x, score


class BaseAttention(nn.Module):
    def __init__(self, hidden_size, n_heads, p_drop):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, n_heads)
        self.norm = nn.LayerNorm(hidden_size)
        self.ref_norm = nn.LayerNorm(hidden_size)
        self.dense = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Dropout(p_drop))


class PreAttention(BaseAttention):
    def forward(self, query, ref, mask):
        x = self.norm(query)
        if ref is None:
            ref = x
        else:
            ref = self.ref_norm(ref)
        x, score = self.attention(x, ref, ref, mask)
        x = query + self.dense(x)
        return x, score


class PostAttention(BaseAttention):
    def forward(self, query, ref, mask):
        if ref is None:
            ref = query

        x, score = self.attention(query, ref, ref, mask)
        x = self.norm(query + self.dense(x))
        return x, score


class FeedForwardNorm(nn.Module):
    def __init__(self, hidden_size, ff_size, p_drop, ln_type):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, ff_size),
            nn.GELU(),
            nn.Linear(ff_size, hidden_size),
            nn.Dropout(p_drop),
        )
        self.ln_type = ln_type

    def forward(self, inputs):
        if self.ln_type == "pre":
            return inputs + self.ff(self.norm(inputs))
        elif self.ln_type == "post":
            return self.norm(inputs + self.ff(inputs))


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, n_heads):
        super().__init__()

        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)

        self.hidden_size = hidden_size
        self.n_heads = n_heads

        self.head_size = hidden_size // n_heads
        self.scale_factor = self.head_size**-0.5

    def forward(self, q, k, v, mask):
        q = self.q(q).view(*q.shape[:2], self.n_heads, self.head_size).permute(0, 2, 1, 3)
        k = self.k(k).view(*k.shape[:2], self.n_heads, self.head_size).permute(0, 2, 3, 1)
        v = self.v(v).view(*v.shape[:2], self.n_heads, self.head_size).permute(0, 2, 1, 3)

        x = q @ k
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(-1)
            x = x + (1.0 - mask.unsqueeze(-1)) * -10000.0
        score = (self.scale_factor * x).softmax(dim=-1)
        x = score @ v
        x = x.permute(0, 2, 1, 3).reshape(x.size(0), -1, self.hidden_size)
        return x, score
