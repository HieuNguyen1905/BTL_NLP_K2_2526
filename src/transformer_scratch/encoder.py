import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .normalization import RMSNorm
    from .residual import ResidualConnection
except ImportError:
    from normalization import RMSNorm
    from residual import ResidualConnection


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_kv_head, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states

    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_head, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_kv_head * n_rep, seq_len, head_dim)


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = F.silu(self.w1(x))
        up = self.w2(x)
        return self.dropout(self.w3(gate * up))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_head: int, dropout: float, num_kv_head: int | None = None):
        super().__init__()
        num_kv_head = num_head if num_kv_head is None else num_kv_head
        if num_head <= 0:
            raise ValueError("num_head must be greater than 0")
        if num_kv_head <= 0:
            raise ValueError("num_kv_head must be greater than 0")
        if d_model % num_head != 0:
            raise ValueError("d_model must be divisible by num_head")
        if num_head % num_kv_head != 0:
            raise ValueError("num_head must be divisible by num_kv_head")

        self.d_model = d_model
        self.num_head = num_head
        self.num_kv_head = num_kv_head
        self.n_rep = num_head // num_kv_head
        self.d_k = d_model // num_head

        self.w_q = nn.Linear(d_model, num_head * self.d_k, bias=False)
        self.w_k = nn.Linear(d_model, num_kv_head * self.d_k, bias=False)
        self.w_v = nn.Linear(d_model, num_kv_head * self.d_k, bias=False)
        self.w_o = nn.Linear(num_head * self.d_k, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.attention_score = None

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout | None):
        d_k = query.shape[-1]
        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e9)

        attention_score = attention_score.softmax(dim=-1)
        if dropout is not None:
            attention_score = dropout(attention_score)

        return attention_score @ value, attention_score

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.num_head, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_kv_head, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_kv_head, self.d_k).transpose(1, 2)

        key = repeat_kv(key, self.n_rep)
        value = repeat_kv(value, self.n_rep)

        x, self.attention_score = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.num_head * self.d_k)
        return self.w_o(x)


class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, d_model: int, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, d_model: int):
        super().__init__()
        self.layers = layers
        self.norm = RMSNorm(d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
