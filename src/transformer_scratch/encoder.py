import math

import torch
import torch.nn as nn

try:
    from .normalization import LayerNormalization
    from .residual import ResidualConnection
except ImportError:
    from normalization import LayerNormalization
    from residual import ResidualConnection


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_head: int, dropout: float):
        super().__init__()
        if d_model % num_head != 0:
            raise ValueError("d_model must be divisible by num_head")

        self.d_model = d_model
        self.num_head = num_head
        self.d_k = d_model // num_head

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
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
        key = key.view(key.shape[0], key.shape[1], self.num_head, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_head, self.d_k).transpose(1, 2)

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
        self.norm = LayerNormalization(d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
