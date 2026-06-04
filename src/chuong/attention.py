import torch
import torch.nn as nn
import math

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Nhân bản tensor K và V để khớp với số lượng Q (Dùng cho GQA)"""
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)

class MultiHeadAttentionBlock(nn.Module):
    """ Create an instance for multi head attention block component.
    """
    def __init__(self, d_model: int, num_head: int, num_kv_head: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.num_kv_head = num_kv_head
        self.n_rep = num_head // num_kv_head

        assert d_model % num_head == 0, 'd_model must be divisible by num_head'
        assert num_head % num_kv_head == 0, 'num_head must be divisible by num_kv_head'

        self.d_k = d_model // num_head

        self.w_q = nn.Linear(d_model, num_head * self.d_k, bias=False)
        self.w_k = nn.Linear(d_model, self.num_kv_head * self.d_k, bias=False)
        self.w_v = nn.Linear(d_model, self.num_kv_head * self.d_k, bias=False)
        self.w_o = nn.Linear(num_head * self.d_k, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_score.masked_fill_(mask == 0, -1e9)
        attention_score = attention_score.softmax(dim=-1)
        if dropout is not None:
            attention_score = dropout(attention_score)

        return (attention_score @ value), attention_score

    def forward(self, q, k, v, mask, freqs_cos=None, freqs_sin=None):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.num_head, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.num_kv_head, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.num_kv_head, self.d_k).transpose(1,2)

        if freqs_cos is not None and freqs_sin is not None:
            # Chỗ này sau này bạn code hàm apply_rotary_emb thì mở ra nhé
            # query, key = apply_rotary_emb(query, key, freqs_cos, freqs_sin)
            pass

        key = repeat_kv(key, self.n_rep)
        value = repeat_kv(value, self.n_rep)

        x, self.attention_score = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.num_head*self.d_k)
        return self.w_o(x)