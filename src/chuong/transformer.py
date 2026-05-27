import torch
import torch.nn as nn
# Import các khối thành phần cùng cấp thư mục
from .attention import MultiHeadAttentionBlock
from .feed_forward import FeedForwardBlock
from .normalization import RMSNorm

class TransformerBlock(nn.Module):
    """ Lắp ráp các block của bạn theo luồng Pre-LN """
    def __init__(self, d_model: int, num_head: int, num_kv_head: int, d_ff: int, dropout: float):
        super().__init__()
        self.attention = MultiHeadAttentionBlock(d_model, num_head, num_kv_head, dropout)
        self.feed_forward = FeedForwardBlock(d_model, d_ff, dropout)

        # [CẢI TIẾN 4]: Dùng RMSNorm thay vì nn.LayerNorm
        self.attention_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)

    def forward(self, x, mask, freqs_cos, freqs_sin):
        # [CẢI TIẾN 4 - Pre-LN]: Norm -> Attention -> Cộng Residual
        x = x + self.attention(self.attention_norm(x),
                               self.attention_norm(x),
                               self.attention_norm(x),
                               mask, freqs_cos, freqs_sin)

        # [CẢI TIẾN 4 - Pre-LN]: Norm -> FFN -> Cộng Residual
        x = x + self.feed_forward(self.ffn_norm(x))
        return x