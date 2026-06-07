import torch.nn as nn

try:
    from .encoder import FeedForwardBlock, MultiHeadAttentionBlock
    from .normalization import RMSNorm
    from .residual import ResidualConnection
except ImportError:
    from encoder import FeedForwardBlock, MultiHeadAttentionBlock
    from normalization import RMSNorm
    from residual import ResidualConnection


class DecoderBlock(nn.Module):
    def __init__(
        self,
        self_attention: MultiHeadAttentionBlock,
        cross_attention: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        d_model: int,
        dropout: float,
    ):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, d_model: int):
        super().__init__()
        self.layers = layers
        self.norm = RMSNorm(d_model)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
