import torch.nn as nn

try:
    from .decoder import Decoder, DecoderBlock
    from .embedding import InputEmbedding
    from .encoder import Encoder, EncoderBlock, FeedForwardBlock, MultiHeadAttentionBlock
    from .positional_encoding import PositionalEncoding
    from .transformer import ProjectionLayer, Transformer
except ImportError:
    from decoder import Decoder, DecoderBlock
    from embedding import InputEmbedding
    from encoder import Encoder, EncoderBlock, FeedForwardBlock, MultiHeadAttentionBlock
    from positional_encoding import PositionalEncoding
    from transformer import ProjectionLayer, Transformer


def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048,
    num_kv_head: int | None = None,
) -> Transformer:
    num_kv_head = h if num_kv_head is None else num_kv_head

    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    encoder_blocks = []
    for _ in range(N):
        self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout, num_kv_head=num_kv_head)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_blocks.append(EncoderBlock(self_attention_block, feed_forward_block, d_model, dropout))

    decoder_blocks = []
    for _ in range(N):
        self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout, num_kv_head=num_kv_head)
        cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout, num_kv_head=num_kv_head)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_blocks.append(
            DecoderBlock(self_attention_block, cross_attention_block, feed_forward_block, d_model, dropout)
        )

    encoder = Encoder(nn.ModuleList(encoder_blocks), d_model)
    decoder = Decoder(nn.ModuleList(decoder_blocks), d_model)
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    for parameter in transformer.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)

    return transformer
