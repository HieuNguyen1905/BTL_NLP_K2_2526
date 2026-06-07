import torch.nn as nn

try:
    from .normalization import RMSNorm
except ImportError:
    from normalization import RMSNorm

class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm(d_model)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
