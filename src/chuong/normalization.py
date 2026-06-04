import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """ [CẢI TIẾN 4]: RMSNorm tính toán cực nhanh vì bỏ qua phép tính mean """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight