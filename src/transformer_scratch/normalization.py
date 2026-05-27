import torch
import torch.nn as nn
import math

class LayerNormalization(nn.Module):
    def __init__(self, d_model: int, epsilon: float = 1e-6):
        super().__init__()
        self.eps = epsilon
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias