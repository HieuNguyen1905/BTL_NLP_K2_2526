import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    def __init__(self, d_model: int, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.alpha * (x - mean) / torch.sqrt(variance + self.epsilon) + self.bias
