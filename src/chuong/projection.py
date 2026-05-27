import torch
import torch.nn as nn

class ProjectionLayer(nn.Module):
    """ Create an instance for projection layer component.
    """
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        return torch.log_softmax(self.projection(x), dim=-1)