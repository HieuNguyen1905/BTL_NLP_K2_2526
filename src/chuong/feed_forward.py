import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardBlock(nn.Module):
    """ Create an instance for feed forward block component.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()

        # ---------------------------------------------------------
        # [CẢI TIẾN 3 - SwiGLU]: Loại bỏ linear1, linear2 gốc.
        # Cần 3 ma trận tuyến tính và bỏ luôn bias.
        # ---------------------------------------------------------
        self.w1 = nn.Linear(d_model, d_ff, bias=False) # Luồng Gating
        self.w2 = nn.Linear(d_model, d_ff, bias=False) # Luồng Dữ liệu
        self.w3 = nn.Linear(d_ff, d_model, bias=False) # Luồng Trộn đầu ra

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # ---------------------------------------------------------
        # [CẢI TIẾN 3 - SwiGLU]: Tách dữ liệu làm 2 luồng.
        # Dùng hàm SiLU (Swish) mượt mà thay thế cho ReLU cứng nhắc.
        # ---------------------------------------------------------
        gate = F.silu(self.w1(x))  # Luồng 1 (Cổng kiểm soát đi qua SiLU)
        up = self.w2(x)            # Luồng 2 (Dữ liệu nguyên bản)

        # Nhân element-wise 2 luồng với nhau, sau đó đi qua w3
        x = self.w3(gate * up)

        return self.dropout(x)