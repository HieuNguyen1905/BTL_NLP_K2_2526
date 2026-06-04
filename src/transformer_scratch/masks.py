import torch


def causal_mask(size: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(1, size, size, dtype=torch.bool), diagonal=1)
    return ~mask


def create_src_mask(src: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)


def create_tgt_mask(tgt: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    tgt_causal_mask = causal_mask(tgt.size(1)).type_as(tgt_pad_mask).to(tgt.device)
    return tgt_pad_mask & tgt_causal_mask


def create_masks(src: torch.Tensor, tgt: torch.Tensor, pad_idx: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    return create_src_mask(src, pad_idx), create_tgt_mask(tgt, pad_idx)


def subsequent_mask(size: int) -> torch.Tensor:
    return causal_mask(size)
