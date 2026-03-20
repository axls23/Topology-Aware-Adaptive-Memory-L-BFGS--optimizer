"""K-FAC utilities for embedding/tied-weight preconditioning."""

from __future__ import annotations

import torch
from torch import Tensor


class KFACEmbedding:
    """K-FAC approximation for tied embedding / LM-head weights."""

    def __init__(self, vocab_size: int, embed_dim: int, ema_decay: float = 0.95):
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.decay = float(ema_decay)
        self.A: Tensor = torch.zeros(self.embed_dim, self.embed_dim)
        self.S: Tensor = torch.zeros(self.embed_dim, self.embed_dim)

    # ADDS: EMA K-FAC factor updates from token-frequency proxy and output curvature.
    # REMOVES: generic block-diagonal approximation for tied embedding weights.
    def update(self, embed_in: Tensor, grad_out: Tensor) -> None:
        x = embed_in.detach().float()
        g = grad_out.detach().float()

        if x.dim() == 2 and x.shape[0] == self.vocab_size:
            token_freq = x.abs().mean(0)
        else:
            token_freq = x.reshape(-1)[: self.embed_dim].abs()
            if token_freq.numel() < self.embed_dim:
                token_freq = torch.nn.functional.pad(token_freq, (0, self.embed_dim - token_freq.numel()))

        if g.dim() == 1:
            g = g.unsqueeze(0)
        if g.shape[-1] != self.embed_dim:
            g = g.reshape(-1, self.embed_dim)
        S_sample = (g.t() @ g) / max(1, g.shape[0])

        A = self.A.to(device=token_freq.device, dtype=token_freq.dtype)
        S = self.S.to(device=S_sample.device, dtype=S_sample.dtype)
        A.diagonal().mul_(self.decay).add_((1.0 - self.decay) * token_freq.clamp_min(1e-8))
        S.mul_(self.decay).add_((1.0 - self.decay) * S_sample)
        self.A = A
        self.S = S

    # ADDS: regularized pinv/lstsq inverse preconditioning for near-singular S factor.
    # REMOVES: use of plain inverse in tied-weight embedding curvature updates.
    def inverse_precondition(self, grad: Tensor) -> Tensor:
        if grad.dim() != 2:
            return grad

        g = grad.float()
        A = self.A.to(device=g.device, dtype=g.dtype)
        S = self.S.to(device=g.device, dtype=g.dtype)
        eps = 1e-5

        A_inv = torch.diag(1.0 / A.diagonal().clamp(min=1e-6))
        S_reg = S + eps * torch.eye(S.size(0), device=S.device, dtype=S.dtype)
        try:
            # Solve S_reg X = grad for X, then right-multiply by A_inv.
            left = torch.linalg.lstsq(S_reg, g).solution
        except RuntimeError:
            left = torch.linalg.pinv(S_reg) @ g
        return (left @ A_inv).to(dtype=grad.dtype)
