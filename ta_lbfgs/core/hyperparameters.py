import math
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


class LayerHPBlock(nn.Module):
    """
    Modular block holding hyperparameters for a single transformer layer.
    Enables isolated L-BFGS optimization for block-diagonal approximations.
    """

    def __init__(
        self,
        initial_lr: float,
        initial_wd: float,
        initial_dropout: float,
        initial_attn_temp: float,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.raw_lr = nn.Parameter(torch.tensor(math.log(initial_lr), device=device, dtype=dtype))
        self.raw_wd = nn.Parameter(torch.tensor(math.log(initial_wd), device=device, dtype=dtype))
        self.raw_dropout = nn.Parameter(torch.tensor(self._logit(initial_dropout), device=device, dtype=dtype))
        self.raw_attn_temp = nn.Parameter(torch.tensor(math.log(initial_attn_temp), device=device, dtype=dtype))

    @staticmethod
    def _logit(p: float) -> float:
        p = max(1e-6, min(1 - 1e-6, p))
        return math.log(p / (1 - p))

    @property
    def lr(self) -> torch.Tensor: return torch.exp(self.raw_lr)
    @property
    def wd(self) -> torch.Tensor: return torch.exp(self.raw_wd)
    @property
    def dropout(self) -> torch.Tensor: return torch.sigmoid(self.raw_dropout)
    @property
    def attn_temp(self) -> torch.Tensor: return torch.exp(self.raw_attn_temp)


class DifferentiableHyperparameters(nn.Module):
    """
    Module that holds all differentiable hyperparameters, organized by layer.
    """
    """
    Module that holds all differentiable hyperparameters.

    Per-layer parameters are stored as 1-D tensors indexed by layer.
    All constrained values use reparameterization:
      - Positive reals (lr, wd, temp): log-space  → exp(raw)
      - Bounded (0,1) (dropout, label_smooth): logit-space → sigmoid(raw)

    Args:
        n_layers: Number of transformer layers.
        initial_lr: Initial learning rate (uniform across layers).
        initial_wd: Initial weight decay (uniform across layers).
        initial_dropout: Initial dropout probability.
        initial_label_smoothing: Initial label smoothing factor.
        initial_attn_temp: Initial attention temperature (1.0 = standard).
        enable_moe_routing: Whether to include MoE routing temperatures.
        device: Torch device.
        dtype: Torch dtype.
    """

    def __init__(
        self,
        n_layers: int = 1,
        initial_lr: float = 1e-4,
        initial_wd: float = 1e-2,
        initial_dropout: float = 0.1,
        initial_label_smoothing: float = 0.1,
        initial_attn_temp: float = 1.0,
        enable_moe_routing: bool = False,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.enable_moe_routing = enable_moe_routing

        # ── Modular Per-Layer Blocks ───────────────────────────
        self.blocks = nn.ModuleList([
            LayerHPBlock(
                initial_lr, initial_wd, initial_dropout, initial_attn_temp,
                device=device, dtype=dtype
            )
            for _ in range(n_layers)
        ])

        # ── Global: Label Smoothing (logit-space) ───────────────
        self.raw_label_smoothing = nn.Parameter(
            torch.tensor(
                [self._logit(initial_label_smoothing)],
                device=device, dtype=dtype,
            )
        )

        if enable_moe_routing:
            self.raw_moe_temp = nn.Parameter(
                torch.full((n_layers,), math.log(1.0), device=device, dtype=dtype)
            )

    def _logit(self, p: float) -> float:
        p = max(1e-6, min(1 - 1e-6, p))
        return math.log(p / (1 - p))

    @staticmethod
    def _logit(p: float) -> float:
        """Inverse sigmoid: logit(p) = log(p / (1-p))."""
        p = max(1e-6, min(1 - 1e-6, p))
        return math.log(p / (1 - p))

    # ── Properties (differentiable transforms) ──────────────────

    @property
    def lr(self) -> torch.Tensor:
        """Per-layer learning rates [n_layers]."""
        return torch.stack([b.lr for b in self.blocks])

    @property
    def wd(self) -> torch.Tensor:
        """Per-layer weight decay [n_layers]."""
        return torch.stack([b.wd for b in self.blocks])

    @property
    def dropout(self) -> torch.Tensor:
        """Per-layer dropout rates [n_layers]."""
        return torch.stack([b.dropout for b in self.blocks])

    @property
    def attn_temp(self) -> torch.Tensor:
        """Per-layer attention temperatures [n_layers]."""
        return torch.stack([b.attn_temp for b in self.blocks])

    @property
    def label_smoothing(self) -> torch.Tensor:
        """Global label smoothing factor. Bounded (0,1)."""
        return torch.sigmoid(self.raw_label_smoothing)

    @property
    def moe_temp(self) -> Optional[torch.Tensor]:
        """Per-layer MoE routing temperatures [n_layers] (if enabled)."""
        if self.enable_moe_routing and hasattr(self, "raw_moe_temp"):
            return torch.exp(self.raw_moe_temp)
        return None

    # ── Convenience Accessors ───────────────────────────────────

    def get_layer_lr(self, layer_idx: int) -> torch.Tensor:
        """Get learning rate for a specific layer (scalar, differentiable)."""
        return self.lr[layer_idx]

    def get_layer_wd(self, layer_idx: int) -> torch.Tensor:
        """Get weight decay for a specific layer."""
        return self.wd[layer_idx]

    def get_layer_dropout(self, layer_idx: int) -> torch.Tensor:
        """Get dropout rate for a specific layer."""
        return self.dropout[layer_idx]

    def get_layer_attn_temp(self, layer_idx: int) -> torch.Tensor:
        """Get attention temperature for a specific layer."""
        return self.attn_temp[layer_idx]

    def as_dict(self) -> Dict[str, torch.Tensor]:
        """Return all hyperparameters as a dict of tensors."""
        d = {
            "lr": self.lr,
            "wd": self.wd,
            "dropout": self.dropout,
            "attn_temp": self.attn_temp,
            "label_smoothing": self.label_smoothing,
        }
        if self.moe_temp is not None:
            d["moe_temp"] = self.moe_temp
        return d

    def as_float_dict(self) -> Dict[str, object]:
        """Return hyperparameters as plain Python objects for logging."""
        d = {
            "lr": self.lr.detach().cpu().tolist(),
            "wd": self.wd.detach().cpu().tolist(),
            "dropout": self.dropout.detach().cpu().tolist(),
            "attn_temp": self.attn_temp.detach().cpu().tolist(),
            "label_smoothing": self.label_smoothing.item(),
        }
        if self.moe_temp is not None:
            d["moe_temp"] = self.moe_temp.detach().cpu().tolist()
        return d

    def apply_to_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        layer_idx: Optional[int] = None,
    ):
        """
        Inject current hyperparameters into a PyTorch optimizer.

        If layer_idx is provided, applies only that layer's values
        to the corresponding param_group. Otherwise, applies layer 0.

        Args:
            optimizer: The inner-loop optimizer to update.
            layer_idx: Which layer's hyperparameters to use.
        """
        idx = layer_idx or 0
        lr_val = self.lr[idx].item()
        wd_val = self.wd[idx].item()
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_val
            param_group["weight_decay"] = wd_val

    def clamp(
        self,
        lr_range: Tuple[float, float] = (1e-7, 1.0),
        wd_range: Tuple[float, float] = (1e-7, 1.0),
        dropout_range: Tuple[float, float] = (0.0, 0.5),
        label_smooth_range: Tuple[float, float] = (0.01, 0.3),
        attn_temp_range: Tuple[float, float] = (0.1, 10.0),
    ):
        """Clamp all raw parameters to keep transformed values in valid ranges."""
        with torch.no_grad():
            for b in self.blocks:
                b.raw_lr.clamp_(math.log(lr_range[0]), math.log(lr_range[1]))
                b.raw_wd.clamp_(math.log(wd_range[0]), math.log(wd_range[1]))
                b.raw_dropout.clamp_(
                    self._logit(max(dropout_range[0], 1e-6)),
                    self._logit(min(dropout_range[1], 1 - 1e-6)),
                )
                b.raw_attn_temp.clamp_(
                    math.log(attn_temp_range[0]),
                    math.log(attn_temp_range[1]),
                )
            
            self.raw_label_smoothing.clamp_(
                self._logit(max(label_smooth_range[0], 1e-6)),
                self._logit(min(label_smooth_range[1], 1 - 1e-6)),
            )

    def total_params(self) -> int:
        """Total number of differentiable hyperparameters."""
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        parts = [f"DifferentiableHyperparameters(n_layers={self.n_layers}"]
        parts.append(f"  lr={self.lr.detach().cpu().tolist()}")
        parts.append(f"  wd={self.wd.detach().cpu().tolist()}")
        parts.append(f"  dropout={self.dropout.detach().cpu().tolist()}")
        parts.append(f"  attn_temp={self.attn_temp.detach().cpu().tolist()}")
        parts.append(f"  label_smoothing={self.label_smoothing.item():.4f}")
        if self.moe_temp is not None:
            parts.append(f"  moe_temp={self.moe_temp.detach().cpu().tolist()}")
        parts.append(f"  total_params={self.total_params()}")
        parts.append(")")
        return "\n".join(parts)
