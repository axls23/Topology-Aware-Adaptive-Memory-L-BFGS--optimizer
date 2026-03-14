"""
VRAM-Aware Memory Management.

Adapted from Chronoscope's adaptive_airllm.py.
Provides GPU memory queries and adaptive L-BFGS history sizing
based on available VRAM.
"""

import torch
from typing import Optional, Tuple


def get_free_vram_bytes() -> int:
    """Get available GPU memory in bytes."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.mem_get_info()[0]


def get_total_vram_bytes() -> int:
    """Get total GPU memory in bytes."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.mem_get_info()[1]


def get_vram_usage() -> Tuple[float, float, float]:
    """
    Get VRAM usage statistics.

    Returns:
        Tuple of (used_gb, free_gb, total_gb).
    """
    if not torch.cuda.is_available():
        return (0.0, 0.0, 0.0)

    free = get_free_vram_bytes()
    total = get_total_vram_bytes()
    used = total - free

    return (used / 1e9, free / 1e9, total / 1e9)


def compute_max_history_entries(
    param_dim: int,
    available_memory_bytes: Optional[int] = None,
    dtype_bytes: int = 4,
    safety_factor: float = 0.8,
) -> int:
    """
    Compute maximum L-BFGS history entries based on available memory.

    Each curvature pair (s_k, y_k) costs 2 * param_dim * dtype_bytes.

    Args:
        param_dim: Dimension of parameter vector per layer.
        available_memory_bytes: Override available memory (default: query GPU).
        dtype_bytes: Bytes per element (4 for float32, 2 for float16).
        safety_factor: Fraction of available memory to use.

    Returns:
        Maximum number of history entries.
    """
    if available_memory_bytes is None:
        available_memory_bytes = get_free_vram_bytes()

    usable = int(available_memory_bytes * safety_factor)
    per_pair_cost = 2 * param_dim * dtype_bytes

    if per_pair_cost == 0:
        return 100  # Arbitrary large number

    max_entries = usable // per_pair_cost
    return max(1, max_entries)


class OOMSafeHistoryManager:
    """
    Manages L-BFGS history with OOM-safe fallback.

    Adapted from Chronoscope's OOM-safe chunk size halving pattern.
    If VRAM is exhausted during optimization, this manager halves
    the history window size gracefully.
    """

    def __init__(self, initial_size: int = 10, min_size: int = 2):
        self.current_size = initial_size
        self.min_size = min_size
        self.oom_count = 0

    def handle_oom(self) -> int:
        """
        Handle an OOM event by halving the history size.

        Returns:
            New (reduced) history size.
        """
        self.oom_count += 1
        new_size = max(self.min_size, self.current_size // 2)
        self.current_size = new_size
        return new_size

    def try_increase(self, max_size: int = 20) -> int:
        """
        Try to increase history size if memory allows.

        Returns:
            New (potentially increased) history size.
        """
        if self.oom_count > 0:
            # Be conservative after OOM events
            return self.current_size

        free_gb = get_free_vram_bytes() / 1e9
        if free_gb > 2.0:  # More than 2GB free
            new_size = min(max_size, self.current_size + 1)
            self.current_size = new_size

        return self.current_size
