"""
Layer Utilities.

Provides iterators for walking model architectures and isolating
parameters by layer block. Adapted from Chronoscope's interceptor
module walker pattern.
"""

import torch
import torch.nn as nn
from typing import Dict, Generator, List, Optional, Tuple


def iterate_lora_layers(
    model: nn.Module,
    target_patterns: Optional[List[str]] = None,
) -> Generator[Tuple[str, Dict[str, nn.Parameter]], None, None]:
    """
    Iterate over LoRA adapter layers grouped by transformer block.

    Walks named_parameters() and groups LoRA weights (lora_A, lora_B)
    by their parent transformer block index.

    Adapted from Chronoscope's _register_hooks() module walker.

    Args:
        model: The model to iterate over.
        target_patterns: Parameter name patterns to match
                         (default: ["lora_A", "lora_B"]).

    Yields:
        (block_name, param_dict) tuples where block_name is e.g.
        "layers.0" and param_dict maps full param names to Parameters.
    """
    if target_patterns is None:
        target_patterns = ["lora_A", "lora_B"]

    target_root = model.model if hasattr(model, "model") else model

    layers: Dict[str, Dict[str, nn.Parameter]] = {}
    for name, param in target_root.named_parameters():
        if any(target in name for target in target_patterns):
            # Group by parent transformer block
            # e.g. "layers.0.self_attn.lora_A" → block_name = "layers.0"
            parts = name.split(".")
            block_idx = None
            for i, p in enumerate(parts):
                if p.isdigit():
                    block_idx = i
                    break

            if block_idx is not None:
                block_name = ".".join(parts[: block_idx + 1])
                layers.setdefault(block_name, {})[name] = param

    for block_name in sorted(
        layers.keys(), key=lambda n: int(n.split(".")[-1])
    ):
        yield block_name, layers[block_name]


def iterate_model_layers(
    model: nn.Module,
    layer_prefix: str = "layers",
) -> Generator[Tuple[str, List[nn.Parameter]], None, None]:
    """
    Generic iterator that groups model parameters by numbered layer blocks.

    Works with any model architecture that uses numbered sub-modules
    (e.g., model.layers.0, model.layers.1, ...).

    Args:
        model: The model to iterate over.
        layer_prefix: The prefix for layer modules (default: "layers").

    Yields:
        (layer_name, param_list) tuples.
    """
    target_root = model.model if hasattr(model, "model") else model

    layers: Dict[str, List[nn.Parameter]] = {}
    for name, param in target_root.named_parameters():
        if not param.requires_grad:
            continue
        parts = name.split(".")
        # Find the layer block
        for i, p in enumerate(parts):
            if p == layer_prefix and i + 1 < len(parts) and parts[i + 1].isdigit():
                block_name = f"{layer_prefix}.{parts[i + 1]}"
                layers.setdefault(block_name, []).append(param)
                break

    for block_name in sorted(
        layers.keys(), key=lambda n: int(n.split(".")[-1])
    ):
        yield block_name, layers[block_name]


def count_parameters(params: List[nn.Parameter]) -> int:
    """Count total number of parameters."""
    return sum(p.numel() for p in params)


def flatten_params(params: List[nn.Parameter]) -> torch.Tensor:
    """Flatten a list of parameters into a single 1-D tensor."""
    return torch.cat([p.data.view(-1) for p in params])
