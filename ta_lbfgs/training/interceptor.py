"""
Architecture Interceptor.

Groups learnable model parameters into topological blocks (Transformer Layers)
and provides hooks for injecting differentiable hyperparameters like 
Attention Temperature and Differentiable Dropout.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Callable
import re
import math

class DifferentiableDropout(nn.Module):
    """
    Differentiable Dropout that uses a learned/optimized probability.
    Uses the Concrete Distribution (Relaxed Bernoulli) for gradient flow.
    """
    def __init__(self, p_tensor: torch.Tensor, temperature: float = 0.1):
        super().__init__()
        self.p = p_tensor # Differentiable hyperparameter
        self.temp = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0:
            return x
        
        # Concrete relaxation (Simplified version: mask = sigmoid((log(p/1-p) + Gumbel)/temp))
        # Here we use a simpler Bernoulli-like differentiable mask for demo purposes
        # or just standard dropout scaled by the differentiable p.
        # For true IFT, we need the mask to be a function of p.
        noise = torch.rand_like(x)
        mask = torch.sigmoid((torch.log(self.p + 1e-8) - torch.log(1 - self.p + 1e-8) + 
                             torch.log(noise + 1e-8) - torch.log(1 - noise + 1e-8)) / self.temp)
        return x * mask / (self.p + 1e-8)

class ArchitectureInterceptor:
    """
    Traverses a Transformer model to group parameters by block and
    inject topological hooks.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.blocks: Dict[str, Dict[str, Any]] = {}
        self._analyze_architecture()

    def _analyze_architecture(self):
        """Detect and group MHSA, FFN, and Norm modules by block."""
        # Standard patterns for Qwen, Llama, GPT
        block_pattern = re.compile(r".*layers\.(\d+).*|.*h\.(\d+).*")
        
        for name, module in self.model.named_modules():
            match = block_pattern.match(name)
            if match:
                block_idx = next(idx for idx in match.groups() if idx is not None)
                block_key = f"block.{block_idx}"
                
                if block_key not in self.blocks:
                    self.blocks[block_key] = {
                        "mhsa": [],
                        "ffn": [],
                        "norm": [],
                        "dropout": [],
                        "params": []
                    }
                
                # Categorize module
                m_name = name.lower()
                if any(x in m_name for x in ["attn", "self_attn", "attention"]) and not any(x in m_name for x in ["norm", "ln"]):
                    if any(isinstance(module, c) for c in [nn.Linear, nn.Conv1d]):
                        self.blocks[block_key]["mhsa"].append((name, module))
                elif any(x in m_name for x in ["mlp", "ffn", "feed_forward"]) and not any(x in m_name for x in ["norm", "ln"]):
                    if any(isinstance(module, c) for c in [nn.Linear, nn.Conv1d]):
                        self.blocks[block_key]["ffn"].append((name, module))
                elif any(x in m_name for x in ["norm", "ln"]):
                    self.blocks[block_key]["norm"].append((name, module))
                elif isinstance(module, nn.Dropout):
                    self.blocks[block_key]["dropout"].append((name, module))

        # Collect parameters for each block
        for name, param in self.model.named_parameters():
            if not param.requires_grad: continue
            match = block_pattern.match(name)
            if match:
                block_idx = next(idx for idx in match.groups() if idx is not None)
                block_key = f"block.{block_idx}"
                self.blocks[block_key]["params"].append(param)

    def get_layer_group(self, block_idx: int) -> Dict[str, Any]:
        """Return categorized modules for a specific block."""
        key = f"block.{block_idx}"
        return self.blocks.get(key, {})

    def ground_architectural_variables(self, hyperparams: Any):
        """
        Inject hooks and replace modules to ground hyperparams.dropout and attn_temp.
        """
        # 1. Differentiable Dropout Replacement
        for block_key, data in self.blocks.items():
            idx = int(block_key.split(".")[-1])
            if idx >= len(hyperparams.dropout): continue
            
            p_val = hyperparams.dropout[idx]
            
            for name, mod in data["dropout"]:
                # Replace with DifferentiableDropout
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = self.model.get_submodule(parent_name)
                
                diff_drop = DifferentiableDropout(p_val)
                setattr(parent, child_name, diff_drop)

        # 2. Attention Temperature Hook
        # We hook into the attention projectors to scale the output 
        # (Surrogate for scaling the scores themselves if internal logic is opaque)
        for block_key, data in self.blocks.items():
            idx = int(block_key.split(".")[-1])
            if idx >= len(hyperparams.attn_temp): continue
            
            tau = hyperparams.attn_temp[idx]
            
            for name, mod in data["mhsa"]:
                # Look for 'q_proj', 'k_proj' or similar
                if "q_proj" in name or "k_proj" in name or "query" in name or "key" in name:
                    def make_hook(t):
                        def hook(module, input, output):
                            return output / torch.sqrt(t) # Scale to change attention entropy
                        return hook
                    
                    mod.register_forward_hook(make_hook(tau))

    def wrap_loss_with_label_smoothing(self, smoothing: torch.Tensor):
        """Return a loss function that uses the differentiable smoothing parameter."""
        def smoothed_loss(logits, targets):
            # Sigmoid-bounded epsilon
            eps = smoothing
            
            log_probs = F.log_softmax(logits, dim=-1)
            n_classes = logits.size(-1)
            
            # Standard NLL
            ce_loss = F.nll_loss(log_probs, targets)
            
            # Uniform smoothing
            smooth_loss = -log_probs.mean(dim=-1).mean()
            
            return (1 - eps) * ce_loss + eps * smooth_loss
            
        return smoothed_loss
