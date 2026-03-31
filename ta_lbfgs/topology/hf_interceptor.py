"""
HF Interceptor utilities for five-axis topology extraction.

This module provides:
1) A pure snapshot builder that extracts topology signals from one
   Hugging Face model output object.
2) A lightweight online prompt hook that captures prefill outputs and
   forwards snapshots to the optimizer on a background worker.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from queue import SimpleQueue
from threading import Thread
from typing import Any, Dict, List, Optional

import torch


@dataclass
class TopologySnapshot:
    step: int
    attn_weights: Optional[Any] = None
    kv_key_norms: Optional[List[List[float]]] = None
    router_logits: Optional[Any] = None
    expert_topk_indices: Optional[List[torch.Tensor]] = None
    hidden_states: Optional[Any] = None
    layer_norm_drift: Optional[List[float]] = None
    logit_entropy: Optional[torch.Tensor] = None
    grad_norm: Optional[float] = None
    has_tied_embeddings: bool = False
    rope_is_tunable: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "attn_weights": self.attn_weights,
            "kv_key_norms": self.kv_key_norms,
            "router_logits": self.router_logits,
            "expert_topk_indices": self.expert_topk_indices,
            "hidden_states": self.hidden_states,
            "layer_norm_drift": self.layer_norm_drift,
            "logit_entropy": self.logit_entropy,
            "grad_norm": self.grad_norm,
            "has_tied_embeddings": self.has_tied_embeddings,
            "rope_is_tunable": self.rope_is_tunable,
        }


def detect_moe_model(model_config: Any) -> bool:
    return any(
        hasattr(model_config, attr)
        for attr in (
            "num_experts",
            "num_local_experts",
            "num_experts_per_tok",
            "moe_num_experts",
            "n_routed_experts",
        )
    )


def can_output_attentions(model_config: Any) -> bool:
    attn_impl = getattr(model_config, "_attn_implementation", "eager")
    return attn_impl not in ("flash_attention_2", "sdpa")


def build_topology_snapshot(
    outputs: Any,
    step: int,
    warmup_steps: int,
    model_config: Any,
) -> Dict[str, Any]:
    """Pure extraction from one HF model output object."""
    snap = TopologySnapshot(step=step)

    snap.attn_weights = getattr(outputs, "attentions", None)

    past_kv = getattr(outputs, "past_key_values", None)
    if past_kv is not None:
        snap.kv_key_norms = []
        for item in past_kv:
            # Handle both (key, value) tuples and DynamicCache or other objects
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                keys = item[0]
            elif hasattr(item, "key"):
                keys = item.key
            else:
                continue
            
            try:
                norms = keys.detach().norm(dim=-1).mean(dim=(0, 2))
                snap.kv_key_norms.append(norms.tolist())
            except Exception:
                continue

    raw_router = getattr(outputs, "router_logits", None)
    if raw_router is not None:
        snap.router_logits = raw_router
        k = int(getattr(model_config, "num_experts_per_tok", 2))
        snap.expert_topk_indices = [
            logits.detach().topk(k, dim=-1).indices for logits in raw_router
        ]

    hidden = getattr(outputs, "hidden_states", None)
    if hidden is not None and len(hidden) > 1:
        snap.hidden_states = hidden
        drifts: List[float] = []
        for lidx in range(len(hidden) - 1):
            h0 = hidden[lidx].detach().float()
            h1 = hidden[lidx + 1].detach().float()
            drift = ((h1 - h0).norm() / (h0.norm() + 1e-8)).item()
            drifts.append(float(drift))
        snap.layer_norm_drift = drifts
    elif snap.kv_key_norms is not None and len(snap.kv_key_norms) > 1:
        mean_norms = [sum(h) / max(1, len(h)) for h in snap.kv_key_norms]
        snap.layer_norm_drift = [
            abs(mean_norms[i + 1] - mean_norms[i]) / (mean_norms[i] + 1e-8)
            for i in range(len(mean_norms) - 1)
        ]

    logits = getattr(outputs, "logits", None)
    if logits is not None:
        with torch.no_grad():
            probs = logits.detach().float().softmax(dim=-1)
            snap.logit_entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)

    tied_word_embeddings = bool(getattr(model_config, "tie_word_embeddings", False))
    rope_scaling = getattr(model_config, "rope_scaling", None)
    snap.has_tied_embeddings = tied_word_embeddings
    snap.rope_is_tunable = rope_scaling is not None

    # This field is always injected by the outer loop after hypergradient.
    snap.grad_norm = None
    return snap.as_dict()


class OnlineTopologyHook:
    """Capture topology signals during prompt prefill and process asynchronously."""

    def __init__(self, model, optimizer, warmup_steps: int = 999) -> None:
        self.model = model
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self._step = 0
        self._queue: SimpleQueue = SimpleQueue()
        self._worker = Thread(target=self._process_loop, daemon=True)
        self._worker.start()

        cfg = model.config
        self._can_attn = can_output_attentions(cfg)
        self._is_moe = detect_moe_model(cfg)
        try:
            self._forward_params = set(inspect.signature(model.forward).parameters.keys())
        except (TypeError, ValueError):
            self._forward_params = set()

    def on_prompt(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """Wrap one prefill forward pass and enqueue a topology snapshot."""
        orig_forward = self.model.forward
        captured: Dict[str, Any] = {}

        def _capturing_forward(*args, **kwargs):
            if "output_attentions" in self._forward_params:
                kwargs["output_attentions"] = self._can_attn
            if self._is_moe and "output_router_logits" in self._forward_params:
                kwargs["output_router_logits"] = True
            if "output_hidden_states" in self._forward_params:
                kwargs["output_hidden_states"] = True
            if "use_cache" in self._forward_params:
                kwargs["use_cache"] = True
            if "return_dict" in self._forward_params:
                kwargs["return_dict"] = True
            out = orig_forward(*args, **kwargs)
            captured["out"] = out
            return out

        self.model.forward = _capturing_forward
        try:
            kwargs = {"input_ids": input_ids}
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask
            outputs = self.model(**kwargs)
        finally:
            self.model.forward = orig_forward

        snap = build_topology_snapshot(
            captured.get("out", outputs),
            step=self._step,
            warmup_steps=self.warmup_steps,
            model_config=self.model.config,
        )
        self._step += 1
        self._queue.put(snap)
        return outputs

    def _process_loop(self) -> None:
        while True:
            snap = self._queue.get()
            if snap is None:
                break
            try:
                if hasattr(self.optimizer, "ingest_topology_snapshot"):
                    self.optimizer.ingest_topology_snapshot(snap)
            except Exception as exc:
                print(f"[OnlineTopologyHook] error: {exc}")

    def shutdown(self) -> None:
        self._queue.put(None)
        self._worker.join(timeout=5)