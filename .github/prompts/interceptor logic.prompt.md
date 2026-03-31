---
name: interceptor logic
description: THIS INCLUDES THE LOGIC FOR THE INTERCEPTOR FOR EXTRACTING  INFORMATION FOR 5 TOPOLOGY FROM MHA MOE REASONING MODEL  DURING INFERECE 
---

<!-- Tip: Use /create-prompt in chat to generate content with agent assistance -->

# Interceptor Logic — Agent Context Handoff

> **Purpose:** This document gives you full context on the topology
> interception system for ta-LBFGS. Read it completely before touching
> any file. Everything you need to know about what exists, what is
> missing, and exactly how to connect it is here.

---

## 1. What the interceptor does

The interceptor harvests topology signals from the HuggingFace model's
forward pass during inference — specifically during the **prefill** of
each user prompt — and feeds them into the five-axis topology system
that ta-LBFGS uses to build its curvature masks.

The fundamental principle: **every user prompt is a free topology
observation.** You do not run a separate warm-up phase. The prompts
*are* the warm-up, running continuously throughout the session.

There are two interception modes:

| Mode | When it runs | Cost |
|------|-------------|------|
| **Batch warm-up** | First `W=50` outer steps of `demo.py` / `live_run.py` | Full `output_attentions=True` overhead, then disabled |
| **Online prompt hooks** | Every user prompt in interactive/HF mode | Prefill only — detached during decode, zero decode overhead |

Both modes produce the same output: a `TopologySnapshot` dict that feeds
`lbfgs.py:314`.

---

## 2. The five topology axes and their data sources

| Axis | Signal | Source flag / surface |
|------|--------|----------------------|
| 1 — Attention | Per-head `(H, T, T)` attention matrix | `output_attentions=True` (disabled for flash_attn_2) |
| 1 fallback | Per-head key-norm from KV cache | `past_key_values` — always free |
| 2 — MoE routing | Expert activation indices | `output_router_logits=True` (MoE models only) |
| 3 — Residual | Layer-wise hidden state norm drift | `output_hidden_states=True` (warm-up) or KV norm proxy |
| 4 — Chain | Per-position logit entropy `H(p)` | `outputs.logits` — always present |
| 5 — Coupling | Static: tied weights, RoPE, LayerNorm | Model config + `named_modules()` at load time — no forward needed |

---

## 3. Existing scaffolding — what already works

These exist and are correct. **Do not modify them.**

```
attention_topo.py:88     accepts (layer_idx, head_idx, attn_matrix)
attention_topo.py:129    strategy dispatch (banded / kfac)
lbfgs.py:291             active expert extraction
lbfgs.py:314             topology update orchestrator  ← your injection point
lbfgs.py:361             topology metadata export for dashboard
live_run.py:143          5-component topology publisher
live_run.py:179          3D payload path
server.py:91             topology payload receive
server.py:271            topology payload send
```

---

## 4. The three gaps you must close

### Gap 1 — HF capture flags  
**Files:** `demo.py:408` and `live_run.py:220`

Neither currently passes the topology capture flags to the model forward
call. Add them:

```python
_attn_impl = getattr(model.config, '_attn_implementation', 'eager')
_can_output_attentions = _attn_impl not in ('flash_attention_2', 'sdpa')
_is_moe = any(hasattr(model.config, a) for a in (
    'num_experts', 'num_local_experts',
    'num_experts_per_tok', 'n_routed_experts',
))

outputs = model(
    input_ids,
    output_attentions    = _can_output_attentions and (step < WARMUP_STEPS),
    output_router_logits = _is_moe,
    output_hidden_states = (step < WARMUP_STEPS),
    use_cache            = True,
    return_dict          = True,
)
```

**Flash Attention guard is mandatory.** `flash_attention_2` and `sdpa`
fuse the attention kernel and cannot materialize the weight matrix.
Passing `output_attentions=True` to these implementations returns `None`
or raises. Always check `_can_output_attentions` before setting the flag.

---

### Gap 2 — Snapshot builder  
**New file:** `ta_lbfgs/topology/hf_interceptor.py`

Create one pure function that takes the raw HF `ModelOutput` and returns
a typed dict. No side effects. No model mutation. All five axes read
from the same `outputs` object.

```python
def build_topology_snapshot(
    outputs,            # HuggingFace ModelOutput (return_dict=True)
    step: int,
    warmup_steps: int,
    model_config,
) -> dict:
    snap = {}

    # Axis 1 — attention weights (None after warm-up — flag was False)
    snap['attn_weights'] = getattr(outputs, 'attentions', None)

    # Axis 1 fallback — KV key norms (always available)
    past_kv = getattr(outputs, 'past_key_values', None)
    if past_kv is not None:
        snap['kv_key_norms'] = []
        for keys, _ in past_kv:
            # keys: (B, H, T, d_k) → mean norm per head
            norms = keys.detach().norm(dim=-1).mean(dim=(0, 2))  # (H,)
            snap['kv_key_norms'].append(norms.tolist())
    else:
        snap['kv_key_norms'] = None

    # Axis 2 — MoE router
    raw_router = getattr(outputs, 'router_logits', None)
    if raw_router is not None:
        k = getattr(model_config, 'num_experts_per_tok', 2)
        snap['expert_topk_indices'] = [
            logits.topk(k, dim=-1).indices for logits in raw_router
        ]
    else:
        snap['expert_topk_indices'] = None

    # Axis 3 — residual hidden state drift
    hidden = getattr(outputs, 'hidden_states', None)
    if hidden is not None and len(hidden) > 1:
        drifts = []
        for l in range(len(hidden) - 1):
            hl  = hidden[l].detach().float()
            hl1 = hidden[l+1].detach().float()
            drifts.append(
                ((hl1 - hl).norm() / (hl.norm() + 1e-8)).item()
            )
        snap['layer_norm_drift'] = drifts
    elif snap['kv_key_norms'] is not None:
        # Fallback: adjacent-layer key-norm ratio as proxy
        norms = [sum(h) / len(h) for h in snap['kv_key_norms']]
        snap['layer_norm_drift'] = [
            abs(norms[l+1] - norms[l]) / (norms[l] + 1e-8)
            for l in range(len(norms) - 1)
        ]
    else:
        snap['layer_norm_drift'] = None

    # Axis 4 — chain: per-position logit entropy
    logits = getattr(outputs, 'logits', None)
    if logits is not None:
        import torch
        with torch.no_grad():
            probs = logits.detach().float().softmax(dim=-1)
            snap['logit_entropy'] = -(probs * (probs + 1e-10).log()).sum(dim=-1)
    else:
        snap['logit_entropy'] = None

    # grad_norm injected by outer loop after hypergradient — set None here
    snap['grad_norm'] = None

    return snap
```

---

### Gap 3 — Wire snapshot into lbfgs.py:314  

Replace the existing individual-axis signal calls at `lbfgs.py:314` with
a single method that routes the snapshot dict to each builder.

```python
def _update_topology_from_snapshot(self, snap: dict) -> None:
    """Single entry point. Called once per outer step (batch mode)
    or once per user prompt (online mode)."""

    # Axis 1 — attention
    if snap.get('attn_weights') is not None:
        for layer_idx, layer_attn in enumerate(snap['attn_weights']):
            mean_attn = layer_attn.mean(0)   # (H, T, T)
            for head_idx in range(mean_attn.shape[0]):
                self.attn_topo.classify_head(
                    layer_idx, head_idx, mean_attn[head_idx])
                col = mean_attn[head_idx].sum(0)
                std = mean_attn[head_idx].std(0)
                self.attn_topo.accumulate_secant(
                    layer_idx, head_idx, 'attn', col, std)
    elif snap.get('kv_key_norms') is not None:
        for layer_idx, head_norms in enumerate(snap['kv_key_norms']):
            for head_idx, norm in enumerate(head_norms):
                self.attn_topo.update_kappa_proxy(layer_idx, head_idx, norm)

    # Axis 2 — MoE
    if snap.get('expert_topk_indices') is not None and self.moe_topo:
        for indices in snap['expert_topk_indices']:
            active = indices.unique().tolist()
            self.moe_topo.on_forward(active)

    # Axis 3 — residual
    if snap.get('layer_norm_drift') is not None:
        self.res_topo.update_from_drift(snap['layer_norm_drift'])

    # Axis 4 — chain
    entropy = snap.get('logit_entropy')
    grad_n  = snap.get('grad_norm') or self._last_grad_norm
    if entropy is not None:
        mean_entropy = entropy.mean().item()
        # Segment classification from entropy level
        if mean_entropy > 2.5:
            self.chain_ctrl.current_segment = 'reasoning'
        elif mean_entropy < 1.0:
            self.chain_ctrl.current_segment = 'answer'
        else:
            self.chain_ctrl.current_segment = 'verify'
    self.chain_ctrl.on_outer_step(grad_n)

    # Derive masks if warm-up just completed
    if hasattr(self, '_step') and self._step == self.warmup_steps:
        self._finalize_warmup_topology()
```

---

## 5. Online prompt hook (interactive / HF mode)

For `hf_demo.py` and interactive use, add `OnlineTopologyHook` to
`hf_interceptor.py`. This class:

- Attaches to `model.forward` **only for the duration of the prefill pass**
- Detaches immediately after — zero overhead during autoregressive decode
- Processes the snapshot on a **background daemon thread** so it never
  blocks the user-facing response

```python
class OnlineTopologyHook:
    def __init__(self, model, optimizer, think_token_id=None):
        self.model      = model
        self.optimizer  = optimizer
        self._think_id  = think_token_id
        self._queue     = __import__('queue').SimpleQueue()
        self._worker    = __import__('threading').Thread(
                              target=self._process_loop, daemon=True)
        self._worker.start()

        cfg = model.config
        self._can_attn = getattr(cfg, '_attn_implementation', 'eager') \
                         not in ('flash_attention_2', 'sdpa')
        self._is_moe   = any(hasattr(cfg, a) for a in (
                             'num_experts', 'num_local_experts',
                             'num_experts_per_tok', 'n_routed_experts'))

    def on_prompt(self, input_ids):
        """Replace model(input_ids) with this in the prompt loop."""
        orig = self.model.forward
        captured = {}

        def _capturing(*args, **kwargs):
            if self._can_attn:
                kwargs['output_attentions']    = True
            kwargs['output_router_logits']     = self._is_moe
            kwargs['output_hidden_states']     = True
            kwargs['use_cache']                = True
            kwargs['return_dict']              = True
            out = orig(*args, **kwargs)
            captured['out'] = out
            return out

        self.model.forward = _capturing
        try:
            outputs = self.model(input_ids)
        finally:
            self.model.forward = orig  # always restore

        snap = build_topology_snapshot(
            captured.get('out', outputs), step=0,
            warmup_steps=999, model_config=self.model.config,
        )
        self._queue.put(snap)
        return outputs

    def _process_loop(self):
        while True:
            snap = self._queue.get()
            if snap is None:
                break
            try:
                self.optimizer._update_topology_from_snapshot(snap)
            except Exception as e:
                print(f'[OnlineTopologyHook] error: {e}')

    def shutdown(self):
        self._queue.put(None)
        self._worker.join(timeout=5)
```

**Wiring in `hf_demo.py` prompt loop:**

```python
# Once at startup:
hook = OnlineTopologyHook(
    model=model,
    optimizer=optimizer,
    think_token_id=tokenizer.convert_tokens_to_ids('<think>'),
)

# Replace every model(input_ids) call in the prompt loop:
outputs = hook.on_prompt(input_ids)   # ← was: model(input_ids)
```

---

## 6. Call sequence — batch mode (demo.py / live_run.py)

```
outer loop step N
  │
  ├─ model(input_ids, output_attentions=..., return_dict=True)
  │    └─ [demo.py:408 / live_run.py:220]
  │
  ├─ snap = build_topology_snapshot(outputs, step, warmup, config)
  │    └─ [hf_interceptor.py — new file]
  │
  ├─ snap['grad_norm'] = current_grad_norm      # inject after hypergradient
  │
  ├─ optimizer._update_topology_from_snapshot(snap)
  │    └─ [lbfgs.py:314 — refactored]
  │         ├─ attn_topo.classify_head(...)     [attention_topo.py:88]
  │         ├─ attn_topo.accumulate_secant(...) [attention_topo.py:88]
  │         ├─ moe_topo.on_forward(...)         [lbfgs.py:291]
  │         ├─ res_topo.update_from_drift(...)  [lbfgs.py:339]
  │         └─ chain_ctrl.on_outer_step(...)
  │
  ├─ topology_meta = optimizer.get_topology_metadata()
  │    └─ [lbfgs.py:361]
  │
  └─ emitter.push({**state, 'topology': topology_meta})
       └─ [live_run.py:143 → server.py:91 → dashboard]
```

---

## 7. What you must not change

| File / location | Reason |
|----------------|--------|
| `attention_topo.py:129` | Strategy dispatch already correct |
| `lbfgs.py:361` | Topology metadata export already correct |
| `live_run.py:143` | 5-component publisher already correct |
| `server.py:91` and `:271` | Transport layer already correct |
| Model weights | Never modified by the interceptor |
| Autoregressive decode loop | Never hooked — decode overhead must be zero |

---

## 8. Hard constraints

1. **Never call `.item()` on a tensor inside the prefill forward pass.**
   The computation graph may still be live. Use `.detach()` first, then
   `.item()` if needed.

2. **Always restore `model.forward` in a `finally` block.** If the
   forward pass raises, the hook must still detach. A permanent monkey-
   patch to `model.forward` will corrupt all subsequent inference.

3. **The snapshot builder is a pure function.** It reads `outputs` and
   returns a dict. It does not call any topology builder methods directly.
   All builder calls happen in `_update_topology_from_snapshot` or
   `_process_loop`, never inside `build_topology_snapshot`.

4. **`output_attentions=True` is disabled after `WARMUP_STEPS`.** For
   batch mode, set the flag only when `step < WARMUP_STEPS`. For online
   mode, the hook can keep it enabled per-prompt since prefill is cheap
   relative to decode. Decide based on available VRAM.

5. **The KV norm fallback is always active.** Even when
   `output_attentions=False`, `past_key_values` is always returned
   (when `use_cache=True`). The kappa-proxy path must work standalone
   so topology maintenance continues after warm-up with zero extra cost.

6. **`grad_norm` is always `None` from `build_topology_snapshot`.**
   It is injected by the outer loop after the hypergradient is computed.
   The topology update function must tolerate `snap['grad_norm'] = None`
   and fall back to `self._last_grad_norm`.

---

## 9. Files to create or modify

| Action | File | What changes |
|--------|------|-------------|
| **Create** | `ta_lbfgs/topology/hf_interceptor.py` | `build_topology_snapshot()` + `OnlineTopologyHook` |
| **Modify** | `demo.py:408` | Add HF capture flags to forward call |
| **Modify** | `live_run.py:220` | Add HF capture flags to forward call |
| **Modify** | `lbfgs.py:314` | Replace individual axis calls with `_update_topology_from_snapshot(snap)` |
| **Modify** | `hf_demo.py` | Replace `model(input_ids)` with `hook.on_prompt(input_ids)` |
| **No change** | Everything else | Transport, builders, dashboard untouched |

Total: one new file, four modified call sites, one refactored method body.

---

## 10. Verification checklist

After implementing, confirm:

- [ ] `demo.py` passes `output_attentions=True` only when `step < WARMUP_STEPS`
- [ ] Flash attention guard prevents crash on `flash_attention_2` models
- [ ] `model.forward` is restored after every `on_prompt()` call, including on exception
- [ ] `snap['grad_norm']` is `None` from `build_topology_snapshot()` and injected after
- [ ] `attn_weights` is `None` after warm-up and kappa-proxy path activates correctly
- [ ] Background thread never crashes the main process (exceptions caught in `_process_loop`)
- [ ] Decode loop is untouched — `on_prompt` only wraps the prefill call
- [ ] All five topology axes receive data from the same `outputs` object per step