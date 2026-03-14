# Development Log

## 2026-03-14

### Environment
- Workspace: C:/dev/Topology-Aware Adaptive-Memory L-BFGS  optimizer
- Python env: .venv (activated via .venv/Scripts/Activate.ps1)
- Environment ID: daf6d1772db9ae0727693f784b05454d

### Model Used
- HF model family: Qwen/Qwen2.5-0.5B
- Local snapshot path used by live run:
  - ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987

### Development Completed
- Implemented online layerwise RSVD pipeline for streaming trajectory projection.
- Added `ta_lbfgs/dashboard/online_rsvd.py` with:
  - Online randomized sketching and covariance updates.
  - Incremental basis updates with forgetting factor.
  - Explained variance ratio and confidence state.
- Extended dashboard config in `ta_lbfgs/config.py` with projection controls:
  - monitored layer, component count, sketch dimension, forgetting factor, warning threshold.
- Updated `ta_lbfgs/dashboard/textual_dashboard.py`:
  - Displays RSVD EVR/confidence in sidebar.
  - Accepts projection payload in `update_data`.
- Reworked `live_run.py` from surrogate vectors to real per-layer HF gradients:
  - Local HF model loading (no WSL/vLLM dependency).
  - Real gradient sketches from transformer blocks.
  - Layerwise kappa from gradient history, secant approximation, landscape labels.
  - RSVD receives real gradient streams for tracked layer trajectory.
- Added non-interactive mode (`--no-dashboard`) for smoke testing.
- Hardened live UI updates:
  - mount warmup before first update,
  - periodic iteration logs,
  - exception-safe dashboard logging.

### Validation Performed
- `python -m py_compile live_run.py` passed.
- `python live_run.py --steps 1 --track-layer block.0 --no-dashboard` ran successfully (exit code 0).
- `python demo.py --model hf --outer-steps 15` completed successfully (exit code 0).

### Current State (Latest Updates)

#### Bilevel and Hypergradient Core
- Implemented mathematically real bilevel objective path for HF mode:
  - Differentiable inner loop over virtual parameters (`params_override`).
  - Implicit outer hypergradients via IFT with CG/HVP.
- Added objective signature guardrails and runtime validity checks in bilevel engine:
  - Finite scalar checks for train/val losses.
  - Finite hypergradient checks.
  - Degenerate-gradient detection.
- Added finite-difference hypergradient sanity check (sign/magnitude consistency).
- Added inner-loop sensitivity diagnostics per step:
  - predicted sensitivity norm,
  - actual autograd sensitivity norm,
  - relative mismatch,
  - first suspected graph-disconnect step.
- Hardened hypergradient solver numerics:
  - safe handling of unused grads (`allow_unused=True` + zeros fallback),
  - NaN/Inf sanitization for final hypergradients,
  - fallback from unstable CG/Neumann outputs to direct gradient proxy.

#### Optimizer Modes and Runtime Control
- Added optimizer mode switch in demo:
  - `--optimizer ta-lbfgs` (topology-aware path),
  - `--optimizer lbfgs` (classic `torch.optim.LBFGS`).
- Fixed classic LBFGS closure behavior for repeated Wolfe evaluations (`retain_graph=True`).
- Added trainable scope control:
  - `--trainable-scope subset`,
  - `--trainable-scope full`,
  - `--trainable-scope hybrid`.
- Added low-VRAM control (`--low-vram`) and compatibility settings:
  - math SDP kernel forcing for second-order gradients,
  - gradient checkpointing enablement,
  - save-on-CPU in distillation path,
  - compact packed sequence defaults for constrained mode.

#### 6GB VRAM OOM Mitigation
- Implemented hybrid hypergradient strategy for low-VRAM full-scope requests:
  - auto-route `full + --low-vram` to hybrid mode,
  - rotating parameter shard selection per outer step,
  - exact IFT solve on active shard only,
  - reduced CG budget for low-VRAM safety.
- Added hybrid config knobs:
  - `hybrid_hypergradient`,
  - `hybrid_shard_fraction`,
  - `hybrid_rotation_steps`.
- Added CLI knob:
  - `--hybrid-shard-fraction`.

#### Dashboard and Monitoring
- Wired HF bilevel progress into live monitoring UI:
  - per-iteration outer state updates,
  - layerwise synthetic proxy table updates for live visibility,
  - trajectory updates,
  - sensitivity-disconnect warnings in log.

#### Data and Distillation Objective
- Replaced surrogate-only HF objective path with true token-level distillation loss for bilevel execution.
- Added differentiable causal label-smoothing loss in HF wrapper.
- Kept offline packed reasoning-trace cache in runtime loop.

#### Testing and Validation (Latest)
- Focused bilevel tests:
  - `python -m pytest tests/test_bilevel_validity.py -q` -> passed (2 tests).
- Classic LBFGS HF run:
  - `python demo.py --outer-steps 3 --no-dashboard --model hf --optimizer lbfgs` -> completed successfully.
- ta-LBFGS HF smoke run:
  - `python demo.py --outer-steps 2 --no-dashboard --model hf --optimizer ta-lbfgs` -> completed successfully.
- Low-VRAM full-scope command path now executes with hybrid routing enabled (OOM-first mitigation in place).

#### Known Active Focus
- For full-model low-VRAM hybrid runs, occasional non-finite hypergradient events can still appear in early iterations under aggressive settings; fallback guards are now in place, and ongoing tuning targets stability vs fidelity trade-offs (shard fraction, CG budget, sequence length).