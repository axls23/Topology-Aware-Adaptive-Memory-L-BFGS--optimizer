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
