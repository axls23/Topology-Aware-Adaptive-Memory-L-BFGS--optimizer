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

## 2026-03-15

### Feature Implemented
- Implemented first production integration of the Talpha-LBFGS remediation plan (core optimizer path), including Auto-Topology discovery state, sparse topology-aware curvature handling, corrected Powell damping path, dynamic spectral scaling, and EDRT refresh lifecycle.

### Code Changes

#### Config Surface
- Extended optimizer config in `ta_lbfgs/config.py` with:
  - `curvature_threshold` (mu, default 0.2).
  - Auto-Topology controls:
    - `auto_topology_enabled`,
    - `auto_topology_warmup_steps` (50),
    - `auto_topology_sketch_dim` (256),
    - `auto_topology_edge_top_percentile` (95.0),
    - `auto_topology_active_coords`,
    - `auto_topology_nnz_per_row`,
    - `auto_topology_edge_budget`.
  - EDRT controls:
    - `edrt_enabled`,
    - `edrt_refresh_interval` (1000),
    - `edrt_mini_warmup` (10),
    - `edrt_beta` (0.9),
    - `edrt_sparse_threshold` (0.05).

#### Baseline LBFGS Engine
- Updated `ta_lbfgs/core/baseline_lbfgs.py`:
  - Added constructor support for `damping`, `damping_eps`, `curvature_threshold`.
  - Added sparse topology mask API and projection helpers:
    - `set_topology_mask(...)`,
    - `_compress_with_topology(...)`,
    - `_expand_from_topology(...)`.
  - Added `old_alpha_scales` buffer and alpha-scaled rho path (`rho_i^alpha`).
  - Added `approximate_Bs(...)` to estimate direct-Hessian action for damping logic.
  - Reworked `curvature_update(...)` to:
    - operate in topology-compressed space,
    - use direct-Hessian action estimate for Powell damping,
    - compute dynamic `alpha_k` against `mu` threshold,
    - store per-pair alpha scales used in recursion.
  - Updated step direction path to run two-loop in compressed space and lift back for dense param updates.
  - Updated history resize logic to keep `old_alpha_scales` aligned with curvature buffers.

#### Layerwise Flow (Production Integration)
- Updated `ta_lbfgs/core/lbfgs.py`:
  - Extended `LayerState` with warmup and topology counters (`in_warmup`, refresh counters, edge count).
  - Added per-layer topology runtime storage (`self._topology_state`).
  - Added sparse Gaussian sketch initialization and online sketch-stat accumulation.
  - Added topology-mask finalization from warmup edge scores with optional EMA blending and hard sparsification.
  - Integrated warmup gate in `step_layer(...)`:
    - bypass curvature-history updates while collecting topology signal,
    - execute first-order warmup updates,
    - materialize mask at warmup completion.
  - Integrated EDRT trigger:
    - on periodic refresh interval, enter mini-warmup,
    - blend refreshed mask with EMA (`beta`),
    - enforce sparse thresholding.
  - Added topology edge count to layer dashboard payload.
  - Passed damping/curvature config through `register_layer(...)` when constructing `FullBatchLBFGS`.

#### Bilevel Wiring
- Updated `ta_lbfgs/training/bilevel.py` to pass:
  - `damping`,
  - `damping_eps`,
  - `curvature_threshold`
  into the outer-loop `FullBatchLBFGS` construction.

### Validation Performed
- Static diagnostics check for edited files: no reported errors.
- Regression tests run:
  - `pytest tests/test_bilevel_validity.py tests/test_online_rsvd.py`
  - Result: passed (4 tests total).

### Current Status
- The optimizer flow now contains production-side scaffolding and active logic for Auto-Topology + sparse-aware L-BFGS updates + EDRT refresh.
- Next tightening phase should add dedicated tests for:
  - alpha-branch correctness,
  - rho-alpha invariants,
  - mask-size/complexity regression (
    history scaling with edge-space dimensionality).

### Update: Real-Model Benchmark Profiling + Stability Hardening

#### Benchmark Profiles (Speed-Tuned vs Full-Fidelity)
- Added explicit benchmark profile semantics in `optuna_benchmark.py`:
  - `speed_tuned` profile for smoke/perf sweeps only.
  - `full_fidelity` profile for claim-valid reported numbers.
- Added `--profile-report` CLI switch (`both`, `speed_tuned`, `full_fidelity`).
- Updated MVP-4 reporting to include explicit side-by-side sparse profile comparison and claim-valid tagging.

#### Requested Speed-Tuned Defaults Applied
- Updated speed-tuned profile knobs:
  - `inner_steps=1`
  - `cg_max_iter=2`
  - `outer_hutchpp_samples=1`
  - `outer_hutchpp_trace_enabled=False`
  - `outer_hutchpp_diagonal_precondition_enabled=False`
  - `hybrid_hypergradient=True`
  - `hybrid_shard_fraction=0.125`
  - shorter sequence length cap (`max_length <= 64`)

#### Runtime Failure Fix (Full-Fidelity)
- Root issue observed during full-fidelity real-model run:
  - `BilevelValidationError: Non-finite hypergradient values`.
- Hardened `ta_lbfgs/training/bilevel.py`:
  - Added hypergradient sanitization helper with finite fallback path.
  - Added safe denominator handling (`nan_to_num` + clamp) for Hutchinson diagonal preconditioning.
  - Applied sanitize + clip flow in the Hutch preconditioning branch before state validation and update.
  - Preserved strict validation while preventing transient non-finite values from aborting long runs.

#### Validation Runs
- Real-model L-BFGS baseline completed:
  - Command:
    - `python optuna_benchmark.py --trials 40 --layers 24 --method lbfgs --signature-source hf_cached --hf-device auto --output-dir outputs/final_paper_20260315/lbfgs`
  - Result:
    - Best loss: `-0.106091`
    - Elapsed: `13.67s`
    - Artifacts under `outputs/final_paper_20260315/lbfgs`.
- Full-fidelity smoke re-run after fix completed:
  - Command:
    - `python optuna_benchmark.py --benchmark-design mvp4 --profile-report full_fidelity --trials 3 --layers 24 --signature-source hf_cached --hf-device auto --output-dir outputs/final_paper_20260315/full_fidelity_smoke`
  - Result:
    - ta-LBFGS best loss: `4.861778`
    - Elapsed: `74.84s`
    - MVP-4 report generated successfully.