# ta-LBFGS Integration Checklist

Last updated: 2026-03-20

## Overall Status
- [x] Phase 0 audit completed and documented
- [~] Phase 1 P0 fixes implemented (core fixes done; one private-function export mismatch remains for tests)
- [ ] Phase 2 new topology files fully implemented and wired
- [ ] Phase 3 parameter dispatch fully implemented and wired
- [ ] Phase 4 chain topology implemented
- [~] Phase 5 tests migrated to phase-based suite (structure done; full pass pending code completion)
- [ ] Final equilibrium ledger completed (target net LOC delta <= +50 not yet met)

## Prime Directive: Codebase Equilibrium
- [~] Delete-then-replace done for several P0 fixes
- [ ] Functional-surface equilibrium not yet achieved globally
- [ ] Net LOC delta target (<= +50) currently failing

## Phase 0 — Audit Before Editing
- [x] File audit table produced
- [x] Status mapping performed (KEEP/MODIFY/ABSORB/DELETE-BODY)
- [x] Equilibrium intent declared at file level

## Phase 1 — P0 Critical Fixes

### 1A hypergradient.py — Neumann series
- [x] Added spectral guard and guarded Neumann path
- [x] Replaced old Neumann expansion path
- [x] Removed silent non-finite fallback in Neumann branch

### 1B bilevel.py / inner_loop.py — inner loop regularization
- [x] Added explicit l2_inner_reg in config
- [x] Replaced raw inner loss with regularized inner loss
- [~] Prompt-level constant naming parity (L2_INNER_REG symbol in bilevel module) pending strict alignment

### 1C saddle.py — saddle detection and escape
- [x] Replaced yTs-only logic with Lanczos-style min-eigen probe
- [x] Replaced random orthogonal escape with eigenvector-directed escape
- [x] Wired perturbation flow in optimizer layer path

### 1D hypergradient.py / bilevel.py — outer preconditioner
- [x] Added hutchinson_diagonal
- [x] Added outer_precondition
- [x] Removed scalar-trace-as-diagonal path from active preconditioning flow

### 1E adaptive_memory.py / baseline_lbfgs.py
- [x] Added compute_window log-clamped mapping
- [x] Added AitkenAccelerator
- [x] Added stricter pair gate with EPS_REL
- [x] Replaced simple ys > 0 acceptance rule

## Phase 2 — New Topology Files
- [ ] Create [ta_lbfgs/topology/attention_topo.py](ta_lbfgs/topology/attention_topo.py)
- [ ] Create [ta_lbfgs/topology/moe_topo.py](ta_lbfgs/topology/moe_topo.py)
- [ ] Create [ta_lbfgs/utils/kfac.py](ta_lbfgs/utils/kfac.py)
- [ ] Wire attention topology into outer loop secant flow
- [ ] Wire MoE topology into expert buffer routing
- [ ] Wire KFAC embedding preconditioning path
- [ ] Consolidate existing topology files so non-init count remains <= 6

## Phase 3 — Parameter Group Dispatch
- [ ] Add PARAM_GROUP_TYPES in [ta_lbfgs/core/lbfgs.py](ta_lbfgs/core/lbfgs.py)
- [ ] Add classify_param_group in [ta_lbfgs/core/lbfgs.py](ta_lbfgs/core/lbfgs.py)
- [ ] Add AdamDiagPreconditioner in [ta_lbfgs/core/lbfgs.py](ta_lbfgs/core/lbfgs.py)
- [ ] Add should_freeze_in_inner_loop in [ta_lbfgs/core/lbfgs.py](ta_lbfgs/core/lbfgs.py)
- [ ] Re-export/align freeze guard in [ta_lbfgs/training/inner_loop.py](ta_lbfgs/training/inner_loop.py)
- [ ] Remove superseded uniform parameter iteration path(s)

## Phase 4 — Chain Topology
- [ ] Create [ta_lbfgs/topology/chain_topo.py](ta_lbfgs/topology/chain_topo.py)
- [ ] Implement ChainTopologyController API (on_outer_step, window_scale)
- [ ] Ensure equilibrium compensation for any new file growth

## Phase 5 — Test Stubs / Test Alignment
- [x] Phase-based test suite files created
- [x] Phase 0 tests passing (with best-effort orphan skip)
- [x] Phase 1 tests mostly passing
- [ ] Resolve Phase 2 import/runtime failures (missing modules)
- [ ] Resolve Phase 3 API surface failures (missing symbols)
- [ ] Resolve Phase 4 import/runtime failures (missing chain module)
- [ ] Resolve Phase 5 integration failures (missing TaLBFGS export and missing modules)
- [ ] Reach full green run for pytest tests -v

## Hard Constraints Status (Current)
- [~] No torch.autograd.grad inside inner training loop: currently not fully aligned (inner loop still uses autograd.grad)
- [~] No .item() on graph-critical tensors: mixed status, requires final audit once all phases are complete
- [x] Inner optimizer remains Adam/SGD in current implementation paths
- [ ] Freeze guard for rope/embedding not yet implemented in core dispatch API
- [x] Curvature pair gate upgraded to _is_valid_pair-style relative threshold
- [x] Lanczos saddle probe receives two_loop function handle in optimizer path
- [ ] KFAC inverse requirement pending (module not implemented yet)

## Current Test Snapshot
- Latest full run: 14 passed, 2 skipped, 1 xfailed, 28 failed, 8 errors
- Primary blockers: missing Phase 2/3/4 modules and API exports

## Next Actions
1. Implement Phase 2 modules (attention_topo, moe_topo, kfac) and wire them.
2. Implement Phase 3 dispatch API and freeze guard exports.
3. Implement Phase 4 chain_topo controller.
4. Re-run full tests and update this checklist + equilibrium ledger.
