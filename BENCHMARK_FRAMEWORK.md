# Benchmarking Framework: Talpha-LBFGS vs L-BFGS

This repository now supports a tiered benchmark protocol aligned with the paper claims.

## Tier 1: Core Claims Validation

### 1.1 Sparse Structured Tasks (expected Talpha wins)
- MoE routing
- Graph node classification
- Sparse transformer attention

Current runnable proxy in this repo:
- Real HF prompt-tuning objective via local cached model
- Command:
  - python optuna_benchmark.py --benchmark-design mvp4 --trials 40 --layers 24 --signature-source hf_cached --hf-device auto --output-dir outputs/bench_protocol

### 1.2 Dense Tasks (expected L-BFGS wins)
- Dense linear regression (implemented)
- Fully connected MLP tabular (design target)

Implemented dense benchmark:
- Dense OLS stress test inside protocol runner
- Reports final MSE and wall-clock for both optimizers

## Tier 2: Component Validation

### 2.1 Complexity claim: O(mN) to O(m|E|)
Implemented in protocol runner as step-time scaling with N:
- N = 10K, 100K, 500K
- Outputs per-step timing table for L-BFGS and Talpha-LBFGS

### 2.2 Hypergradient stability claim
Implemented in protocol runner:
- Uses ta-LBFGS outer-loop grad norm history
- Reports:
  - max norm
  - mean tail norm (last 10)
  - count above threshold 0.8

### 2.3 EDRT mask stability claim
Design target in this repo:
- metric: ||M_t - M_{t-1}||_F / ||M_{t-1}||_F
- next step: expose mask snapshots per EDRT refresh in layer optimizer state

### 2.4 Powell damping effectiveness
Design target in this repo:
- metric: fraction of positive curvature s_tilde^T y_hat
- next step: add curvature histogram logging hook in baseline_lbfgs

## Tier 3: Standard ML Credibility

Recommended external benchmarks (not fully wired in this script):
- CIFAR-10/100 ResNet-18
- GPT-2 small fine-tuning on WikiText-2
- Meta-HPO with Bayesian optimization and random search baselines

## Reporting Format

Protocol report generated at:
- outputs/<run_dir>/benchmark_protocol_report.md

Table schema:
- Task
- Sparsity Type
- L-BFGS
- Talpha-LBFGS
- Winner
- Predicted by Theory?

## Minimum Viable Set (MVP-4)

Implemented command:
- python optuna_benchmark.py --benchmark-design mvp4 --trials 40 --layers 24 --signature-source hf_cached --hf-device auto --output-dir outputs/bench_protocol

MVP-4 covers:
1. Sparse real-model proxy
2. Dense regression limitation check
3. Complexity scaling
4. Hypergradient stability
