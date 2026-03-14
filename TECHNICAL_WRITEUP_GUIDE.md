# Technical Writeup Guide for ta-LBFGS Project Submission

This guide is a template and checklist for writing the final project report without fabricating missing evidence.

Use this rule throughout the report:
- If a result is not measured, write EVIDENCE NEEDED and describe the exact experiment required.
- If a claim is implemented but not benchmarked, write IMPLEMENTED, NOT YET VALIDATED.

## 1. Recommended Report Structure

1. Title and Abstract
2. Problem Statement and Motivation
3. Mathematical Formulation
4. Methodology and System Design
5. Implementation Details
6. Experimental Setup
7. Results
8. Discussion
9. Conclusion
10. Limitations and Future Work
11. Reproducibility Appendix

## 2. Abstract Template

Write in 120-180 words.

Template:
- We formulate continuous hyperparameter optimization as bilevel optimization for LLM distillation.
- We propose layerwise topology-aware adaptive-memory L-BFGS.
- We compute hypergradients using IFT with CG/HVP and low-VRAM hybrid safeguards.
- We evaluate on HF local model workflow and compare with classic LBFGS baseline.
- We report convergence behavior, stability observations, and memory-feasibility outcomes.

If baseline comparison is incomplete, add:
- EVIDENCE NEEDED: full parity benchmark against baseline methods.

## 3. Mathematical Formulation Section

### 3.1 Bilevel Objective

Use:

$$
\min_{\lambda} \; \mathcal{L}_{val}(w^*(\lambda), \lambda)
\quad\text{where}\quad
w^*(\lambda) = \arg\min_w \mathcal{L}_{train}(w, \lambda)
$$

### 3.2 Inner Update (Differentiable)

$$
w_{t+1} = w_t - \eta \nabla_w \mathcal{L}_{train}(w_t, \lambda)
$$

Note in text:
- Inner loop is differentiable through virtual parameter updates.

### 3.3 IFT Hypergradient

$$
\nabla_{\lambda}\mathcal{L}_{val}
=
\frac{\partial \mathcal{L}_{val}}{\partial \lambda}
-
\frac{\partial \mathcal{L}_{val}}{\partial w}
\left(\frac{\partial^2 \mathcal{L}_{train}}{\partial w^2}\right)^{-1}
\frac{\partial^2 \mathcal{L}_{train}}{\partial w\,\partial\lambda}
$$

Explain:
- CG solves linear systems involving Hessian-vector products.
- HVP is computed via double-backward, avoiding explicit Hessian construction.

### 3.4 Layerwise Adaptive Memory

$$
m_l = \mathrm{clip}(\lfloor \log(\kappa_l) \rfloor + m_{base}, \; m_{min}, \; m_{max})
$$

Define:
- $\kappa_l$: layer-local condition number.
- $m_l$: layer-local L-BFGS memory size.

### 3.5 Saddle Detection and Evasion

State decision rule:
- If $y_k^T s_k \le 0$, classify as saddle-like non-convex behavior and trigger perturbation.

If perturbation effectiveness not measured:
- EVIDENCE NEEDED: ablation with and without evasion logic.

## 4. Methodology and System Design

Describe these modules and point to implementation files:

1. Bilevel engine: [ta_lbfgs/training/bilevel.py](ta_lbfgs/training/bilevel.py)
2. Differentiable inner loop: [ta_lbfgs/training/inner_loop.py](ta_lbfgs/training/inner_loop.py)
3. Hypergradient solver: [ta_lbfgs/core/hypergradient.py](ta_lbfgs/core/hypergradient.py)
4. Layerwise ta-LBFGS: [ta_lbfgs/core/lbfgs.py](ta_lbfgs/core/lbfgs.py)
5. Adaptive memory rule: [ta_lbfgs/topology/adaptive_memory.py](ta_lbfgs/topology/adaptive_memory.py)
6. Demo and runtime controls: [demo.py](demo.py)

Include one architecture figure or table:
- Inputs: train/val batches, model weights, hyperparameters
- Inner loop output: adapted virtual weights
- Outer loop output: updated hyperparameters
- Topology signals: kappa, secant, memory size, evasion events

## 5. Implementation Details Section

Document only what is verifiably present:

1. Optimizer modes
- ta-lbfgs mode
- classic lbfgs mode

2. Scope and VRAM controls
- subset, full, hybrid scope controls
- low-vram mode
- hybrid shard fraction control

3. Monitoring UI
- textual dashboard with layer metrics and trajectory views

4. Data pipeline
- packed offline reasoning trace cache

If exact command behavior changed over time, provide current command examples only.

## 6. Experimental Setup Section

Include:

1. Hardware
- GPU model and VRAM (example: 6GB local GPU)

2. Model
- Qwen2.5-0.5B local snapshot workflow

3. Data
- reasoning trace dataset and cache mechanism

4. Optimization budgets
- outer steps, inner steps, sequence length, batch size

5. Evaluation metrics
- best validation loss
- convergence trajectory
- wall-clock time
- stability incidents (OOM, non-finite gradients, fallback count)

If any metric is not logged yet:
- EVIDENCE NEEDED: add logger and rerun.

## 7. Results Section Template

Use this table format and fill only measured values.

| Experiment ID | Mode | Scope | Low VRAM | Outer/Inner | Best Val Loss | Runtime | Stability Notes |
|---|---|---|---|---|---|---|---|
| Exp-1 | ta-lbfgs | subset or hybrid | yes/no | x/y | MEASURED VALUE | MEASURED VALUE | MEASURED VALUE |
| Exp-2 | lbfgs | same budget | yes/no | x/y | MEASURED VALUE | MEASURED VALUE | MEASURED VALUE |

For any missing entries:
- EVIDENCE NEEDED.

### 7.1 Minimum Required Plots

1. Validation loss vs outer iteration
2. Hyperparameter trajectories
3. Layerwise kappa and memory-size trends
4. Optional: 3D trajectory visualization

Reference current artifact locations where applicable:
- [outputs/dynamics.png](outputs/dynamics.png)
- [outputs/hyperparameters.png](outputs/hyperparameters.png)
- [outputs/trajectory_3d.html](outputs/trajectory_3d.html)

Only claim a figure as evidence if the corresponding experiment metadata is preserved.

## 8. Discussion Section

Discuss:

1. Why layerwise adaptive memory is theoretically useful.
2. Why CG/HVP is used instead of explicit Hessian inversion.
3. 6GB feasibility tradeoff:
- exact full-model second-order vs hybrid approximation.
4. Failure modes observed:
- OOM,
- non-finite hypergradients,
- graph-disconnect risk.

If a claim about superiority is not benchmarked, write:
- IMPLEMENTED, NOT YET VALIDATED AGAINST FAIR PARITY BASELINE.

## 9. Conclusion Section

Use a two-part conclusion:

1. What is completed and demonstrated.
2. What remains to claim full research validation.

Template sentence:
- The solution demonstrates end-to-end bilevel optimization with topology-aware controls and low-VRAM execution pathways; however, full parity benchmarking and ablation-backed superiority claims remain EVIDENCE NEEDED.

## 10. Reproducibility Appendix

Include:

1. Environment setup commands
2. Exact run commands used for reported tables
3. Seed strategy
4. Version/date and commit hash (if available)

Example command forms:
- python demo.py --model hf --optimizer ta-lbfgs --trainable-scope hybrid --low-vram --outer-steps N --inner-steps M
- python demo.py --model hf --optimizer lbfgs --outer-steps N --inner-steps M
- python -m pytest tests/test_bilevel_validity.py -q

## 11. Evidence Ledger (Do Not Skip)

Before submission, complete this checklist:

1. Every major claim has at least one measured artifact.
2. Every table row has exact command and run date.
3. Every figure has experiment ID linkage.
4. Any unmeasured claim is labeled EVIDENCE NEEDED.
5. No projected or estimated values are presented as measured results.

## 12. Current Known Status You Can Safely State

These are currently safe to state as implementation status (not final superiority claims):

1. Bilevel engine with differentiable inner loop and IFT hypergradient path is implemented.
2. Layerwise adaptive memory and saddle-condition logic are implemented.
3. ta-lbfgs and classic lbfgs modes are both runnable.
4. Low-VRAM hybrid pathway exists for constrained hardware execution.
5. Focused bilevel validity tests are present in [tests/test_bilevel_validity.py](tests/test_bilevel_validity.py).

Use cautious language for performance claims unless parity experiments are complete.
