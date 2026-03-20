# Project Context: Topology-Aware Adaptive-Memory L-BFGS (ta-LBFGS) Optimizer

This document provides an orientation for engineers and AI assistants to understand the ta-LBFGS codebase, which implements a novel second-order optimization framework for high-dimensional hyperparameter optimization.

---

## **Project Identity**
The **Topology-Aware Adaptive-Memory L-BFGS (ta-LBFGS)** project is an advanced optimization framework designed to bridge the gap between inefficient first-order hyperparameter optimization (HPO) and computationally prohibitive second-order methods. It treats HPO as a **bilevel optimization problem**—where hyperparameters are updated in an outer loop based on a model's performance in an inner training loop. By utilizing the **Implicit Function Theorem (IFT)** and a **layerwise block-diagonal approximation** of the Hessian, it enables efficient, topology-aware tuning of continuous hyperparameters (like learning rate, weight decay, and LoRA parameters) for large-scale models like Transformers and LLMs.

---

## **Architecture Overview**
The codebase is structured around a central package, `ta_lbfgs`, which isolates the mathematical core from the training loop and visualization layers.

- **`ta_lbfgs/core/`**: The heart of the optimizer. Contains the `ta-LBFGS` logic (`lbfgs.py`), the standard `torch.optim.LBFGS` baseline logic (`baseline_lbfgs.py`), and the IFT-based hypergradient computation (`hypergradient.py`).
- **`ta_lbfgs/topology/`**: Modules for analyzing the optimization landscape. Includes condition number estimation (`condition.py`), saddle point detection and evasion (`saddle.py`), and the adaptive history window controller (`adaptive_memory.py`).
- **`ta_lbfgs/training/`**: Logic for managing the bilevel optimization process. `bilevel.py` orchestrates the coordination between gradients of weights and gradients of hyperparameters. `inner_loop.py` defines how the model is trained between hyperparameter updates.
- **`ta_lbfgs/dashboard/`**: Real-time monitoring tools. Includes a **Textual-based TUI** (`textual_dashboard.py`) and a dimensionality reduction pipeline (`online_rsvd.py`) for projecting high-dimensional parameter trajectories into a 3D visualization.
- **`ta_lbfgs/utils/`**: General mathematical utilities and layer-specific management functions.

---

## **Tech Stack**
- **Language**: Python 3.8+
- **Deep Learning**: **PyTorch 2.0+** (Essential for higher-order derivatives and dynamic computation graphs).
- **Core Math**: NumPy, SciPy.
- **TUI & Visualization**: **Rich** and **Textual** (for the terminal dashboard), Plotly/Matplotlib (for offline report generation).
- **LLM Integration**: Hugging Face Ecosystem (`transformers`, `peft`, `datasets`).
- **Optimization Baselines**: **Optuna** (Used for benchmarking against state-of-the-art Bayesian Optimization).

---

## **Entry Points**
- **`demo.py`**: The primary script for running the optimizer. Supports both simple synthetic functions and real Hugging Face models (using `--model hf`).
- **`live_run.py`**: Launches the interactive **Textual TUI** which provides real-time feedback on landscape topology (kappa, saddle point status, and trajectory projection).
- **`hf_demo.py`**: A specialized entry point for optimizing Hugging Face transformers on specific datasets (e.g., Qwen2.5 on reasoning traces).
- **`optuna_benchmark.py`**: A validation script used to compare ta-LBFGS against standard Optuna/Bayesian Optimization basins.

---

## **Data Flow**
1. **Inner Loop**: The system performs a small number of training steps with current hyperparameters $\lambda$ (e.g., learning rate) to update the model weights $w$.
2. **Meta-Loss Computation**: A validation loss $\mathcal{L}_{val}(w(\lambda), \lambda)$ is calculated on a separate validation shard.
3. **Hypergradient Calculation**: Using the **Implicit Function Theorem (IFT)**, the system computes the gradient of the validation loss with respect to $\lambda$. This avoids the memory cost of "unrolling" the entire training loop by solving an inverse Hessian-vector system via Conjugate Gradient (CG) or Neumann series.
4. **Outer Loop Update**: The `ta-LBFGS` optimizer uses these hypergradients to update $\lambda$. 
   - **Topology Analysis**: The system calculates the local condition number $\kappa$ per layer.
   - **Adaptive Memory**: The history window $m_l$ for each layer is adjusted based on $\kappa$ to prevent zigzagging in narrow valleys.
   - **Saddle Evasion**: If the landscape is detected to be flat but non-optimal, a small orthogonal perturbation is injected to escape.

---

## **Domain Glossary**
- **IFT (Implicit Function Theorem)**: A mathematical tool used to compute hypergradients without backpropagating through time.
- **$\kappa$ (Kappa / Condition Number)**: A metric of landscape "flatness" or "steepness." High $\kappa$ indicates a narrow ravine requiring more memory and better curvature awareness.
- **Bilevel Optimization**: An optimization problem where one optimization (hyperparameters) is nested within another (model weights).
- **Layerwise / Block-Diagonal**: An approximation where we assume the Hessian is independent between layers, drastically reducing compute time for massive models.
- **Secant Condition**: $y_k^T s_k > 0$. If violated, it indicates the local region is not strictly convex (potential saddle point).
- **RSVD (Randomized SVD)**: Used for fast, incremental projection of parameter trajectories.

---

## **Key Conventions & Patterns**
- **Differentiable Hyperparameters**: All tuned parameters (Learning Rate, Weight Decay, LoRA Alpha) are registered as differentiable tensors in `ta_lbfgs/core/hyperparameters.py`.
- **Closure Pattern**: Similar to standard PyTorch L-BFGS, the `step()` method often requires a `closure` to re-evaluate loss and gradients.
- **TUI-First Monitoring**: The project prioritizes high-fidelity terminal monitoring over heavy web-based dashboards, making it ideal for remote GPU server environments.
- **Hybrid Sharding**: For very large models on small VRAM (6GB), the optimizer shards the parameter space and updates only a subset per outer iteration.

---

## **External Dependencies & Integrations**
- **Hugging Face (`transformers`/`peft`)**: The code interacts directly with `Model` and `PeftModel` objects to isolate layers and extract gradients.
- **Optuna**: Integrated for side-by-side performance comparisons.
- **vLLM (Legacy/Reference)**: Some parts of the project previously referenced vLLM APIs for inference, though current implementations focus on native PyTorch.

---

## **How to Run It**
### **1. Environment Setup**
```bash
pip install -r requirements.txt
```

### **2. Quick Demo (Synthetic Objective)**
```bash
python demo.py --optimizer ta-lbfgs
```

### **3. Live Optimization with HF Model & TUI**
```bash
python live_run.py --track-layer block.0
```

### **4. Benchmark against Bayesian Optimization**
```bash
python optuna_benchmark.py --trials 20 --layers 12
```

---

## **Known Landmines**
- **Hypergradient Instability**: The IFT solver (CG/Neumann) can occasionally produce non-finite values if the inner loop is extremely unstable. Guardrails are present in `bilevel.py`, but aggressive learning rates can still trigger them.
- **Graph Disconnections**: Care must be taken not to break the PyTorch computational graph during the inner loop (e.g., by calling `.item()` on a tensor being tuned).
- **VRAM Floor**: Running bilevel optimization on LLMs (like Qwen 0.5B or LLaMA 8B) requires significant VRAM. 6GB is the minimum with `--low-vram` and `full-hybrid` flags enabled.
- **Cold-Start Warmup**: The **Auto-Topology** discovery needs ~50 steps of first-order updates to populate the sparse masks correctly.

---

**Objective**: Rapid, curvature-aware tuning for PEFT-based multi head models fine-tuning.
