# Topology-Aware Adaptive Memory L-BFGS Optimizer

## Overview
The **Topology-Aware Adaptive Memory L-BFGS Optimizer** is an advanced optimization algorithm designed to improve the efficiency and convergence behavior of traditional Limited-memory Broyden–Fletcher–Goldfarb–Shanno (L-BFGS) methods. The optimizer integrates topology-aware mechanisms with adaptive memory management to dynamically adjust the stored curvature information during training or optimization.

This approach aims to enhance performance in high-dimensional optimization problems commonly encountered in machine learning, deep learning, and scientific computing. By incorporating structural awareness of the parameter space and adaptive selection of memory updates, the optimizer can achieve better stability and faster convergence compared to standard L-BFGS implementations.

The repository provides an implementation of this optimizer and demonstrates how topology-aware mechanisms can be used to improve second-order optimization techniques.

---

## Key Features

### 1. Topology Awareness
The optimizer analyzes the structure of the parameter space and the gradient landscape to identify meaningful curvature information. This allows it to selectively retain or discard historical updates based on their relevance.

### 2. Adaptive Memory Management
Instead of using a fixed number of stored curvature pairs as in classical L-BFGS, this implementation adapts the memory dynamically based on the quality and usefulness of the information collected during optimization.

### 3. Efficient Large-Scale Optimization
The algorithm is designed to work efficiently with high-dimensional parameter spaces, making it suitable for deep neural networks and other large optimization problems.

### 4. Improved Stability
By incorporating topology-based filtering and adaptive memory policies, the optimizer can reduce noisy updates and avoid poor curvature approximations.

### 5. Modular Implementation
The codebase is structured to allow easy integration with existing machine learning frameworks and experimentation with optimization strategies.

---

## Background

### Limited-memory BFGS (L-BFGS)
L-BFGS is a quasi-Newton optimization algorithm that approximates the inverse Hessian matrix using a limited amount of stored information from previous iterations. Unlike classical Newton methods, it does not require storing or computing the full Hessian matrix.

Instead, it maintains a small set of vector pairs representing parameter updates and gradient differences:

- **s_k = x_{k+1} - x_k**
- **y_k = ∇f(x_{k+1}) - ∇f(x_k)**

These vectors are used to build an approximation of the inverse Hessian matrix.

### Limitations of Standard L-BFGS
Traditional L-BFGS uses a fixed memory size and does not consider the structure or topology of the optimization landscape. This can lead to:

- Inefficient use of stored curvature information
- Poor approximation of the Hessian in noisy or non-convex settings
- Instability during optimization

### Motivation for Topology-Aware Adaptive Memory
The topology-aware adaptive memory mechanism aims to:

- Retain only meaningful curvature information
- Adapt memory size dynamically
- Improve robustness in complex optimization landscapes

---

## Repository Structure
Topology-Aware-Adaptive-Memory-L-BFGS-optimizer/
│
├── README.md
├── optimizer/
│ ├── lbfgs_optimizer.py
│ └── memory_manager.py
│
├── experiments/
│ ├── benchmark.py
│ └── test_cases.py
│
├── utils/
│ ├── math_utils.py
│ └── topology_analysis.py
│
└── requirements.txt


### Directory Description

| Directory/File | Description |
|----------------|-------------|
| `optimizer/` | Core implementation of the topology-aware L-BFGS optimizer |
| `experiments/` | Scripts used for testing and benchmarking the optimizer |
| `utils/` | Helper functions for mathematical operations and topology analysis |
| `requirements.txt` | Python dependencies required to run the project |
| `README.md` | Documentation for the repository |

---

## Installation

### Prerequisites
- Python 3.8 or later
- pip package manager

### Clone the Repository

```bash
git clone https://github.com/axls23/Topology-Aware-Adaptive-Memory-L-BFGS--optimizer.git
cd Topology-Aware-Adaptive-Memory-L-BFGS--optimizer
```
### Install Dependencies
pip install -r requirements.txt

### Usage - Basic Example:

```python
# Below is an example showing how the optimizer can be used in a training loop.
from optimizer.lbfgs_optimizer import TopologyAwareLBFGS

# Initialize optimizer
optimizer = TopologyAwareLBFGS(
    parameters=model.parameters(),
    lr=0.1,
    memory_size=10
)

for epoch in range(num_epochs):

    def closure():
        optimizer.zero_grad()
        loss = model(input_data)
        loss.backward()
        return loss

    optimizer.step(closure)
```

## Algorithm Workflow

The optimizer follows an iterative quasi-Newton optimization process enhanced with topology-aware adaptive memory selection.

1. **Initialization**
   - Initialize model parameters \(x_0\).
   - Initialize empty memory buffers to store curvature pairs.
   - Set hyperparameters such as learning rate, memory size, and topology filtering thresholds.

2. **Gradient Computation**
   - Compute the gradient of the objective function:
     \[
     g_k = \nabla f(x_k)
     \]

3. **Compute Parameter Difference**
   - Calculate the difference between consecutive parameter vectors:
     \[
     s_k = x_{k+1} - x_k
     \]

4. **Compute Gradient Difference**
   - Compute the difference between gradients:
     \[
     y_k = g_{k+1} - g_k
     \]

5. **Topology-Aware Evaluation**
   - Analyze the curvature information represented by \(s_k\) and \(y_k\).
   - Determine whether the curvature pair contains useful structural information.
   - Discard pairs that do not meet the topology criteria.

6. **Adaptive Memory Update**
   - Store accepted curvature pairs in the memory buffer.
   - If the buffer exceeds the maximum memory size:
     - Remove the least useful pair based on topology metrics.

7. **Inverse Hessian Approximation**
   - Use the stored curvature pairs to approximate the inverse Hessian matrix.
   - Apply the two-loop recursion procedure used in L-BFGS.

8. **Compute Search Direction**
   - Compute the search direction:
     \[
     p_k = -H_k g_k
     \]

9. **Line Search**
   - Perform a line search procedure to determine a suitable step size \(\alpha_k\).

10. **Parameter Update**
    - Update the parameters:
      \[
      x_{k+1} = x_k + \alpha_k p_k
      \]

11. **Convergence Check**
    - Stop the optimization if the gradient norm or loss change falls below a specified tolerance.
    - Otherwise repeat the process.

---

## Configuration Parameters

The optimizer exposes several parameters that control its behavior and memory management strategy.

| Parameter | Type | Description |
|----------|------|-------------|
| `lr` | float | Learning rate used to scale parameter updates |
| `memory_size` | int | Maximum number of curvature pairs stored in memory |
| `topology_threshold` | float | Threshold used to determine whether curvature information should be retained |
| `adapt_memory` | bool | Enables adaptive memory resizing during optimization |
| `max_iter` | int | Maximum number of optimization iterations |
| `tolerance_grad` | float | Convergence tolerance based on gradient magnitude |

These parameters can be tuned depending on the optimization problem and model size.

---

## Applications

The topology-aware adaptive memory L-BFGS optimizer can be applied to a wide range of optimization problems.

### Machine Learning
Used for optimizing models where second-order information improves convergence speed.

Examples include:
- Logistic regression
- Support vector machines
- Kernel-based learning models

### Deep Learning
Although first-order methods are commonly used, L-BFGS variants can be effective for:

- Fine-tuning neural networks
- Small to medium neural architectures
- Physics-informed neural networks (PINNs)

### Scientific Computing
Optimization problems arising in scientific domains often benefit from quasi-Newton methods.

Typical applications include:
- Parameter estimation
- Numerical simulations
- Computational physics

### Engineering Optimization
Useful for solving nonlinear optimization problems in engineering systems.

Examples include:
- Control system parameter tuning
- Structural optimization
- Signal processing models

---

## Benchmarking

The repository includes benchmarking scripts to evaluate the performance of the topology-aware L-BFGS optimizer.

### Comparison Algorithms

The optimizer can be compared against common optimization algorithms including:

- Gradient Descent
- Stochastic Gradient Descent (SGD)
- Adam
- Standard L-BFGS

### Evaluation Metrics

Performance is typically evaluated using the following metrics:

| Metric | Description |
|------|-------------|
| Convergence Speed | Number of iterations required to reach a target loss |
| Final Loss | Objective value achieved after optimization |
| Stability | Ability to avoid oscillations or divergence |
| Memory Efficiency | Amount of memory used for curvature storage |

### Benchmark Procedure

1. Select a dataset or test function.
2. Initialize the model parameters.
3. Train the model using different optimizers.
4. Record convergence metrics.
5. Compare performance across algorithms.

---

## License

This project is licensed under the **MIT License**.

The MIT License allows users to freely use, modify, distribute, and sublicense the software with minimal restrictions. Users are permitted to incorporate the software into both open-source and proprietary projects.

For the complete license terms, refer to the `LICENSE` file included in the repository.

---

## Acknowledgements

This project builds upon foundational research in quasi-Newton optimization methods, particularly the development of the **Limited-memory BFGS (L-BFGS)** algorithm designed for large-scale optimization problems.

The repository extends these concepts by introducing topology-aware mechanisms and adaptive memory management strategies aimed at improving optimization stability and efficiency in modern machine learning and scientific computing tasks.