import os
import math
import time
import numpy as np
import torch
import optuna
import argparse
from typing import Dict, List, Tuple

# Import our core components
from ta_lbfgs.core.hyperparameters import DifferentiableHyperparameters
from ta_lbfgs.dashboard.landscape_viz import (
    export_trajectory_3d,
    plot_dynamics,
    plot_hyperparameter_trajectories,
)

# Re-implement a simplified version of the Synthetic Model for the benchmark
# to ensure we are testing on the exact same surface as demo.py
class LayerTarget:
    def __init__(self, idx, n_layers):
        self.idx = idx
        # Target LR: follows a shift per layer
        self.target_lr = 1e-3 * (1.0 + 0.1 * math.sin(idx))
        # Target WD: exponential decay across layers
        self.target_wd = 0.05 * math.exp(-idx / n_layers)

    def loss(self, lr, wd):
        # A non-convex surface with a ravine
        d_lr = (torch.log(lr) - math.log(self.target_lr))
        d_wd = (torch.log(wd) - math.log(self.target_wd))
        # Basic quadratic bowl
        bowl = d_lr**2 + 5.0 * d_wd**2
        # Add a non-convex ripple
        ripple = 0.2 * torch.sin(10 * d_lr) * torch.cos(10 * d_wd)
        return bowl + ripple

class SyntheticBilevelModel:
    def __init__(self, n_layers=4):
        self.layers = [LayerTarget(i, n_layers) for i in range(n_layers)]

    def train_loss(self, hyperparams):
        total = sum(
            layer.loss(hyperparams.get_layer_lr(i), hyperparams.get_layer_wd(i))
            for i, layer in enumerate(self.layers)
        )
        return total / len(self.layers)

    def val_loss(self, hyperparams):
        total = sum(
            layer.loss(
                hyperparams.get_layer_lr(i) * 1.1, # Shifted validation
                hyperparams.get_layer_wd(i) * 0.9,
            )
            for i, layer in enumerate(self.layers)
        )
        return total / len(self.layers)

def run_optuna_baseline(n_trials=40, n_layers=4):
    print("=" * 60)
    print(f"  OPTIMIZATION BASELINE: Optuna (Bayesian / TPE)")
    print("=" * 60)
    
    model = SyntheticBilevelModel(n_layers=n_layers)
    loss_history = []
    hp_history = []
    
    def objective(trial):
        # Construct hyperparams from Optuna suggestions
        # We need to construct a DifferentiableHyperparameters object 
        # to use as_float_dict for the visualization later
        hp_obj = DifferentiableHyperparameters(
            n_layers=n_layers,
            initial_label_smoothing=0.1
        )
        
        # Suggest values for each layer
        for i in range(n_layers):
            # We suggest in log space but Optuna handles that with log=True
            lr = trial.suggest_float(f"layer_{i}_lr", 1e-7, 1.0, log=True)
            wd = trial.suggest_float(f"layer_{i}_wd", 1e-7, 1.0, log=True)
            
            # Manually set the raw data to match Optuna's suggestion
            with torch.no_grad():
                hp_obj.blocks[i].raw_lr.copy_(torch.tensor(math.log(lr)))
                hp_obj.blocks[i].raw_wd.copy_(torch.tensor(math.log(wd)))
        
        loss = model.val_loss(hp_obj).item()
        
        # Record for visualization
        loss_history.append(loss)
        hp_history.append(hp_obj.as_float_dict())
        
        return loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    # Export visualizations
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # 3D Trajectory (Scattered Search)
    hp_array = np.array([
        [x for x in (h["lr"] + h["wd"])]
        for h in hp_history
    ])
    loss_array = np.array(loss_history)
    
    traj_path = os.path.join(output_dir, "optuna_trajectory_3d.html")
    export_trajectory_3d(hp_array, loss_array, traj_path)
    
    # Dynamics (Phase Portrait)
    # Note: Optuna doesn't have gradients, so we'll use "Estimated Step Gradient"
    est_grads = [0.0]
    for i in range(1, len(loss_history)):
        est_grads.append(abs(loss_history[i] - loss_history[i-1]))
        
    dynamics_path = os.path.join(output_dir, "optuna_diagnostics.png")
    plot_dynamics(loss_history, est_grads, [0.0]*len(loss_history), hp_history, dynamics_path)

    hp_plot_path = os.path.join(output_dir, "optuna_hyperparameters.png")
    plot_hyperparameter_trajectories(hp_history, hp_plot_path)
    
    print(f"\n[SUCCESS] Optuna Baseline Complete.")
    print(f"  - Best Loss: {study.best_value:.6f}")
    print(f"  - 3D Map: {traj_path}")
    print(f"  - Diagnostics: {dynamics_path}")
    print(f"  - Hyperparameters: {hp_plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--layers", type=int, default=4)
    args = parser.parse_args()
    
    run_optuna_baseline(args.trials, args.layers)
