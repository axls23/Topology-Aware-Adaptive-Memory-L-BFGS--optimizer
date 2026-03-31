"""
VTK Exporter for TTK / ParaView Offline Analysis.

Exports optimizer state into VTK-compatible file formats that can be
loaded into ParaView with the TTK plugin for full topological analysis:
  - Loss subspace grids → .vti (vtkImageData)
  - Gradient point clouds → .vtp (vtkPolyData)
  - Residual drift signals → .vtp (vtkPolyData)

These files can be loaded as a time series in ParaView to visualize
how the loss landscape topology evolves during training.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def export_loss_grid_vti(
    loss_grid: np.ndarray,
    step: int,
    output_dir: str = "outputs/vtk",
    prefix: str = "loss_surface",
) -> str:
    """Export a 2D loss grid as a VTK ImageData (.vti) file.

    This format is directly consumable by TTK's CubicalPersistence,
    MorseSmaleComplex, and ContourTree filters in ParaView.

    Args:
        loss_grid: 2D array (H, W) of loss values.
        step: Outer optimization step for time-series naming.
        output_dir: Directory to write VTK files.
        prefix: Filename prefix.

    Returns:
        Path to the written file.
    """
    _ensure_dir(output_dir)
    filename = os.path.join(output_dir, f"{prefix}_{step:06d}.vti")
    H, W = loss_grid.shape

    # Write VTK XML ImageData format manually (no vtk dependency needed).
    grid_flat = loss_grid.astype(np.float64).ravel(order="C")

    with open(filename, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write(
            '<VTKFile type="ImageData" version="0.1" '
            'byte_order="LittleEndian">\n'
        )
        f.write(
            f'  <ImageData WholeExtent="0 {W - 1} 0 {H - 1} 0 0" '
            f'Origin="0 0 0" Spacing="1 1 1">\n'
        )
        f.write(
            f'    <Piece Extent="0 {W - 1} 0 {H - 1} 0 0">\n'
        )
        f.write('      <PointData Scalars="Loss">\n')
        f.write(
            '        <DataArray type="Float64" Name="Loss" '
            'format="ascii">\n'
        )
        f.write("          ")
        f.write(" ".join(f"{v:.8e}" for v in grid_flat))
        f.write("\n")
        f.write("        </DataArray>\n")
        f.write("      </PointData>\n")
        f.write("    </Piece>\n")
        f.write("  </ImageData>\n")
        f.write("</VTKFile>\n")

    return filename


def export_attention_matrix_vti(
    attn_weights: np.ndarray,
    layer: int,
    head: int,
    step: int,
    output_dir: str = "outputs/vtk",
) -> str:
    """Export a T×T attention weight matrix as VTK ImageData.

    TTK can compute Contour Trees on this 2D scalar field,
    decomposing the attention pattern into topological features
    (peaks, saddles, basins).

    Args:
        attn_weights: 2D array (T, T) — averaged attention weights.
        layer: Layer index.
        head: Head index.
        step: Outer optimization step.
        output_dir: Output directory.

    Returns:
        Path to the written file.
    """
    prefix = f"attn_L{layer}_H{head}"
    return export_loss_grid_vti(attn_weights, step, output_dir, prefix)


def export_residual_drift_vtp(
    layer_norms: np.ndarray,
    step: int,
    output_dir: str = "outputs/vtk",
) -> str:
    """Export per-layer norm drift as a VTK PolyData line.

    TTK can compute 1D persistence on this signal to find
    critical points where representation coupling changes sharply.

    Args:
        layer_norms: 1D array (L,) of per-layer output norms.
        step: Outer optimization step.
        output_dir: Output directory.

    Returns:
        Path to the written file.
    """
    _ensure_dir(output_dir)
    filename = os.path.join(output_dir, f"residual_drift_{step:06d}.vtp")
    L = len(layer_norms)
    norms = layer_norms.astype(np.float64)

    with open(filename, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write(
            '<VTKFile type="PolyData" version="0.1" '
            'byte_order="LittleEndian">\n'
        )
        f.write(f'  <PolyData>\n')
        f.write(
            f'    <Piece NumberOfPoints="{L}" NumberOfVerts="0" '
            f'NumberOfLines="1" NumberOfStrips="0" NumberOfPolys="0">\n'
        )
        # Points
        f.write("      <Points>\n")
        f.write(
            '        <DataArray type="Float64" '
            'NumberOfComponents="3" format="ascii">\n'
        )
        f.write("          ")
        for i in range(L):
            f.write(f"{float(i):.1f} {float(norms[i]):.8e} 0.0 ")
        f.write("\n")
        f.write("        </DataArray>\n")
        f.write("      </Points>\n")
        # Scalar data
        f.write('      <PointData Scalars="Drift">\n')
        f.write(
            '        <DataArray type="Float64" Name="Drift" '
            'format="ascii">\n'
        )
        f.write("          ")
        f.write(" ".join(f"{v:.8e}" for v in norms))
        f.write("\n")
        f.write("        </DataArray>\n")
        f.write("      </PointData>\n")
        # Line connectivity
        f.write("      <Lines>\n")
        f.write(
            '        <DataArray type="Int32" Name="connectivity" '
            'format="ascii">\n'
        )
        f.write("          ")
        f.write(" ".join(str(i) for i in range(L)))
        f.write("\n")
        f.write("        </DataArray>\n")
        f.write(
            '        <DataArray type="Int32" Name="offsets" '
            'format="ascii">\n'
        )
        f.write(f"          {L}\n")
        f.write("        </DataArray>\n")
        f.write("      </Lines>\n")
        f.write("    </Piece>\n")
        f.write("  </PolyData>\n")
        f.write("</VTKFile>\n")

    return filename


def export_persistence_diagram_json(
    diagram: np.ndarray,
    step: int,
    output_dir: str = "outputs/vtk",
) -> str:
    """Export a persistence diagram as JSON for web visualization.

    Args:
        diagram: Array of shape (N, 3) — (birth, death, dimension).
        step: Outer optimization step.
        output_dir: Output directory.

    Returns:
        Path to the written file.
    """
    _ensure_dir(output_dir)
    filename = os.path.join(output_dir, f"persistence_{step:06d}.json")

    payload = {
        "step": step,
        "points": [
            {
                "birth": float(row[0]),
                "death": float(row[1]),
                "dimension": int(row[2]),
                "persistence": float(row[1] - row[0]),
            }
            for row in diagram
        ],
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return filename
