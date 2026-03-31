#!/usr/bin/env pvpython
"""
TTK Offline Topology Analysis Pipeline.

Run this script with ParaView's pvpython to compute exact topological
invariants on the exported VTK files from ta-LBFGS live_run.py.

Prerequisites:
  conda create -n ttk
  conda activate ttk
  conda install -c conda-forge topologytoolkit

Usage:
  pvpython ttk_analysis.py outputs/vtk/loss_surface_000010.vti

Pipeline (follows TTK tutorial / Figure 1 of TTK paper):
  1. Load loss surface as vtkImageData
  2. Compute PersistenceDiagram
  3. Threshold persistence to keep significant features
  4. TopologicalSimplification using persistent pairs
  5. Compute MorseSmaleComplex on simplified surface
  6. Export results as VTU files + CSV summary
"""

import sys
import os

try:
    import paraview.simple as pvs
except ImportError:
    print("ERROR: This script requires pvpython (ParaView Python).")
    print("Install via: conda install -c conda-forge topologytoolkit")
    sys.exit(1)


def run_ttk_pipeline(
    input_file: str,
    persistence_threshold: float = 5.0,
    output_dir: str = "outputs/ttk_results",
):
    """Execute the full TTK topology pipeline on a loss surface .vti file.

    Args:
        input_file: Path to .vti file (from vtk_exporter.py).
        persistence_threshold: Minimum persistence to keep features.
        output_dir: Directory for output files.
    """
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(input_file))[0]

    # ── Step 1: Load Data ────────────────────────────────────────
    print(f"[TTK] Loading: {input_file}")
    reader = pvs.XMLImageDataReader(FileName=[input_file])
    reader.UpdatePipeline()

    # ── Step 2: Compute Persistence Diagram ──────────────────────
    print("[TTK] Computing Persistence Diagram...")
    persistence = pvs.TTKPersistenceDiagram(Input=reader)
    persistence.ScalarField = ["POINTS", "Loss"]
    persistence.UpdatePipeline()

    # Save the full persistence diagram
    pd_file = os.path.join(output_dir, f"{basename}_persistence.vtm")
    pvs.SaveData(pd_file, persistence)
    print(f"[TTK] Persistence Diagram saved: {pd_file}")

    # ── Step 3: Threshold by Persistence ─────────────────────────
    print(f"[TTK] Filtering persistence > {persistence_threshold}...")
    threshold = pvs.Threshold(Input=persistence)
    threshold.Scalars = ["CELLS", "Persistence"]
    threshold.ThresholdMethod = 'Between'
    threshold.LowerThreshold = float(persistence_threshold)
    threshold.UpperThreshold = 999999.0
    threshold.UpdatePipeline()

    # ── Step 4: Topological Simplification ───────────────────────
    print("[TTK] Topological Simplification...")
    simplification = pvs.TTKTopologicalSimplification(
        Domain=reader,
        Constraints=threshold,
    )
    simplification.ScalarField = ["POINTS", "Loss"]
    simplification.UpdatePipeline()

    # Save simplified surface
    simp_file = os.path.join(output_dir, f"{basename}_simplified.vti")
    pvs.SaveData(simp_file, simplification)
    print(f"[TTK] Simplified surface saved: {simp_file}")

    # ── Step 5: Morse-Smale Complex ──────────────────────────────
    print("[TTK] Computing Morse-Smale Complex...")
    morse_smale = pvs.TTKMorseSmaleComplex(Input=simplification)
    morse_smale.ScalarField = ["POINTS", "Loss"]
    morse_smale.UpdatePipeline()

    # Save critical points (dimension 0)
    cp_file = os.path.join(output_dir, f"{basename}_critical_points.vtp")
    pvs.SaveData(cp_file, pvs.OutputPort(morse_smale, 0))
    print(f"[TTK] Critical Points saved: {cp_file}")

    # Save separatrices / edges (dimension 1)
    sep_file = os.path.join(output_dir, f"{basename}_separatrices.vtp")
    pvs.SaveData(sep_file, pvs.OutputPort(morse_smale, 1))
    print(f"[TTK] Separatrices saved: {sep_file}")

    # Save segmentation / basins (dimension 2)
    seg_file = os.path.join(output_dir, f"{basename}_segmentation.vtm")
    pvs.SaveData(seg_file, pvs.OutputPort(morse_smale, 3))
    print(f"[TTK] Segmentation saved: {seg_file}")

    # ── Step 6: Persistence Curve ────────────────────────────────
    print("[TTK] Computing Persistence Curve...")
    pers_curve = pvs.TTKPersistenceCurve(Input=reader)
    pers_curve.ScalarField = ["POINTS", "Loss"]
    pers_curve.UpdatePipeline()

    pc_file = os.path.join(output_dir, f"{basename}_persistence_curve.csv")
    pvs.SaveData(pc_file, pers_curve)
    print(f"[TTK] Persistence Curve saved: {pc_file}")

    print(f"\n[TTK] Pipeline complete for {basename}")
    print(f"[TTK] Results in: {output_dir}/")
    print(f"[TTK] Open in ParaView for interactive exploration:")
    print(f"       paraview {cp_file}")

    return {
        "persistence_diagram": pd_file,
        "simplified_surface": simp_file,
        "critical_points": cp_file,
        "separatrices": sep_file,
        "segmentation": seg_file,
        "persistence_curve": pc_file,
    }


def batch_analyze(vtk_dir: str, persistence_threshold: float = 5.0):
    """Run TTK analysis on all loss surface .vti files in a directory.

    This creates a time-series of topological analyses, enabling
    Merge Tree Feature Tracking across training steps.
    """
    import glob

    files = sorted(glob.glob(os.path.join(vtk_dir, "loss_surface_*.vti")))
    if not files:
        print(f"[TTK] No loss_surface_*.vti files found in {vtk_dir}")
        return

    print(f"[TTK] Batch analyzing {len(files)} time steps...")
    for f in files:
        run_ttk_pipeline(f, persistence_threshold)

    print(f"\n[TTK] Batch complete. {len(files)} steps analyzed.")
    print("[TTK] For time-varying analysis, load the results as a")
    print("      file series in ParaView and apply TTK TimeTracking.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: pvpython ttk_analysis.py <input.vti> [threshold]")
        print("       pvpython ttk_analysis.py --batch <vtk_dir> [threshold]")
        sys.exit(1)

    if sys.argv[1] == "--batch":
        vtk_dir = sys.argv[2] if len(sys.argv) > 2 else "outputs/vtk"
        threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
        batch_analyze(vtk_dir, threshold)
    else:
        input_file = sys.argv[1]
        threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
        run_ttk_pipeline(input_file, threshold)
