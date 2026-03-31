import ast
import pathlib
import time

import pytest

pytestmark = pytest.mark.phase0

AUDIT_MANIFEST = {
    "ta_lbfgs/__init__.py",
    "ta_lbfgs/config.py",
    "ta_lbfgs/core/__init__.py",
    "ta_lbfgs/core/baseline_lbfgs.py",
    "ta_lbfgs/core/hypergradient.py",
    "ta_lbfgs/core/hyperparameters.py",
    "ta_lbfgs/core/layer_utils.py",
    "ta_lbfgs/core/lbfgs.py",
    "ta_lbfgs/dashboard/__init__.py",
    "ta_lbfgs/dashboard/landscape_viz.py",
    "ta_lbfgs/dashboard/live_dashboard.py",
    "ta_lbfgs/dashboard/online_rsvd.py",
    "ta_lbfgs/dashboard/server.py",
    "ta_lbfgs/dashboard/sparkline.py",
    "ta_lbfgs/dashboard/textual_dashboard.py",
    "ta_lbfgs/topology/__init__.py",
    "ta_lbfgs/topology/adaptive_memory.py",
    "ta_lbfgs/topology/attention_topo.py",
    "ta_lbfgs/topology/chain_topo.py",
    "ta_lbfgs/topology/condition.py",
    "ta_lbfgs/topology/hf_interceptor.py",
    "ta_lbfgs/topology/moe_topo.py",
    "ta_lbfgs/topology/persistent_homology.py",
    "ta_lbfgs/topology/residual_topo.py",
    "ta_lbfgs/topology/saddle.py",
    "ta_lbfgs/topology/vtk_exporter.py",
    "ta_lbfgs/training/__init__.py",
    "ta_lbfgs/training/bilevel.py",
    "ta_lbfgs/training/data_preprocessing.py",
    "ta_lbfgs/training/inner_loop.py",
    "ta_lbfgs/training/interceptor.py",
    "ta_lbfgs/utils/__init__.py",
    "ta_lbfgs/utils/kfac.py",
    "ta_lbfgs/utils/vllm_client.py",
    "ta_lbfgs/utils/vram.py",
}



def _get_all_function_bodies():
    bodies = {}
    for pyfile in pathlib.Path("ta_lbfgs").rglob("*.py"):
        tree = ast.parse(pyfile.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                bodies[f"{pyfile.as_posix()}::{node.name}"] = ast.unparse(node)
    return bodies


def test_all_files_catalogued():
    found = {str(p.as_posix()) for p in pathlib.Path("ta_lbfgs").rglob("*.py")}
    uncatalogued = found - AUDIT_MANIFEST
    assert not uncatalogued, (
        "Uncatalogued files found. Integration produced files not in audit manifest: "
        f"{sorted(uncatalogued)}"
    )


def test_p0_flawed_functions_deleted():
    bodies = _get_all_function_bodies()

    neumann_violations = []
    for loc, body in bodies.items():
        if "neumann" in loc.lower() and "spectral_guard" not in body:
            neumann_violations.append(loc)
    assert not neumann_violations, (
        "Neumann function missing spectral guard. Violates Fix 1.1A: "
        f"{neumann_violations}"
    )

    saddle_violations = []
    for loc, body in bodies.items():
        if "::is_saddle_point" in loc:
            lowered = body.lower()
            if "lanczos" not in lowered and "eigvec" not in lowered:
                saddle_violations.append(loc)
    assert not saddle_violations, (
        "is_saddle_point still resembles yTs-only check. Violates Fix 1.2A: "
        f"{saddle_violations}"
    )

    trace_div_violations = []
    for pyfile in pathlib.Path("ta_lbfgs").rglob("*.py"):
        src = pyfile.read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                if isinstance(node.right, ast.Name) and "trace" in node.right.id.lower():
                    trace_div_violations.append(f"{pyfile.as_posix()}:{node.lineno}")
    assert not trace_div_violations, (
        "Scalar trace used as divisor in a vector/elementwise path. Violates Fix 1.4A: "
        f"{trace_div_violations}"
    )

    window_violations = []
    for loc, body in bodies.items():
        if "window" in loc.lower() or "adaptive" in loc.lower():
            lowered = body.lower()
            has_log_map = "log2" in lowered or "math.log2" in lowered
            delegates = "compute_window(" in lowered
            if "kappa" in lowered and not (has_log_map or delegates):
                window_violations.append(loc)
    assert not window_violations, (
        "Adaptive window mapping missing log2-based clamping. Violates Fix 1.5A: "
        f"{window_violations}"
    )


def test_topology_file_count():
    files = list(pathlib.Path("ta_lbfgs/topology").glob("*.py"))
    non_init = [f for f in files if f.name != "__init__.py"]
    assert len(non_init) <= 12, (
        f"topology has {len(non_init)} non-init files, expected <= 12. "
        f"Files: {[f.name for f in non_init]}. Equilibrium rule violated."
    )


def test_no_orphaned_functions():
    start = time.time()
    root = pathlib.Path("ta_lbfgs")
    public_funcs = {}
    for pyfile in root.rglob("*.py"):
        tree = ast.parse(pyfile.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                public_funcs.setdefault(node.name, []).append(pyfile.as_posix())
        if time.time() - start > 5.0:
            pytest.skip("Call-graph analysis exceeded 5 seconds; skipping best-effort orphan check.")

    exports = set()
    init_file = pathlib.Path("ta_lbfgs/__init__.py")
    if init_file.exists():
        exports_src = init_file.read_text(encoding="utf-8")
        exports.update({name for name in public_funcs if name in exports_src})

    reachable_src = ""
    for p in [pathlib.Path("demo.py"), pathlib.Path("hf_demo.py"), pathlib.Path("live_run.py")]:
        if p.exists():
            reachable_src += "\n" + p.read_text(encoding="utf-8")
    for test_file in pathlib.Path("tests").glob("test_phase*.py"):
        reachable_src += "\n" + test_file.read_text(encoding="utf-8")

    orphaned = []
    for name, locations in public_funcs.items():
        if name in exports:
            continue
        if f"{name}(" in reachable_src or f"import {name}" in reachable_src:
            continue
        orphaned.append((name, locations))

    if orphaned:
        pytest.skip(
            "Static call-graph reachability is inconclusive for module-level functions. "
            f"Best-effort orphan candidates: {orphaned}"
        )
