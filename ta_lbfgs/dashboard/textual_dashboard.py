"""
Textual CLI Dashboard for ta-LBFGS.

Upgraded with a live 3D Trajectory + Static Topology Mesh using Braille characters.
"""

import math
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Static, DataTable, RichLog, Label
from textual.reactive import reactive
from rich.text import Text

from .sparkline import generate_sparkline


class BrailleCanvas:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = np.zeros((height * 4, width * 2), dtype=bool)

    def set_pixel(self, x: int, y: int):
        if 0 <= x < self.width * 2 and 0 <= y < self.height * 4:
            self.grid[y, x] = True

    def render(self) -> str:
        output = []
        for row in range(0, self.height * 4, 4):
            line = []
            for col in range(0, self.width * 2, 2):
                val = 0
                if self.grid[row, col]:     val |= 0x01
                if self.grid[row+1, col]:   val |= 0x02
                if self.grid[row+2, col]:   val |= 0x04
                if self.grid[row, col+1]:   val |= 0x08
                if self.grid[row+1, col+1]: val |= 0x10
                if self.grid[row+2, col+1]: val |= 0x20
                if self.grid[row+3, col]:   val |= 0x40
                if self.grid[row+3, col+1]: val |= 0x80
                line.append(chr(0x2800 + val))
            output.append("".join(line))
        return "\n".join(output)


class Trajectory3D(Static):
    """3D Visualizer showing both a topological mesh and the optimizer path."""
    
    points = reactive([])  # Trajectory points
    mesh = reactive(None)  # Static Topology (X, Y, Z meshes)
    angle = reactive(0.0)

    def on_mount(self):
        self.set_interval(0.05, self.tick)

    def tick(self):
        self.angle += 0.05

    def render(self) -> Text:
        w, h = self.size.width, self.size.height
        if w < 10 or h < 5: return Text("")
            
        canvas = BrailleCanvas(w, h)
        cx, cy = w, h * 2
        scale = min(w, h * 2) * 0.7
        a = self.angle
        s, c = math.sin(a), math.cos(a)

        # 1. Coordinate Normalization
        # We must normalize both mesh and trajectory to the same [-1, 1] cube
        all_pts = []
        if self.mesh:
            MX, MY, MZ = self.mesh
            all_pts.append(np.column_stack([MX.ravel(), MY.ravel(), MZ.ravel()]))
        if self.points:
            all_pts.append(np.array(self.points))
            
        if not all_pts:
            return Text("Initializing Landscape...")

        combined = np.concatenate(all_pts, axis=0)
        p_min, p_max = combined.min(0), combined.max(0)
        p_range = np.where((p_max - p_min) == 0, 1.0, p_max - p_min)

        def normalize(pts):
            return 2.0 * (pts - p_min) / p_range - 1.0

        def project(x, y, z):
            # Rotate Y (around Z-axis basically, but in 3D space)
            rx = x * c - y * s
            ry = x * s + y * c
            # Perspective
            z_proj = 1 / (ry + 3.0)
            px = int(cx + rx * scale * z_proj)
            py = int(cy + z * scale * z_proj * 2)
            return px, py

        # 2. Render Topology Mesh (Static Landscape)
        if self.mesh:
            MX, MY, MZ = self.mesh
            rows, cols = MX.shape
            
            # Pack mesh into [N, 3] and normalize
            mesh_raw = np.column_stack([MX.ravel(), MY.ravel(), MZ.ravel()])
            mesh_norm = normalize(mesh_raw).reshape(rows, cols, 3)
            
            # Project mesh vertices
            mesh_pixels = []
            for r in range(rows):
                row_pix = []
                for col in range(cols):
                    p = mesh_norm[r, col]
                    row_pix.append(project(p[0], p[1], p[2]))
                mesh_pixels.append(row_pix)

            # Draw wireframe
            for r in range(rows):
                for col in range(cols - 1):
                    p1, p2 = mesh_pixels[r][col], mesh_pixels[r][col+1]
                    self._draw_line(canvas, p1[0], p1[1], p2[0], p2[1])
            
            for col in range(cols):
                for r in range(rows - 1):
                    p1, p2 = mesh_pixels[r][col], mesh_pixels[r+1][col]
                    self._draw_line(canvas, p1[0], p1[1], p2[0], p2[1])

        # 3. Render Trajectory Trace (Active Path)
        if self.points and len(self.points) >= 2:
            traj_norm = normalize(np.array(self.points))
            traj_pixels = [project(p[0], p[1], p[2]) for p in traj_norm]
            for i in range(len(traj_pixels) - 1):
                p1, p2 = traj_pixels[i], traj_pixels[i+1]
                # Trajectory is drawn thicker/darker if possible, but here 
                # we just ensure it's on top
                self._draw_line(canvas, p1[0], p1[1], p2[0], p2[1])

        return Text(canvas.render())

    def _draw_line(self, canvas, x0, y0, x1, y1):
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx - dy
        while True:
            canvas.set_pixel(x0, y0)
            if x0 == x1 and y0 == y1: break
            e2 = 2 * err
            if e2 > -dy: err -= dy; x0 += sx
            if e2 < dx: err -= dx; y0 += sy


class MetricsSidebar(Static):
    data = reactive({
        "iteration": 0,
        "total_iterations": 1,
        "loss": 0.0,
        "best_loss": 0.0,
        "lr": 0.0,
        "wd": 0.0,
        "projection": None,
    })

    def render(self) -> Text:
        d = self.data
        it, tot = d.get("iteration", 0), d.get("total_iterations", 1)
        pct = (it / tot) * 100
        bar = "█" * int(pct/5) + "░" * (20 - int(pct/5))
        projection = d.get("projection") or {}
        ratio = float(projection.get("explained_variance_ratio", 0.0))
        proj_layer = projection.get("layer", "-")
        proj_status = projection.get("status", "N/A")
        warning = projection.get("warning")
        ratio_color = "green" if proj_status == "Reliable" else "red"

        projection_block = (
            f"\n\n[bold white]RSVD[/] {proj_layer}\n"
            f"[bold {ratio_color}]EVR[/] {ratio * 100.0:.1f}%\n"
            f"[bold {ratio_color}]STATUS[/] {proj_status}"
        )
        if warning:
            projection_block += f"\n[bold red]WARN[/] low-dimensional map"

        return Text.from_markup(
            f"[bold magenta]PROGRESS[/] {it}/{tot}\n[{pct:.1f}%] {bar}\n\n"
            f"[bold yellow]LOSS[/] {d.get('loss',0.0):.6f}\n"
            f"[bold green]BEST[/] {d.get('best_loss',0.0):.6f}\n\n"
            f"[bold cyan]LR[/] {d.get('lr',0.0):.6f}\n"
            f"[bold cyan]WD[/] {d.get('wd',0.0):.6f}"
            f"{projection_block}"
        )


class TextualDashboard(App):
    CSS = """
    Screen { background: #1a1b26; }
    #left_pane { width: 1fr; border: solid #333; margin: 1; }
    #center_pane { width: 1fr; border: solid #333; margin: 1; background: #16161e; }
    #right_pane { width: 1fr; border: solid #333; margin: 1; background: #24283b; padding: 1; }
    #log_pane { height: 8; border: solid #333; margin: 1; background: #16161e; }
    DataTable { height: 1fr; background: transparent; }
    Label { width: 100%; text-align: center; background: #3d59a1; color: white; text-style: bold; }
    """
    BINDINGS = [("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            with Vertical(id="left_pane"):
                yield Label("TOPOLOGY METRICS")
                yield DataTable(id="metrics_table")
            with Vertical(id="center_pane"):
                yield Label("LIVE TRAJECTORY ON TOPOLOGY")
                yield Trajectory3D(id="viz_3d")
            with Vertical(id="right_pane"):
                yield Label("STATE")
                yield MetricsSidebar(id="sidebar")
        with Vertical(id="log_pane"):
            yield Label("SADDLE EVASION LOG")
            yield RichLog(id="evasion_log", markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(DataTable).add_columns("Layer", "κ", "m_l", "y^Ts", "Status", "History")

    def call_from_thread(self, fn, *args, **kwargs):
        """Thread-safe call to a dashboard method."""
        self.call_next_tick(fn, *args, **kwargs)

    def update_data(
        self,
        layer_data,
        outer_state,
        evasion_events=None,
        trajectory_points=None,
        mesh=None,
        projection_info=None,
    ):
        if projection_info is not None:
            outer_state = {**outer_state, "projection": projection_info}
        self.query_one(MetricsSidebar).data = outer_state
        viz = self.query_one(Trajectory3D)
        if trajectory_points is not None and len(trajectory_points) > 0:
            viz.points = trajectory_points
        if mesh is not None:
            viz.mesh = mesh
        
        table = self.query_one(DataTable)
        table.clear()
        for name, data in layer_data.items():
            kappa, secant = data.get("kappa", 1.0), data.get("secant", 1.0)
            table.add_row(
                name, f"{kappa:.1f}", str(data.get("memory_size", "-")), f"{secant:.4f}",
                Text(data.get("landscape", "Unknown"), style="yellow"),
                generate_sparkline(data.get("kappa_history", []), 10)
            )

        if evasion_events:
            log = self.query_one(RichLog)
            for e in evasion_events:
                log.write(f"[bold red]EVASION[/]: {e['layer']} | y^Ts={e['ys']:.4f}")

    def update_log(self, message: str):
        self.query_one(RichLog).write(message)
