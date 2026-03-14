"""
Rich Live CLI Dashboard.

Real-time terminal visualization of the layerwise ta-LBFGS optimizer state.
Adapted from Chronoscope's exp4_live_dashboard.py and exp2_live_heatmap.py.

Layout:
  ┌─── Header ──────────────────────────────────────────────┐
  │ ta-LBFGS Optimizer | Layerwise Topology Dashboard       │
  ├─── Layer Metrics (2/3) ──────┬── Optimizer State (1/3) ─┤
  │ Layer | κ | m_l | ‖∇‖ | ...  │  Loss: 0.1234           │
  │ ...                          │  Outer Iter: 5/50        │
  │                              │  Progress ████░░░        │
  ├─── Saddle Evasion Log ──────────────────────────────────┤
  │ [03:14:15] EVASION: layers.3 | y^Ts=-0.02 | κ=142.5    │
  └─────────────────────────────────────────────────────────┘
"""

from datetime import datetime
from typing import Dict, List, Optional, Any

from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.console import Group, Console
from rich.align import Align
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn

from .sparkline import generate_sparkline


class OptimizerDashboard:
    """
    Real-time CLI dashboard for monitoring the ta-LBFGS optimizer.

    Uses Rich Live to render a multi-pane terminal dashboard with:
    - Layerwise topology metrics table
    - Outer-loop progress and loss tracking
    - Saddle-evasion event log

    Usage:
        dashboard = OptimizerDashboard()
        with dashboard.live():
            for outer_iter in range(n_iters):
                # ... optimization step ...
                dashboard.update(layer_data, outer_state, events)
    """

    def __init__(
        self,
        refresh_rate: float = 0.25,
        sparkline_width: int = 30,
    ):
        self.refresh_rate = refresh_rate
        self.sparkline_width = sparkline_width
        self.log_messages: List[str] = []
        self.layout = self._build_layout()
        self._live: Optional[Live] = None

    def _build_layout(self) -> Layout:
        """Create the Rich layout skeleton."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=8),
        )
        layout["main"].split_row(
            Layout(name="layer_metrics", ratio=2),
            Layout(name="optimizer_state", ratio=1),
        )

        layout["header"].update(
            Panel(
                Align.center(
                    "[bold cyan]ta-LBFGS Optimizer | "
                    "Layerwise Topology Dashboard[/]"
                ),
                style="white on blue",
            )
        )

        layout["layer_metrics"].update(
            Panel("Initializing...", title="[yellow]Layer Metrics[/]", border_style="yellow")
        )
        layout["optimizer_state"].update(
            Panel("Waiting for data...", title="[magenta]Optimizer State[/]", border_style="magenta")
        )
        layout["footer"].update(
            Panel("Booting optimizer...", title="[green]Saddle Evasion Log[/]", border_style="green")
        )

        return layout

    def live(self) -> Live:
        """Create and return a Rich Live context manager."""
        self._live = Live(
            self.layout,
            refresh_per_second=int(1 / self.refresh_rate),
            screen=False,
        )
        return self._live

    def add_log(self, msg: str, style: str = "green"):
        """Add a timestamped message to the evasion log panel."""
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_messages.append(f"[{ts}] {msg}")
        if len(self.log_messages) > 6:
            self.log_messages.pop(0)

        log_text = Text.from_markup(
            "\n".join(f"[{style}]{m}[/{style}]" for m in self.log_messages)
        )
        self.layout["footer"].update(
            Panel(log_text, title="[green]Saddle Evasion Log[/]", border_style="green")
        )

    def update(
        self,
        layer_data: Dict[str, Dict],
        outer_state: Dict[str, Any],
        evasion_events: Optional[List[Dict]] = None,
    ):
        """
        Update all dashboard panes.

        Args:
            layer_data: Per-layer metrics dict from LayerwiseTaLBFGS.get_all_layer_data().
            outer_state: Dict with keys 'iteration', 'total_iterations', 'loss', 'best_loss'.
            evasion_events: New saddle-evasion events to log.
        """
        # ── Layer Metrics Table ─────────────────────────────────────
        table = self._create_layer_table(layer_data)
        self.layout["layer_metrics"].update(Panel(table, padding=(0, 1)))

        # ── Optimizer State Panel ───────────────────────────────────
        state_text = self._create_state_panel(outer_state)
        self.layout["optimizer_state"].update(
            Panel(state_text, title="[magenta]Optimizer State[/]", border_style="magenta")
        )

        # ── Evasion Events ──────────────────────────────────────────
        if evasion_events:
            for event in evasion_events:
                self.add_log(
                    f"EVASION: {event['layer']} | "
                    f"y^Ts={event.get('ys', 0):.4f} | "
                    f"κ={event.get('kappa', 0):.1f}",
                    style="bold red",
                )

    def _create_layer_table(self, layer_data: Dict[str, Dict]) -> Table:
        """Render per-layer optimizer state as a Rich table."""
        table = Table(
            title="[bold cyan]Layerwise Optimizer State[/]",
            border_style="dim",
        )
        table.add_column("Layer", style="cyan", width=15)
        table.add_column("κ", justify="right", width=8)
        table.add_column("m_l", justify="right", width=4)
        table.add_column("‖∇‖", justify="right", width=10)
        table.add_column("y^Ts", justify="right", width=10)
        table.add_column("Status", width=16)
        table.add_column("κ History", width=self.sparkline_width)

        for name, data in layer_data.items():
            kappa = data.get("kappa", 1.0)
            secant = data.get("secant", 1.0)

            kappa_color = (
                "red" if kappa > 100
                else "yellow" if kappa > 10
                else "green"
            )
            secant_color = "red" if secant <= 0 else "green"

            landscape = data.get("landscape", "Unknown")
            landscape_colors = {
                "Convex Bowl": "green",
                "Narrow Ravine": "yellow",
                "Ill-Conditioned": "yellow",
                "Saddle Point": "red",
                "Converged": "cyan",
            }
            land_color = landscape_colors.get(landscape, "white")

            table.add_row(
                name,
                f"[{kappa_color}]{kappa:.1f}[/]",
                str(data.get("memory_size", "-")),
                f"{data.get('grad_norm', 0):.4f}",
                f"[{secant_color}]{secant:.4f}[/]",
                f"[{land_color}]{landscape}[/]",
                generate_sparkline(
                    data.get("kappa_history", []),
                    self.sparkline_width,
                ),
            )

        return table

    def _create_state_panel(self, outer_state: Dict[str, Any]) -> Group:
        """Create the optimizer state side panel."""
        iteration = outer_state.get("iteration", 0)
        total = outer_state.get("total_iterations", 1)
        loss = outer_state.get("loss", float("nan"))
        best_loss = outer_state.get("best_loss", float("nan"))
        lr = outer_state.get("lr", 0.0)
        wd = outer_state.get("wd", 0.0)

        pct = iteration / max(total, 1) * 100
        bar_filled = int(pct / 5)
        bar_str = "█" * bar_filled + "░" * (20 - bar_filled)

        return Group(
            Text(f"\n  Outer Iteration", style="bold white"),
            Text(f"  {iteration} / {total}"),
            Text(f"\n  [{bar_str}] {pct:.0f}%", style="cyan"),
            Text(f"\n  Current Loss", style="bold yellow"),
            Text(f"  {loss:.6f}"),
            Text(f"\n  Best Loss", style="bold green"),
            Text(f"  {best_loss:.6f}"),
            Text(f"\n  Hyperparameters", style="bold magenta"),
            Text(f"  lr = {lr:.6f}"),
            Text(f"  wd = {wd:.6f}"),
        )
