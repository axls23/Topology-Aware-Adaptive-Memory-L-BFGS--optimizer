"""
Unicode Sparkline Generator.

Adapted from Chronoscope's exp4_live_dashboard.py.
Renders compact inline sparklines for per-layer metrics in the CLI dashboard.
"""


def generate_sparkline(values: list, max_points: int = 30) -> str:
    """
    Generate a unicode sparkline for a list of numeric values.

    Uses 8 levels of unicode block characters for resolution.

    Args:
        values: List of numeric values to plot.
        max_points: Maximum number of points to display.

    Returns:
        Unicode sparkline string.
    """
    if not values:
        return ""

    values = values[-max_points:]
    v_min, v_max = min(values), max(values)

    if v_max == v_min:
        return "▃" * len(values)

    bars = [" ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]

    sparkline = ""
    for v in values:
        idx = int((v - v_min) / (v_max - v_min + 1e-9) * 7)
        sparkline += bars[idx]

    return sparkline
