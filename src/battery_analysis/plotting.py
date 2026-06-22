"""Shared plotting style and a figure-saving helper.

Keeping the style in one place means every figure in the notebook (and every
figure embedded in the PowerPoint) looks consistent, and the notebook stays
focused on analysis rather than matplotlib boilerplate.
"""

from __future__ import annotations

import calendar
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns

# Month / weekday name labels (so axes read "Jan, Feb, …" / "Mon, …" not numbers).
MONTH_ABBR = list(calendar.month_abbr)  # index 1..12 -> 'Jan'..'Dec' (index 0 = '')
WEEKDAY_ABBR = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def month_labels(months) -> list[str]:
    """Map an iterable of month numbers (1-12) to abbreviations."""
    return [calendar.month_abbr[int(m)] for m in months]


def weekday_labels(days) -> list[str]:
    """Map an iterable of weekday numbers (0=Mon … 6=Sun) to abbreviations."""
    return [WEEKDAY_ABBR[int(d)] for d in days]


def date_axis(ax, fmt: str = "%b") -> None:
    """Format a datetime x-axis with month ticks labelled by name."""
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))


# Navy palette: deep navy accent + a clean qualitative set.
NAVY = "#1b2a4a"
ACCENT = "#2e6fdb"
PALETTE = ["#2e6fdb", "#e4572e", "#17bebb", "#ffc914", "#76b041", "#7d3c98"]

FIG_DIR = Path(__file__).resolve().parents[2] / "figures"


def set_style() -> None:
    """Apply the shared seaborn/matplotlib style. Call once per notebook."""
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.figsize": (11, 5.5),
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "axes.titleweight": "bold",
            "axes.titlecolor": NAVY,
            "axes.edgecolor": "#cccccc",
            "axes.prop_cycle": plt.cycler(color=PALETTE),
            "grid.color": "#e6e6e6",
            "font.size": 12,
        }
    )


def save_fig(fig, name: str, fig_dir: Path | str = FIG_DIR) -> Path:
    """Save ``fig`` as a PNG into the figures directory and return its path."""
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / (name if name.endswith(".png") else f"{name}.png")
    fig.savefig(path)
    return path
