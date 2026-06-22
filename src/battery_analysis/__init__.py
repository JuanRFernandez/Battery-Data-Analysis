"""Battery & PV market analysis - reusable analysis utilities.

Modules
-------
io        : load and clean the price, PV and weather CSVs.
arbitrage : battery arbitrage models (greedy perfect-foresight backtest and an
            exact LP dispatch optimisation) plus a shared ``BatterySpec``.
plotting  : shared matplotlib/seaborn styling and a ``save_fig`` helper.
"""

from . import arbitrage, io, metrics, plotting
from .arbitrage import BatterySpec, greedy_backtest, lp_optimize
from .metrics import battery_capture, capture_metrics

__all__ = [
    "arbitrage",
    "io",
    "metrics",
    "plotting",
    "BatterySpec",
    "greedy_backtest",
    "lp_optimize",
    "capture_metrics",
    "battery_capture",
]
