"""Unit tests for the battery arbitrage models.

These guard the economic and physical invariants the business recommendation
relies on:

* the battery never violates its usable SoC window (physical feasibility),
* the LP optimum is at least as good as the feasible greedy heuristic
  (otherwise the "value of optimisation" story would be wrong),
* a flat price curve yields ~zero profit (no spread => no arbitrage),
* adding a degradation cost can only reduce net revenue.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from battery_analysis.arbitrage import BatterySpec, greedy_backtest, lp_optimize

DT = 1.0  # hourly test data


def _make_prices(n_days: int = 4, seed: int = 0) -> pd.Series:
    """Synthetic hourly prices with a clear daily arbitrage spread."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=24 * n_days, freq="h")
    hours = idx.hour.to_numpy()
    # Cheap at night/midday, expensive in the evening peak (~17-20h).
    base = 50 + 30 * np.sin((hours - 8) / 24 * 2 * np.pi)
    peak = np.where((hours >= 17) & (hours <= 20), 60, 0)
    noise = rng.normal(0, 5, size=len(idx))
    return pd.Series(base + peak + noise, index=idx, name="price")


def _reconstruct_soc(schedule: pd.DataFrame, spec: BatterySpec) -> np.ndarray:
    """Replay the dispatch day by day and return the SoC trajectory (MWh)."""
    eta = spec.eta_one_way
    socs = []
    for _, day in schedule.groupby(schedule.index.normalize()):
        soc = spec.soc_min_mwh
        for _, row in day.iterrows():
            soc += eta * row["charge_grid"] - row["discharge_grid"] / eta
            socs.append(soc)
    return np.array(socs)


@pytest.fixture
def spec() -> BatterySpec:
    return BatterySpec()


def test_soc_within_window_greedy(spec):
    res = greedy_backtest(_make_prices(), spec, DT, max_cycles_per_day=1.0)
    soc = _reconstruct_soc(res["schedule"], spec)
    assert soc.min() >= spec.soc_min_mwh - 1e-6
    assert soc.max() <= spec.soc_max_mwh + 1e-6


def test_soc_within_window_lp(spec):
    res = lp_optimize(_make_prices(), spec, DT, max_cycles_per_day=2.0)
    soc = _reconstruct_soc(res["schedule"], spec)
    assert soc.min() >= spec.soc_min_mwh - 1e-6
    assert soc.max() <= spec.soc_max_mwh + 1e-6


def test_lp_beats_or_matches_greedy(spec):
    prices = _make_prices()
    greedy = greedy_backtest(prices, spec, DT, max_cycles_per_day=2.0)
    lp = lp_optimize(prices, spec, DT, max_cycles_per_day=2.0)
    assert lp["total_revenue_eur"] >= greedy["total_revenue_eur"] - 1e-6


def test_flat_prices_zero_profit(spec):
    idx = pd.date_range("2022-01-01", periods=48, freq="h")
    flat = pd.Series(50.0, index=idx, name="price")
    greedy = greedy_backtest(flat, spec, DT, max_cycles_per_day=2.0)
    lp = lp_optimize(flat, spec, DT, max_cycles_per_day=2.0)
    assert greedy["total_revenue_eur"] == pytest.approx(0.0, abs=1e-6)
    assert lp["total_revenue_eur"] == pytest.approx(0.0, abs=1e-3)


def test_degradation_reduces_revenue(spec):
    prices = _make_prices()
    with_deg = lp_optimize(prices, spec, DT, max_cycles_per_day=2.0, use_degradation=True)
    without_deg = lp_optimize(prices, spec, DT, max_cycles_per_day=2.0, use_degradation=False)
    assert without_deg["total_revenue_eur"] >= with_deg["total_revenue_eur"] - 1e-6


def test_usable_capacity_respects_soc_window():
    spec = BatterySpec(capacity_mwh=2.0, soc_min_frac=0.1, soc_max_frac=0.9)
    assert spec.usable_capacity_mwh == pytest.approx(1.6)
    assert spec.eta_one_way == pytest.approx(0.9486832, rel=1e-6)
