"""Unit tests for the capture-price / capture-rate metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from battery_analysis.metrics import battery_capture, capture_metrics, granularity_match


def test_flat_production_capture_rate_is_one():
    # Constant production earns exactly the baseload, regardless of price shape.
    price = np.array([10.0, 50.0, 90.0, 30.0])
    production = np.ones(4)
    m = capture_metrics(price, production)
    assert m["capture_rate"] == pytest.approx(1.0)
    assert m["capture_price"] == pytest.approx(price.mean())


def test_production_in_cheap_hours_cannibalised():
    # Production concentrated when prices are low -> capture rate below 1.
    price = np.array([10.0, 20.0, 100.0, 120.0])
    production = np.array([10.0, 8.0, 0.5, 0.0])
    m = capture_metrics(price, production)
    assert m["capture_rate"] < 1.0
    assert m["capture_price"] < m["baseload_price"]


def test_production_in_expensive_hours_premium():
    price = np.array([10.0, 20.0, 100.0, 120.0])
    production = np.array([0.0, 0.5, 8.0, 10.0])
    m = capture_metrics(price, production)
    assert m["capture_rate"] > 1.0


def test_capture_metrics_ignores_nan():
    price = np.array([10.0, np.nan, 90.0])
    production = np.array([1.0, 1.0, 1.0])
    m = capture_metrics(price, production)
    assert m["capture_price"] == pytest.approx(50.0)  # mean of 10 and 90


def test_battery_buys_low_sells_high():
    # A sensible battery charges at low prices and discharges at high prices.
    price = np.array([10.0, 20.0, 100.0, 120.0])
    charge = np.array([1.0, 1.0, 0.0, 0.0])
    discharge = np.array([0.0, 0.0, 1.0, 1.0])
    b = battery_capture(price, charge, discharge)
    assert b["sell_price"] > b["baseload_price"] > b["buy_price"]
    assert b["capture_spread"] > 0
    assert b["sell_over_baseload"] > 1.0 > b["buy_over_baseload"]


def _hourly_index(n_hours: int) -> pd.DatetimeIndex:
    return pd.date_range("2022-01-01", periods=n_hours, freq="60min")


def test_granularity_match_perfect_aggregation():
    # 15-min series whose four quarter-hours in each hour average *exactly* to
    # the 60-min series -> a mechanical aggregation: corr 1, every diff 0.
    n_hours = 24
    hourly = pd.Series(
        np.linspace(10.0, 200.0, n_hours), index=_hourly_index(n_hours)
    )
    # Spread each hour's value into 4 QHs that still average to it (h-3,h-1,h+1,h+3).
    qh_offsets = np.array([-3.0, -1.0, 1.0, 3.0])
    qh_index = pd.date_range("2022-01-01", periods=n_hours * 4, freq="15min")
    qh_values = (hourly.to_numpy()[:, None] + qh_offsets).ravel()
    p15 = pd.Series(qh_values, index=qh_index)

    m = granularity_match(p15, hourly)
    assert m["n_hours"] == n_hours
    assert m["near_equal_rate"] == pytest.approx(1.0)
    assert m["mean_abs_diff"] == pytest.approx(0.0, abs=1e-9)
    assert m["corr"] == pytest.approx(1.0)


def test_granularity_match_separate_market_differs():
    # A 60-min series that is NOT the aggregate of the 15-min series -> the two
    # are essentially never equal and the mean absolute difference is positive.
    n_hours = 24
    rng = np.random.default_rng(0)
    qh_index = pd.date_range("2022-01-01", periods=n_hours * 4, freq="15min")
    p15 = pd.Series(rng.normal(50.0, 20.0, n_hours * 4), index=qh_index)
    p60 = pd.Series(rng.normal(50.0, 20.0, n_hours), index=_hourly_index(n_hours))

    m = granularity_match(p15, p60)
    assert m["n_hours"] == n_hours
    assert m["near_equal_rate"] < 1.0
    assert m["mean_abs_diff"] > 0.0
