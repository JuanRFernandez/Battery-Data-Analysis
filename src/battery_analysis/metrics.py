"""Price metrics: capture-price / capture-rate, plus a 15-vs-60-min granularity check.

These are the standard energy-industry "value factor" metrics:

* **Baseload price** — the simple time-weighted average price over the period.
* **Capture price** — the *production-weighted* average price an asset actually
  earns: ``sum(price * volume) / sum(volume)``.
* **Capture rate** (a.k.a. *value factor* / *Quality Factor*) — capture price
  divided by baseload. For solar a rate **< 100 %** signals **cannibalisation**
  (it produces when prices are low). A battery is the mirror image: it should
  **buy below** and **sell above** baseload, so its sell-capture rate is > 100 %
  and its buy-capture rate < 100 %.

``granularity_match`` additionally cross-checks whether the 60-min day-ahead
series is merely the 15-min series aggregated (it is not — they are separate
auctions).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _clean(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Align arrays and drop rows where any value is NaN."""
    cols = [np.asarray(a, dtype=float) for a in arrays]
    mask = np.ones(len(cols[0]), dtype=bool)
    for a in cols:
        mask &= ~np.isnan(a)
    return tuple(a[mask] for a in cols)


def capture_metrics(price, production) -> dict:
    """Capture price & capture rate of a production profile (e.g. PV).

    Parameters
    ----------
    price : array-like of EUR/MWh.
    production : array-like of generation (any consistent unit).

    Returns
    -------
    dict with ``baseload_price``, ``capture_price``, ``capture_rate``.
    A capture rate < 1 means the asset earns less than the average price
    (cannibalisation); = 1 means it is price-neutral.
    """
    price, production = _clean(price, production)
    baseload = float(price.mean())
    total = float(production.sum())
    capture_price = float((price * production).sum() / total) if total > 0 else float("nan")
    capture_rate = capture_price / baseload if baseload != 0 else float("nan")
    return {
        "baseload_price": baseload,
        "capture_price": capture_price,
        "capture_rate": capture_rate,
    }


def battery_capture(price, charge, discharge) -> dict:
    """Volume-weighted buy/sell prices and capture spread for a battery.

    ``charge`` / ``discharge`` are per-interval grid energies (MWh) from a
    dispatch schedule. Returns the average **buy** price (charge-weighted),
    average **sell** price (discharge-weighted), the **capture spread**
    (sell − buy), and each leg relative to baseload.
    """
    price, charge, discharge = _clean(price, charge, discharge)
    baseload = float(price.mean())
    csum, dsum = float(charge.sum()), float(discharge.sum())
    buy = float((price * charge).sum() / csum) if csum > 0 else float("nan")
    sell = float((price * discharge).sum() / dsum) if dsum > 0 else float("nan")
    return {
        "baseload_price": baseload,
        "buy_price": buy,
        "sell_price": sell,
        "capture_spread": sell - buy,
        "sell_over_baseload": sell / baseload if baseload else float("nan"),
        "buy_over_baseload": buy / baseload if baseload else float("nan"),
    }


def granularity_match(p15_price: pd.Series, p60_price: pd.Series) -> dict:
    """Test whether the 60-min day-ahead series is just the 15-min series aggregated.

    Averages each hour's four quarter-hours up to hourly resolution
    (``p15.resample("60min").mean()``), aligns it with the native 60-min series
    and returns agreement statistics. A *mechanical* aggregation would give a
    correlation of 1 and a difference of 0 in every hour; in reality the two are
    **separate auctions** (the hourly and the quarter-hour day-ahead auctions),
    so they track closely (corr ~0.96) but are essentially never equal.

    Parameters
    ----------
    p15_price, p60_price : pd.Series
        Day-ahead prices (EUR/MWh) indexed by a ``DatetimeIndex`` at 15- and
        60-minute resolution respectively (as returned by ``io.load_prices``).

    Returns
    -------
    dict with
        ``aligned`` — DataFrame with columns ``agg15_to_60``, ``da_60`` and
        ``diff`` (= ``da_60 - agg15_to_60``), inner-joined on the hourly grid;
        ``n_hours`` — number of aligned hours;
        ``corr`` — Pearson correlation of aggregated-15 vs native-60;
        ``near_equal_rate`` — fraction of hours with ``|diff| < 0.01``;
        ``within_1_rate`` — fraction of hours with ``|diff| < 1``;
        ``mean_diff``, ``mean_abs_diff``, ``rmse``, ``max_abs_diff`` — EUR/MWh.
    """
    agg = p15_price.resample("60min").mean().rename("agg15_to_60")
    aligned = pd.concat([agg, p60_price.rename("da_60")], axis=1).dropna()
    aligned["diff"] = aligned["da_60"] - aligned["agg15_to_60"]
    d = aligned["diff"].to_numpy()
    return {
        "aligned": aligned,
        "n_hours": int(len(aligned)),
        "corr": float(aligned["agg15_to_60"].corr(aligned["da_60"])),
        "near_equal_rate": float((np.abs(d) < 0.01).mean()),
        "within_1_rate": float((np.abs(d) < 1.0).mean()),
        "mean_diff": float(d.mean()),
        "mean_abs_diff": float(np.abs(d).mean()),
        "rmse": float(np.sqrt((d**2).mean())),
        "max_abs_diff": float(np.abs(d).max()),
    }
