"""Battery arbitrage models for the day-ahead wholesale market.

Two models share a single :class:`BatterySpec`:

* :func:`greedy_backtest` - a transparent, **feasible** perfect-foresight
  heuristic.  Each day it charges during the cheapest intervals and discharges
  during the dearest ones, but it then *simulates the state of charge (SoC)
  chronologically*, so the dispatch it reports is physically achievable
  (power, SoC window and ordering are all respected).  It is therefore a lower
  bound on the achievable arbitrage value.

* :func:`lp_optimize` - the exact linear-program optimum (PuLP / CBC) for the
  same battery and constraints.  Because the greedy dispatch is a feasible
  point of the LP, the LP optimum is always >= the greedy result; the gap is
  the value of optimisation ("the value of granularity / foresight").

Energy bookkeeping
------------------
Power (``power_mw``) is the grid-meter (AC) limit, so per interval at most
``power_mw * dt_hours`` MWh can flow in or out.  A one-way efficiency
``eta = sqrt(round_trip_eff)`` is applied on both charge and discharge, giving
the quoted round-trip loss.  The battery only swings within the usable SoC
window ``[soc_min_frac, soc_max_frac]``.  A degradation cost
(``degradation_cost_eur_per_mwh``) is charged per MWh delivered to the grid
(throughput), which raises the spread a trade must clear and discourages
low-value cycling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BatterySpec:
    """Physical and economic specification of the battery.

    Defaults are the case base case: a 1 MW / 2 MWh (2-hour) lithium BESS with
    90% round-trip efficiency, operated within a 10-90% SoC window, and a
    ~7 EUR/MWh degradation cost on throughput.
    """

    power_mw: float = 1.0
    capacity_mwh: float = 2.0
    round_trip_eff: float = 0.90
    soc_min_frac: float = 0.10
    soc_max_frac: float = 0.90
    degradation_cost_eur_per_mwh: float = 7.0

    @property
    def eta_one_way(self) -> float:
        """One-way efficiency such that eta_one_way**2 == round_trip_eff."""
        return math.sqrt(self.round_trip_eff)

    @property
    def usable_capacity_mwh(self) -> float:
        """Energy swing available between the SoC limits."""
        return self.capacity_mwh * (self.soc_max_frac - self.soc_min_frac)

    @property
    def soc_min_mwh(self) -> float:
        """Lower SoC limit in MWh (``capacity_mwh * soc_min_frac``)."""
        return self.capacity_mwh * self.soc_min_frac

    @property
    def soc_max_mwh(self) -> float:
        """Upper SoC limit in MWh (``capacity_mwh * soc_max_frac``)."""
        return self.capacity_mwh * self.soc_max_frac


def _summarise(
    label: str, schedule: pd.DataFrame, spec: BatterySpec, deg_cost_per_mwh: float
) -> dict:
    """Build the standard result dict from a per-interval dispatch schedule.

    ``deg_cost_per_mwh`` is the *effective* degradation cost used by the model
    (0.0 when ``use_degradation=False``), so the reported net revenue is
    consistent with the objective that produced the schedule.
    """
    n_days = schedule.index.normalize().nunique()
    discharged_grid = float(schedule["discharge_grid"].sum())
    charged_grid = float(schedule["charge_grid"].sum())
    gross = float(
        (schedule["discharge_grid"] * schedule["price"]).sum()
        - (schedule["charge_grid"] * schedule["price"]).sum()
    )
    deg_cost = deg_cost_per_mwh * discharged_grid
    net = gross - deg_cost
    # Equivalent full cycles = total battery-side discharge / nameplate capacity.
    battery_discharge = discharged_grid / spec.eta_one_way
    efc = battery_discharge / spec.capacity_mwh
    return {
        "label": label,
        "total_revenue_eur": net,
        "gross_revenue_eur": gross,
        "degradation_cost_eur": deg_cost,
        "n_days": int(n_days),
        "avg_daily_revenue_eur": net / n_days if n_days else float("nan"),
        "charged_grid_mwh": charged_grid,
        "discharged_grid_mwh": discharged_grid,
        "equivalent_full_cycles": efc,
        "cycles_per_day": efc / n_days if n_days else float("nan"),
        "revenue_per_mwh_discharged": net / discharged_grid if discharged_grid else float("nan"),
        "schedule": schedule,
    }


def greedy_backtest(
    prices: pd.Series,
    spec: BatterySpec,
    dt_hours: float,
    max_cycles_per_day: float = 1.0,
    use_degradation: bool = True,
) -> dict:
    """Feasible perfect-foresight greedy dispatch, day by day.

    Parameters
    ----------
    prices : Series indexed by datetime, EUR/MWh.
    spec : BatterySpec.
    dt_hours : interval length in hours (0.25 for 15-min, 1.0 for 60-min).
    max_cycles_per_day : daily charge/discharge budget in equivalent full cycles.
    use_degradation : if False, the degradation cost is set to zero.
    """
    prices = prices.dropna().sort_index()
    g_max = spec.power_mw * dt_hours  # max grid MWh per interval
    eta = spec.eta_one_way
    deg = spec.degradation_cost_eur_per_mwh if use_degradation else 0.0
    budget_batt = max_cycles_per_day * spec.usable_capacity_mwh  # battery-side per day

    rows: list[dict] = []
    for _, day_prices in prices.groupby(prices.index.normalize()):
        vals = day_prices.to_numpy()
        times = day_prices.index
        n = len(vals)
        charge_grid = np.zeros(n)
        discharge_grid = np.zeros(n)

        # --- target selection (cheapest charge / dearest discharge) ---
        # Charge: fill the usable window from the cheapest intervals.
        remaining = budget_batt
        for t in np.argsort(vals, kind="stable"):  # ascending price
            if remaining <= 1e-9:
                break
            gain = min(eta * g_max, remaining)  # battery-side energy gained
            charge_grid[t] = gain / eta  # grid-side energy bought
            remaining -= gain
        # Discharge: empty the usable window into the dearest intervals.
        remaining = budget_batt
        for t in np.argsort(-vals, kind="stable"):  # descending price
            if remaining <= 1e-9:
                break
            removal = min(g_max / eta, remaining)  # battery-side energy removed
            discharge_grid[t] = removal * eta  # grid-side energy sold
            remaining -= removal

        # --- chronological SoC simulation to guarantee feasibility ---
        soc = spec.soc_min_mwh
        c_exec = np.zeros(n)
        d_exec = np.zeros(n)
        for t in range(n):
            if charge_grid[t] > 0:
                gain = min(eta * charge_grid[t], spec.soc_max_mwh - soc)
                gain = max(gain, 0.0)
                c_exec[t] = gain / eta
                soc += gain
            if discharge_grid[t] > 0:
                removal = min(discharge_grid[t] / eta, soc - spec.soc_min_mwh)
                removal = max(removal, 0.0)
                d_exec[t] = removal * eta
                soc -= removal

        # --- per-day go/no-go: idle if the day is not profitable after costs ---
        day_profit = float((d_exec * vals).sum() - (c_exec * vals).sum() - deg * d_exec.sum())
        if day_profit < 0:
            c_exec[:] = 0.0
            d_exec[:] = 0.0

        rows.append(
            pd.DataFrame(
                {"price": vals, "charge_grid": c_exec, "discharge_grid": d_exec},
                index=times,
            )
        )

    schedule = pd.concat(rows).sort_index()
    return _summarise(f"greedy ({max_cycles_per_day:g} cyc/day)", schedule, spec, deg)


def lp_optimize(
    prices: pd.Series,
    spec: BatterySpec,
    dt_hours: float,
    max_cycles_per_day: float = 2.0,
    use_degradation: bool = True,
    verbose: bool = False,
) -> dict:
    """Exact LP dispatch optimisation, solved independently per day.

    Maximises ``sum(price * (discharge - charge)) - deg * sum(discharge)``
    subject to grid-power limits, the usable SoC window, SoC dynamics with
    efficiency, and a per-day throughput cap of ``max_cycles_per_day``. SoC
    starts each day at ``soc_min`` (consistent with the greedy backtest), so
    the LP optimum dominates the greedy result day by day.
    """
    import pulp

    prices = prices.dropna().sort_index()
    g_max = spec.power_mw * dt_hours
    eta = spec.eta_one_way
    deg = spec.degradation_cost_eur_per_mwh if use_degradation else 0.0
    solver = pulp.PULP_CBC_CMD(msg=1 if verbose else 0)

    rows: list[pd.DataFrame] = []
    for _, day_prices in prices.groupby(prices.index.normalize()):
        vals = day_prices.to_numpy()
        times = day_prices.index
        n = len(vals)

        prob = pulp.LpProblem("battery_arbitrage", pulp.LpMaximize)
        c = [pulp.LpVariable(f"c{t}", lowBound=0, upBound=g_max) for t in range(n)]
        d = [pulp.LpVariable(f"d{t}", lowBound=0, upBound=g_max) for t in range(n)]
        soc = [
            pulp.LpVariable(f"s{t}", lowBound=spec.soc_min_mwh, upBound=spec.soc_max_mwh)
            for t in range(n)
        ]

        prev = spec.soc_min_mwh
        for t in range(n):
            prob += soc[t] == prev + eta * c[t] - d[t] / eta
            prev = soc[t]

        # Daily throughput cap: battery-side discharge <= cycles * usable energy.
        prob += pulp.lpSum(d) <= max_cycles_per_day * spec.usable_capacity_mwh * eta

        prob += pulp.lpSum(vals[t] * (d[t] - c[t]) for t in range(n)) - deg * pulp.lpSum(d)
        prob.solve(solver)

        c_val = np.array([v.value() or 0.0 for v in c])
        d_val = np.array([v.value() or 0.0 for v in d])
        rows.append(
            pd.DataFrame(
                {"price": vals, "charge_grid": c_val, "discharge_grid": d_val},
                index=times,
            )
        )

    schedule = pd.concat(rows).sort_index()
    return _summarise(f"LP ({max_cycles_per_day:g} cyc/day)", schedule, spec, deg)
