# Battery & PV Market Analysis

A two-part energy-data study. Both tasks live in a single notebook,
[`notebooks/01_battery_pv_analysis.ipynb`](notebooks/01_battery_pv_analysis.ipynb). The task brief
is in [`CASE.md`](CASE.md).

- **Task 1 — Battery Business Case.** Statistically compare 15-minute vs 60-minute day-ahead
  price contracts (DE-LU, Jan–Jun 2022) and recommend which is more beneficial for trading a
  battery for arbitrage. Includes a feasible perfect-foresight backtest and, as a bonus, an exact
  LP dispatch optimisation.
- **Task 2 — Weather & PV Exploration.** How weather drives PV pool production: daily and seasonal
  patterns, anomalies, and a conceptual forecast model. Optional: how PV affects market prices.

## Deliverables

- `notebooks/01_battery_pv_analysis.ipynb` — the full analysis, narrative and figures.
- `notebooks/01_battery_pv_analysis.html` — a **pre-run, self-contained HTML render** of the notebook
  (every figure embedded); open in any browser to read the whole analysis without installing anything.

## Layout

```
data/                   source CSVs (prices, PV, weather)
src/battery_analysis/   reusable, unit-tested analysis code (io, arbitrage, metrics, plotting)
tests/                  pytest unit tests
notebooks/              01_battery_pv_analysis.ipynb — the deliverable
figures/                PNGs produced by the notebook
CASE.md                 the task brief
```

## Run it (uv)

```bash
uv sync --extra dev                                   # create the env & install deps
uv run pytest                                         # run the unit tests
uv run jupyter nbconvert --to notebook --execute \
  --inplace notebooks/01_battery_pv_analysis.ipynb    # run the analysis end-to-end
```

The notebook imports the local `battery_analysis` package from `src/`, so run it from inside the repo
(the whole folder is needed, not just the `.ipynb`). Figures are written to `figures/`.

## Key assumptions (full list at the top of the notebook)

- Prices are local CET/CEST, indexed at the interval start; PV and weather are hourly UTC.
- Battery base case: 1 MW / 2 MWh, 90% round-trip efficiency, 10–90% usable SoC, ~€7/MWh wear cost.
- Perfect price foresight ⇒ revenue is an upper bound on a single revenue stream.
- PV production has no stated unit, so it is treated as a relative index.
