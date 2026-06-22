"""Data loading and cleaning for the battery & PV analysis.

The four source CSVs live in ``data/`` and have slightly different conventions:

* The two price files encode the delivery period as a text interval
  ``"01.01.2022 00:00 - 01.01.2022 00:15"`` in local CET/CEST
  time.  We parse the *start* of each interval and keep it as a naive local
  timestamp so that intraday / solar seasonality lines up with German clock
  time (the alternative, UTC, would smear the midday solar dip across two
  hours depending on DST).

* The PV and weather files are clean hourly series in **UTC**.

All loaders return a ``DatetimeIndex``-ed frame so the downstream resampling,
rolling and decomposition code can rely on a proper time index.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Repository root = two levels up from this file (src/battery_analysis/io.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

PRICE_FILES = {
    "15min": "1_Day-ahead_Prices_15min.csv",
    "60min": "1_Day-ahead_Prices_60min.csv",
}
PV_FILE = "2_pv_production_pool.csv"
WEATHER_FILE = "2_weather.csv"

# Step length in hours for each contract granularity (used by the battery
# models to convert MW <-> MWh).
DELTA_T_HOURS = {"15min": 0.25, "60min": 1.0}


def load_prices(freq: str, data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Load a day-ahead price file and return a clean, indexed series.

    Parameters
    ----------
    freq : {"15min", "60min"}
        Which contract granularity to load.
    data_dir : path
        Directory containing the CSVs.

    Returns
    -------
    DataFrame indexed by the (local, naive) interval-start timestamp with a
    single ``price`` column in EUR/MWh.  Missing / non-numeric prices (e.g. the
    DST spring-forward gap) are coerced to NaN and dropped, and the index is
    sorted and de-duplicated.
    """
    if freq not in PRICE_FILES:
        raise ValueError(f"freq must be one of {list(PRICE_FILES)}, got {freq!r}")

    path = Path(data_dir) / PRICE_FILES[freq]
    raw = pd.read_csv(path)

    mtu_col = next(c for c in raw.columns if c.startswith("MTU"))
    price_col = next(c for c in raw.columns if "Price" in c)

    # Interval start = text before the " - " separator.
    start_str = raw[mtu_col].str.split(" - ", regex=False).str[0].str.strip()
    ts = pd.to_datetime(start_str, format="%d.%m.%Y %H:%M", errors="coerce")
    price = pd.to_numeric(raw[price_col], errors="coerce")

    df = (
        pd.DataFrame({"price": price.to_numpy()}, index=pd.DatetimeIndex(ts, name="datetime"))
        .dropna()
        .sort_index()
    )
    df = df[~df.index.duplicated(keep="first")]
    return df


def load_pv(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Load the PV pool production series (hourly, UTC)."""
    path = Path(data_dir) / PV_FILE
    df = pd.read_csv(path, parse_dates=["datetime_utc"])
    df = df.rename(columns={"datetime_utc": "datetime"}).set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def load_weather(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Load the weather series (hourly, UTC)."""
    path = Path(data_dir) / WEATHER_FILE
    df = pd.read_csv(path, parse_dates=["forecast_datetime_utc"])
    df = df.rename(columns={"forecast_datetime_utc": "datetime"}).set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def load_pv_weather(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Inner-join PV production and weather on their shared UTC timestamp."""
    pv = load_pv(data_dir)
    weather = load_weather(data_dir)
    merged = pv.join(weather, how="inner")
    return merged


def to_local(df: pd.DataFrame, tz: str = "Europe/Berlin") -> pd.DataFrame:
    """Convert a UTC-indexed frame to local wall-clock time (tz-naive).

    Used to align the UTC PV series with the local-time day-ahead prices for
    the optional price-cannibalisation subtask.
    """
    out = df.copy()
    idx = out.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    out.index = idx.tz_convert(tz).tz_localize(None)
    return out
