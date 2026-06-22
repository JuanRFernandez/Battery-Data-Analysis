# Case Brief — Energy Data Analyst

**Dataset coverage:** DE-LU day-ahead electricity market, January–June 2022.

## Task 1 — The Battery Business Case

Two price datasets are provided:

- **15-minute contracts** (`data/1_Day-ahead_Prices_15min.csv`) — delivery of electricity at a
  constant power over 15 minutes.
- **60-minute contracts** (`data/1_Day-ahead_Prices_60min.csv`) — same for 60-minute contracts.

Summarise and compare both datasets statistically. Which patterns do they share, and which differ?
Considering a battery traded for arbitrage / wholesale revenue, recommend which contract type is
more beneficial. Document the logic and present key insights with visualisations.

## Task 2 — Weather & PV Exploration

Two datasets for the same period:

- **PV production** (`data/2_pv_production_pool.csv`) — hourly output from a pool of residential
  PV installations in one geographic region.
- **Weather** (`data/2_weather.csv`) — temperature, cloud cover, solar radiation, wind, etc.

Analyse how weather variables influence PV production. Provide a statistical summary, examine
daily and seasonal patterns, identify anomalies, and conceptualise a forecast model based on the
key predictive variables.

**Optional:** How does observed PV production affect market prices?

## Weather field descriptions

| Field | Description |
| --- | --- |
| `forecast_datetime` | Start of the 1-hour forecast period. |
| `temperature` | Air temperature (°C). |
| `dewpoint` | Dew point temperature (°C). |
| `cloudcover_[low/mid/high/total]` | Sky cover by altitude band (0–2 km, 2–6 km, 6+ km, total). |
| `10_metre_[u/v]_wind_component` | Eastward (`u`) / northward (`v`) wind at 10 m. |
| `direct_solar_radiation` | Direct radiation on a plane perpendicular to the Sun, Wh/m² per hour. |
| `surface_solar_radiation_downwards` | Direct + diffuse radiation on a horizontal surface, Wh/m² per hour. |
| `snowfall` | Snowfall over the hour, metres of water equivalent. |
| `total_precipitation` | Accumulated liquid reaching the surface over the hour, metres. |
