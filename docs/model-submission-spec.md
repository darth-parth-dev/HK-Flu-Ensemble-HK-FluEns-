# HK-FluEns Model Submission Specification

**Version:** 0.1 (Draft — Phase 1)
**Status:** Draft — pending target variable decision (Decision Point 1 in planning document)

---

## Overview

All models contributing to HK-FluEns must submit forecasts in a standardised CSV format
derived from the [CDC FluSight format](https://github.com/cdcepi/FluSight-forecast-hub).
This format ensures that all models can be evaluated on the same metrics (WIS, PI coverage)
and combined by the AWBE ensemble engine.

---

## File Naming Convention

```
hub/model-output/<forecast_date>-<model_id>.csv
```

Example:
```
hub/model-output/2026-09-05-HKU_SARIMA.csv
hub/model-output/2026-09-05-HKU_ETS.csv
hub/model-output/2026-09-05-CUHK_SEIR.csv
```

- `forecast_date`: ISO 8601 date of the **Saturday** ending the week in which the forecast was produced
- `model_id`: Registered model identifier (format: `<institution>_<method>`)

---

## CSV Format

### Required Columns

| Column | Type | Description |
|--------|------|-------------|
| `forecast_date` | date (YYYY-MM-DD) | Date forecast was produced (Saturday of forecast week) |
| `target` | string | Forecast target (see Targets below) |
| `horizon` | string | Forecast horizon (e.g. `"1 wk ahead"`) |
| `type` | string | Row type: `"quantile"` or `"point"` |
| `quantile` | float or `"NA"` | Quantile level (for type=quantile rows); `"NA"` for point rows |
| `value` | float | Forecast value in target units |

### Example

```csv
forecast_date,target,horizon,type,quantile,value
2026-09-05,lab_positivity_pct,1 wk ahead,quantile,0.010,1.2
2026-09-05,lab_positivity_pct,1 wk ahead,quantile,0.025,2.1
2026-09-05,lab_positivity_pct,1 wk ahead,quantile,0.050,3.4
2026-09-05,lab_positivity_pct,1 wk ahead,quantile,0.100,5.0
2026-09-05,lab_positivity_pct,1 wk ahead,quantile,0.150,6.2
2026-09-05,lab_positivity_pct,1 wk ahead,quantile,0.200,7.1
2026-09-05,lab_positivity_pct,1 wk ahead,quantile,0.250,7.9
2026-09-05,lab_positivity_pct,1 wk ahead,quantile,0.300,8.6
2026-09-05,lab_positivity_pct,1 wk ahead,quantile,0.350,9.3
2026-09-05,lab_positivity_pct,1 wk ahead,quantile,0.400,10.0
2026-09-05,lab_positivity_pct,1 wk ahead,quantile,0.450,10.6
2026-09-05,lab_positivity_pct,1 wk ahead,quantile,0.500,11.2
2026-09-05,lab_positivity_pct,1 wk ahead,quantile,0.550,11.9
2026-09-05,lab_positivity_pct,1 wk ahead,quantile,0.600,12.7
2026-09-05,lab_positivity_pct,1 wk ahead,quantile,0.650,13.6
2026-09-05,lab_positivity_pct,1 wk ahead,quantile,0.700,14.6
2026-09-05,lab_positivity_pct,1 wk ahead,quantile,0.750,15.9
2026-09-05,lab_positivity_pct,1 wk ahead,quantile,0.800,17.5
2026-09-05,lab_positivity_pct,1 wk ahead,quantile,0.850,19.6
2026-09-05,lab_positivity_pct,1 wk ahead,quantile,0.900,22.4
2026-09-05,lab_positivity_pct,1 wk ahead,quantile,0.950,26.8
2026-09-05,lab_positivity_pct,1 wk ahead,quantile,0.975,31.2
2026-09-05,lab_positivity_pct,1 wk ahead,quantile,0.990,36.0
2026-09-05,lab_positivity_pct,1 wk ahead,point,NA,11.2
```

---

## Targets

| Target string | Definition | Units |
|--------------|------------|-------|
| `lab_positivity_pct` | Weekly lab positivity: positive specimens / total specimens tested | % (0–100) |
| `gopc_ili_rate_per1000` | GOPC ILI consultations per 1,000 attendances | Rate per 1,000 |

> **Note:** `lab_positivity_pct` is the **primary target** for ensemble combination and AWBE re-weighting. `gopc_ili_rate_per1000` is the secondary target. Models must submit both targets if they wish to participate in the full ensemble.

---

## Required Quantile Levels

All 23 quantile levels must be present for each (target, horizon) combination:

```
0.010, 0.025, 0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450,
0.500, 0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 0.950, 0.975, 0.990
```

Plus one `"point"` row per (target, horizon) combination.

---

## Required Horizons

| Horizon string | Definition |
|----------------|------------|
| `"1 wk ahead"` | Forecast for the week starting 7 days after `forecast_date` |
| `"2 wk ahead"` | Forecast for the week starting 14 days after `forecast_date` |
| `"3 wk ahead"` | Forecast for the week starting 21 days after `forecast_date` |
| `"4 wk ahead"` | Forecast for the week starting 28 days after `forecast_date` |

---

## Validation Rules

The hub validator (`scoring/wis.py`, Phase 0) checks:

1. **Schema:** All required columns present; correct data types
2. **Completeness:** All 23 quantile levels present for each (target, horizon) pair
3. **Monotonicity:** `value` must be non-decreasing as `quantile` increases
4. **Range:** `lab_positivity_pct` values must be in [0, 100]; `gopc_ili_rate_per1000` in [0, 500]
5. **Date alignment:** `forecast_date` must be a Saturday; target weeks must align with CHP reporting weeks

Submissions failing validation are rejected with a structured error message.

---

## Weighted Interval Score (WIS)

WIS is the primary metric used for:
- Ranking models in the AWBE ensemble
- Producing the ensemble evaluation report
- Benchmarking against published Hong Kong flu forecasting studies

For a forecast with K symmetric intervals at levels α_k ∈ {0.02, 0.05, ..., 0.98}:

```
WIS = (1/K) × Σ_k [ (u_k - l_k) + (2/α_k)(l_k - y)·1[y < l_k] + (2/α_k)(y - u_k)·1[y > u_k] ]
    + (1/2) |y - ŷ|
```

where y is the observed value, ŷ is the point forecast, and (l_k, u_k) are the lower and upper
bounds of the k-th prediction interval.

**Lower WIS = better.** Models with narrower, well-calibrated intervals achieve lower WIS.

---

## Model Registration

To register a model with HK-FluEns, create a file:
```
models/<model_id>/metadata.json
```

with the following structure:

```json
{
  "model_id": "HKU_SARIMA",
  "model_abbr": "SARIMA",
  "institution": "HKU School of Public Health",
  "contact": "email@hku.hk",
  "method": "SARIMA(1,0,1)(1,1,0)[52] with absolute humidity regressors",
  "data_sources": ["CHP Flu Express", "WHO FluNet", "HKO climate data"],
  "target": ["lab_positivity_pct", "gopc_ili_rate_per1000"],
  "horizons": ["1 wk ahead", "2 wk ahead", "3 wk ahead", "4 wk ahead"],
  "first_submission_date": "2026-09-06",
  "ensemble_eligible": true,
  "notes": "Baseline SARIMA model. No statsmodels dependency."
}
```

---

*For questions, contact Parth Singh. This specification will be finalised in Phase 1 following the target variable decision workshop.*
