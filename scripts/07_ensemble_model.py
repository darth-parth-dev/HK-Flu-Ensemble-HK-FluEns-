"""
07_ensemble_model.py
====================
Three-model ensemble for Hong Kong influenza forecasting 2026-27.

DESIGN RATIONALE (Supervisor-approved, March 2026)
---------------------------------------------------
No single model performs adequately across peak timing, peak magnitude,
and strain-specific severity. The ensemble addresses this by combining
three structurally complementary models:

  Model 1 — SARIMA(1,0,1)(1,1,0)[52]      weight = 1/3
    Role: seasonal timing anchor.
    ARIMA(1,0,1) with seasonal differencing at lag 52 captures Hong
    Kong's bimodal pattern. Load pre-fitted forecast; do not refit.
    Weakness: magnitude calibration (post-COVID baseline shift).

  Model 2 — Historical Percentile Benchmark  weight = 1/3
    Role: magnitude anchor.
    Pre-pandemic weekly median (2014-2019). Corrects for COVID-era level
    distortion. Used as an independent predictor, not a reference line.

  Model 3 — ETS(A,A,A)                      weight = 1/3
    Role: adaptive seasonal updating.
    Additive Holt-Winters (additive error, additive trend, additive
    seasonality) on log-transformed series. ETS updates seasonal
    components incrementally; SARIMA does not. Structurally complementary.

Equal weights are used. Defensible for narrative review contexts where
model complexity differences cannot be justified to non-technical audiences.

Post-hoc strain multiplier (deterministic scalar, applied after averaging):
  H3N2-dominant:       ×1.65
  H1N1pdm09-dominant:  ×0.85
  Influenza B-dominant: ×0.60
Derived from historical ILI+ ratios in FluNet data.

EXCLUDED: Prophet (R²=0.02; biologically implausible trend extrapolation).

Inputs:
  ../data/processed/processed_master_dataset.csv   — full historical series
  ../data/processed/processed_model_ready.csv      — log-transformed series
  ../outputs/forecasts/forecast_78wk_2026_27.csv   — pre-fitted SARIMA forecast

Outputs:
  ../outputs/forecasts/ensemble_forecast_78wk.csv         — ensemble (no strain adj.)
  ../outputs/forecasts/ensemble_strain_adjusted_78wk.csv  — ensemble + strain multiplier
  ../outputs/metrics/ensemble_model_weights.csv           — weight table
  ../outputs/metrics/ets_fitted_values.csv                — ETS in-sample fit
  ../figures/forecasts/fig_ensemble_forecast.png          — ensemble vs components
  ../figures/forecasts/fig_ensemble_components.png        — individual model tracks
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.optimize import minimize

# ── Paths ───────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent.parent
DATA_DIR    = BASE / "data" / "processed"
OUTPUT_DIR  = BASE / "outputs"
FIG_DIR     = BASE / "figures" / "forecasts"
FIG_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "forecasts").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "metrics").mkdir(parents=True, exist_ok=True)

# ── Configuration ────────────────────────────────────────────────────────────
SEASONAL_PERIOD = 52       # Weekly data, annual seasonality
FORECAST_HORIZON = 78      # 78 weeks ≈ 18 months

# Strain multiplier table (post-hoc scalar applied to ensemble point forecast)
# Source: historical ILI+ ratios from FluNet HK data
STRAIN_MULTIPLIERS = {
    "H3N2":      1.65,
    "H1N1pdm09": 0.85,
    "B":         0.60,
}

# ── Default dominant strain ───────────────────────────────────────────────────
# Derive from the most recent weeks of FluNet data in master dataset.
# Can be overridden by setting DOMINANT_STRAIN directly below.
DOMINANT_STRAIN = None   # Set to "H3N2", "H1N1pdm09", or "B" to override

FLOOR = 1e-6   # Log-transform floor


# ═══════════════════════════════════════════════════════════════════════════
#  ETS (A,A,A) — Additive Holt-Winters
#  Implemented from scratch using scipy.optimize (no statsmodels dependency)
# ═══════════════════════════════════════════════════════════════════════════

def ets_fit(y, m=52):
    """
    Fit ETS(A,A,A) (additive error, additive trend, additive seasonality).

    Parameters
    ----------
    y : 1-D array, training observations (already log-transformed)
    m : int, seasonal period (52 for weekly data)

    Returns
    -------
    dict with keys:
      alpha, beta, gamma  — smoothing parameters
      l, b                — final level and trend
      s                   — array of m seasonal factors
      fitted              — in-sample fitted values
      sse                 — sum of squared errors
    """
    y = np.asarray(y, dtype=float)
    n = len(y)

    # ── Initialisation (classic Holt-Winters decomposition) ──────────────
    # Level: average of first season
    l0 = np.mean(y[:m])
    # Trend: average slope between first and second season averages
    if n >= 2 * m:
        b0 = (np.mean(y[m:2*m]) - np.mean(y[:m])) / m
    else:
        b0 = 0.0
    # Seasonal: first-season deviations from level
    s0 = y[:m] - l0

    def _sse(params):
        alpha, beta, gamma = params
        l, b = l0, b0
        s = s0.copy()
        sse = 0.0
        for t in range(n):
            # One-step-ahead forecast
            yhat = l + b + s[t % m]
            err  = y[t] - yhat
            sse += err * err
            # State updates (additive error form)
            l_new = l + b + alpha * err
            b_new = b + alpha * beta * err
            s[t % m] = s[t % m] + gamma * (1 - alpha) * err
            l, b = l_new, b_new
        return sse

    # Optimise smoothing parameters in (0, 1)
    best_result = None
    best_sse    = np.inf
    for a0 in [0.1, 0.3, 0.5]:
        for b0_ in [0.01, 0.05, 0.1]:
            for g0 in [0.1, 0.3, 0.5]:
                res = minimize(
                    _sse,
                    x0=[a0, b0_, g0],
                    method="L-BFGS-B",
                    bounds=[(1e-4, 0.999)] * 3,
                    options={"maxiter": 500, "ftol": 1e-10},
                )
                if res.fun < best_sse:
                    best_sse    = res.fun
                    best_result = res

    alpha, beta, gamma = best_result.x

    # Recover fitted values and final states
    l, b = l0, b0
    s    = s0.copy()
    fitted = np.zeros(n)
    for t in range(n):
        fitted[t] = l + b + s[t % m]
        err = y[t] - fitted[t]
        l_new = l + b + alpha * err
        b_new = b + alpha * beta * err
        s[t % m] = s[t % m] + gamma * (1 - alpha) * err
        l, b = l_new, b_new

    return {
        "alpha": alpha, "beta": beta, "gamma": gamma,
        "l": l, "b": b, "s": s.copy(),
        "fitted": fitted, "sse": best_sse,
        "n_train": n, "m": m,
    }


def ets_forecast(model, h):
    """
    Generate h-step-ahead point forecasts from a fitted ETS(A,A,A) model.

    Parameters
    ----------
    model : dict from ets_fit()
    h     : int, forecast horizon

    Returns
    -------
    np.ndarray of length h (log scale)
    """
    l, b = model["l"], model["b"]
    s    = model["s"].copy()
    m    = model["m"]
    fcast = np.zeros(h)
    for i in range(1, h + 1):
        s_idx = (model["n_train"] + i - 1) % m
        fcast[i - 1] = l + i * b + s[s_idx]
    return fcast


# ═══════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("SCRIPT 07 — Three-Model Ensemble Forecast")
print("=" * 60)

# Full master dataset (2014-2026, ~630 weeks)
print("\nLoading data...")
master = pd.read_csv(
    DATA_DIR / "processed_master_dataset.csv",
    parse_dates=["date"],
)
master = master.sort_values("date").reset_index(drop=True)

# Model-ready series (log-transformed, COVID-imputed)
model_ready = pd.read_csv(
    DATA_DIR / "processed_model_ready.csv",
    parse_dates=["date"],
)
model_ready = model_ready.sort_values("date").reset_index(drop=True)

# Pre-fitted SARIMA forecast
sarima_fc = pd.read_csv(
    OUTPUT_DIR / "forecasts" / "forecast_78wk_2026_27.csv",
    parse_dates=["date"],
)
sarima_fc = sarima_fc.sort_values("date").reset_index(drop=True)

print(f"  Master dataset:    {len(master)} weeks  "
      f"({master['date'].min().date()} to {master['date'].max().date()})")
print(f"  Model-ready series:{len(model_ready)} weeks  "
      f"({model_ready['date'].min().date()} to {model_ready['date'].max().date()})")
print(f"  SARIMA forecast:   {len(sarima_fc)} weeks  "
      f"({sarima_fc['date'].min().date()} to {sarima_fc['date'].max().date()})")


# ═══════════════════════════════════════════════════════════════════════════
#  AUTO-DETECT DOMINANT STRAIN from most recent 8 weeks of FluNet data
# ═══════════════════════════════════════════════════════════════════════════

if DOMINANT_STRAIN is None:
    recent = master.dropna(subset=["dominant_strain"]).tail(8)
    if len(recent) > 0:
        strain_counts = recent["dominant_strain"].value_counts()
        top_strain = strain_counts.index[0]
        # Map to multiplier key
        if "H3" in top_strain:
            DOMINANT_STRAIN = "H3N2"
        elif "H1" in top_strain or "pdm" in top_strain.lower():
            DOMINANT_STRAIN = "H1N1pdm09"
        elif "B" in top_strain or "Influenza_B" in top_strain:
            DOMINANT_STRAIN = "B"
        else:
            DOMINANT_STRAIN = "H3N2"   # conservative default
        print(f"\n  Auto-detected dominant strain: {top_strain} → multiplier key: {DOMINANT_STRAIN}")
    else:
        DOMINANT_STRAIN = "H3N2"
        print(f"\n  Dominant strain: defaulting to H3N2 (no recent FluNet data)")
else:
    print(f"\n  Dominant strain: {DOMINANT_STRAIN} (manually set)")

STRAIN_MULT = STRAIN_MULTIPLIERS[DOMINANT_STRAIN]
print(f"  Strain multiplier: ×{STRAIN_MULT}")


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL 1 — SARIMA (load pre-fitted forecast)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "-" * 40)
print("Model 1: SARIMA — loading pre-fitted forecast")

# The SARIMA forecast is on the original scale
# Convert to log scale for ensemble combination
sarima_yhat      = sarima_fc["forecast"].values
sarima_yhat_log  = np.log(np.clip(sarima_yhat, FLOOR, None))
forecast_dates   = sarima_fc["date"].values

print(f"  Loaded {len(sarima_yhat)} forecast weeks from outputs/forecasts/forecast_78wk_2026_27.csv")
print(f"  Forecast range (original scale): "
      f"{sarima_yhat.min():.6f} – {sarima_yhat.max():.6f}")


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL 2 — Historical Percentile Benchmark (2014-2019 weekly median)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "-" * 40)
print("Model 2: Historical Percentile Benchmark (pre-pandemic 2014-2019 median)")

# Use ILI+ values from 2014-2019 (pre-pandemic baseline period)
# ili_plus_model is the CHP-sourced, pre-imputed series
prepandemic = master[
    (master["date"].dt.year >= 2014) &
    (master["date"].dt.year <= 2019) &
    (master["ili_plus_model"].notna())
].copy()

# Compute median ILI+ by ISO week number (1-52)
prepandemic["iso_week"] = prepandemic["date"].dt.isocalendar().week.astype(int)
weekly_median = (
    prepandemic.groupby("iso_week")["ili_plus_model"]
    .median()
    .reindex(range(1, 53))
    .interpolate(method="linear")
    .fillna(method="bfill")
    .fillna(method="ffill")
)

print(f"  Pre-pandemic weeks: {len(prepandemic)}  "
      f"({prepandemic['date'].min().date()} to {prepandemic['date'].max().date()})")
print(f"  Weekly median range: {weekly_median.min():.6f} – {weekly_median.max():.6f}")

# Project benchmark onto forecast dates by matching ISO week
fc_weeks = pd.to_datetime(forecast_dates)
benchmark_yhat = np.array([
    weekly_median.get(int(d.isocalendar()[1]), weekly_median.median())
    for d in fc_weeks
])
benchmark_yhat_log = np.log(np.clip(benchmark_yhat, FLOOR, None))

print(f"  Benchmark projected over {len(benchmark_yhat)} forecast weeks")


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL 3 — ETS(A,A,A) on log-transformed series
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "-" * 40)
print("Model 3: ETS(A,A,A) — fitting on log-transformed series")

y_log  = model_ready["y_log"].values
y_dates = model_ready["date"].values

print(f"  Training series: {len(y_log)} weeks  "
      f"({model_ready['date'].min().date()} to {model_ready['date'].max().date()})")
print(f"  Fitting ETS(A,A,A) with m={SEASONAL_PERIOD}  (this may take ~60 sec)...")

ets_model = ets_fit(y_log, m=SEASONAL_PERIOD)

print(f"  Fitted parameters:")
print(f"    α (level)      = {ets_model['alpha']:.4f}")
print(f"    β (trend)      = {ets_model['beta']:.4f}")
print(f"    γ (seasonality)= {ets_model['gamma']:.4f}")
print(f"    SSE            = {ets_model['sse']:.6f}")

ets_yhat_log = ets_forecast(ets_model, h=FORECAST_HORIZON)
ets_yhat     = np.exp(ets_yhat_log)

# Save ETS in-sample fitted values
ets_fitted_df = pd.DataFrame({
    "date":       y_dates,
    "y_log_actual": y_log,
    "ets_fitted_log": ets_model["fitted"],
    "ets_fitted":    np.exp(ets_model["fitted"]),
})
ets_fitted_df.to_csv(OUTPUT_DIR / "metrics" / "ets_fitted_values.csv", index=False)
print(f"  ETS in-sample fit saved → outputs/metrics/ets_fitted_values.csv")


# ═══════════════════════════════════════════════════════════════════════════
#  ENSEMBLE COMBINATION (equal weights, log scale)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "-" * 40)
print("Combining models (equal weights = 1/3 each, on log scale)")

w = 1.0 / 3.0

ensemble_log = (
    w * sarima_yhat_log +
    w * benchmark_yhat_log +
    w * ets_yhat_log
)
ensemble_yhat = np.exp(ensemble_log)

# ── Prediction interval ───────────────────────────────────────────────────
# Propagate SARIMA PI by scaling proportionally to ensemble/SARIMA ratio.
# This preserves the calibrated bootstrap uncertainty from script 05 while
# re-centring on the ensemble mean.
# On log scale: pi_width_log = (log(sarima_hi95) - log(sarima_lo95)) / 2

sarima_lo95 = np.clip(sarima_fc["lo95"].values, FLOOR, None)
sarima_hi95 = np.clip(sarima_fc["hi95"].values, FLOOR, None)
sarima_lo80 = np.clip(sarima_fc["lo80"].values, FLOOR, None)
sarima_hi80 = np.clip(sarima_fc["hi80"].values, FLOOR, None)

half_width_95 = (np.log(sarima_hi95) - np.log(sarima_lo95)) / 2.0
half_width_80 = (np.log(sarima_hi80) - np.log(sarima_lo80)) / 2.0

ensemble_lo95 = np.exp(ensemble_log - half_width_95)
ensemble_hi95 = np.exp(ensemble_log + half_width_95)
ensemble_lo80 = np.exp(ensemble_log - half_width_80)
ensemble_hi80 = np.exp(ensemble_log + half_width_80)

print(f"  Ensemble point forecast range: "
      f"{ensemble_yhat.min():.6f} – {ensemble_yhat.max():.6f}")


# ═══════════════════════════════════════════════════════════════════════════
#  STRAIN MULTIPLIER (post-hoc, applied to point forecast only)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "-" * 40)
print(f"Applying strain multiplier: {DOMINANT_STRAIN} × {STRAIN_MULT}")

ensemble_strain_adj     = ensemble_yhat     * STRAIN_MULT
ensemble_lo95_strain    = ensemble_lo95     * STRAIN_MULT
ensemble_hi95_strain    = ensemble_hi95     * STRAIN_MULT
ensemble_lo80_strain    = ensemble_lo80     * STRAIN_MULT
ensemble_hi80_strain    = ensemble_hi80     * STRAIN_MULT

print(f"  Strain-adjusted forecast range: "
      f"{ensemble_strain_adj.min():.6f} – {ensemble_strain_adj.max():.6f}")


# ═══════════════════════════════════════════════════════════════════════════
#  SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "-" * 40)
print("Saving outputs...")

# ── Ensemble forecast (no strain adjustment) ──────────────────────────────
ensemble_df = pd.DataFrame({
    "date":          forecast_dates,
    "week":          range(1, FORECAST_HORIZON + 1),
    "sarima_yhat":   sarima_yhat,
    "benchmark_yhat":benchmark_yhat,
    "ets_yhat":      ets_yhat,
    "ensemble_yhat": ensemble_yhat,
    "lo80":          ensemble_lo80,
    "hi80":          ensemble_hi80,
    "lo95":          ensemble_lo95,
    "hi95":          ensemble_hi95,
})
ensemble_df.to_csv(
    OUTPUT_DIR / "forecasts" / "ensemble_forecast_78wk.csv", index=False
)
print(f"  Saved: outputs/forecasts/ensemble_forecast_78wk.csv")

# ── Strain-adjusted ensemble forecast ─────────────────────────────────────
strain_df = pd.DataFrame({
    "date":                  forecast_dates,
    "week":                  range(1, FORECAST_HORIZON + 1),
    "ensemble_yhat":         ensemble_yhat,
    "ensemble_yhat_adjusted":ensemble_strain_adj,
    "dominant_strain":       DOMINANT_STRAIN,
    "strain_multiplier":     STRAIN_MULT,
    "lo80":                  ensemble_lo80_strain,
    "hi80":                  ensemble_hi80_strain,
    "lo95":                  ensemble_lo95_strain,
    "hi95":                  ensemble_hi95_strain,
})
strain_df.to_csv(
    OUTPUT_DIR / "forecasts" / "ensemble_strain_adjusted_78wk.csv", index=False
)
print(f"  Saved: outputs/forecasts/ensemble_strain_adjusted_78wk.csv")

# ── Model weight table ─────────────────────────────────────────────────────
weights_df = pd.DataFrame({
    "model":       ["SARIMA(1,0,1)(1,1,0)[52]",
                    "Historical Percentile Benchmark (2014-2019)",
                    "ETS(A,A,A)"],
    "role":        ["Seasonal timing anchor",
                    "Magnitude anchor (pre-pandemic median)",
                    "Adaptive seasonal updating"],
    "weight":      [w, w, w],
    "notes":       ["Pre-fitted; loaded from outputs/forecasts/forecast_78wk_2026_27.csv",
                    f"Weekly median of ILI+ 2014-2019 ({len(prepandemic)} obs)",
                    f"α={ets_model['alpha']:.4f}, β={ets_model['beta']:.4f}, γ={ets_model['gamma']:.4f}"],
})
weights_df.to_csv(
    OUTPUT_DIR / "metrics" / "ensemble_model_weights.csv", index=False
)
print(f"  Saved: outputs/metrics/ensemble_model_weights.csv")
print(f"\n  Strain multiplier: {DOMINANT_STRAIN} (×{STRAIN_MULT})")


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURES
# ═══════════════════════════════════════════════════════════════════════════

print("\nGenerating figures...")

# Historical observed series for context (post-COVID only)
hist = master[
    (master["date"] >= "2022-01-01") &
    (master["ili_plus_model"].notna())
][["date", "ili_plus_model"]].copy()

fc_dates_dt = pd.to_datetime(forecast_dates)

# ── Figure 07a: Ensemble forecast with PI bands ───────────────────────────
fig, ax = plt.subplots(figsize=(16, 6))

ax.fill_between(fc_dates_dt,
                strain_df["lo95"], strain_df["hi95"],
                alpha=0.18, color="#4dac26", label="95% PI (strain-adjusted)")
ax.fill_between(fc_dates_dt,
                strain_df["lo80"], strain_df["hi80"],
                alpha=0.32, color="#4dac26", label="80% PI (strain-adjusted)")

ax.plot(fc_dates_dt, ensemble_strain_adj,
        color="#1a1a1a", lw=2.2, label=f"Ensemble + strain adj. (×{STRAIN_MULT}, {DOMINANT_STRAIN})")
ax.plot(fc_dates_dt, ensemble_yhat,
        color="#4dac26", lw=1.5, linestyle="--", alpha=0.7, label="Ensemble (no strain adj.)")

ax.plot(hist["date"], hist["ili_plus_model"],
        color="#636363", lw=1.3, alpha=0.9, label="Observed ILI+ (post-COVID)")

ax.axvline(fc_dates_dt[0], color="#bdbdbd", lw=1.0, linestyle=":", alpha=0.8)
ax.text(fc_dates_dt[0], ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.03,
        " Forecast start", fontsize=8, color="#636363", va="top")

ax.set_xlabel("Date", fontsize=10)
ax.set_ylabel("ILI+ rate", fontsize=10)
ax.set_title(
    f"Script 07 — Three-Model Ensemble Forecast with Strain Adjustment\n"
    f"SARIMA (1/3) + Benchmark (1/3) + ETS(A,A,A) (1/3)  ×  {DOMINANT_STRAIN} multiplier ×{STRAIN_MULT}  |  "
    f"Horizon: {FORECAST_HORIZON} weeks",
    fontsize=10,
)
ax.legend(fontsize=8, loc="upper left")
ax.set_ylim(bottom=0)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig_ensemble_forecast.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/forecasts/fig_ensemble_forecast.png")

# ── Figure 07b: Individual component tracks ───────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
fig.suptitle(
    "Script 07 — Ensemble Components: Individual Model Forecasts",
    fontsize=11, y=0.98,
)

component_specs = [
    (sarima_yhat,      "#2c7bb6", "Model 1: SARIMA(1,0,1)(1,1,0)[52]",
     "Seasonal timing anchor | pre-fitted"),
    (benchmark_yhat,   "#d7191c", "Model 2: Historical Percentile Benchmark",
     "Pre-pandemic 2014-2019 weekly median | magnitude anchor"),
    (ets_yhat,         "#1a9641", f"Model 3: ETS(A,A,A)",
     f"α={ets_model['alpha']:.3f}, β={ets_model['beta']:.3f}, γ={ets_model['gamma']:.3f} | adaptive seasonality"),
]

for ax, (comp_yhat, colour, title, subtitle) in zip(axes, component_specs):
    ax.plot(hist["date"], hist["ili_plus_model"],
            color="#bdbdbd", lw=1.0, alpha=0.8, label="Observed (post-COVID)")
    ax.plot(fc_dates_dt, comp_yhat,
            color=colour, lw=2.0, label="Model forecast")
    ax.plot(fc_dates_dt, ensemble_yhat,
            color="black", lw=1.0, linestyle=":", alpha=0.5, label="Ensemble mean")
    ax.set_title(f"{title}\n{subtitle}", fontsize=9, loc="left")
    ax.set_ylabel("ILI+ rate", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(bottom=0)
    ax.axvline(fc_dates_dt[0], color="#bdbdbd", lw=0.8, linestyle=":")

axes[-1].set_xlabel("Date", fontsize=10)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig_ensemble_components.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/forecasts/fig_ensemble_components.png")


# ═══════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SCRIPT 07 COMPLETE — Three-Model Ensemble")
print("=" * 60)
print(f"  Ensemble components  : SARIMA + Benchmark + ETS(A,A,A)")
print(f"  Weights              : 1/3 each (equal)")
print(f"  Dominant strain      : {DOMINANT_STRAIN}")
print(f"  Strain multiplier    : ×{STRAIN_MULT}")
print(f"  Forecast horizon     : {FORECAST_HORIZON} weeks")
print(f"  Outputs:")
print(f"    outputs/forecasts/ensemble_forecast_78wk.csv")
print(f"    outputs/forecasts/ensemble_strain_adjusted_78wk.csv")
print(f"    outputs/metrics/ensemble_model_weights.csv")
print(f"    outputs/metrics/ets_fitted_values.csv")
print(f"    figures/forecasts/fig_ensemble_forecast.png")
print(f"    figures/forecasts/fig_ensemble_components.png")
print("=" * 60)
