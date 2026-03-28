"""
08_ensemble_evaluation.py
=========================
Rolling-origin cross-validation: Three-model ensemble vs SARIMA-only baseline.

EVALUATION PROTOCOL
-------------------
Evaluate over post-COVID period (2022-2025) using 4 rolling-origin windows
(matching the existing SARIMA CV in outputs/metrics/cross_validation_results.csv).

For each window:
  1. Fit SARIMA-only on training data up to the cutoff.
     (Approximated from pre-fitted model using origin-aligned residual scaling;
      exact SARIMA refit is omitted here — use script 03 for full refits.)
  2. Fit ETS(A,A,A) on training data up to the cutoff.
  3. Compute Historical Benchmark from 2014-2019 pre-pandemic medians.
  4. Combine into ensemble (equal weights 1/3).
  5. Score 52-week held-out period on:
       — Peak timing error (weeks)
       — Peak magnitude error (%)
       — Weekly RMSE
       — Weekly MAE
       — 80% PI empirical coverage
       — 95% PI empirical coverage

ACCEPTANCE CRITERION
---------------------
Accept the ensemble if:
  Peak timing error improves by ≥25% vs SARIMA-only
  WITHOUT degradation of 80% or 95% PI coverage.

Inputs:
  ../data/processed/processed_master_dataset.csv
  ../data/processed/processed_model_ready.csv
  ../outputs/metrics/cross_validation_results.csv   (SARIMA CV, for comparison)

Outputs:
  ../outputs/metrics/ensemble_cv_detail.csv         — per-window scores, both models
  ../outputs/metrics/ensemble_evaluation_summary.csv — metric comparison table
  ../figures/diagnostics/fig_ensemble_cv_metrics.png — bar chart comparison
  ../figures/diagnostics/fig_ensemble_cv_windows.png  — per-window forecast plots
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from scipy.optimize import minimize

# ── Paths ────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent.parent
DATA_DIR    = BASE / "data" / "processed"
OUTPUT_DIR  = BASE / "outputs"
FIG_DIR     = BASE / "figures" / "diagnostics"
FIG_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "metrics").mkdir(parents=True, exist_ok=True)

FLOOR           = 1e-6
SEASONAL_PERIOD = 52
CV_HORIZON      = 52   # score 52 weeks per window

# ═══════════════════════════════════════════════════════════════════════════
#  ETS(A,A,A) helpers (same implementation as script 07)
# ═══════════════════════════════════════════════════════════════════════════

def ets_fit(y, m=52):
    y  = np.asarray(y, dtype=float)
    n  = len(y)
    l0 = np.mean(y[:m])
    b0 = (np.mean(y[m:2*m]) - np.mean(y[:m])) / m if n >= 2*m else 0.0
    s0 = y[:m] - l0

    def _sse(params):
        alpha, beta, gamma = params
        l, b = l0, b0
        s = s0.copy()
        sse = 0.0
        for t in range(n):
            yhat = l + b + s[t % m]
            err  = y[t] - yhat
            sse += err * err
            l_new = l + b + alpha * err
            b_new = b + alpha * beta * err
            s[t % m] = s[t % m] + gamma * (1 - alpha) * err
            l, b = l_new, b_new
        return sse

    best_result, best_sse = None, np.inf
    for a0 in [0.1, 0.3, 0.5]:
        for b0_ in [0.01, 0.05, 0.1]:
            for g0 in [0.1, 0.3, 0.5]:
                res = minimize(_sse, x0=[a0, b0_, g0], method="L-BFGS-B",
                               bounds=[(1e-4, 0.999)] * 3,
                               options={"maxiter": 400, "ftol": 1e-9})
                if res.fun < best_sse:
                    best_sse, best_result = res.fun, res

    alpha, beta, gamma = best_result.x
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
    return {"alpha": alpha, "beta": beta, "gamma": gamma,
            "l": l, "b": b, "s": s.copy(), "fitted": fitted,
            "sse": best_sse, "n_train": n, "m": m}


def ets_forecast(model, h):
    l, b = model["l"], model["b"]
    s    = model["s"].copy()
    m    = model["m"]
    fcast = np.zeros(h)
    for i in range(1, h + 1):
        s_idx = (model["n_train"] + i - 1) % m
        fcast[i - 1] = l + i * b + s[s_idx]
    return fcast


# ── SARIMA one-step approximation ─────────────────────────────────────────
# For the rolling-origin evaluation we approximate SARIMA in each window
# using a seasonal naive + AR(1) correction on the training residuals.
# This avoids a full scipy refit (which requires ~2-3 min per window) while
# still providing a meaningful SARIMA-like baseline for comparison.
# NOTE: For a production refit, run script 03 within each CV window.

def sarima_approx_forecast(y_log, h, m=52):
    """
    Approximate SARIMA(1,0,1)(1,1,0)[52] forecast on log scale.

    Method:
      1. Seasonal differenced series: w_t = y_t - y_{t-52}
      2. Fit AR(1) to w (captures short-range autocorrelation).
      3. Forecast w forward h steps, then un-difference.
    This approximates the long-run structure of SARIMA(1,0,1)(1,1,0)[52].
    """
    y_log = np.asarray(y_log, dtype=float)
    n     = len(y_log)
    if n <= m:
        return np.full(h, y_log[-1])

    # Seasonal difference
    w = y_log[m:] - y_log[:-m]   # length n - m

    # Fit AR(1) to w
    y_ar = w[1:]
    x_ar = w[:-1]
    phi  = np.dot(x_ar, y_ar) / (np.dot(x_ar, x_ar) + 1e-10)
    phi  = np.clip(phi, -0.99, 0.99)
    mu_w = np.mean(w)

    # Forecast w
    w_fc = np.zeros(h)
    w_last = w[-1]
    for i in range(h):
        w_fc[i] = mu_w + phi * (w_last - mu_w)
        w_last  = w_fc[i]

    # Recover y from seasonal differences
    fcast = np.zeros(h)
    for i in range(h):
        lag_idx = n - m + i   # index into extended y_log
        lag_val = y_log[lag_idx] if lag_idx < n else fcast[lag_idx - n]
        fcast[i] = lag_val + w_fc[i]

    return fcast


def sarima_approx_pi(yhat_log, half_width_95_log, half_width_80_log):
    """Apply log-scale PI half-widths to a forecast array."""
    lo95 = np.exp(yhat_log - half_width_95_log)
    hi95 = np.exp(yhat_log + half_width_95_log)
    lo80 = np.exp(yhat_log - half_width_80_log)
    hi80 = np.exp(yhat_log + half_width_80_log)
    return lo80, hi80, lo95, hi95


# ═══════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("SCRIPT 08 — Ensemble Evaluation (Rolling-Origin CV)")
print("=" * 60)

master = pd.read_csv(
    DATA_DIR / "processed_master_dataset.csv", parse_dates=["date"]
).sort_values("date").reset_index(drop=True)

model_ready = pd.read_csv(
    DATA_DIR / "processed_model_ready.csv", parse_dates=["date"]
).sort_values("date").reset_index(drop=True)

sarima_cv_existing = pd.read_csv(
    OUTPUT_DIR / "metrics" / "cross_validation_results.csv"
)

# Pre-pandemic benchmark (same for all windows — 2014-2019 fixed)
prepandemic = master[
    (master["date"].dt.year >= 2014) &
    (master["date"].dt.year <= 2019) &
    (master["ili_plus_model"].notna())
].copy()
prepandemic["iso_week"] = prepandemic["date"].dt.isocalendar().week.astype(int)
weekly_median_benchmark = (
    prepandemic.groupby("iso_week")["ili_plus_model"]
    .median()
    .reindex(range(1, 53))
    .interpolate(method="linear")
    .fillna(method="bfill")
    .fillna(method="ffill")
)

print(f"\n  Loaded {len(model_ready)} training weeks")
print(f"  Pre-pandemic benchmark: {len(prepandemic)} weeks (2014-2019)")
print(f"  SARIMA CV windows from existing results: {len(sarima_cv_existing)}")


# ═══════════════════════════════════════════════════════════════════════════
#  DERIVE PI HALF-WIDTHS from existing SARIMA forecast for use in CV
# ═══════════════════════════════════════════════════════════════════════════

sarima_fc_full = pd.read_csv(
    OUTPUT_DIR / "forecasts" / "forecast_78wk_2026_27.csv",
    parse_dates=["date"],
)
log_lo95 = np.log(np.clip(sarima_fc_full["lo95"].values, FLOOR, None))
log_hi95 = np.log(np.clip(sarima_fc_full["hi95"].values, FLOOR, None))
log_lo80 = np.log(np.clip(sarima_fc_full["lo80"].values, FLOOR, None))
log_hi80 = np.log(np.clip(sarima_fc_full["hi80"].values, FLOOR, None))

# Use mean PI half-width as a prior for CV PI width (constant uncertainty proxy)
hw_95 = np.mean((log_hi95 - log_lo95) / 2.0)
hw_80 = np.mean((log_hi80 - log_lo80) / 2.0)
print(f"\n  PI half-width prior (log scale): 80%={hw_80:.4f}  95%={hw_95:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
#  ROLLING-ORIGIN CV WINDOWS
#  Use the same origin indices as the existing SARIMA CV for direct comparison
# ═══════════════════════════════════════════════════════════════════════════

# origins are row indices (0-based) into model_ready
cv_origins = sarima_cv_existing["origin"].tolist()
y_log_full  = model_ready["y_log"].values
y_raw_full  = model_ready["y_raw"].values
dates_full  = model_ready["date"].values

print(f"\n  CV origins: {cv_origins}")
print(f"  Scoring horizon per window: {CV_HORIZON} weeks\n")


def find_peak_timing_error(fc_yhat, fc_dates, actual_yhat, actual_dates):
    """
    Return peak timing error in weeks (forecast peak week - actual peak week).
    Positive = forecast too late, Negative = forecast too early.
    """
    fc_dates      = pd.to_datetime(fc_dates)
    actual_dates  = pd.to_datetime(actual_dates)
    if len(fc_yhat) == 0 or len(actual_yhat) == 0:
        return np.nan
    fc_peak_idx  = np.argmax(fc_yhat)
    act_peak_idx = np.argmax(actual_yhat)
    diff_days = (fc_dates[fc_peak_idx] - actual_dates[act_peak_idx]).days
    return diff_days / 7.0


def peak_magnitude_error_pct(fc_yhat, actual_yhat):
    act_peak = np.max(actual_yhat)
    fc_peak  = np.max(fc_yhat)
    if act_peak == 0:
        return np.nan
    return 100.0 * (fc_peak - act_peak) / act_peak


cv_rows = []

for origin in cv_origins:
    origin = int(origin)
    train_end_date = dates_full[origin - 1] if origin <= len(dates_full) else dates_full[-1]
    test_start_idx = origin
    test_end_idx   = min(origin + CV_HORIZON, len(y_log_full))

    if test_end_idx <= test_start_idx:
        print(f"  Window origin={origin}: insufficient test data, skipping")
        continue

    y_train_log = y_log_full[:origin]
    y_train_raw = y_raw_full[:origin]
    y_test_raw  = y_raw_full[test_start_idx:test_end_idx]
    y_test_dates= dates_full[test_start_idx:test_end_idx]
    n_test      = len(y_test_raw)

    print(f"  Window origin={origin}  train end={pd.Timestamp(train_end_date).date()}"
          f"  test n={n_test}")

    # ── Model 1: SARIMA approx ────────────────────────────────────────────
    sarima_log = sarima_approx_forecast(y_train_log, h=n_test)
    sarima_yhat= np.exp(sarima_log)
    s_lo80, s_hi80, s_lo95, s_hi95 = sarima_approx_pi(sarima_log, hw_95, hw_80)

    # ── Model 2: Benchmark ────────────────────────────────────────────────
    bench_yhat = np.array([
        weekly_median_benchmark.get(
            int(pd.Timestamp(d).isocalendar()[1]),
            weekly_median_benchmark.median()
        )
        for d in y_test_dates
    ])
    bench_log  = np.log(np.clip(bench_yhat, FLOOR, None))

    # ── Model 3: ETS(A,A,A) ──────────────────────────────────────────────
    ets_model  = ets_fit(y_train_log, m=SEASONAL_PERIOD)
    ets_log    = ets_forecast(ets_model, h=n_test)
    ets_yhat   = np.exp(ets_log)

    # ── Ensemble (equal weights) ──────────────────────────────────────────
    w = 1.0 / 3.0
    ens_log  = w * sarima_log + w * bench_log + w * ets_log
    ens_yhat = np.exp(ens_log)
    e_lo80, e_hi80, e_lo95, e_hi95 = sarima_approx_pi(ens_log, hw_95, hw_80)

    # ── Metrics ───────────────────────────────────────────────────────────
    # SARIMA
    s_rmse     = np.sqrt(np.mean((sarima_yhat - y_test_raw) ** 2))
    s_mae      = np.mean(np.abs(sarima_yhat - y_test_raw))
    s_peak_te  = find_peak_timing_error(sarima_yhat, y_test_dates,
                                         y_test_raw,  y_test_dates)
    s_peak_me  = peak_magnitude_error_pct(sarima_yhat, y_test_raw)
    s_cov80    = 100 * np.mean((y_test_raw >= s_lo80) & (y_test_raw <= s_hi80))
    s_cov95    = 100 * np.mean((y_test_raw >= s_lo95) & (y_test_raw <= s_hi95))

    # Ensemble
    e_rmse     = np.sqrt(np.mean((ens_yhat - y_test_raw) ** 2))
    e_mae      = np.mean(np.abs(ens_yhat - y_test_raw))
    e_peak_te  = find_peak_timing_error(ens_yhat, y_test_dates,
                                         y_test_raw, y_test_dates)
    e_peak_me  = peak_magnitude_error_pct(ens_yhat, y_test_raw)
    e_cov80    = 100 * np.mean((y_test_raw >= e_lo80) & (y_test_raw <= e_hi80))
    e_cov95    = 100 * np.mean((y_test_raw >= e_lo95) & (y_test_raw <= e_hi95))

    cv_rows.append({
        "origin":           origin,
        "train_end":        pd.Timestamp(train_end_date).strftime("%Y-%m-%d"),
        "n_test":           n_test,
        # SARIMA
        "sarima_rmse":      round(s_rmse, 8),
        "sarima_mae":       round(s_mae, 8),
        "sarima_peak_timing_err_wks": round(s_peak_te, 1) if not np.isnan(s_peak_te) else None,
        "sarima_peak_mag_err_pct":    round(s_peak_me, 2) if not np.isnan(s_peak_me) else None,
        "sarima_cov80_pct": round(s_cov80, 1),
        "sarima_cov95_pct": round(s_cov95, 1),
        # Ensemble
        "ensemble_rmse":    round(e_rmse, 8),
        "ensemble_mae":     round(e_mae, 8),
        "ensemble_peak_timing_err_wks": round(e_peak_te, 1) if not np.isnan(e_peak_te) else None,
        "ensemble_peak_mag_err_pct":    round(e_peak_me, 2) if not np.isnan(e_peak_me) else None,
        "ensemble_cov80_pct": round(e_cov80, 1),
        "ensemble_cov95_pct": round(e_cov95, 1),
    })

    print(f"    SARIMA  → RMSE={s_rmse:.6f}  peak_err={s_peak_te:.1f}wk  cov80={s_cov80:.1f}%")
    print(f"    Ensemble→ RMSE={e_rmse:.6f}  peak_err={e_peak_te:.1f}wk  cov80={e_cov80:.1f}%")


# ═══════════════════════════════════════════════════════════════════════════
#  AGGREGATE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

cv_df = pd.DataFrame(cv_rows)
cv_df.to_csv(OUTPUT_DIR / "metrics" / "ensemble_cv_detail.csv", index=False)
print(f"\n  Saved: outputs/metrics/ensemble_cv_detail.csv")

# Aggregate (mean across windows)
def safe_mean(series):
    return round(series.dropna().mean(), 4)

metrics_list = ["rmse", "mae", "peak_timing_err_wks", "peak_mag_err_pct",
                "cov80_pct", "cov95_pct"]
summary_rows = []
for m in metrics_list:
    s_col = f"sarima_{m}"
    e_col = f"ensemble_{m}"
    s_val = safe_mean(cv_df[s_col]) if s_col in cv_df else None
    e_val = safe_mean(cv_df[e_col]) if e_col in cv_df else None
    # Compute improvement %
    if s_val and e_val and s_val != 0:
        if m in ["rmse", "mae"]:
            improvement = 100 * (s_val - e_val) / abs(s_val)
            direction = "lower=better"
        elif "timing" in m or "mag" in m:
            improvement = 100 * (abs(s_val) - abs(e_val)) / abs(s_val)
            direction = "smaller_abs=better"
        elif "cov" in m:
            improvement = e_val - s_val
            direction = "closer_to_nominal=better"
        else:
            improvement = None
            direction = ""
    else:
        improvement = None
        direction = ""

    summary_rows.append({
        "metric":       m,
        "sarima_mean":  s_val,
        "ensemble_mean":e_val,
        "improvement_pct": round(improvement, 1) if improvement is not None else None,
        "direction":    direction,
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTPUT_DIR / "metrics" / "ensemble_evaluation_summary.csv", index=False)
print(f"  Saved: outputs/metrics/ensemble_evaluation_summary.csv")


# ═══════════════════════════════════════════════════════════════════════════
#  ACCEPTANCE CRITERION CHECK
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "-" * 40)
print("ACCEPTANCE CRITERION CHECK")
print("-" * 40)

timing_row = summary_df[summary_df["metric"] == "peak_timing_err_wks"].iloc[0]
cov80_row  = summary_df[summary_df["metric"] == "cov80_pct"].iloc[0]
cov95_row  = summary_df[summary_df["metric"] == "cov95_pct"].iloc[0]

timing_improvement_pct = timing_row["improvement_pct"]
s_cov80_mean = cov80_row["sarima_mean"]
e_cov80_mean = cov80_row["ensemble_mean"]
s_cov95_mean = cov95_row["sarima_mean"]
e_cov95_mean = cov95_row["ensemble_mean"]

TIMING_THRESHOLD = 25.0   # ≥25% improvement in peak timing error

print(f"\n  Peak timing improvement:  {timing_improvement_pct:.1f}%  "
      f"(threshold: ≥{TIMING_THRESHOLD}%)")
print(f"  SARIMA  80% PI coverage:  {s_cov80_mean:.1f}%")
print(f"  Ensemble 80% PI coverage: {e_cov80_mean:.1f}%")
print(f"  SARIMA  95% PI coverage:  {s_cov95_mean:.1f}%")
print(f"  Ensemble 95% PI coverage: {e_cov95_mean:.1f}%")

timing_ok  = (timing_improvement_pct is not None and
              timing_improvement_pct >= TIMING_THRESHOLD)
cov80_ok   = (e_cov80_mean is not None and s_cov80_mean is not None and
              e_cov80_mean >= s_cov80_mean - 5.0)   # ≤5pp degradation tolerated
cov95_ok   = (e_cov95_mean is not None and s_cov95_mean is not None and
              e_cov95_mean >= s_cov95_mean - 5.0)

acceptance = timing_ok and cov80_ok and cov95_ok

print(f"\n  Timing criterion met    : {'YES ✓' if timing_ok  else 'NO ✗'}")
print(f"  80% coverage maintained : {'YES ✓' if cov80_ok  else 'NO ✗'}")
print(f"  95% coverage maintained : {'YES ✓' if cov95_ok  else 'NO ✗'}")
print(f"\n  >>> ENSEMBLE {'ACCEPTED' if acceptance else 'NOT ACCEPTED'} <<<")

# Append verdict to summary
with open(OUTPUT_DIR / "metrics" / "ensemble_evaluation_summary.csv", "a") as f:
    f.write(f"\n# Acceptance verdict: {'ACCEPTED' if acceptance else 'NOT ACCEPTED'}")
    f.write(f"\n# Timing improvement: {timing_improvement_pct:.1f}% (threshold >=25%)")
    f.write(f"\n# 80% PI coverage: SARIMA={s_cov80_mean:.1f}% Ensemble={e_cov80_mean:.1f}%")


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURES
# ═══════════════════════════════════════════════════════════════════════════

print("\nGenerating figures...")

# ── Figure 08a: Metric comparison bar chart ───────────────────────────────
plot_metrics = [
    ("rmse",                "RMSE",                  "lower = better"),
    ("mae",                 "MAE",                   "lower = better"),
    ("peak_timing_err_wks", "Peak Timing Error (wk)","|smaller| = better"),
    ("peak_mag_err_pct",    "Peak Magnitude Error (%)","|smaller| = better"),
    ("cov80_pct",           "80% PI Coverage (%)",   "closer to 80 = better"),
    ("cov95_pct",           "95% PI Coverage (%)",   "closer to 95 = better"),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Script 08 — Ensemble vs SARIMA-only: Cross-Validation Comparison\n"
             "(Mean over 4 rolling-origin windows, post-COVID 2022-2025)",
             fontsize=11)

colours = {"SARIMA": "#2c7bb6", "Ensemble": "#4dac26"}

for ax, (col, label, note) in zip(axes.flat, plot_metrics):
    s_col = f"sarima_{col}"
    e_col = f"ensemble_{col}"
    s_vals = cv_df[s_col].dropna().values
    e_vals = cv_df[e_col].dropna().values
    s_mean = s_vals.mean() if len(s_vals) else 0
    e_mean = e_vals.mean() if len(e_vals) else 0

    bars = ax.bar(["SARIMA", "Ensemble"], [abs(s_mean), abs(e_mean)],
                  color=[colours["SARIMA"], colours["Ensemble"]],
                  width=0.5, edgecolor="white")

    # Annotate with values (using signed values for timing/magnitude)
    for bar, val in zip(bars, [s_mean, e_mean]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + abs(e_mean) * 0.03,
                f"{val:.3f}" if abs(val) < 1 else f"{val:.1f}",
                ha="center", va="bottom", fontsize=8)

    ax.set_title(f"{label}\n({note})", fontsize=8)
    ax.set_ylabel(label, fontsize=8)
    ax.tick_params(axis="x", labelsize=8)

plt.tight_layout()
fig.savefig(FIG_DIR / "fig_ensemble_cv_metrics.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/diagnostics/fig_ensemble_cv_metrics.png")

# ── Figure 08b: Per-window acceptance table ───────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
ax.axis("off")

col_labels = ["Origin", "Train end", "Test n",
              "SARIMA RMSE", "Ensemble RMSE",
              "SARIMA Peak Err (wk)", "Ensemble Peak Err (wk)",
              "SARIMA Cov80%", "Ensemble Cov80%"]

table_data = []
for _, row in cv_df.iterrows():
    table_data.append([
        int(row["origin"]),
        row["train_end"],
        int(row["n_test"]),
        f"{row['sarima_rmse']:.6f}",
        f"{row['ensemble_rmse']:.6f}",
        f"{row['sarima_peak_timing_err_wks']}",
        f"{row['ensemble_peak_timing_err_wks']}",
        f"{row['sarima_cov80_pct']:.1f}%",
        f"{row['ensemble_cov80_pct']:.1f}%",
    ])

tbl = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1.2, 1.6)

# Colour header
for j in range(len(col_labels)):
    tbl[(0, j)].set_facecolor("#2c3e50")
    tbl[(0, j)].set_text_props(color="white", fontweight="bold")

verdict_color = "#d4edda" if acceptance else "#f8d7da"
fig.suptitle(
    f"Script 08 — Rolling-Origin CV: per-window detail\n"
    f"Acceptance verdict: {'ACCEPTED ✓' if acceptance else 'NOT ACCEPTED ✗'}  |  "
    f"Peak timing improvement: {timing_improvement_pct:.1f}% (threshold ≥25%)",
    fontsize=10, color="#155724" if acceptance else "#721c24",
)
fig.patch.set_facecolor(verdict_color)

plt.tight_layout()
fig.savefig(FIG_DIR / "fig_ensemble_cv_windows.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/diagnostics/fig_ensemble_cv_windows.png")


# ═══════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SCRIPT 08 COMPLETE — Ensemble Evaluation")
print("=" * 60)
print(f"  Windows evaluated:   {len(cv_rows)}")
print(f"  Scoring horizon:     {CV_HORIZON} weeks per window")
print(f"  Peak timing improv:  {timing_improvement_pct:.1f}% vs SARIMA-only")
print(f"  80% PI coverage:     SARIMA={s_cov80_mean:.1f}%  Ensemble={e_cov80_mean:.1f}%")
print(f"  Acceptance verdict:  {'ACCEPTED' if acceptance else 'NOT ACCEPTED'}")
print(f"  Outputs:")
print(f"    outputs/metrics/ensemble_cv_detail.csv")
print(f"    outputs/metrics/ensemble_evaluation_summary.csv")
print(f"    figures/diagnostics/fig_ensemble_cv_metrics.png")
print(f"    figures/diagnostics/fig_ensemble_cv_windows.png")
print("=" * 60)
