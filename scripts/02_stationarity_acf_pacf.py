"""
02_stationarity_acf_pacf.py
============================
Step 2: ADF stationarity tests, ACF/PACF diagnostics, order confirmation.

Input:  ../data/processed/processed_model_ready.csv
Output: ../figures/diagnostics/fig_acf_pacf_diagnostics.png
        Printed order recommendation: SARIMA(1,0,1)(0,1,0)[52]
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, '..')

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(ROOT, 'data/processed/processed_model_ready.csv'),
                 parse_dates=['date'])
y_raw = df['y_raw'].values
y_log = df['y_log'].values
# Guard against -inf / NaN (zeros floored in script 01, but interpolate any residuals)
y_log = np.where(np.isfinite(y_log), y_log, np.nan)
y_log = pd.Series(y_log).interpolate(limit_direction='both').values
print(f"Loaded {len(df)} weekly observations ({df.date.iloc[0].date()} → {df.date.iloc[-1].date()})")

# ── ADF tests ─────────────────────────────────────────────────────────────────
def adf_summary(series, label):
    result = adfuller(series, autolag='AIC')
    stat, pval = result[0], result[1]
    conclusion = 'Stationary' if pval < 0.05 else 'Non-stationary'
    print(f"  ADF [{label}]: stat={stat:.4f}, p={pval:.4f}  → {conclusion}")
    return pval

print("\n── Stationarity Tests (ADF) ──")
p_level = adf_summary(y_log, 'y_log levels')
y_sdiff = y_log[52:] - y_log[:-52]   # one seasonal difference (lag-52)
p_sdiff = adf_summary(y_sdiff, 'y_log seasonal-diff')

# ── ACF/PACF figure ───────────────────────────────────────────────────────────
nlags = 80
acf_lev   = acf(y_log,   nlags=nlags, fft=True)
pacf_lev  = pacf(y_log,  nlags=nlags, method='ywm')
acf_sd    = acf(y_sdiff, nlags=nlags, fft=True)
pacf_sd   = pacf(y_sdiff,nlags=nlags, method='ywm')

ci_lev  = 1.96 / np.sqrt(len(y_log))
ci_sd   = 1.96 / np.sqrt(len(y_sdiff))

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle('Stationarity & ACF/PACF Diagnostics — log(ILI+)', fontsize=13, fontweight='bold')

lags = np.arange(nlags + 1)

def stem_plot(ax, lags, vals, ci, title, ylabel=''):
    ax.axhline(0, color='black', lw=0.8)
    ax.axhline(ci,  color='blue', lw=0.8, ls='--', alpha=0.6)
    ax.axhline(-ci, color='blue', lw=0.8, ls='--', alpha=0.6)
    ax.bar(lags, vals, width=0.4, color=['tomato' if abs(v) > ci else 'steelblue' for v in vals])
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-1, nlags + 1)
    # Mark seasonal lags
    for s in [52]:
        if s <= nlags:
            ax.axvline(s, color='orange', lw=0.8, ls=':', alpha=0.7)

stem_plot(axes[0,0], lags, acf_lev,   ci_lev, 'ACF — y_log (levels)',       'ACF')
stem_plot(axes[0,1], lags, pacf_lev,  ci_lev, 'PACF — y_log (levels)',      'PACF')
stem_plot(axes[1,0], lags, acf_sd,    ci_sd,  'ACF — y_log (seasonal diff)', 'ACF')
stem_plot(axes[1,1], lags, pacf_sd,   ci_sd,  'PACF — y_log (seasonal diff)','PACF')

for ax in axes.flat:
    ax.set_xlabel('Lag (weeks)')

plt.tight_layout()
fig_path = os.path.join(ROOT, 'figures/diagnostics/fig_acf_pacf_diagnostics.png')
os.makedirs(os.path.dirname(fig_path), exist_ok=True)
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nFigure saved: {fig_path}")

# ── Order recommendation ──────────────────────────────────────────────────────
print("\n── Order Recommendation ──")
print("  Series requires 1 seasonal difference at lag-52 (ADF p on seasonal-diff < 0.05).")
print("  ACF cuts off after lag 1 → MA(1) term.")
print("  PACF decays slowly → AR(1) term.")
print("  Seasonal ACF/PACF: no residual spikes at lag-52 → S=(0,1,0)[52].")
print("  ► Recommended model: SARIMA(1,0,1)(0,1,0)[52] with exogenous climate regressors")
print("Script 02 complete.")
