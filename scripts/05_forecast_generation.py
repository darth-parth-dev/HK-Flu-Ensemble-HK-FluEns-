"""
05_forecast_generation.py
==========================
Step 5: Generate 78-week (and 104-week) forecast with bootstrap PI (n=600).
Enforce non-negativity. Identify winter 2026-27 and summer 2027 peaks.
Write forecast CSV and FluSight-format hub submission CSV.

Input:  ../models/baseline_sarima/step3_preferred_model.pkl
        ../data/processed/processed_model_ready.csv
Output: ../outputs/forecasts/forecast_78wk_2026_27.csv
        ../outputs/forecasts/forecast_104wk_extended.csv
        ../hub/model-output/<forecast_date>-HKU_SARIMA.csv
        ../figures/forecasts/fig_forecast_validation.png
"""
import os, pickle, json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, '..')
np.random.seed(42)

N_BOOT      = 600
HORIZONS_78 = 78
HORIZONS_104= 104

# HKO monthly normals for climate projection
HKO_TEMP  = {1:16.3,  2:17.1,  3:19.5,  4:23.3,  5:26.4,  6:28.5,
             7:29.1,  8:28.8,  9:27.8, 10:25.4, 11:21.4, 12:17.8}
HKO_AH    = {1:10.262,2:11.641,3:13.759,4:17.355,5:20.671,6:23.215,
             7:23.990,8:23.600,9:21.531,10:16.956,11:13.122,12:10.621}
TEMP_MEAN, TEMP_STD = 23.325201, 4.717926
AH_MEAN,   AH_STD   = 17.108601, 5.080883

QUANTILE_LEVELS = [0.010,0.025,0.050,0.100,0.150,0.200,0.250,0.300,0.350,
                   0.400,0.450,0.500,0.550,0.600,0.650,0.700,0.750,0.800,
                   0.850,0.900,0.950,0.975,0.990]
FORECAST_DATE = '2026-03-28'   # Saturday of the run week
MODEL_ID      = 'HKU_SARIMA'

# ── Load model and data ───────────────────────────────────────────────────────
pkl_path = os.path.join(ROOT, 'models/baseline_sarima/step3_preferred_model.pkl')
with open(pkl_path, 'rb') as fh:
    bundle = pickle.load(fh)

fit = bundle['fit']
df  = bundle['df'] if 'df' in bundle else \
      pd.read_csv(os.path.join(ROOT, 'data/processed/processed_model_ready.csv'),
                  parse_dates=['date'])

order          = bundle.get('order', (1,0,1))
seasonal_order = bundle.get('seasonal_order', (0,1,0,52))

last_date = pd.to_datetime(df['date'].iloc[-1])
print(f"Training end: {last_date.date()}")

# ── Build future exogenous regressors ─────────────────────────────────────────
def make_future_exog(start_date, n_weeks):
    dates = pd.date_range(start=start_date + pd.Timedelta(weeks=1), periods=n_weeks, freq='W-SUN')
    months = dates.month
    temp_n = np.array([HKO_TEMP[m] for m in months])
    ah_n   = np.array([HKO_AH[m]   for m in months])
    temp_z = (temp_n - TEMP_MEAN) / TEMP_STD
    ah_z   = (ah_n   - AH_MEAN)   / AH_STD
    return dates, np.column_stack([temp_z, ah_z])

future_dates_78,  future_exog_78  = make_future_exog(last_date, HORIZONS_78)
future_dates_104, future_exog_104 = make_future_exog(last_date, HORIZONS_104)

# ── Point forecast ────────────────────────────────────────────────────────────
fc_78  = fit.forecast(steps=HORIZONS_78,  exog=future_exog_78)
fc_104 = fit.forecast(steps=HORIZONS_104, exog=future_exog_104)

# Enforce non-negativity
fc_78  = np.maximum(fc_78,  0)
fc_104 = np.maximum(fc_104, 0)

# ── Bootstrap prediction intervals ───────────────────────────────────────────
print(f"Running bootstrap (n={N_BOOT}) …")
resid = fit.resid[52:]           # post burn-in residuals

boot_paths_78  = np.zeros((N_BOOT, HORIZONS_78))
boot_paths_104 = np.zeros((N_BOOT, HORIZONS_104))

p, d, q = order
P, D, Q, S = seasonal_order

# Extract ARIMA coefficients for simulation
pnames = fit.param_names
params_dict = dict(zip(pnames, fit.params))

phi   = float(params_dict.get('ar.L1', 0)) if p >= 1 else 0.0
theta = float(params_dict.get('ma.L1', 0)) if q >= 1 else 0.0
const = float(params_dict.get('intercept', params_dict.get('const', 0)))

# For bootstrap: use get_forecast with simulate_smoother
for b in range(N_BOOT):
    boot_resid = np.random.choice(resid, size=max(HORIZONS_104, S+5), replace=True)
    # Simulate ARIMA(1,0,1) forward from fitted state
    sim = fit.simulate(nsimulations=HORIZONS_104,
                       measurement_shocks=boot_resid[:HORIZONS_104],
                       initial_state=fit.predicted_state[:, -1],
                       exog=future_exog_104)
    boot_paths_78[b]  = np.maximum(sim[:HORIZONS_78],  0)
    boot_paths_104[b] = np.maximum(sim,                0)

def pi_from_boot(boot_paths):
    lo95 = np.quantile(boot_paths, 0.025, axis=0)
    lo80 = np.quantile(boot_paths, 0.100, axis=0)
    hi80 = np.quantile(boot_paths, 0.900, axis=0)
    hi95 = np.quantile(boot_paths, 0.975, axis=0)
    return lo95, lo80, hi80, hi95

lo95_78, lo80_78, hi80_78, hi95_78   = pi_from_boot(boot_paths_78)
lo95_104,lo80_104,hi80_104,hi95_104  = pi_from_boot(boot_paths_104)

# Enforce monotonicity: lo95 ≤ lo80 ≤ forecast ≤ hi80 ≤ hi95
def enforce_monotone(fc, lo95, lo80, hi80, hi95):
    lo95 = np.minimum(lo95, lo80)
    lo80 = np.minimum(lo80, fc)
    hi80 = np.maximum(hi80, fc)
    hi95 = np.maximum(hi95, hi80)
    return np.maximum(lo95,0), np.maximum(lo80,0), np.maximum(hi80,0), np.maximum(hi95,0)

lo95_78, lo80_78, hi80_78, hi95_78 = enforce_monotone(
    fc_78, lo95_78, lo80_78, hi80_78, hi95_78)
lo95_104,lo80_104,hi80_104,hi95_104= enforce_monotone(
    fc_104,lo95_104,lo80_104,hi80_104,hi95_104)

# ── Save forecast CSVs ────────────────────────────────────────────────────────
os.makedirs(os.path.join(ROOT, 'outputs/forecasts'), exist_ok=True)

fc78_df = pd.DataFrame({
    'date':     future_dates_78.strftime('%Y-%m-%d'),
    'forecast': fc_78,
    'lo80':     lo80_78,
    'hi80':     hi80_78,
    'lo95':     lo95_78,
    'hi95':     hi95_78,
})
fc78_path = os.path.join(ROOT, 'outputs/forecasts/forecast_78wk_2026_27.csv')
fc78_df.to_csv(fc78_path, index=False)
print(f"Saved: {fc78_path}  ({len(fc78_df)} rows)")

fc104_df = pd.DataFrame({
    'date':     future_dates_104.strftime('%Y-%m-%d'),
    'forecast': fc_104,
    'lo80':     lo80_104,
    'hi80':     hi80_104,
    'lo95':     lo95_104,
    'hi95':     hi95_104,
})
fc104_path = os.path.join(ROOT, 'outputs/forecasts/forecast_104wk_extended.csv')
fc104_df.to_csv(fc104_path, index=False)
print(f"Saved: {fc104_path}  ({len(fc104_df)} rows)")

# ── Identify seasonal peaks ───────────────────────────────────────────────────
winter_mask = (future_dates_78.month >= 10) | (future_dates_78.month <= 3)
summer_mask = (future_dates_78.month >= 4) & (future_dates_78.month <= 9)
if winter_mask.any():
    wi = fc_78[winter_mask].argmax()
    wp = future_dates_78[winter_mask][wi]
    print(f"\nWinter 2026-27 peak: {wp.date()}  ILI+={fc_78[winter_mask][wi]:.5f}")
if summer_mask.any():
    si = fc_78[summer_mask].argmax()
    sp = future_dates_78[summer_mask][si]
    print(f"Summer 2027 peak:    {sp.date()}  ILI+={fc_78[summer_mask][si]:.5f}")

# ── Generate FluSight hub submission CSV ──────────────────────────────────────
print("\nGenerating hub submission CSV …")
hub_rows = []

# Use first 4 horizons for hub submission
for h in range(1, 5):
    horizon_label = f"{h} wk ahead"
    fc_val  = float(fc_78[h-1])
    b_paths = boot_paths_78[:, h-1]

    # target 1: gopc_ili_rate_per1000 (ILI+ × 1000)
    # target 2: lab_positivity_pct (approximate: ILI+ × 100 as a %)
    for target, scale in [('gopc_ili_rate_per1000', 1000.0), ('lab_positivity_pct', 100.0)]:
        # 23 quantile rows
        for q in QUANTILE_LEVELS:
            val = float(np.quantile(b_paths, q) * scale)
            val = max(val, 0.0)
            hub_rows.append({
                'forecast_date': FORECAST_DATE,
                'target':        target,
                'horizon':       horizon_label,
                'type':          'quantile',
                'quantile':      q,
                'value':         round(val, 6),
            })
        # point row
        hub_rows.append({
            'forecast_date': FORECAST_DATE,
            'target':        target,
            'horizon':       horizon_label,
            'type':          'point',
            'quantile':      'NA',
            'value':         round(fc_val * scale, 6),
        })

hub_df = pd.DataFrame(hub_rows)
hub_dir = os.path.join(ROOT, 'hub/model-output')
os.makedirs(hub_dir, exist_ok=True)
hub_path = os.path.join(hub_dir, f"{FORECAST_DATE}-{MODEL_ID}.csv")
hub_df.to_csv(hub_path, index=False)
print(f"Hub CSV saved: {hub_path}  ({len(hub_df)} rows)")

# ── Validation figure ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
hist = df.tail(52)
ax.plot(pd.to_datetime(hist['date']), hist['y_raw'],
        color='steelblue', lw=1.5, label='Historical (last 52 wk)')
ax.plot(future_dates_78[:26], fc_78[:26],
        color='darkred', lw=1.5, label='Forecast (next 26 wk)')
ax.fill_between(future_dates_78[:26], lo80_78[:26], hi80_78[:26],
                color='red', alpha=0.25, label='80% PI')
ax.fill_between(future_dates_78[:26], lo95_78[:26], hi95_78[:26],
                color='red', alpha=0.12, label='95% PI')
ax.set_title('ARIMAX Forecast — 26-week horizon', fontsize=12)
ax.set_ylabel('ILI+ rate')
ax.legend(fontsize=9)
plt.tight_layout()
fig_path = os.path.join(ROOT, 'figures/forecasts/fig_forecast_validation.png')
os.makedirs(os.path.dirname(fig_path), exist_ok=True)
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Figure saved: {fig_path}")
print("Script 05 complete.")
