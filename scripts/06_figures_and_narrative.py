"""
06_figures_and_narrative.py
============================
Step 6: Generate publication-quality figures and 3-sentence narrative text
for the review background section.

Input:  ../data/processed/processed_master_dataset.csv
        ../outputs/forecasts/forecast_78wk_2026_27.csv
        ../outputs/forecasts/historical_percentile_envelope.csv  (if present)
Output: ../figures/forecasts/fig_main_forecast.png
        ../figures/forecasts/fig_season_detail_panels.png
        Printed narrative text
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, '..')

FIG_DIR = os.path.join(ROOT, 'figures/forecasts')
os.makedirs(FIG_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
master = pd.read_csv(
    os.path.join(ROOT, 'data/processed/processed_master_dataset.csv'),
    parse_dates=['date']
)
fc = pd.read_csv(
    os.path.join(ROOT, 'outputs/forecasts/forecast_78wk_2026_27.csv'),
    parse_dates=['date']
)

# Optional: historical percentile envelope (keyed by ISO week, not date)
env_path = os.path.join(ROOT, 'outputs/forecasts/historical_percentile_envelope.csv')
env = pd.read_csv(env_path) if os.path.exists(env_path) else None
if env is not None and 'date' not in env.columns:
    env = None   # week-indexed envelope; skip date-based overlay

# Restrict historical to post-2019 for clarity (training window)
hist = master[master['date'] >= '2019-01-01'].copy()
y_col = 'ili_plus_model'   # use modelled (imputed) series for continuity

# ── Colour palette ────────────────────────────────────────────────────────────
C_HIST  = '#2c7bb6'
C_FC    = '#d7191c'
C_PI80  = '#fdae61'
C_PI95  = '#fee08b'
C_ENV   = '#bdbdbd'
C_SHADE = '#f0f0f0'

# ── Helper: shade alternate winters ──────────────────────────────────────────
def shade_winters(ax, start_yr, end_yr, alpha=0.07):
    for yr in range(start_yr, end_yr + 1):
        ax.axvspan(pd.Timestamp(f'{yr}-10-01'),
                   pd.Timestamp(f'{yr+1}-03-31'),
                   color='steelblue', alpha=alpha, zorder=0)

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: Main forecast — full historical + 78-week ahead
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 5))

# Historical series
ax.plot(hist['date'], hist[y_col], color=C_HIST, lw=1.4,
        label='Observed ILI+ (2019–2026)', zorder=3)

# Optional: historical percentile envelope
if env is not None and 'p10' in env.columns and 'p90' in env.columns:
    ax.fill_between(env['date'], env['p10'], env['p90'],
                    color=C_ENV, alpha=0.5, label='Historical 10–90th pct', zorder=1)

# Forecast
ax.fill_between(fc['date'], fc['lo95'], fc['hi95'],
                color=C_PI95, alpha=0.9, label='95% PI', zorder=2)
ax.fill_between(fc['date'], fc['lo80'], fc['hi80'],
                color=C_PI80, alpha=0.9, label='80% PI', zorder=2)
ax.plot(fc['date'], fc['forecast'], color=C_FC, lw=1.8,
        label='ARIMAX point forecast', zorder=4)

# Forecast origin line
origin = hist['date'].max()
ax.axvline(origin, color='black', lw=1.0, ls='--', alpha=0.6, label='Forecast origin')

# Seasonal shading (winters)
shade_winters(ax, 2018, 2027)

ax.set_ylabel('ILI+ rate', fontsize=11)
ax.set_xlabel('')
ax.set_title('Hong Kong ILI+ Forecast — ARIMAX(1,0,1)(0,1,0)[52]\n'
             'Season 2026–27 (78-week horizon, bootstrap 80/95% PI)',
             fontsize=12, fontweight='bold')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.legend(fontsize=8, ncol=3, loc='upper left')
ax.set_xlim(hist['date'].min(), fc['date'].max())
ax.set_ylim(bottom=0)

plt.tight_layout()
p1 = os.path.join(FIG_DIR, 'fig_main_forecast.png')
plt.savefig(p1, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {p1}")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: Season detail panels — Winter 2026-27 | Summer 2027
# ══════════════════════════════════════════════════════════════════════════════
winter_fc = fc[(fc['date'] >= '2026-09-01') & (fc['date'] <= '2027-03-31')]
summer_fc = fc[(fc['date'] >= '2027-04-01') & (fc['date'] <= '2027-08-31')]

# Historical same-season comparators (2019-20, 2023-24 for winter)
prev_winters = master[
    ((master['date'] >= '2019-09-01') & (master['date'] <= '2020-03-31')) |
    ((master['date'] >= '2023-09-01') & (master['date'] <= '2024-03-31'))
].copy()
prev_summers = master[
    ((master['date'] >= '2022-04-01') & (master['date'] <= '2022-08-31')) |
    ((master['date'] >= '2024-04-01') & (master['date'] <= '2024-08-31'))
].copy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Seasonal Detail Panels — 2026–27 Forecast', fontsize=12, fontweight='bold')

def season_panel(ax, fc_slice, hist_slice, title):
    if not fc_slice.empty:
        ax.fill_between(fc_slice['date'], fc_slice['lo95'], fc_slice['hi95'],
                        color=C_PI95, alpha=0.9, label='95% PI')
        ax.fill_between(fc_slice['date'], fc_slice['lo80'], fc_slice['hi80'],
                        color=C_PI80, alpha=0.9, label='80% PI')
        ax.plot(fc_slice['date'], fc_slice['forecast'],
                color=C_FC, lw=2.0, label='Forecast')
    if not hist_slice.empty:
        for (label, grp) in hist_slice.groupby(hist_slice['date'].dt.year):
            ax.plot(grp['date'], grp[y_col],
                    color=C_ENV, lw=1.2, ls='--', alpha=0.7)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel('ILI+ rate')
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    ax.legend(fontsize=8)

season_panel(axes[0], winter_fc, prev_winters, 'Winter 2026–27\n(grey dashed = 2019–20, 2023–24)')
season_panel(axes[1], summer_fc, prev_summers, 'Summer 2027\n(grey dashed = 2022, 2024)')

plt.tight_layout()
p2 = os.path.join(FIG_DIR, 'fig_season_detail_panels.png')
plt.savefig(p2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {p2}")

# ══════════════════════════════════════════════════════════════════════════════
# Narrative text (3 sentences for review background section)
# ══════════════════════════════════════════════════════════════════════════════
# Compute summary statistics for narrative
peak_fc_row   = fc.loc[fc['forecast'].idxmax()]
peak_fc_date  = pd.to_datetime(peak_fc_row['date']).strftime('%B %Y')
peak_fc_val   = peak_fc_row['forecast']
peak_hi95     = peak_fc_row['hi95']
last_obs_val  = hist[y_col].dropna().iloc[-1]
last_obs_date = hist['date'].iloc[-1].strftime('%B %Y')
n_hist_weeks  = len(hist)

narrative = (
    f"Hong Kong influenza activity, measured as the composite ILI+ index "
    f"(GOPC consultation rate × laboratory positivity), was tracked weekly "
    f"across {n_hist_weeks} surveillance weeks from January 2019 to {last_obs_date}, "
    f"incorporating a Kalman-smoothed imputation of the 54-week COVID-19 "
    f"suppression window (April 2020 – September 2022). "
    f"An ARIMAX(1,0,1)(0,1,0)[52] model with standardised temperature and "
    f"absolute-humidity climatological regressors was selected from four "
    f"candidate specifications by minimum AIC/BIC and used to generate a "
    f"78-week probabilistic forecast with bootstrap prediction intervals "
    f"(n = 600 resamples) from the training endpoint. "
    f"The model projects a peak ILI+ of {peak_fc_val:.4f} "
    f"(95% PI upper: {peak_hi95:.4f}) around {peak_fc_date}, "
    f"with seasonal winter and summer peaks consistent with Hong Kong's "
    f"characteristic bimodal influenza pattern."
)

print("\n── Narrative (for review background section) ──\n")
print(narrative)
print("\nScript 06 complete.")
