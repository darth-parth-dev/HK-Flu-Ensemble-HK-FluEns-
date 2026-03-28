"""
01_data_preparation.py
======================
Step 1: Load CHP surveillance data, compute ILI+, handle COVID gap,
apply log transformation, create structural break regressors.

Input:  ../data/raw/raw_chp_weekly.xlsx
        ../data/processed/processed_covid_imputed_series.csv
Output: ../data/processed/processed_model_ready.csv
        ../figures/diagnostics/fig_prepared_series.png
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, '..')

# ── HKO 30-year climatological normals (1991-2020) ───────────────────────────
# Monthly: [Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec]
HKO_TEMP  = {1:16.3, 2:17.1, 3:19.5, 4:23.3, 5:26.4, 6:28.5,
             7:29.1, 8:28.8, 9:27.8, 10:25.4, 11:21.4, 12:17.8}
HKO_AH    = {1:10.262, 2:11.641, 3:13.759, 4:17.355, 5:20.671, 6:23.215,
             7:23.990, 8:23.600, 9:21.531, 10:16.956, 11:13.122, 12:10.621}
# Z-score parameters (from full training series, precomputed)
TEMP_MEAN, TEMP_STD = 23.325201, 4.717926
AH_MEAN,   AH_STD   = 17.108601, 5.080883

TRAINING_START = '2019-01-06'
TRAINING_END   = '2026-02-22'

# ── 1. Parse CHP Excel ────────────────────────────────────────────────────────
print("Loading CHP Excel surveillance data …")
xl  = pd.ExcelFile(os.path.join(ROOT, 'data/raw/raw_chp_weekly.xlsx'))
raw = xl.parse('Weekly surveillance data 每周監測數據', header=None)

# Identify data rows (rows where col 0 is a 4-digit year)
mask = pd.to_numeric(raw[0], errors='coerce').notna()
data = raw[mask].copy().reset_index(drop=True)
data.columns = range(data.shape[1])

# Extract key columns
chp = pd.DataFrame({
    'year':         pd.to_numeric(data[0]),
    'week':         pd.to_numeric(data[1]),
    'date_from':    pd.to_datetime(data[2], errors='coerce'),
    'date_to':      pd.to_datetime(data[3], errors='coerce'),
    'ili_fmc_raw':  pd.to_numeric(data[4], errors='coerce'),   # Sentinel FMC per 1000
    'ili_pmp_raw':  pd.to_numeric(data[5], errors='coerce'),   # Sentinel PMP per 1000
    'pos_H1':       pd.to_numeric(data[6], errors='coerce'),
    'pos_H3':       pd.to_numeric(data[7], errors='coerce'),
    'pos_B':        pd.to_numeric(data[8], errors='coerce'),
    'pos_all':      pd.to_numeric(data[9], errors='coerce'),   # positivity fraction
    'pos_pct_H1':   pd.to_numeric(data[10], errors='coerce'),
    'pos_pct_H3':   pd.to_numeric(data[11], errors='coerce'),
    'pos_pct_B':    pd.to_numeric(data[12], errors='coerce'),
    'pos_pct_all':  pd.to_numeric(data[13], errors='coerce'),  # overall positivity %
})
chp['date'] = chp['date_from']  # Use week-start date as index

# ILI+ = Sentinel FMC rate (per 1000) × positivity fraction / 100
# When FMC is NaN (pre-2020 for some weeks), fall back to PMP column
ili_src = chp['ili_fmc_raw'].where(chp['ili_fmc_raw'].notna(), chp['ili_pmp_raw'])
chp['ili_plus_raw'] = ili_src * chp['pos_pct_all'] / 100

print(f"  CHP rows parsed: {len(chp)} (range {chp.date.min().date()} → {chp.date.max().date()})")

# ── 2. Load COVID-imputed series ──────────────────────────────────────────────
print("Loading COVID-imputed series …")
imp = pd.read_csv(
    os.path.join(ROOT, 'data/processed/processed_covid_imputed_series.csv'),
    parse_dates=['date']
)
# Use ili_plus_imputed as the cleaned model target
imp = imp[['date','year','week','ili_plus_imputed','suppression_flag']].copy()
imp.columns = ['date','year','week','y_raw','suppression_flag_filled']
imp['suppression_flag_filled'] = imp['suppression_flag_filled'].astype(int)

# ── 3. Filter to training window ──────────────────────────────────────────────
imp = imp[(imp['date'] >= TRAINING_START) & (imp['date'] <= TRAINING_END)].reset_index(drop=True)
print(f"  Training rows: {len(imp)} ({imp.date.iloc[0].date()} → {imp.date.iloc[-1].date()})")

# ── 4. Add month, post-COVID flag ─────────────────────────────────────────────
imp['month'] = imp['date'].dt.month
imp['post_covid'] = (imp['date'] >= '2022-10-01').astype(int)

# ── 5. Log transform (ILI+ → log scale) ──────────────────────────────────────
imp['y_log'] = np.log(imp['y_raw'])

# ── 6. Add HKO climate regressors ────────────────────────────────────────────
imp['hko_temp_norm'] = imp['month'].map(HKO_TEMP)
imp['hko_ah_norm']   = imp['month'].map(HKO_AH)
imp['temp_z'] = (imp['hko_temp_norm'] - TEMP_MEAN) / TEMP_STD
imp['ah_z']   = (imp['hko_ah_norm']   - AH_MEAN)   / AH_STD

# ── 7. Column ordering matching downstream expectations ───────────────────────
out = imp[['date','year','week','month','y_raw','y_log',
           'suppression_flag_filled','post_covid',
           'temp_z','ah_z','hko_temp_norm','hko_ah_norm']].copy()

out_path = os.path.join(ROOT, 'data/processed/processed_model_ready.csv')
out.to_csv(out_path, index=False)
print(f"  Saved: {out_path}")

# ── 8. Figure: prepared series ────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle('HK Influenza ILI+ — Prepared Training Series', fontsize=13, fontweight='bold')

# Panel 1: raw ILI+
axes[0].plot(out['date'], out['y_raw'], color='steelblue', lw=1.2)
imp_mask = out['suppression_flag_filled'] == 1
axes[0].fill_between(out['date'], 0, out['y_raw'].max(),
                     where=imp_mask, alpha=0.15, color='orange', label='COVID-imputed')
axes[0].set_ylabel('ILI+ rate')
axes[0].set_title('ILI+ (raw)')
axes[0].legend(fontsize=8)

# Panel 2: log-transformed
axes[1].plot(out['date'], out['y_log'], color='darkgreen', lw=1.2)
axes[1].set_ylabel('log(ILI+)')
axes[1].set_title('Log-transformed ILI+')

# Panel 3: climate regressors
ax3b = axes[2].twinx()
axes[2].plot(out['date'], out['temp_z'], color='tomato', lw=1.0, label='Temp z-score')
ax3b.plot(out['date'], out['ah_z'],   color='royalblue', lw=1.0, linestyle='--', label='AH z-score')
axes[2].set_ylabel('Temp z-score', color='tomato')
ax3b.set_ylabel('AH z-score', color='royalblue')
axes[2].set_title('Climatological regressors (standardised)')
lines1, labels1 = axes[2].get_legend_handles_labels()
lines2, labels2 = ax3b.get_legend_handles_labels()
axes[2].legend(lines1 + lines2, labels1 + labels2, fontsize=8)

plt.tight_layout()
fig_path = os.path.join(ROOT, 'figures/diagnostics/fig_prepared_series.png')
os.makedirs(os.path.dirname(fig_path), exist_ok=True)
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Figure saved: {fig_path}")
print("Script 01 complete.")
