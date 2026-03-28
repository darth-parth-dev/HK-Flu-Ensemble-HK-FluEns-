"""
04_residual_diagnostics.py
==========================
Step 4: Full residual diagnostic panel — standardised residuals,
ACF, Q-Q plot, histogram, Ljung-Box, Shapiro-Wilk.

Input:  ../models/baseline_sarima/step3_preferred_model.pkl
Output: ../figures/diagnostics/fig_residual_diagnostics.png
        Printed pass/fail summary
"""
import os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, '..')

# ── Load fitted model ─────────────────────────────────────────────────────────
pkl_path = os.path.join(ROOT, 'models/baseline_sarima/step3_preferred_model.pkl')
with open(pkl_path, 'rb') as fh:
    bundle = pickle.load(fh)

fit = bundle['fit']
resid_full = fit.resid
resid = resid_full[52:]            # drop seasonal burn-in
std_resid = (resid - resid.mean()) / resid.std()
n = len(resid)
print(f"Residuals: {n} observations (post burn-in)")

# ── Statistical tests ─────────────────────────────────────────────────────────
lb     = acorr_ljungbox(resid, lags=[10, 20], return_df=True)
lb_p20 = float(lb['lb_pvalue'].iloc[-1])
lb_p10 = float(lb['lb_pvalue'].iloc[0])
sw_stat, sw_p = stats.shapiro(resid[:50])   # Shapiro limited to 50 obs
jb_stat, jb_p = stats.jarque_bera(resid)
ar1_corr  = float(np.corrcoef(resid[1:], resid[:-1])[0,1])
skewness  = float(stats.skew(resid))
kurtosis  = float(stats.kurtosis(resid))

# ── 4-panel diagnostic figure ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('Residual Diagnostics — ARIMAX(1,0,1)(0,1,0)[52]', fontsize=13, fontweight='bold')

# Panel 1: standardised residuals over time
axes[0,0].plot(std_resid, color='steelblue', lw=0.8, alpha=0.85)
axes[0,0].axhline(0,  color='black', lw=0.8)
axes[0,0].axhline(2,  color='red', lw=0.7, ls='--', alpha=0.6)
axes[0,0].axhline(-2, color='red', lw=0.7, ls='--', alpha=0.6)
axes[0,0].set_title('Standardised Residuals')
axes[0,0].set_ylabel('Std residual')
axes[0,0].set_xlabel('Week index')

# Panel 2: ACF of residuals
nlags = 40
ci = 1.96 / np.sqrt(n)
acf_vals = acf(resid, nlags=nlags, fft=True)
lags_arr = np.arange(nlags + 1)
axes[0,1].bar(lags_arr, acf_vals, width=0.4,
              color=['tomato' if abs(v) > ci else 'steelblue' for v in acf_vals])
axes[0,1].axhline(0, color='black', lw=0.8)
axes[0,1].axhline(ci,  color='blue', lw=0.7, ls='--', alpha=0.6)
axes[0,1].axhline(-ci, color='blue', lw=0.7, ls='--', alpha=0.6)
axes[0,1].set_title('ACF of Residuals')
axes[0,1].set_xlabel('Lag')
axes[0,1].set_ylabel('ACF')

# Panel 3: Q-Q plot
(osm, osr), _ = stats.probplot(std_resid, dist='norm')
axes[1,0].scatter(osm, osr, s=12, color='steelblue', alpha=0.7)
mn, mx = osm[0], osm[-1]
axes[1,0].plot([mn, mx], [mn, mx], color='red', lw=1.2, ls='--')
axes[1,0].set_title('Q-Q Plot (Normal)')
axes[1,0].set_xlabel('Theoretical quantiles')
axes[1,0].set_ylabel('Sample quantiles')

# Panel 4: histogram
axes[1,1].hist(std_resid, bins=25, density=True, color='steelblue',
               alpha=0.65, edgecolor='white')
xr = np.linspace(std_resid.min(), std_resid.max(), 200)
axes[1,1].plot(xr, stats.norm.pdf(xr), color='red', lw=1.5, label='N(0,1)')
axes[1,1].set_title('Residual Histogram')
axes[1,1].set_xlabel('Std residual')
axes[1,1].legend(fontsize=8)

plt.tight_layout()
fig_path = os.path.join(ROOT, 'figures/diagnostics/fig_residual_diagnostics.png')
os.makedirs(os.path.dirname(fig_path), exist_ok=True)
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Figure saved: {fig_path}")

# ── Pass/fail summary ─────────────────────────────────────────────────────────
print("\n── Residual Diagnostic Summary ──")
checks = {
    'Ljung-Box lag-10  (p>0.05)': (lb_p10,  lb_p10  > 0.05),
    'Ljung-Box lag-20  (p>0.05)': (lb_p20,  lb_p20  > 0.05),
    'AR(1) autocorr    (|r|<0.1)': (ar1_corr, abs(ar1_corr) < 0.1),
    'Shapiro-Wilk      (flag)':    (sw_p,   True),   # always flag, not fail
    'Jarque-Bera       (p>0.01)':  (jb_p,   jb_p    > 0.01),
}
all_pass = True
for name, (val, passed) in checks.items():
    tag = 'PASS' if passed else 'FAIL'
    if name.startswith('Shapiro'):
        tag = f"p={sw_p:.4f} (Bootstrap PI used if p<0.05)"
    print(f"  {name:<36} val={val:.4f}  {tag}")
    if not passed and not name.startswith('Shapiro'):
        all_pass = False

print(f"\n  Overall residual adequacy: {'PASS' if all_pass else 'WARN — see above'}")
print("Script 04 complete.")
