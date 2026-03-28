"""
03_model_fitting.py
===================
Step 3: Fit 4 candidate SARIMA/ARIMAX models, compare by AIC/BIC/LB-p,
save preferred model object and coefficients.

Input:  ../data/processed/processed_model_ready.csv
Output: ../models/baseline_sarima/step3_preferred_model.pkl
        ../outputs/coefficients/model_coefficients.json
        Printed model comparison table
"""
import os, json, pickle
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, '..')
np.random.seed(42)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(ROOT, 'data/processed/processed_model_ready.csv'),
                 parse_dates=['date'])
y    = df['y_raw'].values          # ILI+ rate (model target)
exog = df[['temp_z','ah_z']].values
n    = len(y)
print(f"Fitting on {n} observations ({df.date.iloc[0].date()} → {df.date.iloc[-1].date()})")

# ── Candidate models: ARIMAX(p,0,q)(0,1,0)[52], p,q ∈ {0,1} ─────────────────
candidates = [(0,0), (1,0), (0,1), (1,1)]
results = []

for p, q in candidates:
    label = f"ARIMAX({p},0,{q})(0,1,0)[52]"
    try:
        mod = SARIMAX(y, exog=exog, order=(p,0,q), seasonal_order=(0,1,0,52),
                      trend='c', enforce_stationarity=False, enforce_invertibility=False)
        fit = mod.fit(method='lbfgs', maxiter=3000, disp=False,
                      optim_score=None, low_memory=False)
        resid = fit.resid[52:]           # drop seasonal-diff burn-in
        lb    = acorr_ljungbox(resid, lags=20, return_df=True)
        lb_p  = float(lb['lb_pvalue'].iloc[-1])
        sw_p  = float(stats.shapiro(resid[:50])[1])  # Shapiro on first 50 (power limit)
        results.append({
            'label': label, 'p': p, 'q': q, 'fit': fit,
            'aic': fit.aic, 'bic': fit.bic, 'lb_p': lb_p, 'sw_p': sw_p,
            'sigma2': float(fit.params[-1]),
        })
        print(f"  {label}: AIC={fit.aic:.2f}, BIC={fit.bic:.2f}, LB-p={lb_p:.4f}")
    except Exception as e:
        print(f"  {label}: FAILED — {e}")

# ── Select preferred model (lowest AIC) ───────────────────────────────────────
best = min(results, key=lambda r: r['aic'])
print(f"\n  ► Preferred: {best['label']}  (AIC={best['aic']:.2f}, BIC={best['bic']:.2f})")

fit = best['fit']
pnames = fit.param_names

# ── Print comparison table ────────────────────────────────────────────────────
print("\n  Model comparison table:")
print(f"  {'Model':<30} {'AIC':>10} {'BIC':>10} {'LB-p':>8} {'SW-p':>8}")
print("  " + "-"*70)
for r in sorted(results, key=lambda x: x['aic']):
    marker = " ◄" if r['label'] == best['label'] else ""
    print(f"  {r['label']:<30} {r['aic']:>10.2f} {r['bic']:>10.2f} "
          f"{r['lb_p']:>8.4f} {r['sw_p']:>8.4f}{marker}")

# ── Extract parameters ────────────────────────────────────────────────────────
params = dict(zip(pnames, fit.params))

intercept   = float(params.get('intercept', params.get('const', 0)))
phi1        = float(params.get('ar.L1', 0))
theta1      = float(params.get('ma.L1', 0))
beta_temp   = float(params.get('x1', params.get('temp_z', 0)))
beta_ah     = float(params.get('x2', params.get('ah_z', 0)))
sigma2      = float(fit.params[-1])

resid = fit.resid[52:]
lb_p  = float(acorr_ljungbox(resid, lags=20, return_df=True)['lb_pvalue'].iloc[-1])
resid_ar1 = float(np.corrcoef(resid[1:], resid[:-1])[0,1])
resid_skew = float(stats.skew(resid))

print(f"\n  phi1_AR1    = {phi1:.4f}")
print(f"  theta1_MA1  = {theta1:.4f}")
print(f"  AIC         = {fit.aic:.2f}")
print(f"  BIC         = {fit.bic:.2f}")
print(f"  LB-p (lag20)= {lb_p:.4f}")
print(f"  sigma       = {np.sqrt(sigma2):.6f}")

# ── Save model pickle ─────────────────────────────────────────────────────────
pkl_dir = os.path.join(ROOT, 'models/baseline_sarima')
os.makedirs(pkl_dir, exist_ok=True)
pkl_path = os.path.join(pkl_dir, 'step3_preferred_model.pkl')
with open(pkl_path, 'wb') as fh:
    pickle.dump({'fit': fit, 'df': df, 'order': (best['p'], 0, best['q']),
                 'seasonal_order': (0, 1, 0, 52)}, fh)
print(f"\n  Model pickle saved: {pkl_path}")

# ── Save coefficients JSON ────────────────────────────────────────────────────
coef = {
    "model_spec": f"SARIMA({best['p']},0,{best['q']})(0,1,0)[52] with exogenous regressors",
    "parameters": {
        "mu_intercept":  round(intercept, 8),
        "phi1_AR1":      round(phi1,  4),
        "theta1_MA1":    round(theta1, 4),
        "beta_temp_z":   round(beta_temp, 5),
        "beta_ah_z":     round(beta_ah, 5),
        "sigma":         round(float(np.sqrt(sigma2)), 8),
        "sigma_squared": round(sigma2, 8),
    },
    "model_diagnostics": {
        "AIC":              round(fit.aic, 2),
        "BIC":              round(fit.bic, 2),
        "ljung_box_p":      round(lb_p, 4),
        "residual_AR1":     round(resid_ar1, 6),
        "residual_skewness":round(resid_skew, 4),
    },
    "training_data": {
        "n_weeks": int(n),
        "start":   str(df.date.iloc[0].date()),
        "end":     str(df.date.iloc[-1].date()),
    }
}
out_dir = os.path.join(ROOT, 'outputs/coefficients')
os.makedirs(out_dir, exist_ok=True)
coef_path = os.path.join(out_dir, 'model_coefficients.json')
with open(coef_path, 'w') as fh:
    json.dump(coef, fh, indent=2)
print(f"  Coefficients saved: {coef_path}")
print("Script 03 complete.")
