"""
scoring/wis.py
==============
HK-FluEns hub submission validator and Weighted Interval Score (WIS) calculator.

Implements the validation rules from docs/model-submission-spec.md (v0.1):
  1. Schema       — required columns present with correct types
  2. Completeness — all 23 quantile levels for every (target, horizon) pair
  3. Monotonicity — quantile values non-decreasing within each (target, horizon)
  4. Range        — lab_positivity_pct in [0,100]; gopc_ili_rate_per1000 in [0,500]
  5. Date         — forecast_date is a Saturday; horizons align to CHP weeks

Public API
----------
validate_submission(df) -> (ok: bool, errors: list[str])
wis(quantile_df, observed, target_col='value') -> float
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Tuple

# ── Constants ─────────────────────────────────────────────────────────────────
REQUIRED_COLS   = {'forecast_date', 'target', 'horizon', 'type', 'quantile', 'value'}
REQUIRED_QUANTS = [0.010, 0.025, 0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350,
                   0.400, 0.450, 0.500, 0.550, 0.600, 0.650, 0.700, 0.750, 0.800,
                   0.850, 0.900, 0.950, 0.975, 0.990]
VALID_TARGETS   = {'lab_positivity_pct', 'gopc_ili_rate_per1000'}
VALID_HORIZONS  = {'1 wk ahead', '2 wk ahead', '3 wk ahead', '4 wk ahead'}
TARGET_RANGES   = {
    'lab_positivity_pct':      (0.0, 100.0),
    'gopc_ili_rate_per1000':   (0.0, 500.0),
}
QUANT_TOL = 1e-9   # floating-point tolerance for quantile comparison


# ─────────────────────────────────────────────────────────────────────────────
def validate_submission(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate a FluSight-format submission DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Contents of a hub/model-output/<date>-<model>.csv file.

    Returns
    -------
    ok : bool
        True if all checks pass.
    errors : list of str
        Empty when ok=True; one entry per failed check otherwise.
    """
    errors: List[str] = []

    # ── 1. Schema ─────────────────────────────────────────────────────────────
    missing_cols = REQUIRED_COLS - set(df.columns)
    if missing_cols:
        errors.append(f"Schema: missing columns {sorted(missing_cols)}")
        return False, errors   # cannot proceed without required columns

    # Coerce types
    df = df.copy()
    df['forecast_date'] = pd.to_datetime(df['forecast_date'], errors='coerce')
    df['value']         = pd.to_numeric(df['value'], errors='coerce')
    quant_numeric = pd.to_numeric(df['quantile'], errors='coerce')

    if df['forecast_date'].isna().any():
        errors.append("Schema: 'forecast_date' contains unparseable dates")
    if df['value'].isna().any():
        errors.append(f"Schema: 'value' has {df['value'].isna().sum()} non-numeric entries")

    # ── 2. Date alignment ─────────────────────────────────────────────────────
    for fd in df['forecast_date'].dropna().unique():
        dow = pd.Timestamp(fd).dayofweek   # Monday=0, Saturday=5
        if dow != 5:
            errors.append(
                f"Date: forecast_date {fd.date()} is not a Saturday "
                f"(day-of-week={dow})"
            )

    bad_targets  = set(df['target'].unique()) - VALID_TARGETS
    bad_horizons = set(df['horizon'].unique()) - VALID_HORIZONS
    if bad_targets:
        errors.append(f"Schema: unrecognised target(s): {bad_targets}")
    if bad_horizons:
        errors.append(f"Schema: unrecognised horizon(s): {bad_horizons}")

    bad_types = set(df['type'].unique()) - {'quantile', 'point'}
    if bad_types:
        errors.append(f"Schema: unrecognised type(s): {bad_types}")

    # ── 3. Completeness ───────────────────────────────────────────────────────
    q_rows = df[df['type'] == 'quantile'].copy()
    q_rows['_q'] = pd.to_numeric(q_rows['quantile'], errors='coerce')

    for target in df['target'].unique():
        for horizon in df['horizon'].unique():
            mask = (q_rows['target'] == target) & (q_rows['horizon'] == horizon)
            present = sorted(q_rows.loc[mask, '_q'].dropna().round(4).tolist())
            required = [round(q, 4) for q in REQUIRED_QUANTS]
            missing  = sorted(set(required) - set(present))
            extra    = sorted(set(present) - set(required))
            if missing:
                errors.append(
                    f"Completeness: ({target}, {horizon}) missing quantiles {missing}"
                )
            if extra:
                errors.append(
                    f"Completeness: ({target}, {horizon}) unexpected quantiles {extra}"
                )
        # Check point row present
        pt_mask = (df['target'] == target) & (df['type'] == 'point')
        for horizon in df['horizon'].unique():
            hmask = pt_mask & (df['horizon'] == horizon)
            if not hmask.any():
                errors.append(
                    f"Completeness: ({target}, {horizon}) missing 'point' row"
                )

    # ── 4. Monotonicity ───────────────────────────────────────────────────────
    for target in q_rows['target'].unique():
        for horizon in q_rows['horizon'].unique():
            mask = (q_rows['target'] == target) & (q_rows['horizon'] == horizon)
            sub  = q_rows[mask].sort_values('_q')
            vals = sub['value'].values
            if len(vals) > 1:
                diffs = np.diff(vals)
                if (diffs < -QUANT_TOL).any():
                    first_viol = int(sub.iloc[np.argmax(diffs < -QUANT_TOL)]['_q'] * 1000)
                    errors.append(
                        f"Monotonicity: ({target}, {horizon}) values decrease at "
                        f"quantile ~0.{first_viol:03d}"
                    )

    # ── 5. Range ──────────────────────────────────────────────────────────────
    for target, (lo, hi) in TARGET_RANGES.items():
        mask = df['target'] == target
        if not mask.any():
            continue
        vals = df.loc[mask, 'value'].dropna()
        out_of_range = ((vals < lo - QUANT_TOL) | (vals > hi + QUANT_TOL))
        if out_of_range.any():
            bad_vals = vals[out_of_range].unique()[:3]
            errors.append(
                f"Range: {target} has {out_of_range.sum()} value(s) outside "
                f"[{lo}, {hi}]; examples: {list(np.round(bad_vals, 4))}"
            )

    return len(errors) == 0, errors


# ─────────────────────────────────────────────────────────────────────────────
def wis(quantile_df: pd.DataFrame,
        observed: float,
        quantile_col: str = 'quantile',
        value_col: str = 'value') -> float:
    """
    Compute the Weighted Interval Score (WIS) for a single forecast.

    WIS = (1/K) Σ_k [ (u_k - l_k)
                      + (2/α_k)(l_k - y)·1[y < l_k]
                      + (2/α_k)(y - u_k)·1[y > u_k] ]
          + (1/2)|y - ŷ|

    where K is the number of symmetric prediction intervals,
    α_k is the nominal non-coverage (1 - interval level),
    l_k/u_k are the lower/upper bounds, y is observed, ŷ is the point forecast.

    Parameters
    ----------
    quantile_df : pd.DataFrame
        Rows with columns [quantile_col, value_col] for a single
        (target, horizon) combination.  Must include the 23 standard quantile
        levels.  May also include a 'type' column; 'point' rows are used for ŷ.
    observed : float
        The realised value for this target-horizon.

    Returns
    -------
    float
        WIS (lower = better).
    """
    df = quantile_df.copy()

    # Extract point forecast if available
    if 'type' in df.columns:
        pt = df[df['type'] == 'point'][value_col]
        y_hat = float(pt.iloc[0]) if not pt.empty else float(
            df[pd.to_numeric(df[quantile_col], errors='coerce') == 0.5][value_col].iloc[0])
        df = df[df['type'] == 'quantile'].copy()
    else:
        y_hat = float(df[pd.to_numeric(df[quantile_col], errors='coerce') == 0.5][value_col].iloc[0])

    df[quantile_col] = pd.to_numeric(df[quantile_col], errors='coerce')
    df = df.dropna(subset=[quantile_col]).sort_values(quantile_col)

    qs   = df[quantile_col].values
    vals = df[value_col].values
    y    = float(observed)

    # Build symmetric intervals from complementary quantile pairs
    # Pair (α/2, 1-α/2)  →  interval level = 1-α,  non-coverage = α
    scores = []
    for i, q_lo in enumerate(qs):
        if q_lo >= 0.5:
            break
        q_hi = round(1.0 - q_lo, 4)
        # find matching upper quantile
        hi_idx = np.where(np.abs(qs - q_hi) < QUANT_TOL)[0]
        if len(hi_idx) == 0:
            continue
        l_k   = vals[i]
        u_k   = vals[hi_idx[0]]
        alpha = round(2 * q_lo, 4)        # non-coverage probability

        interval_score = (u_k - l_k)
        if y < l_k:
            interval_score += (2 / alpha) * (l_k - y)
        elif y > u_k:
            interval_score += (2 / alpha) * (y - u_k)
        scores.append(interval_score)

    K = len(scores)
    if K == 0:
        return abs(y - y_hat) / 2.0

    return (sum(scores) / K) + 0.5 * abs(y - y_hat)


# ─────────────────────────────────────────────────────────────────────────────
def score_submission(submission_df: pd.DataFrame,
                     observed_df: pd.DataFrame,
                     date_col: str = 'date',
                     target_col: str = 'target',
                     value_col: str = 'value') -> pd.DataFrame:
    """
    Compute WIS for every (target, horizon) in a submission against observed data.

    Parameters
    ----------
    submission_df : pd.DataFrame  — validated hub submission
    observed_df   : pd.DataFrame  — columns [date, target, value]

    Returns
    -------
    pd.DataFrame with columns [target, horizon, forecast_date, observed, wis]
    """
    rows = []
    for (target, horizon), grp in submission_df[submission_df['type'] == 'quantile'].groupby(
            ['target', 'horizon']):
        # Determine the target end date from the horizon label
        h_num = int(horizon.split()[0])
        fd    = pd.to_datetime(grp['forecast_date'].iloc[0])
        target_date = fd + pd.Timedelta(weeks=h_num)

        obs_mask = (observed_df[date_col] == target_date) & \
                   (observed_df[target_col] == target)
        if not obs_mask.any():
            continue

        obs_val = float(observed_df.loc[obs_mask, value_col].iloc[0])
        # Include point row for ŷ
        full_grp = submission_df[
            (submission_df['target']  == target) &
            (submission_df['horizon'] == horizon)
        ]
        w = wis(full_grp, obs_val)
        rows.append({'target': target, 'horizon': horizon,
                     'forecast_date': fd.date(), 'target_date': target_date.date(),
                     'observed': obs_val, 'wis': w})

    return pd.DataFrame(rows)
