"""
05_forecast_generation.py
==========================
Step 5: Generate 78-week (or 104-week) forecast with bootstrap PI (n=600).
Back-transform from log scale, enforce non-negativity.
Identify winter 2026-27 and summer 2027 seasonal peaks.

Input:  step3_preferred_model.pkl
        ../data/processed/processed_model_ready.csv
Output: ../outputs/forecasts/forecast_78wk_2026_27.csv
        ../outputs/forecasts/forecast_104wk_extended.csv
        ../figures/forecasts/fig_forecast_validation.png
"""
import pandas as pd, numpy as np, pickle
