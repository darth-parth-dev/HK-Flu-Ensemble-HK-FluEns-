"""
02_stationarity_acf_pacf.py
============================
Step 2: ADF stationarity tests, ACF/PACF diagnostics, order confirmation.

Input:  ../data/processed/processed_model_ready.csv
Output: ../figures/diagnostics/fig_acf_pacf_diagnostics.png
        Printed order recommendation: SARIMA(1,0,1)(1,1,0)[52]
"""
import pandas as pd, numpy as np
from scipy import stats
