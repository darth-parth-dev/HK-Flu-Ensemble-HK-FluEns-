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
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

# Load CHP data
xl  = pd.ExcelFile('../data/raw/raw_chp_weekly.xlsx')
raw = xl.parse('Weekly surveillance data 每周監測數據', header=None)
# ... (see full pipeline in session notes)
