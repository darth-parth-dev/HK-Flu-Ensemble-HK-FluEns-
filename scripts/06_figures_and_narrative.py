"""
06_figures_and_narrative.py
============================
Step 6: Generate publication-quality figures and 3-sentence narrative text
for the review background section.

Input:  ../data/processed/processed_master_dataset.csv
        ../outputs/forecasts/forecast_78wk_2026_27.csv
        ../outputs/forecasts/historical_percentile_envelope.csv
Output: ../figures/forecasts/fig_main_forecast.png
        ../figures/forecasts/fig_season_detail_panels.png
        Printed narrative text
"""
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
