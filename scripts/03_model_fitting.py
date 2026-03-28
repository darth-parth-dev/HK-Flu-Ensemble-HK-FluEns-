"""
03_model_fitting.py
===================
Step 3: Fit 4 candidate SARIMA/ARIMAX models, compare by AIC/BIC/LB-p,
save preferred model object.

Input:  ../data/processed/processed_model_ready.csv
Output: step3_preferred_model.pkl
        Printed model comparison table
"""
import pandas as pd, numpy as np
from scipy import optimize, stats
import pickle
