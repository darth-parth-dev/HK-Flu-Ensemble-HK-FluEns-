"""
04_residual_diagnostics.py
==========================
Step 4: Full residual diagnostic panel — standardised residuals,
ACF, Q-Q plot, histogram, Ljung-Box, Shapiro-Wilk.

Input:  step3_preferred_model.pkl
Output: ../figures/diagnostics/fig_residual_diagnostics.png
        Printed pass/fail summary
"""
import pickle, numpy as np
from scipy import stats
