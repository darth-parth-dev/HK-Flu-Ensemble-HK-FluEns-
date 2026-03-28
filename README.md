# HK Flu Ensemble (HK-FluEns)

**Hong Kong Influenza Forecasting Network**

[![Status: Planning Phase](https://img.shields.io/badge/Status-Planning%20Phase-yellow)](docs/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](scripts/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

HK-FluEns is a hub-and-spoke probabilistic influenza forecasting network for Hong Kong, structurally inspired by the [CDC FluSight network](https://github.com/cdcepi/FluSight-forecast-hub) but adapted to Hong Kong's unique epidemiological context: bimodal seasonality (winter and summer peaks), subtropical climate drivers, PDF-based CHP surveillance data, and an emerging wastewater surveillance programme.

The project is currently in **Planning Phase**. The immediate focus is reconstructing a reproducible baseline pipeline before building the full hub infrastructure.

---

## Repository Structure

```
HK-FluEns/
│
├── README.md                          ← You are here
├── .gitignore
├── LICENSE
│
├── scripts/                           ← Baseline SARIMA + ensemble pipeline (Phase 0)
│   ├── 01_data_preparation.py         Step 1: Load CHP data, compute ILI+, handle COVID gap
│   ├── 02_stationarity_acf_pacf.py    Step 2: ADF test, ACF/PACF, order selection
│   ├── 03_model_fitting.py            Step 3: Fit SARIMA candidates, select by AIC/BIC
│   ├── 04_residual_diagnostics.py     Step 4: Residual diagnostic panel
│   ├── 05_forecast_generation.py      Step 5: Bootstrap forecast + 23-quantile output
│   ├── 06_figures_and_narrative.py    Step 6: Publication figures
│   ├── 07_ensemble_model.py           Step 7: 3-model ensemble (SARIMA + ETS + Benchmark)
│   └── 08_ensemble_evaluation.py      Step 8: Rolling-origin CV, WIS scoring
│
├── models/                            ← Model implementations (hub spoke layer)
│   └── baseline_sarima/               Baseline SARIMA model — Phase 0 reference
│
├── hub/                               ← Hub infrastructure (Phase 1+)
│   ├── model-output/                  Validated FluSight-format submissions from all models
│   ├── target-data/                   Ground-truth weekly observations (CHP + FluNet)
│   ├── ensemble-output/               AWBE ensemble forecasts
│   └── evaluation/                    WIS scores, wave-stratified metrics, model weights
│
├── data-pipeline/                     ← Automated ETL (Phase 1)
│   │                                  CHP scraper, FluNet pull, HKO weather feed
│   └── (coming in Phase 1)
│
├── scoring/                           ← WIS and evaluation utilities (Phase 0/1)
│   └── (coming in Phase 0)
│
├── data/
│   ├── raw/
│   │   ├── raw_chp_weekly.xlsx        CHP Weekly Influenza Surveillance 2014-2026
│   │   ├── raw_flunet_hk.csv          WHO FluNet Hong Kong 1997-2026
│   │   └── flunet_data_dictionary.csv FluNet column definitions
│   └── processed/
│       ├── processed_master_dataset.csv     All sources merged (636 weeks × 55 cols)
│       ├── processed_model_ready.csv        Log-transformed series + regressors
│       ├── processed_covid_imputed_series.csv  COVID-gap filled ILI+ series
│       └── processed_hk_flu_clean.csv       Clean CHP series 2019-2026
│
├── outputs/
│   ├── forecasts/                     Model forecast outputs (CSV)
│   ├── metrics/                       CV results, ensemble weights, evaluation summaries
│   └── coefficients/                  Fitted model parameters (JSON)
│
├── figures/
│   ├── diagnostics/                   Model diagnostic plots
│   ├── forecasts/                     Forecast visualisations
│   └── comparison/                    Model comparison figures
│
└── docs/
    ├── HK_FluEns_Planning_Document_March2026.docx   Full planning document
    ├── model-submission-spec.md        (coming Phase 1) FluSight-format submission spec
    └── hub-governance.md              (coming Phase 3) Governance and participation terms
```

---

## Forecast Targets

| Target | Definition | Priority |
|--------|------------|----------|
| `lab_positivity_pct` | Weekly lab positivity rate: positive specimens / total specimens (%) | Primary |
| `gopc_ili_rate_per1000` | GOPC ILI consultations per 1,000 visits | Secondary |

**Forecast horizons:** 1, 2, 3, 4 weeks ahead
**Output format:** 23 quantiles (0.01 – 0.99) per horizon, FluSight CSV format
**Update cadence:** Weekly, year-round (no seasonal window cutoff)

---

## Current Ensemble Design

| Component | Method | Weight | Role |
|-----------|--------|--------|------|
| Model A | SARIMA(1,0,1)(1,1,0)[52] + AH regressors | Adaptive (AWBE) | Seasonal timing anchor |
| Model B | ETS(A,A,A) — additive Holt-Winters | Adaptive (AWBE) | Adaptive seasonal updating |
| Model C | Historical Percentile Benchmark (2014-2019) | Adaptive (AWBE) | Magnitude anchor |

**Ensemble method:** Adaptive Weight Blending Ensemble (AWBE) — re-weights based on rolling Weighted Interval Score (WIS) over a sliding 8-week window. Falls back to equal weights when history < 16 weeks.

**Post-hoc strain multiplier:** H3N2 ×1.65 | H1N1pdm09 ×0.85 | Influenza B ×0.60 (derived from FluNet HK historical strain-severity ratios).

---

## Epidemiological Context

Hong Kong presents a distinct set of forecasting challenges not present in temperate-climate systems like US FluSight:

- **Bimodal seasonality:** Two distinct influenza peaks annually — winter (January–March) and summer (June–August)
- **Subtropical climate drivers:** Absolute humidity drives transmission non-linearly; hot-humid summers suppress winter-pattern viruses while summer strains circulate
- **Year-round surveillance:** Forecasting cannot be bounded by a "flu season" window; the system operates 52 weeks per year
- **COVID structural break:** Influenza was nearly absent from April 2020 to September 2022, requiring imputation and careful handling of the post-COVID baseline shift
- **School term effects:** School-age children are disproportionately affected; school term/holiday calendar is a strong modulator of incidence
- **Cross-border dynamics:** High HK–Mainland China travel volumes introduce cross-border importation risk

---

## Evaluation Framework

| Metric | Definition | Primary Use |
|--------|------------|-------------|
| **WIS** | Weighted Interval Score (FluSight primary) | AWBE re-weighting; overall comparison |
| MAE | Mean Absolute Error (ILI+ units) | Point forecast accuracy |
| Peak timing error | Forecast peak week − observed peak week (weeks) | Wave detection accuracy |
| Peak magnitude error | % error at the forecast peak | Severity calibration |
| 80% / 95% PI coverage | Empirical coverage of prediction intervals | Calibration check |
| Wave-stratified WIS | WIS computed separately for winter/summer peaks and troughs | Seasonal performance |

**Validation protocol:** Leave-one-season-out cross-validation over ≥5 complete epi-years (Oct–Sep), scoring non-imputed weeks only.

---

## Implementation Roadmap

| Phase | Weeks | Focus | Status |
|-------|-------|-------|--------|
| **Phase 0: Foundation** | 1-4 | Reconstruct scripts 01-06; WIS scoring; FluSight quantile output | 🔄 In Progress |
| **Phase 1: Core Infrastructure** | 5-14 | Automated ETL; target variable agreement; hub skeleton | ⏳ Pending |
| **Phase 2: Model Enhancement** | 15-28 | Novel data streams; AWBE; gradient boosting model; LOSO CV | ⏳ Pending |
| **Phase 3: Hub & Distribution** | 29-42 | Public API; external contributors; CHP dashboard | ⏳ Pending |
| **Phase 4: Governance** | 43+ | CHP MOU; RGC/ITF grant; publications | ⏳ Pending |

See [`docs/HK_FluEns_Planning_Document_March2026.docx`](docs/HK_FluEns_Planning_Document_March2026.docx) for the full planning document including risk assessment and open decision points.

---

## Data Sources

| Source | Coverage | Access |
|--------|----------|--------|
| [CHP Flu Express](https://www.chp.gov.hk/en/statistics/data/10/641/642.html) | 2000–present, weekly | Public PDF → ETL pipeline |
| [WHO FluNet](https://www.who.int/tools/flunet) | 1997–present, weekly | Public API (CSV download) |
| [HKO Climate Data](https://www.hko.gov.hk/en/cis/climat.htm) | 1991–present | Open data API |
| Wastewater Surveillance | TBD | Data-sharing agreement required |
| HK–Mainland travel volumes | 2010–present | IMMD / HKIA open data |

---

## Software Requirements

```
Python 3.10+
numpy >= 1.24
pandas >= 1.5
scipy >= 1.10
matplotlib >= 3.7
```

All models are implemented from scratch using `scipy.optimize` (L-BFGS-B). No `statsmodels` dependency — this ensures reproducibility across Python versions.

---

## How to Run (Phase 0 Baseline Pipeline)

```bash
cd scripts/

# Step 1-6: SARIMA pipeline
python 01_data_preparation.py
python 02_stationarity_acf_pacf.py
python 03_model_fitting.py
python 04_residual_diagnostics.py
python 05_forecast_generation.py    # → outputs 23-quantile FluSight CSV
python 06_figures_and_narrative.py

# Step 7-8: Ensemble
python 07_ensemble_model.py
python 08_ensemble_evaluation.py    # → WIS scores + acceptance verdict
```

---

## Contributing

HK-FluEns is designed as a multi-institutional hub. We plan to onboard model contributors from HKU, CUHK, PolyU, and other Hong Kong institutions in Phase 3. See `docs/model-submission-spec.md` (coming Phase 1) for submission format requirements.

**Contact:** Parth Singh | CityU School of Public Health
**Supervisor:** Mr. Chen JunQiao

---

## Acknowledgements

This project draws structural inspiration from the [CDC FluSight Network](https://github.com/cdcepi/FluSight-forecast-hub). The AWBE ensemble strategy follows the approach validated in published Hong Kong influenza forecasting research. Surveillance data is sourced from CHP, WHO FluNet, and HKO.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
