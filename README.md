# Olympic Medal Forecasting with Recency-Aware Machine Learning

A reproducible quantitative forecasting pipeline for **Summer Olympic medal counts**. The project first reproduces the Grid-Search Random Forest (GSRF) framework from an MCM/ICM 2025 Problem C paper, then extends it with **recency-weighted features, out-of-time validation, Poisson count modeling, and sensitivity analysis**.

> **Project context.** This was completed as a collaborative academic modeling project. The baseline methodology is reproduced from the referenced MCM/ICM paper; the recency-aware EMA/Hybrid specifications, robustness analysis, and Poisson extension are subsequent modeling extensions in this repository.

## Why this project is interesting

Olympic performance is a non-stationary forecasting problem: delegation size, host effects, event availability, long-run national strength, and recent momentum all change over time. A cumulative-history model can overweight results from decades ago, so this project explicitly tests whether **recent performance should receive more weight**.

The workflow is deliberately structured like a small quantitative research pipeline:

1. construct country–Olympics panel data;
2. engineer historical and participation features;
3. train models only on information available before the test Olympics;
4. hold out **2024 as an out-of-time test set**;
5. compare competing specifications;
6. run hyperparameter and sensitivity analysis;
7. generate forward-looking 2028 forecasts.

## Methods

### 1. Baseline GSRF reproduction

Separate `RandomForestRegressor` models forecast **gold medals** and **total medals** using:

- athlete/delegation count,
- cumulative historical medal counts,
- total Olympic events,
- host-country indicator.

Hyperparameters are selected using `GridSearchCV`, with cross-validation performed on pre-2024 data.

### 2. Recency-aware historical signals

The enhanced pipeline replaces the assumption that all historical Olympics are equally informative with **exponential moving-average (EMA)** features.

Three specifications are compared:

- **Cumulative** — long-run medal history only;
- **EMA** — recency-weighted history only;
- **Hybrid** — both cumulative and EMA signals.

The EMA decay parameter alpha is stress-tested across multiple values rather than selected from a single arbitrary choice.

### 3. First-time medal modeling

For countries with no prior medals, the enhanced pipeline uses **Poisson regression** to model medal counts from participation-based covariates. The fitted expected count lambda is converted into a probability of winning at least one medal:

`P(Y >= 1) = 1 - exp(-lambda)`

The repository also contains the paper-faithful logistic-regression implementation for comparison.

### 4. Sport attribution and coaching model

Additional components reproduce the paper's:

- country–sport medal-importance analysis;
- Lasso-regularized **"Great Coach"** model;
- perturbation-based sensitivity analysis.

## Out-of-time results

Using **2024 as the held-out test Olympics**, the recency-aware models materially improve predictive performance relative to the cumulative baseline.

| Target | Model | Test R² | MAE |
|---|---:|---:|---:|
| Gold medals | Hybrid | **0.948** | **0.404** |
| Total medals | Hybrid | **0.979** | **0.703** |

The EMA-only specification also outperformed the cumulative baseline, supporting the hypothesis that **recent Olympic performance contains incremental predictive signal**. Alpha-sensitivity and k-fold experiments are saved under `Output_Enhanced/`.

These are historical holdout results, not guarantees of future forecasting accuracy.

## Repository structure

```text
.
├── Data/                         # Olympic source data
├── Model/
│   ├── preprocess_paper.py
│   ├── GSRF.py
│   ├── run_paper_pipeline.py
│   ├── run_paper_pipeline_enhanced.py
│   ├── model_comparison.py
│   ├── alpha_sensitivity.py
│   ├── kfold_sensitivity.py
│   ├── poisson_medal_count.py
│   ├── LassoRegression.py
│   └── sensitivity.py
├── Output/                       # Baseline outputs
├── Output_Enhanced/              # Enhanced-model diagnostics and forecasts
├── Markdowns/                    # Reproduction and methodology notes
├── verify_pipeline.py
├── verify_enhanced_pipeline.py
└── requirements.txt
```

## Reproduce the analysis

### Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Paper-faithful pipeline

```bash
python Model/run_paper_pipeline.py
python verify_pipeline.py
```

### Enhanced pipeline

```bash
python Model/run_paper_pipeline_enhanced.py
python verify_enhanced_pipeline.py
```

The enhanced pipeline writes model comparisons, alpha/k-fold sensitivity tables, feature importances, Poisson first-medal estimates, and 2028 forecasts to `Output_Enhanced/`.

## Selected techniques

`Python` · `pandas` · `NumPy` · `scikit-learn` · `Random Forests` · `Grid Search` · `Cross-Validation` · `Time-Weighted Features` · `Poisson Regression` · `Logistic Regression` · `Lasso` · `Feature Importance` · `Sensitivity Analysis` · `Out-of-Time Validation`

## Attribution

The baseline reproduction follows the methodology of **"Olympic Medals Unveiled: A Mathematical Exploration of Achievement Trends"** (MCM/ICM 2025, Problem C). The repository contains reproduction notes under `Markdowns/` and clearly separates the paper-faithful pipeline from the enhanced modeling pipeline.

This repository is intended as an academic modeling and reproducibility project.
