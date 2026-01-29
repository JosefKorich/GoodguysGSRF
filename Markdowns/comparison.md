# Comparison Analysis: GSRF Model Results vs. Olympic Medals Paper

This document compares the GSRF model output from a current run (terminal results) with the results reported in **olympic_medals_paper.md** (2025 MCM/ICM Problem C).

---

## 1. Model Performance Metrics

### 1.1 Cross-Validation / Out-of-Fold Evaluation

| Metric | **Paper (Gold)** | **Current Run (Gold)** | **Paper (Total)** | **Current Run (Total)** |
|--------|------------------|------------------------|-------------------|-------------------------|
| **R²** | 0.710 (CV set) | 0.625 | 0.784 (CV set) | 0.706 |
| **MAE** | 0.994 (CV) | 0.867 | 2.744 (CV) | 2.075 |
| **RMSE** | 3.145 (CV) | 3.033 | 7.162 (CV) | 7.124 |

**Findings:**

- **Gold model:** The current run has **lower R²** (0.625 vs. 0.710) but **better MAE and RMSE** (0.867 vs. 0.994, 3.033 vs. 3.145). The paper’s summary reports Gold R² = 0.710; the reproduction is ~0.08 lower in R² but slightly more accurate in absolute error.
- **Total model:** The current run has **slightly lower R²** (0.706 vs. 0.784) and **better MAE** (2.075 vs. 2.744), with **very similar RMSE** (7.124 vs. 7.162). Again, the reproduction is somewhat worse in explained variance but comparable or better in error magnitude.

**Interpretation:** The reproduction likely uses a different train/validation split or methodology (e.g., “out-of-fold” vs. the paper’s “cross validation set”). The paper also reports separate training, CV, and test metrics; the current script appears to report a single cross-validated evaluation. Differences in data preprocessing, feature construction, or hyperparameters (see Section 2) can explain the R² gap while still yielding similar RMSE/MAE.

---

## 2. Hyperparameters

The paper does not state the chosen Random Forest hyperparameters. The current run reports:

- **Gold model:** `max_depth=10`, `min_samples_leaf=2`, `min_samples_split=5`, `n_estimators=500`
- **Total model:** `max_depth=10`, `min_samples_leaf=2`, `min_samples_split=5`, `n_estimators=200`

Any difference in grid definition or search strategy between the paper’s implementation and the current code would affect both metrics and predictions.

---

## 3. 2028 Medal Predictions

### 3.1 Top 10 Gold Medals

| Rank | **Paper (Table 5)** | **Current Run** |
|------|---------------------|-----------------|
| 1 | USA 51 | United States 32.18 |
| 2 | CHN 34 | France 26.34 |
| 3 | ITA 29 | Australia 25.64 |
| 4 | FRA 28 | China 20.91 |
| 5 | GBR 26 | Japan 19.50 |
| 6 | AUS 20 | Great Britain 11.55 |
| 7 | GER 19 | Germany 18.65 |
| 8 | JPN 15 | Italy 15.54 |
| 9 | CAN 15 | Spain 11.81 |
| 10 | — | Netherlands 9.53 |

**Observations:**

- **USA** stays first in both, but the paper predicts **51** gold vs. current **~32**—a large gap.
- **China** is 2nd in the paper (34 gold) but **4th** in the current run (~21 gold).
- **France** and **Australia** are much higher in the current run (2nd and 3rd by gold) than in the paper.
- **Italy** is 3rd in the paper (29) but 8th in the current run (~15.5). **Great Britain** drops from 5th (26) to 6th (~11.5).
- The current run adds **Spain** and **Netherlands** in the gold top 10; **Canada** is in the paper’s top 10 but not in the current run’s gold top 10 (7.23 gold, 12th by gold).

### 3.2 Top 10–11 Total Medals

| Rank | **Paper (Table 6)** | **Current Run** |
|------|---------------------|-----------------|
| 1 | USA 124 | United States 99.84 |
| 2 | FRA 81 | France 79.22 |
| 3 | CHN 74 | Australia 66.55 |
| 4 | GBR 72 | China 55.72 |
| 5 | AUS 65 | Japan 49.72 |
| 6 | GER 61 | Great Britain 49.13 |
| 7 | ITA 55 | Germany 46.70 |
| 8 | JPN 52 | Italy 44.49 |
| 9 | CAN 35 | Spain 36.68 |
| 10 | NED 34 | Netherlands 29.05 |
| 11 | BRA 34 | Brazil 27.53 |

**Observations:**

- **USA** is first in both, but the paper gives **124** total medals vs. current **~100**—about 24 medals lower in the reproduction.
- **France** is second in both, with very close totals (81 vs. 79.22).
- **Australia** rises to 3rd in the current run (66.55) vs. 5th in the paper (65); **China** falls to 4th (55.72 vs. 74).
- **Great Britain** is 4th in the paper (72) but 6th in the current run (49.13). **Japan** is 5th in the current run (49.72) vs. 8th in the paper (52).
- **Germany, Italy, Japan** are in similar “middle” ranks in both, but with generally lower totals in the current run.
- **Canada, Netherlands, Brazil** are in the paper’s top 10–11; the current run has them in the next ranks with lower totals (e.g., Canada 24.01, Netherlands 29.05, Brazil 27.53).

### 3.3 Prediction Intervals

- **Paper (Section 5.2.2):** 95% intervals given as **ŷ ± 7** for gold and **ŷ ± 15** for total medals (from t-distribution with reported SE and degrees of freedom).
- **Current run:** Prints team-level **TotalMedals_lower95** and **TotalMedals_upper95** (e.g., USA 85.87–113.80, France 65.25–93.18). Widths are about **±14** for the USA total, consistent with a “rough 95% interval” on the same order as the paper’s ±15 for totals.

The paper illustrates these intervals in “Figure 10: Error bar chart of prediction model”; the current run provides numeric bounds in tabular form.

---

## 4. Progress and Regression (Paper Figure 11 vs. Current Run)

- **Paper:** “Great Britain had the most significant increase in gold medals and Germany had the most significant increase in total medals”; “South Korea had the largest decrease in gold medals and China had the largest decrease in total medals.”
- **Current run:** Does not explicitly list “countries most likely to advance or regress.” To make a direct comparison, the same definition of “change vs. 2024” (or vs. last Olympics) would need to be applied to the current predictions and then compared to the paper’s narrative and figures.

---

## 5. Summary Table

| Dimension | Paper | Current Run | Comment |
|-----------|--------|-------------|---------|
| Gold R² (CV/oof) | 0.710 | 0.625 | Lower in reproduction |
| Total R² (CV/oof) | 0.784 | 0.706 | Lower in reproduction |
| Gold MAE | 0.994 | 0.867 | Better in reproduction |
| Total MAE | 2.744 | 2.075 | Better in reproduction |
| Gold RMSE | 3.145 | 3.033 | Slightly better in reproduction |
| Total RMSE | 7.162 | 7.124 | Very similar |
| USA 2028 gold | 51 | ~32 | Much lower in reproduction |
| USA 2028 total | 124 | ~100 | Lower in reproduction |
| China 2028 gold | 34 | ~21 | Lower in reproduction |
| France 2028 total | 81 | ~79 | Very close |
| 95% interval (total) | ŷ ± 15 | Team-specific (~±14 for USA) | Consistent order of magnitude |

---

## 6. Possible Reasons for Differences

1. **Data and preprocessing:** Different vintages of `summerOly_medal_counts`, `summerOly_athletes`, host files, or ISO mappings; different handling of Russia/USSR/neutral athletes; different outlier rules.
2. **Train/validation design:** The paper uses “first 80% / last 20%” for some tables and “10-fold CV” for others. The current script labels results as “out-of-fold predictions,” which may not match the paper’s exact split or aggregation.
3. **Features and targets:** Slight differences in how “medals before the j-th Olympics,” “current Olympiad,” or host/event variables are defined would change both metrics and 2028 forecasts.
4. **Hyperparameters and software:** Different grid search space or sklearn version can change the chosen model and all downstream numbers.
5. **Paper conventions:** Table 5 has two “GBR” columns (likely a typo); country ordering and rounding in the paper can create small apparent differences even if the underlying model were the same.

---

## 7. Conclusions

- **Metrics:** The current GSRF run is **slightly worse in R²** (Gold and Total) but **similar or better in MAE/RMSE** compared to the paper’s CV metrics. That suggests a comparable or slightly different choice of model or evaluation scheme rather than a strictly “worse” model.
- **2028 predictions:** The reproduction gives **substantially lower** gold and total medals for **USA** and **China**, and **different rankings** (e.g., Australia, France, and Japan higher; China and Great Britain lower). **France** and **Australia** are especially elevated in the current run relative to the paper.
- **Intervals:** The 95% intervals for total medals in the current run are in line with the paper’s reported **ŷ ± 15** for total medals.

A full reconciliation would require aligning data sources, feature definitions, and evaluation protocol (splits, folds, and reporting) with the paper’s description and, where possible, with the authors’ original code or data.
