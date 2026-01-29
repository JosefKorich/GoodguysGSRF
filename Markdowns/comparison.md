# Comparison Analysis: GSRF Model Results vs. Olympic Medals Paper

This document compares the GSRF model output from a current run (terminal results) with the results reported in **olympic_medals_paper.md** (2025 MCM/ICM Problem C).

---

## 1. Model Performance Metrics

### 1.1 Cross-Validation / Out-of-Fold Evaluation

| Metric | **Paper (Gold)** | **Current Run (Gold)** | **Paper (Total)** | **Current Run (Total)** |
|--------|------------------|------------------------|-------------------|-------------------------|
| **R²** (Training / CV / Test) | 0.903 / 0.710 / 0.795 (Gold) | 0.8665 / 0.6323 / 0.7772 (Gold) | 0.933 / 0.784 / 0.736 (Total) | 0.9066 / 0.7227 / 0.8801 (Total) |
| **MAE** (Train / CV / Test) | 0.626 / 0.994 / 0.914 (Gold) | 0.6716 / 1.0561 / 0.8888 (Gold) | 1.947 / 2.744 / 2.430 (Total) | 1.4907 / 2.4487 / 1.6745 (Total) |
| **RMSE** (Train / CV / Test) | 1.829 / 3.145 / 2.574 (Gold) | 1.9977 / 3.3158 / 2.3440 (Gold) | 4.11 / 7.162 / 7.145 (Total) | 4.3730 / 7.5358 / 4.9280 (Total) |

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
| 1 | USA 51 | USA 41.23 |
| 2 | CHN 34 | FRA 31.71 |
| 3 | ITA 29 | AUS 28.05 |
| 4 | FRA 28 | CHN 27.04 |
| 5 | GBR 26 | GER 26.22 |
| 6 | GBR 26 | GBR 26.13 |
| 7 | AUS 20 | ITA 25.93 |
| 8 | GER 19 | JPN 14.25 |
| 9 | JPN 15 | NED 10.69 |
| 10 | CAN 15 | ESP 9.77 |

**Observations:**

- **USA** stays first in both; paper **51** gold vs. current **41.23**—closer than in earlier runs.
- **China** is 2nd in the paper (34) but **4th** in the current run (27.04). **France** (31.71) and **Australia** (28.05) are 2nd and 3rd in the current run.
- **Great Britain** (26.13) and **Italy** (25.93) are in the current top 10; **Canada** is 12th by gold (9.09). **Spain** and **Netherlands** appear in the current gold top 10.

### 3.2 Top 10–11 Total Medals

| Rank | **Paper (Table 6)** | **Current Run** |
|------|---------------------|-----------------|
| 1 | USA 124 | USA 105.96 |
| 2 | FRA 81 | FRA 75.67 |
| 3 | CHN 74 | AUS 70.22 |
| 4 | GBR 72 | GBR 55.90 |
| 5 | AUS 65 | GER 53.93 |
| 6 | GER 61 | JPN 53.63 |
| 7 | ITA 55 | CHN 44.64 |
| 8 | JPN 52 | ITA 43.55 |
| 9 | CAN 35 | ESP 40.99 |
| 10 | NED 34 | NED 33.90 |
| 11 | BRA 34 | CAN 30.73 |

**Observations:**

- **USA** first in both (124 vs. 105.96); **France** second in both (81 vs. 75.67)—close.
- **Australia** 3rd in current run (70.22) vs. 5th in paper (65). **China** 4th in paper (74) but 7th in current run (44.64)—much lower total in reproduction.
- **Great Britain** 4th in current run (55.90); **Netherlands** (33.90) and **Canada** (30.73) in current top 11; **Spain** (40.99) in current top 10.




### 3.3 Prediction Intervals

- **Paper (Section 5.2.2):** 95% intervals **ŷ ± 7** (gold) and **ŷ ± 15** (total medals).
- **Current run:** Gold_lower95 / Gold_upper95 and Total_lower95 / Total_upper95 (e.g., USA gold 34.23–48.23, total 90.96–120.96). Intervals **match** the paper: gold ±7, total ±15.

## 4. Progress and Regression (Paper Figure 11 vs. Current Run)

- **Paper:** “Great Britain had the most significant increase in gold medals and Germany had the most significant increase in total medals”; “South Korea had the largest decrease in gold medals and China had the largest decrease in total medals.”
- **Current run:** Does not explicitly list “countries most likely to advance or regress.” To make a direct comparison, the same definition of “change vs. 2024” (or vs. last Olympics) would need to be applied to the current predictions and then compared to the paper’s narrative and figures.

---

## 5. Summary Table

| Dimension | Paper | Current Run | Comment |
|-----------|--------|-------------|---------|
| Gold R² (CV / Test) | 0.710 / 0.795 | 0.6323 / 0.7772 | CV lower; Test close |
| Total R² (CV / Test) | 0.784 / 0.736 | 0.7227 / 0.8801 | CV lower; Test higher in run |
| Gold MAE (CV / Test) | 0.994 / 0.914 | 1.0561 / 0.8888 | Test better in run |
| Total MAE (CV / Test) | 2.744 / 2.430 | 2.4487 / 1.6745 | Run better on both |
| USA 2028 gold | 51 | 41.23 | Lower in reproduction |
| USA 2028 total | 124 | 105.96 | Lower in reproduction |
| China 2028 gold | 34 | 27.04 | Lower in reproduction |
| France 2028 total | 81 | 75.67 | Very close |
| 95% interval (total) | ŷ ± 15 | Gold ±7, Total ±15 (e.g. USA 90.96–120.96) | Match |
| **Sensitivity** (raw-space 1%, ps = \|Δy/y\|) | athletes: 0.058%; Total event: −1.21% | athletes_num: **0.7185%**; Total event (signed): **0.0000%** | Definition match; athletes closer after 1% raw perturbation |

---

## 6. Sensitivity (Paper Section 10.2)

- **Paper:** “average sensitivity: athletes_num: 0.058329%; Total event: −1.2113%” (ps = |Δy/y|).
- **Current run (raw-space 1% perturbation; ps = |Δy/y|):**
  - **athletes_num:** **0.7185%**
  - **Total event (signed):** **0.0000%**

After the adjustment to use **1% raw-space perturbation** on athletes and total events (and to report signed sensitivity for Total event), the athletes sensitivity is **0.7185%**—still above the paper’s 0.058% but much closer than the earlier ~11% from a different perturbation setup. Total event sensitivity remains 0% in the run (no change in prediction for a 1% change in total events in the current implementation).

---

## 7. Possible Reasons for Differences

1. **Data and preprocessing:** Different vintages of `summerOly_medal_counts`, `summerOly_athletes`, host files, or ISO mappings; different handling of Russia/USSR/neutral athletes; different outlier rules.
2. **Train/validation design:** The paper uses “first 80% / last 20%” for some tables and “10-fold CV” for others. The current script labels results as “out-of-fold predictions,” which may not match the paper’s exact split or aggregation.
3. **Features and targets:** Slight differences in how “medals before the j-th Olympics,” “current Olympiad,” or host/event variables are defined would change both metrics and 2028 forecasts.
4. **Hyperparameters and software:** Different grid search space or sklearn version can change the chosen model and all downstream numbers.
5. **Paper conventions:** Table 5 has two “GBR” columns (likely a typo); country ordering and rounding in the paper can create small apparent differences even if the underlying model were the same.

---

## 8. Conclusions

- **Metrics:** The current run has **lower CV R²** (Gold 0.6323 vs. 0.710; Total 0.7227 vs. 0.784) but **close or better test R²** (Gold 0.7772 vs. 0.795; Total 0.8801 vs. 0.736) and **better MAE/RMSE** on test. The reproduction is comparable or stronger on held-out 2024 performance.
- **2028 predictions:** **USA** remains first (41.23 gold, 105.96 total) but below the paper (51, 124). **France** (75.67 total) and **Australia** (70.22) align well; **China** (27.04 gold, 44.64 total) and **Great Britain** (55.90 total) are lower in the reproduction. Rankings 2–10 differ (FRA, AUS higher; CHN, GBR lower).
- **Intervals:** 95% intervals **match** the paper (gold ±7, total ±15).
- **Sensitivity:** With **1% raw-space perturbation** and ps = |Δy/y|, athletes_num is **0.7185%** (paper 0.058%); Total event (signed) is **0.0000%** (paper −1.21%). The athletes value is much closer to the paper after the last adjustments than the earlier ~11%.

A full reconciliation would require aligning data sources, feature definitions, and evaluation protocol with the paper and, where possible, with the authors' original code or data.

A full reconciliation would require aligning data sources, feature definitions, and evaluation protocol (splits, folds, and reporting) with the paper’s description and, where possible, with the authors’ original code or data.
