# Model Reproduction Analysis: Paper vs. Repo

## 1. Executive Summary

The repository implements portions of the pipeline described in **olympic_medals_paper.md** (GSRF, logistic regression, Lasso “Great Coach,” sport importance, sensitivity) but **deviates in several core design choices**: Task 1 omits cumulative medals (G_ij, T_ij) and standardization; Task 2 uses different features and is not wired to the data pipeline; Task 3 is partial; Task 4 uses a simplified coach variable and different country/sport cases; sensitivity is not integrated. Under the scoring heuristic below, the reproduction is **~47% faithful**.

---

## 2. Score Heuristic

- **Scope:** One score per “component” (equations, features, train/test design, outputs).
- **Weights:** By importance to the stated model:
  - **Critical (weight 3):** Correct feature set, correct target definition, core equations.
  - **Important (weight 2):** Train/test design, evaluation metrics, preprocessing.
  - **Supporting (weight 1):** Visuals, exact tables, minor formula details.
- **Score per component:**
  - **1.0** = matches paper (same features/equations/splits/design).
  - **0.5** = partial (same idea, different params or minor omissions).
  - **0.0** = missing, wrong, or opposite to paper.
- **Overall:**

\[
\text{Match \%} = 100 \times \frac{\sum (\text{component score} \times \text{weight})}{\sum \text{weight}}.
\]

---

## 3. Exact Matches

| # | Component | Paper reference | Repo location | Weight | Score |
|---|-----------|-----------------|---------------|--------|-------|
| 1 | Random Forest + GridSearchCV for medals | §5.1.2, Eq (2), Table 2 | `GSRF.py`: `RandomForestRegressor`, `GridSearchCV` | 3 | 1.0 |
| 2 | 10-fold cross-validation | §5.1.2, Figure 5 | `GSRF.py`: `KFold(n_splits=10)` | 2 | 1.0 |
| 3 | Separate models for gold vs total | Eq (2): Y_g, Y_t | `GSRF.py`: `grid_gold`, `grid_total` | 3 | 1.0 |
| 4 | Use of Athletes (A_ij) and Host (H_ij) | §5.1.2, Table 2 | `GSRF.py`: `Athletes`, `Host` in `features` | 3 | 1.0 |
| 5 | Events total per year (E_j) | §5.1.2 | `GSRF.py`: `build_events_per_year`, `EventsTotal` | 2 | 1.0 |
| 6 | 2028 host = USA (Los Angeles) | §5.2.1 | `GSRF.py`: `Host = (Team == "United States")` | 1 | 1.0 |
| 7 | Prediction interval using t and residual spread | §5.2.2, Eq (6) | `GSRF.py`: `t.ppf(0.975, df)`, residual std | 2 | 0.5 |
| 8 | Sport importance I_j = m_k / M_k | §7.2, Eq (13) | `Medalchart.py`: `Importance = MedalCount / TotalMedals` | 3 | 1.0 |
| 9 | Medal points (10, 6, 3) | Table 13 | `LassoRegression.py`: `Gold*10 + Silver*6 + Bronze*3` | 2 | 1.0 |
| 10 | Lasso regression for “Great Coach” | §8.1.2, Eq (15)–(18) | `LassoRegression.py`: `Lasso`, L1 | 3 | 1.0 |
| 11 | Sensitivity as \|Δy/y\| | §10.1, Eq (20) | `sensitivity.py`: percent change in predictions | 2 | 1.0 |
| 12 | Binary logistic regression for non-medal countries | §6.1, Eq (7)–(10) | `Regression.py`: `LogisticRegression`, sigmoid via predict_proba | 3 | 1.0 |
| 13 | L1 regularization in logistic model | §6, “L1 regularization” | `Regression.py`: `penalty='l1', solver='liblinear'` | 2 | 1.0 |

---

## 4. Mismatches and Missing Pieces

### 4.1 Task 1 – GSRF (critical)

| # | Component | Paper | Repo | Weight | Score |
|---|-----------|------|------|--------|-------|
| 14 | **Cumulative medals G_ij, T_ij** | Eq (2), Table 2: “gold medals … **before** the j-th Olympics,” “total medals … **before** the j-th” | Only current-year medals used as targets; **no cumulative G_ij or T_ij as features** | 3 | 0.0 |
| 15 | **Feature set** | Four inputs: A_ij, G_ij (or T_ij), E_j, H_ij | Three: Athletes, EventsTotal, Host — **G_ij and T_ij missing** | 3 | 0.0 |
| 16 | **Data standardization** | Eq (0): x' = (x − x̄) / SD | No standardization of features in `GSRF.py` | 2 | 0.0 |
| 17 | **Train/test split** | “Data prior to 2024 as training, 2024 as test”; also “first 80%, second 20%” for R²/MAE/RMSE | All years in CV; no explicit “pre-2024 train / 2024 test” or 80/20 split | 2 | 0.0 |
| 18 | **Prediction intervals** | Eq (6), Table 8: SE(ŷ), t_{α/2,n-p}; gold ±7, total ±15 | Single interval from total-medal residual std; no paper-style SE(ŷ) or ±7/±15 | 2 | 0.5 |
| 19 | **Progress/regression (Fig 11)** | “Difference between predicted and 2024”; “countries most likely to advance and regress” | Not implemented | 1 | 0.0 |
| 20 | **ARIMA pre-step** | §5.1.1, Eq (1), Table 1 | Not implemented (paper then rejects ARIMA for GSRF) | 1 | 0.0 |

### 4.2 Task 2 – Logistic regression for non-medal countries

| # | Component | Paper | Repo | Weight | Score |
|---|-----------|------|------|--------|-------|
| 21 | **Features** | a_k, e_k, p_k (athletes, **events participated**, **historical participation**), Eq (8), Table 9 | Athletes, **PrevGames** (≈ historical participations), **Host** — **no “events participated” (e_k)**; Host not in paper formula | 3 | 0.0 |
| 22 | **Final equation** | Eq (11): z = -0.012 a_k - 0.068 e_k + 2.425 | Different feature set → different equation; paper coefficients not reproduced | 2 | 0.0 |
| 23 | **Output / Table 11** | Probability > 0.2; LBN, GUM, PLE, ANG, ESA | “Top 10 by probability”; no 0.2 threshold or paper country list | 1 | 0.5 |
| 24 | **Runnable pipeline** | Standalone model with defined data | `Regression.py` assumes `data`, `athlete_counts`, `medals_df`, `host_map` from elsewhere; not callable from repo as-is | 2 | 0.0 |

### 4.3 Task 3 – Sports and medals

| # | Component | Paper | Repo | Weight | Score |
|---|-----------|------|------|--------|-------|
| 25 | **Relationship sports ↔ medals (Table 12)** | “Some countries and sports they are good at” (USA/CHN/JPN/KOR/AUS/GBR, sport–medal counts) | No equivalent table; Medalchart uses `country_sport_medals` but not in this form | 1 | 0.0 |
| 26 | **Radial histogram / “radial histogram”** | Fig 15 (USA, CHN), Fig 16 (single-sport countries) | `Medalchart.py`: polar/radar plot and single-sport list — same idea, different layout/labels | 2 | 0.5 |
| 27 | **Host-selected sports (7.3)** | “Host countries usually add events … explore the effect” | No analysis of host choice of sports or effect on others | 1 | 0.0 |

### 4.4 Task 4 – “Great Coach” / Lasso

| # | Component | Paper | Repo | Weight | Score |
|---|-----------|------|------|--------|-------|
| 28 | **Coach variable C** | Eq (16): C = 1+0.1t (start) or e^{-t} (stop) with t = Olympic cycles | Binary C ∈ {0, 1} only | 3 | 0.0 |
| 29 | **Countries/sports** | CHN Women’s Volleyball, BRA Women’s Soccer, ROU Women’s Gymnastics; Table 15 | Romania Gymnastics, **India Athletics, Nigeria Boxing**; different cases and scores | 2 | 0.0 |
| 30 | **Table 15 values** | CHN 0→5.33, BRA 6→8.86, ROU 3→36.65 | Uses hypothetical BaseScore/PostScore; not tied to paper’s numbers | 2 | 0.0 |
| 31 | **λ selection** | “Cross-validation to select λ”; Fig 18–19 | Fixed `alpha=0.1`; no CV for λ | 2 | 0.5 |
| 32 | **Data flow** | Lasso fit on constructed (I,V,S,A,H,C) from full pipeline | `LassoRegression.py` needs `medals_df`, `athlete_counts`, `country_sport_medals` from elsewhere; not integrated | 2 | 0.0 |

### 4.5 Preprocessing and sensitivity

| # | Component | Paper | Repo | Weight | Score |
|---|-----------|------|------|--------|-------|
| 33 | **Outlier / country-name handling** | Outlier cleaning; ISO mapping; exclude USSR/banned; drop AIN | GSRF uses Team/Year as-is; no explicit ISO map, AIN drop, or outlier step in code | 1 | 0.5 |
| 34 | **Sensitivity layout** | Vary A_ij and E_ij; report “athletes_num: 0.058329%; Total event: -1.2113%” | `sensitivity.py` uses +5% perturbations; requires `best_total_model`, `data_2024`, `features` — not called from `GSRF.py` | 2 | 0.5 |

---

## 5. Score Summary

Using the heuristic above:

- **Critical (weight 3):** 8 components → max 24. From above: 8×1 + 2×0 = 10 (two failures: G_ij/T_ij as features, logistic features).
- **Important (weight 2):** 14 components → max 28. Sum of (score × 2) from components above ≈ 15.
- **Supporting (weight 1):** 6 components → max 6. Sum ≈ 2.5.

Rough total score:

\[
\text{Score} \approx 10 + 15 + 2.5 = 27.5,\quad
\text{Total weight} \approx 24 + 28 + 6 = 58,\quad
\text{Match \%} \approx 100 \times \frac{27.5}{58} \approx \mathbf{47\%}.
\]

The reproduction is **~47% faithful**: correct on RF+grid search, separate gold/total models, 10-fold CV, sport importance, Lasso and logistic frameworks, and sensitivity definition, but **not** on the exact GSRF feature set (missing G_ij, T_ij), standardization, train/test design, coach variable C, chosen countries/sports, or a fully integrated, runnable pipeline for Tasks 2–4 and sensitivity.

---

## 6. Recommendations to Raise the Match Percentage

1. **Task 1:** Add cumulative gold and total medals **before** each Olympic year as features (G_ij, T_ij); apply Eq (0) standardization; implement “pre-2024 train / 2024 test” and optionally 80/20 for metrics; align prediction intervals with Eq (6) and Table 8; add “progress/regression” vs 2024.
2. **Task 2:** Use features a_k, e_k, p_k (and drop or clearly justify Host); replicate Eq (11) and Table 11 (p > 0.2, paper countries); make `Regression.py` runnable from loaded medal/athlete/host data.
3. **Task 3:** Add a Table-12-style “countries and sports they are good at”; keep current importance I_j; add a minimal analysis of host-selected sports (7.3).
4. **Task 4:** Implement C as in Eq (16); use CHN Women’s Volleyball, BRA Women’s Soccer, ROU Women’s Gymnastics and Table 15 targets; select λ via CV; wire Lasso to the same data pipeline as Task 1/3.
5. **Globally:** Add preprocessing (outliers, ISO mapping, AIN exclusion); standardize features; expose one entry point (or notebook) that runs Task 1 → 2 → 3 → 4 and sensitivity so all scripts use the same `medals_df`, `athlete_counts`, `host_map`, etc.
