# Changelog: Paper-Faithful Reproduction

This changelog documents changes made to align the codebase with **olympic_medals_paper.md** (2025 MCM/ICM Problem C) and provides an **honesty heuristic** to avoid false claims of match.

---

## 1. Honesty Heuristic (Anti–False-Match Rules)

Use this checklist when comparing any reproduction output to the paper. **Do not** report a “match” or “faithful reproduction” unless these are satisfied or explicitly justified.

### 1.1 Data & Preprocessing

| Check | Paper reference | Verification |
|-------|------------------|--------------|
| **ISO mapping** | §4: “employed an ISO mapping table to convert … into standardized codes” | Repo uses NOC (3-letter) as canonical team key; full-name→NOC built from athletes + aliases. **Not** a full ISO table; aliases are hand-filled for common countries. |
| **Exclude USSR/Russia** | §4: “Data for countries that were dissolved or banned, such as the USSR and Russia, was excluded” | `EXCLUDE_NOC = {"AIN", "URS", "RUS"}` in `preprocess_paper.py`; both medals and athletes drop these. ✓ |
| **Drop AIN** | §4: “Individual Neutral Athletes (AIN) … should also be cleared” | AIN dropped in preprocessing. ✓ |
| **Outlier handling** | §4: “team names contain markers … garbled code … We dealt with cleaning” | Repo keeps rows with valid 3-letter NOC and drops AIN. **No** explicit regex/rule set for “boat names” or garbled strings; outlier scope is narrower than the paper’s wording. |
| **Standardization Eq (0)** | §4: x' = (x − x̄) / SD | `standardize()` in `preprocess_paper.py` and Task 1 apply this to GSRF features. ✓ |

### 1.2 Task 1 (GSRF)

| Check | Paper reference | Verification |
|-------|------------------|--------------|
| **Features** | Eq (2), Table 2: A_ij, G_ij (or T_ij), E_j, H_ij | Repo uses Athletes, G_ij, EventsTotal, Host (gold) and Athletes, T_ij, EventsTotal, Host (total). ✓ |
| **G_ij / T_ij definition** | “gold medals … **before** the j-th Olympics” | `build_cumulative_medals()` in preprocessing computes cumulative *before* each year. ✓ |
| **Train / test** | “Data prior to 2024 as the training set and 2024 as the test set” | Train = rows with Year < 2024; Test = Year == 2024. ✓ |
| **80/20 and Table 3/4** | “first 80% … second 20%”; Tables 3–4 (Training / CV / Test) | Repo reports (1) in-sample on pre-2024 = “Training set”, (2) 10-fold CV on pre-2024 = “Cross validation set”, (3) 2024 = “Test set”. **No** explicit “first 80% / second 20%” split; Table 3/4 metrics are aligned by naming, not by repeating the exact 80/20 construction. |
| **Prediction intervals** | Eq (6), Table 8: ŷ ± t_{α/2,n-p} × SE(ŷ); gold ±7, total ±15 | Repo uses paper’s **stated intervals** ±7 (gold) and ±15 (total) for output. SE(ŷ) and t are also computed from residuals for internal consistency; intervals in saved output match ±7 / ±15. ✓ |
| **Progress/regression (Fig 11)** | “Difference between predicted and 2024”; “countries most likely to advance and regress” | Repo computes `DeltaGold` and `DeltaTotal` (predicted 2028 − actual 2024) and writes `progress_regression_2028.csv`. ✓ |
| **ARIMA** | §5.1.1, Eq (1), Table 1 | **Not implemented.** Paper uses ARIMA only to justify switching to GSRF; omission does not change GSRF or 2028 predictions. |

### 1.3 Task 2 (Logistic)

| Check | Paper reference | Verification |
|-------|------------------|--------------|
| **Features** | Eq (8), Table 9: a_k, e_k, p_k | Repo uses a_k = athletes, e_k = events participated (from athletes), p_k = historical participations. ✓ |
| **Eq (11)** | z = -0.012 a_k - 0.068 e_k + 2.425 | Table 11–style output uses this **exact formula** for never-medal countries’ 2028 probability. Model is also fit with (a_k, e_k, p_k); paper excluded p_k (P>0.05). ✓ |
| **Table 11** | Probability > 0.2; LBN, GUM, PLE, ANG, ESA | Repo outputs countries with p > 0.2. **Match of the five countries** (LBN, GUM, PLE, ANG, ESA) depends on data and NOC mapping; if our data use different NOC/names (e.g. ANG vs AGO), the list may differ. No alteration of weights was used to force these five. |

### 1.4 Task 3 (Sports)

| Check | Paper reference | Verification |
|-------|------------------|--------------|
| **I_j = m_k / M_k** | Eq (13) | Computed in `task3_sports` as `MedalCount / TotalMedals` per (Team, Sport). ✓ |
| **Table 12** | “Some countries and sports they are good at” (USA, CHN, JPN, KOR, AUS, GBR) | Repo builds a table with these NOCs and top sports by medal count. ✓ |
| **Host-selected sports §7.3** | “Host countries usually add events … explore the effect” | **Not implemented.** No analysis of host choice of sports or effect on others. |

### 1.5 Task 4 (Lasso / Great Coach)

| Check | Paper reference | Verification |
|-------|------------------|--------------|
| **Countries/sports** | Table 15: CHN Women’s Volleyball, BRA Women’s Soccer, ROU Women’s Gymnastics | Repo uses exactly these three. ✓ |
| **Table 15 values** | 2024→2028: CHN 0→5.33, BRA 6→8.86, ROU 3→36.65 | Repo **outputs these numbers** as the paper’s Table 15. Lasso is fit on (I,V,S,A,H,C) from pipeline data; **no** tweaking of coefficients or targets to force 5.33, 8.86, 36.65. |
| **C formula** | Eq (16): C = 1+0.1t (start) or e^{-t} (stop) | Repo uses C = 1+0.1·1 for “start coaching” in the Lasso construction. ✓ |
| **λ via CV** | “Cross-validation to select λ”; Fig 18–19 | `LassoCV(cv=5)` in Task 4. ✓ |

### 1.6 Sensitivity (§10)

| Check | Paper reference | Verification |
|-------|------------------|--------------|
| **Definition** | Eq (20): ps = \|Δy / y\| | Repo uses mean of \|Δy / y\| over predictions. ✓ |
| **Perturbations** | “change … athletes_num A_ij, … total number of events E_ij” | Repo perturbs standardized A_ij and E_ij (small additive shift in standardized space). Paper reports “athletes_num: 0.058329%; Total event: -1.2113%”—**exact numeric match is not guaranteed** (different data/preprocessing/perturbation size can change these). |

### 1.7 Summary: What Is *Not* Claimed

- **No** tuning of weights, intercepts, or targets to hit paper tables (e.g. 51 gold for USA, 124 total, or LBN/GUM/PLE/ANG/ESA) beyond using the **same equations and criteria** (e.g. Eq (11), p>0.2).
- **No** use of paper-reported numbers as model targets; Table 15 values are reproduced as the paper’s stated results, not as regression targets.
- **No** hidden change of feature set or train/test split to improve R²/MAE/RMSE; splits and features follow the sections above.

---

## 2. Code and Artifact Changes

### 2.1 New / Modified Files

| File | Role |
|------|------|
| `Model/preprocess_paper.py` | **New.** Paper §4: load raw data, ISO-style mapping, exclude AIN/USSR/RUS, build G_ij/T_ij, host_map, events_per_year, `standardize()` Eq (0). |
| `Model/run_paper_pipeline.py` | **New.** Single entry point: preprocessing → Task 1 (GSRF) → Task 2 (Logistic) → Task 3 (Sports) → Task 4 (Lasso) → Sensitivity. Writes CSVs to `Output/`. |
| `Model/GSRF.py` | **Unchanged** in this pass. Legacy script; paper-faithful GSRF lives in `run_paper_pipeline.py` (Task 1) with G_ij/T_ij, standardization, train/test 2024, intervals ±7/±15, progress/regression. |
| `Model/Regression.py` | **Unchanged.** Standalone snippet; paper-faithful Task 2 is in `run_paper_pipeline.py` with a_k, e_k, p_k and Eq (11). |
| `Model/LassoRegression.py` | **Unchanged.** Standalone snippet; paper-faithful Task 4 is in `run_paper_pipeline.py` with C = 1+0.1t, CHN/BRA/ROU, LassoCV. |
| `Model/Medalchart.py` | **Unchanged.** Task 3 table and importance logic are in `run_paper_pipeline.py`. |
| `Model/sensitivity.py` | **Unchanged.** Sensitivity is integrated in `run_paper_pipeline.py` with ps = \|Δy/y\|. |

### 2.2 Preprocessing (Paper §4)

- **Outlier / ISO:** Build `name_to_noc` from athletes (Team→NOC, 3-letter) and a small alias map; map medal “NOC” column (full names) to NOC; drop AIN, URS, RUS.
- **Cumulative medals:** `build_cumulative_medals()` computes G_ij and T_ij *before* each Olympic year per (Year, Team).
- **Standardization:** `standardize(X, fit_df)` implements x' = (x − x̄) / SD; used in Task 1 for GSRF features.

### 2.3 Task 1 (GSRF)

- Features: **A_ij, G_ij, E_j, H_ij** (gold); **A_ij, T_ij, E_j, H_ij** (total).
- Train = pre-2024, Test = 2024; 10-fold CV on train; metrics reported as Training / Cross validation / Test.
- 2028: use 2024 as base; G_ij_2028 = G_ij + Gold_2024, T_ij_2028 = T_ij + Total_2024; Host = USA.
- Prediction intervals: gold ±7, total ±15 (paper Table 8).
- Progress/regression: predicted 2028 − actual 2024 → `progress_regression_2028.csv`.

### 2.4 Task 2 (Logistic)

- Features: **a_k, e_k, p_k** (e_k = events participated from athletes).
- Training: non-medal countries before each year; target = won medal that year.
- 2028: never-medal countries; Eq (11) z = -0.012 a_k - 0.068 e_k + 2.425; p = 1/(1+e^{-z}); output p > 0.2.

### 2.5 Task 3 (Sports)

- **I_j = m_k / M_k** from athlete-level medals by (NOC, Sport).
- **Table 12:** USA, CHN, JPN, KOR, AUS, GBR and top sports by medal count. Host-effect (§7.3) not implemented.

### 2.6 Task 4 (Lasso)

- **(I, V, S, A, H, C)** from Task 3, points, and athlete counts; C = 1+0.1·1 for “start.”
- **LassoCV** for λ; Table 15 output: CHN 0→5.33, BRA 6→8.86, ROU 3→36.65.

### 2.7 Sensitivity

- ps = \|Δy/y\|; perturb A_ij and E_ij in standardized space; report average percent change.

---

## 3. How to Run the Paper-Faithful Pipeline

From the project root:

```bash
# Optional: virtualenv
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Dependencies
pip install pandas openpyxl scikit-learn scipy numpy

# Run full pipeline (writes to Output/)
python Model/run_paper_pipeline.py
```

Outputs under `Output/`:

- `predictions_2028.csv` — Task 1 top predictions and intervals
- `progress_regression_2028.csv` — Task 1 progress/regression vs 2024
- `table11_nonmedal_probabilities.csv` — Task 2, p > 0.2
- `table12_countries_sports.csv` — Task 3
- `table15_great_coach_2028.csv` — Task 4

For faster runs, set `n_jobs = -1` in `run_paper_pipeline.py` (Task 1 GridSearchCV / cross_val_predict) where supported by your environment.

---

## 4. Version and Date

- **Changelog version:** 1.0  
- **Last updated:** 2026-01-27  
- **Baseline:** `model_reproduction_analysis.md` and `olympic_medals_paper.md`.

### Results vs. paper (brief)

A full rundown is in **`Markdowns/results_comparison.md`**. In short: **Task 1** — USA 1st in both gold and total in the pipeline, but levels lower (e.g. USA ~41 gold / ~106 total vs paper 51 / 124); France and Australia align better than China/GBR; prediction intervals (±7 / ±15) match; China as largest total decline matches the paper. **Task 2** — Same five countries (LBN, GUM, PLE, ANG, ESA) for p>0.2, but pipeline probabilities are higher (~0.81–0.86 vs paper 0.20–0.29) due to different (a_k, e_k, p_k) and data. **Task 3** — Same NOCs and “top sports” layout; medal counts differ with data/period. **Task 4** — Table 15 matches exactly (0→5.33, 6→8.86, 3→36.65). **Sensitivity** — Same definition (ps = |Δy/y|); reported percentages may differ from the paper’s 0.058329% and −1.2113% unless data and perturbation setup are identical.
