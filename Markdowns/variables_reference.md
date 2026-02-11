# Variables Reference — Olympic Medals Pipeline

All variables used in the GSRF model calculations, preprocessing, and paper-faithful pipeline. Section numbers refer to `olympic_medals_paper.md`.

---

## 1. Preprocessing & standardization (§4)

| Variable | Definition | Code / usage |
|----------|------------|--------------|
| **x** | Raw feature value | Input to standardization |
| **x̄** | Mean of the feature (over fit set) | `fit_df.mean()` |
| **SD** | Standard deviation of the feature | `fit_df.std()`; zeros replaced with 1 |
| **x′** | Standardized feature | Eq (0): \( x' = (x - \bar{x}) / \text{SD} \) |
| **EXCLUDE_NOC** | NOCs excluded from analysis | `{AIN, URS, RUS}` (paper §4) |
| **name_to_noc** | Mapping from country name to 3-letter NOC | Built from athletes; aliases for USA, CHN, GBR, etc. |
| **athletes_clean** | Athlete-level data after dropping AIN/URS/RUS and invalid NOCs | Preprocessed athletes |
| **medals_clean** | Medal counts with country names converted to NOC; excludes EXCLUDE_NOC | Preprocessed medals |
| **athlete_counts** | Rows (Year, Team) with count of athletes | `build_athlete_counts()` |
| **events_per_year** | Rows (Year, EventsTotal); 1906 skipped | `build_events_per_year()` |
| **host_map** | Year → NOC of host; 2028 → USA | `build_host_map()` |

---

## 2. Task 1 — GSRF (gold & total medals) (§5)

### 2.1 Core symbols (paper §3, §5.1.2)

| Symbol | Definition | Code column / usage |
|--------|------------|----------------------|
| **A_ij** | Total number of athletes from country \(i\) in the \(j\)-th Olympic Games | `Athletes` |
| **G_ij** | Number of gold medals won by country \(i\) **before** the \(j\)-th Olympics (cumulative) | `G_ij` from `build_cumulative_medals()` |
| **T_ij** | Total number of medals won by country \(i\) **before** the \(j\)-th Olympics (cumulative) | `T_ij` from `build_cumulative_medals()` |
| **E_j** | Total number of events in the \(j\)-th Olympic Games | `EventsTotal` (from programs) |
| **H_ij** | Host indicator: 1 if country \(i\) hosts the \(j\)-th Olympics, else 0 | `Host` |
| **Y_g** | Predicted number of **gold** medals | Output of gold GSRF model |
| **Y_t** | Predicted **total** number of medals | Output of total GSRF model |

### 2.2 GSRF model (Eq 2)

- **Gold model inputs:** \(A_{ij,g}, G_{ij}, E_{j,g}, H_{ij,g}\) → **Y_g**
- **Total model inputs:** \(A_{ij,t}, T_{ij,t}, E_{j,t}, H_{ij,t}\) → **Y_t**

Features used in code:
- Gold: `["Athletes", "G_ij", "EventsTotal", "Host"]`
- Total: `["Athletes", "T_ij", "EventsTotal", "Host"]`

### 2.3 Standardization for GSRF

- **μ_g, sd_g** — Mean and standard deviation of gold features (fit on training set).
- **μ_t, sd_t** — Mean and standard deviation of total features (fit on training set).
- Training: pre-2024; test: 2024; 10-fold CV on training.

### 2.4 Evaluation metrics (Eq 3–5)

| Symbol | Definition | Formula |
|--------|------------|---------|
| **R²** | Coefficient of determination | \( R^2 = 1 - \frac{\sum(Y_i - \hat{Y}_i)^2}{\sum(Y_i - \bar{Y}_i)^2} \) |
| **MAE** | Mean absolute error | \( \text{MAE} = \frac{1}{n}\sum|Y_i - \hat{Y}_i| \) |
| **RMSE** | Root mean square error | \( \text{RMSE} = \sqrt{\frac{1}{n}\sum(Y_i - \hat{Y}_i)^2} \) |
| **Y_i** | True value | Observed medals |
| **Ŷ_i** | Predicted value | Model output |
| **Ȳ_i** | Mean of true values | — |

### 2.5 Prediction intervals (Eq 6, Table 8)

| Symbol | Definition | Usage |
|--------|------------|--------|
| **ŷ** | Point prediction | GSRF output |
| **t_{α/2,n−p}** | t-distribution quantile (e.g. α = 0.05, 95% interval) | `scipy.stats.t.ppf(0.975, df)` |
| **n** | Number of training samples | — |
| **p** | Number of model parameters (features) | len(feat_g) or len(feat_t) |
| **SE(ŷ)** | Standard error of prediction (from residuals) | Std of CV residuals, ddof = n−p |
| **Gold interval** | Paper Table 8 | ŷ ± 7 |
| **Total interval** | Paper Table 8 | ŷ ± 15 |

### 2.6 2028 projections

| Variable | Definition | Code |
|----------|------------|------|
| **G_ij_2028** | Cumulative gold before 2028 | G_ij(2024) + Gold(2024) |
| **T_ij_2028** | Cumulative total before 2028 | T_ij(2024) + Total(2024) |
| **Host_2028** | 1 for USA (Los Angeles), 0 otherwise | — |
| **DeltaGold** | Predicted 2028 gold − actual 2024 gold | Progress/regression |
| **DeltaTotal** | Predicted 2028 total − actual 2024 total | Progress/regression |

---

## 3. Task 2 — Logistic regression (never-medal countries) (§6)

### 3.1 Input features (Eq 8, Table 9)

| Symbol | Definition | Code / source |
|--------|------------|----------------|
| **a_k** | Number of athletes from the \(k\)-th country (in that Olympics) | `Athletes` from athlete_counts |
| **e_k** | Number of **events** in which the \(k\)-th country participates in that Olympics | `EventsParticipated` (unique events per Year, NOC from athletes) |
| **p_k** | Historical participation: number of Olympic editions the \(k\)-th country has participated in (before that year) | Count of distinct Year for that Team with Year < current year |

### 3.2 Logistic model (Eq 7, 11, 12)

| Symbol | Definition | Formula / value |
|--------|------------|------------------|
| **z** | Linear predictor | \( z = \beta_0 + \beta_1 a_k + \beta_2 e_k + \beta_3 p_k \); paper uses \(z = -0.012\,a_k - 0.068\,e_k + 2.425\) (p_k excluded, P>0.05) |
| **σ(z)** | Probability of winning a medal | \( \sigma(z) = 1/(1 + e^{-z}) \) (Eq 7) |
| **WonMedal** | Binary target: 1 if country won first medal that year, 0 otherwise | Training label |
| **MedalWin_Prob2028** | P(win first medal in 2028) for never-medal countries | \( 1/(1 + e^{-z}) \) with 2024 proxy for \(a_k, e_k\) |

### 3.3 Table 11

- Countries with **MedalWin_Prob2028 > 0.2** (paper lists LBN, GUM, PLE, ANG, ESA).

---

## 4. Task 3 — Sport importance (§7)

### 4.1 Sport–country medals

| Symbol | Definition | Code |
|--------|------------|------|
| **m_k** | Number of medals won by the country in a given **sport** | `MedalCount` per (NOC, Sport) from athletes (medal rows only) |
| **M_k** | Total number of medals won by the country (all sports) | `TotalMedals` per NOC |
| **I_j** | Importance of that sport to the country (Eq 13) | \( I_j = m_k / M_k \) → `Importance` = MedalCount / TotalMedals |

### 4.2 Outputs

- **table12** — Selected countries (USA, CHN, JPN, KOR, AUS, GBR) and top sports by medal count.
- **country_sport_medals** — (Team, Sport, MedalCount, TotalMedals, Importance).

---

## 5. Task 4 — Lasso “Great Coach” (§8)

### 5.1 Points (Table 13)

| Medal | Points |
|-------|--------|
| Gold | 10 |
| Silver | 6 |
| Bronze | 3 |
| No medal | 0 |

**TotalPoints** = Gold×10 + Silver×6 + Bronze×3 (per country-year).

### 5.2 Lasso regression variables (Eq 15, Table 14)

| Symbol | Definition | Code / computation |
|--------|------------|---------------------|
| **I** | Importance of the sport to the country | Same as Task 3 \(I_j\) for that (Team, Sport) |
| **V** | Average total points scored by the country in the **last two Olympic Games** | Mean of `TotalPoints` for last 2 years for that Team |
| **S** | Theoretical points for that sport | \( S = I \times V \) |
| **A** | Number of athletes from that country at the current Olympics (e.g. 2024) | Count of athlete rows (Year, NOC) |
| **H** | Historical average points for that country–sport (baseline without coach) | In pipeline: `BaseScore` from paper Table 15 (0, 6, 3 for CHN/BRA/ROU) |
| **C** | Coaching impact (Eq 16) | Start: \( C = 1 + 0.1t \) (e.g. t=1 → C=1.1); Stop: \( C = e^{-t} \). Pipeline uses C=0 (no coach) and C=1.1 (start, t=1). |
| **y** | Score (points) for that country–sport | Regression target; Table 15: 2024 vs 2028 scores |

### 5.3 Table 15 cases

| Country_Sport | 2024 (base) | 2028 (with Great Coach) |
|---------------|-------------|---------------------------|
| CHN, Women's Volleyball | 0 | 5.33 |
| BRA, Women's Soccer | 6 | 8.86 |
| ROU, Women's Gymnastics | 3 | 36.65 |

### 5.4 Lasso (Eq 17, 18)

| Symbol | Definition |
|--------|------------|
| **λ** | Regularization strength (chosen by cross-validation, e.g. LassoCV) |
| **RSS** | Residual sum of squares: \( \sum(y_i - \hat{y}_i)^2 \) |
| **L1** | \( \lambda \sum|\beta_i| \) |
| **Objective** | min(RSS + L1) |

---

## 6. Sensitivity analysis (§10)

### 6.1 Definition (Eq 20)

| Symbol | Definition | Formula |
|--------|------------|---------|
| **y** | Baseline predicted total medals (2024 features) | GSRF total model output |
| **Δy** | Change in prediction after perturbing an input | \( y_{\text{perturbed}} - y \) |
| **ps** | Average sensitivity (as %) | \( p_s = |\Delta y / y| \), then mean over countries × 100 |

### 6.2 Perturbations (pipeline)

- **SENSITIVITY_PERTURB_PCT** = 0.01 (1% in **raw** space).
- **Athletes:** \(A_{ij}\) → \(A_{ij} \times (1 + 0.01)\); then standardize and predict → **athletes_num_pct**.
- **Total events:** \(E_j\) (EventsTotal) → \(E_j \times (1 + 0.01)\); then standardize and predict → **total_event_pct** (signed) and **total_event_pct_abs**.

---

## 7. Data columns (merged “data” used in pipeline)

Preprocessing produces one row per (Year, Team) with (among others):

| Column | Meaning |
|--------|---------|
| Year | Olympic year (1916, 1940, 1944 excluded) |
| Team | NOC (3-letter) |
| Athletes | A_ij |
| Gold, Silver, Bronze, Total | Medals that year |
| G_ij | Cumulative gold before that year |
| T_ij | Cumulative total before that year |
| Host | H_ij (0/1) |
| EventsTotal | E_j |

---

## 8. ARIMA (paper §5.1.1, not implemented)

| Symbol | Definition |
|--------|------------|
| X_t | Time series value at time t |
| c | Constant |
| φ₁…φ_p | AR parameters |
| θ₁…θ_q | MA parameters |
| ε_t | Error at t |

Used in paper only to justify switching to GSRF; not used in this codebase.
