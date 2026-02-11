# Results Comparison: Paper vs. Last Pipeline vs. New (Enhanced v2.0)

This document compares three sets of results:

1. **Paper** — Values reported in **olympic_medals_paper.md** (2025 MCM/ICM Problem C).
2. **Last** — Outputs from **`run_paper_pipeline.py`** (paper-faithful reproduction), in `Output/`.
3. **New** — Outputs from **`run_paper_pipeline_enhanced.py`** (v2.0: EMA, model comparison, alpha/k-fold sensitivity, Poisson, hybrid 2028), in `Output_Enhanced/`.

---

## Brief summary

- **Paper:** GSRF with cumulative G_ij/T_ij; CV R² Gold 0.710 / Total 0.784; Test R² 0.795 / 0.736; USA 51 gold / 124 total in 2028; Table 11 logistic for non-medal countries; Table 15 Great Coach; sensitivity athletes 0.058%, Total event −1.21%.
- **Last (paper-faithful):** Same GSRF design; CV R² lower (0.6323 / 0.7227), Test R² close or higher (0.7772 / 0.8801); USA 41.23 gold / 105.96 total; same five NOCs in Table 11 with higher probabilities; Table 15 exact match; sensitivity athletes 0.7185%, Total event 0%.
- **New (enhanced):** Three model variants (Cumulative, EMA, Hybrid); **Hybrid** best: Gold Test R² 0.948, Total Test R² 0.979; 2028 predictions from Hybrid: USA 49.94 gold / 133.04 total; alpha sensitivity (α ∈ {0.1–0.5}); k-fold sensitivity (k ∈ {3,5,7,10,15,20}); Poisson regression for never-medal countries (expected medals, P(≥1)); no Table 11/15 or paper-style sensitivity in enhanced run.

---

## 1. Task 1: GSRF Model Performance (Tables 3 & 4)

### 1.1 Gold Medal Model

| Set                | Paper (Table 3) | Last (paper-pipeline) | New (enhanced) |
|--------------------|-----------------|------------------------|----------------|
| **R²** Training    | 0.903           | 0.8665                 | Cumulative 0.845; **EMA 0.909; Hybrid 0.919** |
| **R²** Cross-val   | 0.710           | 0.6323                 | Cumulative 0.648; **EMA 0.837; Hybrid 0.865** |
| **R²** Test (2024) | 0.795           | 0.7772                 | Cumulative 0.779; **EMA 0.930; Hybrid 0.948** |
| **MAE** Test       | 0.914           | 0.8888                 | Cumulative 0.890; **EMA 0.449; Hybrid 0.404** |
| **RMSE** Test      | 2.574           | 2.3440                 | Cumulative 2.336; **EMA 1.317; Hybrid 1.135** |

**New (enhanced):** Best performer is **Hybrid** (G_ij + EMA_Gold + Athletes + EventsTotal + Host). EMA and Hybrid substantially outperform Cumulative and exceed paper Test R².

### 1.2 Total Medal Model

| Set                | Paper (Table 4) | Last (paper-pipeline) | New (enhanced) |
|--------------------|-----------------|------------------------|----------------|
| **R²** Training    | 0.933           | 0.9066                 | Cumulative 0.898; **EMA 0.936; Hybrid 0.939** |
| **R²** Cross-val   | 0.784           | 0.7227                 | Cumulative 0.719; **EMA 0.858; Hybrid 0.875** |
| **R²** Test        | 0.736           | 0.8801                 | Cumulative 0.872; **EMA 0.979; Hybrid 0.979** |
| **MAE** Test       | 2.430           | 1.6745                 | Cumulative 1.746; **EMA 0.732; Hybrid 0.703** |
| **RMSE** Test      | 7.145           | 4.9280                 | Cumulative 5.101; **EMA 2.076; Hybrid 2.047** |

**New (enhanced):** Again **Hybrid** (T_ij + EMA_Total + …) and **EMA** are best; Test R² and MAE/RMSE much better than paper and last pipeline.

---

## 2. 2028 Predictions (Tables 5 & 6 style)

### 2.1 Top 10 Gold Medals

| Rank | Paper (Table 5) | Last (paper-pipeline) | New (enhanced, Hybrid) |
|------|-----------------|------------------------|-------------------------|
| 1    | USA 51          | USA 41.23              | USA 49.94               |
| 2    | CHN 34          | FRA 31.71              | CHN 38.33               |
| 3    | ITA 29          | AUS 28.05              | JPN 21.25               |
| 4    | FRA 28          | CHN 27.04              | AUS 18.44               |
| 5    | GBR 26          | GER 26.22              | FRA 17.98               |
| 6    | GBR 26          | GBR 26.13              | GBR 17.18               |
| 7    | AUS 20          | ITA 25.93              | GER 16.84               |
| 8    | GER 19          | JPN 14.25              | NED 12.81               |
| 9    | JPN 15          | NED 10.69              | ITA 10.95               |
| 10   | CAN 15         | ESP 9.77               | KOR 9.12                |

**New (enhanced):** USA and CHN top two; USA (49.94) and CHN (38.33) closer to paper levels than last run; order differs (e.g. JPN 3rd, AUS 4th).

### 2.2 Top 10 Total Medals

| Rank | Paper (Table 6) | Last (paper-pipeline) | New (enhanced, Hybrid) |
|------|-----------------|------------------------|-------------------------|
| 1    | USA 124         | USA 105.96             | USA 133.04              |
| 2    | FRA 81          | FRA 75.67              | CHN 97.69               |
| 3    | CHN 74          | AUS 70.22              | JPN 46.30               |
| 4    | GBR 72          | GBR 55.90              | AUS 54.74               |
| 5    | AUS 65          | GER 53.93              | FRA 63.99               |
| 6    | GER 61          | JPN 53.63              | GBR 51.94               |
| 7    | ITA 55          | CHN 44.64              | GER 52.27               |
| 8    | JPN 52          | ITA 43.55              | NED 36.64               |
| 9    | CAN 35          | ESP 40.99              | ITA 35.89               |
| 10   | NED 34         | NED 33.90              | KOR 27.58               |

**New (enhanced):** USA total (133.04) above paper (124); CHN second (97.69) above paper (74). Rankings differ from both paper and last.

### 2.3 Prediction Intervals

| Quantity       | Paper      | Last (paper-pipeline)     | New (enhanced)     |
|----------------|------------|----------------------------|--------------------|
| Gold interval  | ŷ ± 7      | ±7 (e.g. USA 34.23–48.23)  | Not reported in enhanced output |
| Total interval | ŷ ± 15     | ±15 (e.g. USA 90.96–120.96)| Not reported in enhanced output |

---

## 3. Progress and Regression (Figure 11 style)

| Phenomenon             | Paper | Last (paper-pipeline) | New (enhanced, from DeltaGold/DeltaTotal) |
|------------------------|-------|------------------------|-------------------------------------------|
| Largest gold increase  | GBR   | FRA +15.7, GER +14.2   | USA +9.94, GER +4.84, GBR +3.18           |
| Largest total increase | GER   | GER +20.9, AUS +17.2   | GER +19.27, USA +7.04, CZE +3.47          |
| Largest gold decrease  | KOR   | CHN −13.0, KOR −9.5    | KOR −3.88, NED −2.19, NZL −2.93           |
| Largest total decrease | CHN   | CHN −46.4              | GBR −13.06                                |

**New (enhanced):** Deltas from `predictions_2028_hybrid.csv` (Predicted2028 − 2024 actual). GER shows largest total increase; GBR largest total decrease in this run.

---

## 4. Task 2: Non-Medal Countries

### 4.1 Paper & Last — Logistic, p > 0.2 (Table 11)

| NOC  | Paper prob | Last (paper-pipeline) |
|------|------------|------------------------|
| LBN  | 0.290      | 0.8462                 |
| GUM  | 0.230      | 0.8493                 |
| PLE  | 0.220      | 0.8563                 |
| ANG  | 0.206      | 0.8092                 |
| ESA  | 0.201      | 0.8477                 |

### 4.2 New (enhanced) — Poisson: Expected Medals & P(≥1)

The enhanced pipeline uses **Poisson regression** for never-medal countries (expected medal count and P(≥1 medal)), not the paper’s logistic Table 11.

**Top never-medal countries by expected medals (New):**

| Team | Athletes | EventsParticipated | ExpectedMedals | ProbAtLeastOne |
|------|----------|--------------------|----------------|----------------|
| GUI  | 25       | 7                  | 2.53           | 0.920          |
| MLI  | 24       | 6                  | 2.53           | 0.920          |
| ANG  | 26       | 10                 | 2.49           | 0.917          |
| SSD  | 14       | 3                  | 2.36           | 0.906          |
| LBR  | 10       | 5                  | 2.23           | 0.893          |
| LBN  | 11       | 9                  | 2.18           | 0.887          |
| NEP  | 7        | 6                  | 2.15           | 0.884          |
| MDV  | 5        | 4                  | 2.15           | 0.883          |
| ESA  | 9        | 9                  | 2.14           | 0.882          |
| GUM  | 9        | 9                  | 2.14           | 0.882          |
| PLE  | 8        | 8                  | 2.13           | 0.882          |

Same NOCs (LBN, GUM, PLE, ANG, ESA) appear among high-potential never-medal countries; New adds expected counts and P(≥1).

---

## 5. Task 3: Countries and Sports (Table 12)

**Paper:** USA/CHN/JPN/KOR/AUS/GBR and sport medal counts.  
**Last:** Same NOCs and structure; counts differ (data/period).  
**New (enhanced):** No Table 12–style output in the enhanced pipeline (focus is EMA, model comparison, alpha/k-fold, Poisson, 2028 hybrid).

---

## 6. Task 4: Great Coach (Table 15)

**Paper:** CHN 0→5.33, BRA 6→8.86, ROU 3→36.65.  
**Last:** Exact match (table15_great_coach_2028.csv).  
**New (enhanced):** Table 15 is not produced by the enhanced pipeline.

---

## 7. Sensitivity (Paper §10.2)

| Quantity            | Paper      | Last (paper-pipeline) | New (enhanced) |
|---------------------|------------|------------------------|----------------|
| athletes_num        | 0.058329%  | 0.7185%                | Not run in enhanced pipeline |
| Total event (signed)| −1.2113%   | 0.0000%                | Not run in enhanced pipeline |

---

## 8. New (Enhanced) Only: Model Comparison, Alpha, K-Fold, Poisson

### 8.1 Model Comparison (Cumulative vs EMA vs Hybrid)

From `Output_Enhanced/model_comparison.csv`:

| Target | Model      | Train_R2 | CV_R2  | Test_R2 | Test_MAE | Test_RMSE |
|--------|------------|----------|--------|---------|----------|-----------|
| Gold   | Cumulative | 0.845    | 0.648  | 0.779   | 0.890    | 2.336     |
| Gold   | Ema        | 0.909    | 0.837  | **0.930** | **0.449** | **1.317** |
| Gold   | Hybrid     | 0.919    | 0.865  | **0.948** | **0.404** | **1.135** |
| Total  | Cumulative | 0.898    | 0.719  | 0.872   | 1.746    | 5.101     |
| Total  | Ema        | 0.936    | 0.858  | **0.979** | **0.732** | **2.076** |
| Total  | Hybrid     | 0.939    | 0.875  | **0.979** | **0.703** | **2.047** |

### 8.2 Alpha Sensitivity (EMA)

From `Output_Enhanced/alpha_sensitivity.csv` (α ∈ {0.1, 0.2, 0.3, 0.4, 0.5}):

| Target | Alpha | Test_R2  | Test_MAE | CV_R2  |
|--------|-------|----------|----------|--------|
| Gold   | 0.1   | 0.862    | 0.655   | 0.737  |
| Gold   | 0.2   | 0.909    | 0.545   | 0.780  |
| Gold   | 0.3   | 0.920    | 0.462   | 0.822  |
| Gold   | 0.4   | 0.947    | 0.385   | 0.862  |
| Gold   | 0.5   | **0.978**| **0.283** | 0.885 |
| Total  | 0.1   | 0.942    | 1.176   | 0.781  |
| Total  | 0.3   | 0.980    | 0.735   | 0.840  |
| Total  | 0.5   | **0.987**| **0.610** | 0.918 |

Higher α (more weight on recent performance) gives better Test R² and MAE in this run.

### 8.3 K-Fold Sensitivity (k = 3, 5, 7, 10, 15, 20)

From `Output_Enhanced/kfold_sensitivity.csv` — Mean R² at k=10 (paper’s choice):

| Model             | Mean_R2 (k=10) | CV_Stability (k=10) |
|-------------------|----------------|---------------------|
| Cumulative_Gold   | 0.633          | 0.283               |
| EMA_Gold          | 0.827          | 0.214               |
| Cumulative_Total  | 0.713          | 0.264               |
| EMA_Total         | 0.854          | 0.205               |

EMA models show higher CV R² and similar or better stability at k=10.

### 8.4 Poisson: Never-Medal Predictions

From `Output_Enhanced/poisson_never_medal_predictions.csv`: 67 countries; top by ExpectedMedals include GUI, MLI, ANG, SSD, LBR, LBN, NEP, MDV, ESA, GUM, PLE (see table in §4.2).

---

## 9. Summary Table (Paper | Last | New)

| Dimension              | Paper           | Last (paper-pipeline)   | New (enhanced) |
|------------------------|-----------------|--------------------------|----------------|
| **Gold R² (CV/Test)**  | 0.710 / 0.795   | 0.6323 / 0.7772          | Hybrid 0.865 / **0.948** |
| **Total R² (CV/Test)** | 0.784 / 0.736   | 0.7227 / 0.8801          | Hybrid 0.875 / **0.979** |
| **USA 2028 gold**      | 51              | 41.23                    | **49.94** (Hybrid) |
| **USA 2028 total**     | 124             | 105.96                   | **133.04** (Hybrid) |
| **CHN 2028 gold**      | 34              | 27.04                    | **38.33** (Hybrid) |
| **Intervals**          | ±7 / ±15        | ±7 / ±15                 | Not in enhanced output |
| **Task 2**             | Table 11 (logistic p>0.2) | Same NOCs, higher probs | Poisson: expected medals, P(≥1) |
| **Task 3 Table 12**    | USA/CHN/… + sports | Same structure          | Not in enhanced run |
| **Task 4 Table 15**   | 0→5.33, 6→8.86, 3→36.65 | Exact match        | Not in enhanced run |
| **Sensitivity**       | 0.058%; −1.21%  | 0.7185%; 0%              | Not in enhanced run |
| **Extra in New**       | —               | —                        | Model comparison (Cum/EMA/Hybrid), alpha sensitivity, k-fold sensitivity, Poisson never-medal |

---

## 10. How to Regenerate

**Last (paper-faithful):**
```bash
python Model/run_paper_pipeline.py
```
Outputs in `Output/`.

**New (enhanced v2.0):**
```bash
python Model/run_paper_pipeline_enhanced.py
```
Outputs in `Output_Enhanced/`. Then:
```bash
python verify_enhanced_pipeline.py --output_dir Output_Enhanced
```

Use `Output_Enhanced/model_comparison.csv`, `predictions_2028_hybrid.csv`, `alpha_sensitivity.csv`, `kfold_sensitivity.csv`, and `poisson_never_medal_predictions.csv` to refresh the “New (enhanced)” columns above.
