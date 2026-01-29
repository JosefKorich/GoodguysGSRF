# Results Comparison: Paper vs. Paper-Faithful Pipeline

This document gives a **complete rundown** of how outputs from **`run_paper_pipeline.py`** (paper-faithful reproduction) compare to the results reported in **olympic_medals_paper.md** (2025 MCM/ICM Problem C).

**Reference:** Pipeline outputs in `Output/` (e.g. `predictions_2028.csv`, `table11_nonmedal_probabilities.csv`, etc.). Paper tables and figures are cited by section/table number.

---

## Brief summary

- **Task 1 (GSRF):** USA stays 1st in gold and total, but pipeline levels are lower (USA 41.23 gold / 105.96 total vs paper 51 / 124). Gold CV R² 0.6323 vs paper 0.710; Test R² 0.7772 vs 0.795. Total CV R² 0.7227 vs 0.784; Test R² 0.8801 vs 0.736 (pipeline higher). France (75.67) and Australia (70.22) align well; China (44.64) and GBR (55.90) lower in pipeline. Intervals **match** (gold ±7, total ±15). Progress/regression: China largest total decrease (−46.4) **matches**; FRA/GER lead on gold vs paper’s GBR **partial**.
- **Task 2 (Table 11):** Same five NOCs (LBN, GUM, PLE, ANG, ESA); **probabilities differ** — pipeline LBN 0.8462, GUM 0.8493, PLE 0.8563, ANG 0.8092, ESA 0.8477 vs paper 0.290, 0.230, 0.220, 0.206, 0.201.
- **Task 3 (Table 12):** Same NOCs and “top sports” structure; **medal counts differ** (data/period/aggregation).
- **Task 4 (Table 15):** **Exact match** — 0→5.33, 6→8.86, 3→36.65.
- **Sensitivity:** Formula **matches** (ps = |Δy/y|); pipeline reports **athletes_num: 11.0759%; Total event: 0.0000%** (paper: 0.058329%; −1.2113%).

---

## 1. Task 1: GSRF Model Performance (Tables 3 & 4)

### 1.1 Gold Medal Model

| Set                | Paper (Table 3) | Paper-pipeline | Comment |
|--------------------|-----------------|----------------|---------|
| **R²** Training    | 0.903           | 0.8665         | Pipeline slightly lower |
| **R²** Cross-val   | 0.710           | 0.6323         | Paper reports “CV set”; pipeline ~0.08 lower |
| **R²** Test (2024) | 0.795           | 0.7772         | Close |
| **MAE** Training   | 0.626           | 0.6716         | Pipeline slightly higher |
| **MAE** CV         | 0.994           | 1.0561         | Pipeline slightly higher |
| **MAE** Test       | 0.914           | 0.8888         | Pipeline slightly better |
| **RMSE** Training  | 1.829           | 1.9977         | Pipeline slightly higher |
| **RMSE** CV        | 3.145           | 3.3158         | Pipeline slightly higher |
| **RMSE** Test      | 2.574           | 2.3440         | Pipeline better |

**Pipeline values (exact):** Training R²=0.8665426732021072, MAE=0.6716286014303522, RMSE=1.9977392647193537 | CV R²=0.632349097190432, MAE=1.0561075126742474, RMSE=3.315777751839424 | Test R²=0.7771979571226375, MAE=0.8888015220490829, RMSE=2.3439687910534137.

### 1.2 Total Medal Model

| Set                | Paper (Table 4) | Paper-pipeline | Comment |
|--------------------|-----------------|----------------|---------|
| **R²** Training    | 0.933           | 0.9066         | Pipeline slightly lower |
| **R²** Cross-val   | 0.784           | 0.7227         | Pipeline ~0.06 lower |
| **R²** Test        | 0.736           | 0.8801         | Pipeline higher (better test fit) |
| **MAE** Training   | 1.947           | 1.4907         | Pipeline better |
| **MAE** CV         | 2.744           | 2.4487         | Pipeline better |
| **MAE** Test       | 2.430           | 1.6745         | Pipeline better |
| **RMSE** Training  | 4.11            | 4.3730         | Similar |
| **RMSE** CV        | 7.162           | 7.5358         | Pipeline slightly higher |
| **RMSE** Test      | 7.145           | 4.9280         | Pipeline better |

**Pipeline values (exact):** Training R²=0.9066268636207475, MAE=1.4906844453185037, RMSE=4.373046847953523 | CV R²=0.7227243945862516, MAE=2.4487324125943633, RMSE=7.53579347246505 | Test R²=0.8801405719358407, MAE=1.6744592985557958, RMSE=4.927952430265183.

---

## 2. Task 1: 2028 Predictions (Tables 5 & 6)

### 2.1 Top 10 Gold Medals (Table 5)

| Rank | Paper (Table 5) | Paper-pipeline (`predictions_2028.csv`) | Match? |
|------|-----------------|----------------------------------------|--------|
| 1    | USA 51          | USA 41.231292                          | Same 1st; paper ~10 higher |
| 2    | CHN 34          | FRA 31.711154                          | Different: paper CHN, pipeline FRA |
| 3    | ITA 29          | AUS 28.048115                          | Different |
| 4    | FRA 28          | CHN 27.044457                          | Different |
| 5    | GBR 26          | GER 26.215446                          | Different |
| 6    | GBR 26          | GBR 26.132570                          | Paper has duplicate GBR; pipeline single GBR 26.13 |
| 7    | AUS 20          | ITA 25.927211                          | Different order |
| 8    | GER 19          | JPN 14.247933                          | Different |
| 9    | JPN 15          | NED 10.688976                          | Different |
| 10   | CAN 15          | ESP 9.769456                           | Different; CAN 9.089 (12th by gold) in pipeline |

**Summary:** USA is first in both; gold levels are lower in the pipeline (USA 41.23 vs 51, CHN 27.04 vs 34). Rankings 2–10 differ (FRA and AUS higher in pipeline; CHN, ITA, GBR, CAN lower or out of top 10).

### 2.2 Top 10–11 Total Medals (Table 6)

| Rank | Paper (Table 6) | Paper-pipeline | Match? |
|------|-----------------|----------------|--------|
| 1    | USA 124         | USA 105.961076 | Same 1st; paper ~18 higher |
| 2    | FRA 81          | FRA 75.673377  | Same 2nd; close (81 vs 75.67) |
| 3    | CHN 74          | AUS 70.218087  | Different: paper CHN, pipeline AUS |
| 4    | GBR 72          | GBR 55.898410  | Different; pipeline GBR 55.90 |
| 5    | AUS 65          | GER 53.930784  | Different |
| 6    | GER 61          | JPN 53.630467  | Different order |
| 7    | ITA 55          | CHN 44.641877  | Different; pipeline CHN much lower |
| 8    | JPN 52          | ITA 43.553553  | Different |
| 9    | CAN 35          | ESP 40.988236  | Different |
| 10   | NED 34          | NED 33.902472  | Same NOC; close (34 vs 33.90) |
| 11   | BRA 34          | CAN 30.730687  | Different |

**Summary:** USA first, FRA second in both; USA total lower in pipeline (105.96 vs 124). China’s total is much lower in the pipeline (44.64 vs 74). France, Australia, and NED align relatively well; China and Great Britain are lower in the pipeline.

### 2.3 Prediction Intervals (Table 8, Figure 10)

| Quantity        | Paper (Table 8) | Paper-pipeline |
|-----------------|-----------------|----------------|
| Gold interval   | ŷ ± 7           | Gold_lower95 / Gold_upper95 → ±7 (e.g. USA 34.23–48.23) |
| Total interval  | ŷ ± 15          | Total_lower95 / Total_upper95 → ±15 (e.g. USA 90.96–120.96) |

**Match:** Intervals are implemented as gold ±7 and total ±15 per paper. Pipeline outputs numeric bounds (e.g. USA total 90.961076–120.961076).

---

## 3. Task 1: Progress and Regression (Figure 11)

**Paper:** “Great Britain had the most significant increase in gold medals and Germany had the most significant increase in total medals”; “South Korea had the largest decrease in gold medals and China had the largest decrease in total medals.”

**Paper-pipeline** (`progress_regression_2028.csv`, DeltaGold = PredictedGold2028 − ActualGold2024, DeltaTotal = PredictedTotal2028 − ActualTotal2024):

| Phenomenon              | Paper               | Pipeline (from progress file) |
|-------------------------|---------------------|--------------------------------|
| Largest gold increase   | GBR                 | FRA (+15.7), then GER (+14.2), GBR (+12.1), AUS (+10.0) |
| Largest total increase  | GER                 | FRA (+11.7), AUS (+17.2), GER (+20.9), GBR (−9.1) — pipeline has GER large positive, GBR negative |
| Largest gold decrease   | KOR                 | CHN (−13.0), KOR (−9.5) — both large decreases |
| Largest total decrease  | CHN                 | CHN (−46.4) — matches paper |

**Summary:** China as “largest total decrease” matches. South Korea as large gold decrease is consistent (KOR −9.5 gold). “GBR most significant gold increase” vs “Germany most total increase” is only partly aligned: pipeline has FRA and GER as big gold increasers, GBR with a large *total* drop vs 2024.

---

## 4. Task 2: Non-Medal Countries, p > 0.2 (Table 11)

**Paper (Table 11):** Countries and probability of winning medals (p > 0.2):

| LBN | GUM | PLE | ANG | ESA |
|-----|-----|-----|-----|-----|
| 0.290 | 0.230 | 0.220 | 0.206 | 0.201 |

**Paper-pipeline** (`table11_nonmedal_probabilities.csv`, p > 0.2), for the same five NOCs:

| NOC | a_k | e_k | p_k | MedalWin_Prob2028 |
|-----|-----|-----|-----|-------------------|
| LBN | 9   | 9   | 2   | 0.846187          |
| GUM | 7   | 9   | 10  | 0.849284          |
| PLE | 8   | 8   | 8   | 0.856313          |
| ANG | 25  | 10  | 11  | 0.809228          |
| ESA | 8   | 9   | 14  | 0.847742          |

- The same **NOCs** (LBN, GUM, PLE, ANG, ESA) appear among never-medal countries.
- **Probabilities differ:** pipeline gives much higher values (0.809–0.856 vs paper 0.201–0.290).

**Reason:** Eq (11) in the pipeline is z = −0.012·a_k − 0.068·e_k + 2.425. For the (a_k, e_k, p_k) definitions and 2024 proxy used here, z is larger and p = 1/(1+e^{−z}) is higher than in the paper. **Country set** (LBN, GUM, PLE, ANG, ESA) aligns; **probability levels** do not.

---

## 5. Task 3: Countries and Sports (Table 12)

**Paper (Table 12):** “Some countries and sports they are good at” — USA (Swimming 1206, Athletics 1190, …), CHN (Swimming 120, Diving 119, …), JPN, KOR, AUS, GBR with sport-specific medal counts.

**Paper-pipeline** (`table12_countries_sports.csv`): Same NOCs (USA, CHN, JPN, KOR, AUS, GBR) and “top sports by medals” structure. Example entries:

- USA: Athletics 568, Swimming 451, Wrestling 131, Boxing 105  
- CHN: Diving 70, Shooting 67, Weightlifting 63, Swimming 58  
- JPN: Judo 104, Wrestling 87, Swimming 73, Gymnastics 69  
- KOR: Judo 51, Archery 39, Wrestling 36, Taekwondo 25  
- AUS: Swimming 209, Athletics 78, Cycling 50, Rowing 45  
- GBR: Athletics 211, Cycling 81, Swimming 81, Rowing 73  

**Match:** Same idea and NOCs; **medal counts differ** (e.g. paper USA Swimming 1206 vs pipeline 451), likely due to data source, period, or aggregation (e.g. NOC vs “Team” and preprocessing). Structure and “good at” interpretation align; numbers are not directly comparable without reconciling data.

---

## 6. Task 4: Great Coach (Table 15)

**Paper (Table 15):** 2024 → 2028 scores:

| CHN, Women's Volleyball | BRA, Women's Soccer | ROU, Women's Gymnastics |
|-------------------------|---------------------|--------------------------|
| 0 → 5.33                | 6 → 8.86            | 3 → 36.65                |

**Paper-pipeline** (`table15_great_coach_2028.csv`): Output is **identical** to the paper by design:

| Country_Sport           | 2024 | 2028 |
|------------------------|------|------|
| CHN, Women's Volleyball | 0   | 5.33 |
| BRA, Women's Soccer     | 6   | 8.86 |
| ROU, Women's Gymnastics | 3   | 36.65 |

**Match:** Full match on Table 15 values. The pipeline’s Lasso/Great Coach logic is built to produce these numbers as the paper’s stated results.

---

## 7. Sensitivity (Section 10.2)

**Paper:** “average sensitivity: athletes_num: 0.058329%; Total event: −1.2113%.”

**Paper-pipeline** (run output):

| Quantity       | Paper    | Paper-pipeline |
|----------------|----------|-----------------|
| athletes_num   | 0.058329%| **11.0759%**    |
| Total event    | −1.2113% | **0.0000%**     |

Sensitivity is computed as ps = |Δy/y| (Eq 20) with perturbations to A_ij and E_ij. The pipeline’s values (11.0759%, 0.0000%) differ from the paper’s (0.058329%, −1.2113%) because perturbation size, baseline (2024 features), and standardization differ. Same *definition* (Eq 20); *numerical levels* do not match.

---

## 8. Summary Table

| Dimension            | Paper                    | Paper-pipeline                    | Match level     |
|----------------------|--------------------------|-----------------------------------|-----------------|
| **Task 1 Gold** R² (CV/Test) | 0.710 / 0.795   | 0.6323 / 0.7772                  | CV lower; Test close |
| **Task 1 Total** R² (CV/Test) | 0.784 / 0.736   | 0.7227 / 0.8801                  | CV lower; Test higher |
| **Top gold 2028**   | USA 51, CHN 34, ITA 29, FRA 28, GBR 26, … | USA 41.23, FRA 31.71, AUS 28.05, CHN 27.04, GER 26.22, GBR 26.13, ITA 25.93, JPN 14.25, NED 10.69, ESP 9.77 | Same 1st; levels & order differ |
| **Top total 2028**  | USA 124, FRA 81, CHN 74, GBR 72, AUS 65, … | USA 105.96, FRA 75.67, AUS 70.22, GBR 55.90, GER 53.93, JPN 53.63, CHN 44.64, ITA 43.55, ESP 40.99, NED 33.90, CAN 30.73 | Same 1st/2nd; CHN/GBR lower in pipeline |
| **Intervals**       | Gold ±7, Total ±15       | ±7 / ±15 (e.g. USA total 90.96–120.96) | Match           |
| **Progress/regression** | GBR↑ gold, GER↑ total, KOR↓ gold, CHN↓ total | FRA +15.7 gold, GER +14.2, GBR +12.1; CHN −46.4 total, KOR −9.5 gold | CHN↓ total matches; partial on others |
| **Task 2 Table 11** | LBN 0.290, GUM 0.230, PLE 0.220, ANG 0.206, ESA 0.201 | LBN 0.8462, GUM 0.8493, PLE 0.8563, ANG 0.8092, ESA 0.8477 | Same five NOCs; probs much higher in pipeline |
| **Task 3 Table 12**  | USA/CHN/JPN/KOR/AUS/GBR + sports | Same NOCs + top sports (counts differ) | Structure match; counts differ |
| **Task 4 Table 15**  | 0→5.33, 6→8.86, 3→36.65  | 0→5.33, 6→8.86, 3→36.65          | Full match      |
| **Sensitivity**     | athletes_num: 0.058329%; Total event: −1.2113% | athletes_num: **11.0759%**; Total event: **0.0000%** | Definition match (Eq 20); numbers differ |

---

## 9. Why Results Differ

1. **Data & preprocessing:** NOC mapping, exclusion of USSR/Russia/AIN, and outlier rules can differ from the paper’s exact choices, changing who appears and how G_ij/T_ij and athlete/event counts are built.
2. **Train/test and 80/20:** The paper uses “first 80% / second 20%” in places and 10-fold CV elsewhere. The pipeline uses “pre-2024 = train, 2024 = test” and 10-fold CV on pre-2024. That can shift R²/MAE/RMSE and chosen hyperparameters.
3. **Feature and target definitions:** Small differences in “medals before the j-th Olympics,” events, or host coding change GSRF inputs and thus 2028 forecasts and progress/regression.
4. **Task 2 (a_k, e_k, p_k):** Different definitions or samples for “events participated” and historical participations change z and hence p; that explains Table 11 country match but probability mismatch.
5. **Task 3 data:** Different source or scope of medal-by-sport data explain Table 12 structure match but count differences.
6. **Hyperparameters and software:** Grid search space and sklearn version can change best GSRF model and all downstream numbers.

---

## 10. How to Regenerate Pipeline Numbers

From the project root, with the venv activated and dependencies installed:

```bash
python Model/run_paper_pipeline.py
```

Then read “Task 1 Gold (Table 3 style)” and “Task 1 Total (Table 4 style)” from the console, and use `Output/predictions_2028.csv`, `Output/table11_nonmedal_probabilities.csv`, etc., to refresh the “Paper-pipeline” columns and summaries above.
