# GSRF Model — Olympic Medal Prediction

A reproducible pipeline for predicting Summer Olympic medal standings, implementing the methodology from the paper **"Olympic Medals Unveiled: A Mathematical Exploration of Achievement Trends"** (MCM/ICM 2025, Problem C). The project uses a **Grid-Search Random Forest (GSRF)** model for gold and total medal prediction, plus logistic regression for first-time medal probability, sport importance analysis, and a Lasso-based "Great Coach" effect model.

## Features

- **Task 1 — GSRF**: Predicts gold and total medals using athlete counts, cumulative medals (G_ij, T_ij), total events, and host indicator. Train on pre-2024 data, test on 2024; 10-fold CV; 95% prediction intervals; 2028 projections (Los Angeles).
- **Task 2 — Logistic**: Classifies never-medal countries into likely/not-likely to win a first medal in 2028 (Table 11: countries with p > 0.2).
- **Task 3 — Sports**: Country–sport medal importance (I_j = m_k/M_k); Table 12-style breakdown for USA, CHN, JPN, KOR, AUS, GBR.
- **Task 4 — Lasso “Great Coach”**: Points-based model (Gold×10, Silver×6, Bronze×3); Table 15 projections for CHN Women’s Volleyball, BRA Women’s Soccer, ROU Women’s Gymnastics.
- **Sensitivity (§10)**: 1% perturbation in athletes and total events; reports relative change in predictions.

## Requirements

- Python 3.9+
- See `requirements.txt`:

```
pandas>=2.0
openpyxl>=3.0
scikit-learn>=1.0
scipy>=1.9
numpy>=1.21
```

## Installation

From the project root:

```bash
# Optional: use a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Data

Place the following files in the `Data/` directory:

| File | Description |
|------|-------------|
| `summerOly_athletes.xlsx` | Athlete-level data (NOC, Year, Sport, Event, Medal, etc.) |
| `summerOly_medal_counts.xlsx` | Country-year medal counts (Gold, Silver, Bronze, Total) |
| `summerOly_programs.xlsx` | Events per year (sport/discipline × year) |
| `summerOly_hosts.csv` | Host country per Olympic year |

Preprocessing (§4) excludes AIN, URS, and RUS; standardizes features per Eq (0); and builds merged country-year inputs for the models.

## Usage

**Run the full paper-faithful pipeline** (preprocessing → Task 1 → Task 2 → Task 3 → Task 4 → sensitivity). From the project root:

```bash
python Model/run_paper_pipeline.py
```

Optional arguments:

```bash
python Model/run_paper_pipeline.py --data_dir Data --output_dir Output
```

**Verify outputs** (exclusions, USA in top 10, Table 15 values):

```bash
python verify_pipeline.py [--data_dir Data] [--output_dir Output]
```

**Standalone GSRF script** (Task 1 only, no paper tables):

```bash
cd Model && python GSRF.py
```

## Project Structure

```
├── Data/                    # Input data (athletes, medals, programs, hosts)
├── Model/
│   ├── run_paper_pipeline.py   # Main pipeline (preprocess + Tasks 1–4 + sensitivity)
│   ├── preprocess_paper.py     # Paper §4 preprocessing
│   ├── GSRF.py                 # Standalone GSRF medal prediction
│   ├── LassoRegression.py      # Lasso / Great Coach components
│   ├── Regression.py           # Regression utilities
│   ├── Medalchart.py           # Medal chart helpers
│   └── sensitivity.py          # Sensitivity analysis
├── Output/                  # Generated CSVs (predictions, tables)
├── Markdowns/               # Paper and analysis notes (e.g. olympic_medals_paper.md)
├── requirements.txt
├── verify_pipeline.py       # Checks preprocessing and key outputs
└── README.md
```

## Outputs

After running the pipeline, `Output/` contains:

| File | Description |
|------|-------------|
| `predictions_2028.csv` | 2028 gold/total predictions and 95% intervals by country |
| `progress_regression_2028.csv` | Change vs 2024 (delta gold/total) for progress/regress view |
| `table11_nonmedal_probabilities.csv` | Never-medal countries with medal probability > 0.2 |
| `table12_countries_sports.csv` | Top sports by medals for selected countries |
| `table15_great_coach_2028.csv` | Great Coach 2024→2028 score projections (CHN, BRA, ROU) |

## References

- Methodology and task descriptions: `Markdowns/olympic_medals_paper.md`
- Reproducibility notes: `changelog.md`, `Markdowns/model_reproduction_analysis.md`

## License

For academic use in connection with MCM/ICM 2025 Problem C.
