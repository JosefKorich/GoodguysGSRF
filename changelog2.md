# Changelog 2: Next-Step Improvements

This document records changes made after the initial paper-faithful reproduction (changelog.md). All edits are marked by file and purpose.

---

## 1. Sensitivity: Raw-Space Perturbation (Paper §10)

**File:** `Model/run_paper_pipeline.py`

**Change:** Sensitivity was previously computed by adding a small constant (0.01) in **standardized** feature space, which produced large percentages (e.g. athletes_num ~11%, Total event ~0%). The paper reports athletes_num: 0.058329%; Total event: −1.2113%, which suggests perturbations in **raw** space and a signed average for events.

**Updates:**
- Perturbations are now done in **raw** space: Athletes and EventsTotal are multiplied by `(1 + 0.01)` (1% increase), then standardized with the training-set mean/std before prediction.
- A constant `SENSITIVITY_PERTURB_PCT = 0.01` is used so the perturbation size can be tuned.
- For **Total event**, the pipeline now reports both a **signed** average `(y_e - y_base)/y_base * 100` (to capture the paper’s −1.2113%) and an absolute average; the console prints the signed value.
- Formula remains ps = |Δy/y| per Eq (20); the average over countries is reported as a percentage.

**Rationale:** Aligning with the paper’s “change A_ij, E_ij” in raw units and reporting a signed sensitivity for events makes the pipeline comparable to the paper’s reported numbers (0.058329%; −1.2113%). Exact match still depends on data and aggregation; raw-space 1% is a defensible choice.

---

## 2. CLI: `--data_dir` and `--output_dir`

**File:** `Model/run_paper_pipeline.py`

**Change:** The pipeline previously used fixed `DATA_DIR` and `OUT_DIR` (project `Data/` and `Output/`). It now accepts optional command-line arguments so paths can be overridden without editing code.

**Updates:**
- `argparse` is used to define:
  - `--data_dir`: directory containing `summerOly_athletes.xlsx`, `summerOly_medal_counts.xlsx`, `summerOly_programs.xlsx`, `summerOly_hosts.csv`. Default: `{project_root}/Data`.
  - `--output_dir`: directory where all output CSVs are written. Default: `{project_root}/Output`.
- `main()` signature is `main(data_dir=None, output_dir=None)`. If `None`, the module defaults `DATA_DIR` and `OUT_DIR` are used.
- `_ensure_out(out_dir)` takes the chosen output directory and creates it if missing.
- All CSV writes use the chosen `output_dir` (passed into `main`).

**Usage:**
```bash
python Model/run_paper_pipeline.py
python Model/run_paper_pipeline.py --data_dir /path/to/data --output_dir /path/to/out
```

---

## 3. Verification Script

**File:** `verify_pipeline.py` (new, project root)

**Change:** A small verification script was added to check that (1) preprocessing excludes AIN, URS, RUS; (2) USA appears in the top 10 for 2028 gold and total; (3) Table 15 (Great Coach) matches the paper’s 0→5.33, 6→8.86, 3→36.65.

**Updates:**
- **Check 1:** Runs `run_preprocessing(data_dir)` and asserts that no team in the preprocessed data has `Team` in `EXCLUDE_NOC` (AIN, URS, RUS).
- **Check 2:** Reads `{output_dir}/predictions_2028.csv`, takes the top 10 by `PredictedGold2028` and by `PredictedTotal2028`, and asserts that USA appears in both.
- **Check 3:** Reads `{output_dir}/table15_great_coach_2028.csv` and asserts that the three rows have 2024 and 2028 values (0, 5.33), (6, 8.86), (3, 36.65) within 0.01.
- The script accepts optional `--data_dir` and `--output_dir` (defaults: `Data`, `Output`). Exit code 0 if all checks pass, 1 otherwise.

**Usage:**
```bash
python verify_pipeline.py
python verify_pipeline.py --data_dir Data --output_dir Output
```

---

## 4. Main Return Value and Sensitivity Print

**File:** `Model/run_paper_pipeline.py`

**Change:** `main()` now returns a single dictionary (instead of a tuple) and the sensitivity line in the console labels the Total event value as signed.

**Updates:**
- Return value is `{"prep": prep, "task1": task1, "task2": task2, "task3": task3, "task4": task4, "sens": sens}`.
- Console print: `"Total event (signed): ..."` so it is clear the event sensitivity can be negative (as in the paper’s −1.2113%).

---

## 5. Summary Table

| Item | File(s) | Purpose |
|------|---------|---------|
| Sensitivity in raw space | `Model/run_paper_pipeline.py` | Match paper §10: perturb A_ij and E_ij in raw units; report signed event sensitivity. |
| CLI `--data_dir`, `--output_dir` | `Model/run_paper_pipeline.py` | Override data and output paths without editing code. |
| Verification script | `verify_pipeline.py` | Check AIN/URS/RUS excluded, USA in top 10, Table 15 matches paper. |
| Main return + sensitivity label | `Model/run_paper_pipeline.py` | Consistent API and clearer console output. |

---

## 6. Version and Date

- **Changelog 2 version:** 1.0  
- **Date:** 2026-01-27  
- **Baseline:** changelog.md (paper-faithful reproduction); this changelog covers the next-step improvements above.
