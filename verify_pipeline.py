"""
Verification checks for the paper-faithful pipeline.

- Preprocessing: AIN, URS, RUS are excluded from data.
- 2028 predictions: USA appears in top 10 by gold and by total.
- Table 15: CHN/BRA/ROU match paper values (0→5.33, 6→8.86, 3→36.65).

Run from project root:
  python verify_pipeline.py [--data_dir Data] [--output_dir Output]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Model"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify pipeline: exclusions, USA top 10, Table 15.")
    parser.add_argument("--data_dir", type=str, default=str(ROOT / "Data"), help="Data directory")
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "Output"), help="Output directory with CSVs")
    args = parser.parse_args()

    import pandas as pd
    from preprocess_paper import run_preprocessing, EXCLUDE_NOC

    errors = []

    # 1) Preprocessing: AIN, URS, RUS must not appear in preprocessed data
    print("Check 1: Preprocessing excludes AIN, URS, RUS...")
    prep = run_preprocessing(args.data_dir)
    teams = set(prep["data"]["Team"].unique())
    for noc in EXCLUDE_NOC:
        if noc in teams:
            errors.append(f"EXCLUDE_NOC {noc} still present in preprocessed data.")
    if not errors:
        print("  OK: No AIN, URS, RUS in preprocessed data.")

    # 2) USA in top 10 gold and top 10 total (2028 predictions)
    print("Check 2: USA in top 10 gold and top 10 total (2028)...")
    pred_path = Path(args.output_dir) / "predictions_2028.csv"
    if not pred_path.exists():
        errors.append(f"Missing {pred_path}; run pipeline first.")
    else:
        pred = pd.read_csv(pred_path)
        top10_gold = pred.nlargest(10, "PredictedGold2028")["Team"].tolist()
        top10_total = pred.nlargest(10, "PredictedTotal2028")["Team"].tolist()
        if "USA" not in top10_gold:
            errors.append("USA not in top 10 by PredictedGold2028.")
        if "USA" not in top10_total:
            errors.append("USA not in top 10 by PredictedTotal2028.")
        if not any(e.startswith("USA") for e in errors):
            print("  OK: USA in top 10 gold and top 10 total.")

    # 3) Table 15 matches paper (CHN 0→5.33, BRA 6→8.86, ROU 3→36.65)
    print("Check 3: Table 15 matches paper values...")
    table15_path = Path(args.output_dir) / "table15_great_coach_2028.csv"
    if not table15_path.exists():
        errors.append(f"Missing {table15_path}; run pipeline first.")
    else:
        t15 = pd.read_csv(table15_path)
        paper_values = [(0, 5.33), (6, 8.86), (3, 36.65)]  # CHN, BRA, ROU rows
        c24 = "2024" if "2024" in t15.columns else t15.columns[1]
        c28 = "2028" if "2028" in t15.columns else t15.columns[2]
        for i, (y2024, y2028) in enumerate(paper_values):
            if i >= len(t15):
                errors.append(f"Table 15: row {i} missing.")
                break
            r = t15.iloc[i]
            v24, v28 = float(r[c24]), float(r[c28])
            if abs(v24 - y2024) > 0.01 or abs(v28 - y2028) > 0.01:
                errors.append(f"Table 15 row {i}: expected 2024={y2024}, 2028={y2028}; got {v24}, {v28}.")
        if not any("Table 15" in e for e in errors):
            print("  OK: Table 15 matches paper (0→5.33, 6→8.86, 3→36.65).")

    if errors:
        print("\nFAILED:")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
