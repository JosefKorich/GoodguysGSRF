"""
Enhanced Pipeline with EMA, Alpha Sensitivity, K-Fold Analysis, and Poisson Regression

This is the improved version of run_paper_pipeline.py with model innovations (v2.0).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Model"))

from preprocess_paper import run_preprocessing
from model_comparison import (
    compare_three_models,
    create_comparison_table,
    print_comparison_summary,
    standardize,
)
from alpha_sensitivity import run_alpha_sensitivity, create_alpha_comparison_table
from kfold_sensitivity import run_kfold_sensitivity_analysis
from poisson_medal_count import (
    prepare_first_timer_data,
    train_poisson_model,
    predict_never_medal_countries,
)


def _ensure_out(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def main(data_dir=None, output_dir=None):
    """
    Run enhanced pipeline with all model improvements.

    Args:
        data_dir: Directory containing input data files
        output_dir: Directory for output CSV files
    """
    if data_dir is None:
        data_dir = ROOT / "Data"
    else:
        data_dir = Path(data_dir)
    if output_dir is None:
        output_dir = ROOT / "Output_Enhanced"
    else:
        output_dir = Path(output_dir)

    _ensure_out(output_dir)

    print("=" * 80)
    print("ENHANCED OLYMPIC MEDALS PREDICTION PIPELINE")
    print("=" * 80)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # STEP 1: PREPROCESSING (including EMA)
    print("\n" + "=" * 80)
    print("STEP 1: PREPROCESSING & EMA COMPUTATION")
    print("=" * 80)

    prep = run_preprocessing(str(data_dir))
    data = prep["data"]

    print(f"\nPreprocessed data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")

    # STEP 2: MODEL COMPARISON (Cumulative vs EMA vs Hybrid)
    print("\n" + "=" * 80)
    print("STEP 2: MODEL COMPARISON")
    print("=" * 80)

    results_gold = compare_three_models(data, target="Gold")
    results_total = compare_three_models(data, target="Total")

    comparison_df = create_comparison_table(results_gold, results_total)
    comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)
    print_comparison_summary(comparison_df)

    print(f"\n✓ Saved: {output_dir / 'model_comparison.csv'}")

    for target in ["gold", "total"]:
        results = results_gold if target == "gold" else results_total
        for model_type in ["cumulative", "ema", "hybrid"]:
            fi_df = results[model_type]["feature_importance"]
            fi_df.to_csv(
                output_dir / f"feature_importance_{target}_{model_type}.csv",
                index=False,
            )

    # STEP 3: ALPHA SENSITIVITY ANALYSIS
    print("\n" + "=" * 80)
    print("STEP 3: ALPHA SENSITIVITY ANALYSIS")
    print("=" * 80)

    alphas = [0.1, 0.2, 0.3, 0.4, 0.5]
    alpha_results = run_alpha_sensitivity(data, prep["medals_clean"], alphas)
    alpha_df = create_alpha_comparison_table(alpha_results)
    alpha_df.to_csv(output_dir / "alpha_sensitivity.csv", index=False)

    print(f"\n✓ Saved: {output_dir / 'alpha_sensitivity.csv'}")

    # STEP 4: K-FOLD SENSITIVITY ANALYSIS
    print("\n" + "=" * 80)
    print("STEP 4: K-FOLD SENSITIVITY ANALYSIS")
    print("=" * 80)

    kfold_df = run_kfold_sensitivity_analysis(data)
    kfold_df.to_csv(output_dir / "kfold_sensitivity.csv", index=False)

    print(f"\n✓ Saved: {output_dir / 'kfold_sensitivity.csv'}")

    # STEP 5: POISSON REGRESSION FOR FIRST-TIME WINNERS
    print("\n" + "=" * 80)
    print("STEP 5: POISSON REGRESSION FOR MEDAL COUNTS")
    print("=" * 80)

    first_timer_data = prepare_first_timer_data(data, prep["athletes_clean"])
    first_timer_data.to_csv(
        output_dir / "first_timer_training_data.csv", index=False
    )

    if len(first_timer_data) > 0:
        poisson_model, poisson_metrics = train_poisson_model(first_timer_data)
        never_medal_pred = predict_never_medal_countries(
            poisson_model, data, prep["athletes_clean"], year=2024
        )
    else:
        never_medal_pred = pd.DataFrame(
            columns=[
                "Team",
                "Athletes",
                "EventsParticipated",
                "ExpectedMedals",
                "ProbAtLeastOne",
            ]
        )
        print("No first-timer training data; skipping Poisson predictions.")

    high_potential = never_medal_pred[
        (never_medal_pred["ExpectedMedals"] > 0.5)
        | (never_medal_pred["ProbAtLeastOne"] > 0.2)
    ]
    if len(high_potential) > 0:
        print("\nCountries with high medal potential (Expected > 0.5 OR Prob > 0.2):")
        print(high_potential.to_string(index=False))

    never_medal_pred.to_csv(
        output_dir / "poisson_never_medal_predictions.csv", index=False
    )
    print(f"\n✓ Saved: {output_dir / 'poisson_never_medal_predictions.csv'}")

    # STEP 6: 2028 PREDICTIONS (using hybrid model)
    print("\n" + "=" * 80)
    print("STEP 6: 2028 PREDICTIONS")
    print("=" * 80)

    data_2024 = data[data["Year"] == 2024].copy()
    if len(data_2024) == 0:
        print("No 2024 data; skipping 2028 predictions.")
        predictions_2028_sorted = pd.DataFrame()
    else:
        data_2028 = data_2024.copy()
        data_2028["Year"] = 2028
        data_2028["G_ij"] = data_2024["G_ij"] + data_2024["Gold"]
        data_2028["T_ij"] = data_2024["T_ij"] + data_2024["Total"]

        alpha_default = 0.3
        data_2028["EMA_Gold"] = (
            alpha_default * data_2024["Gold"]
            + (1 - alpha_default) * data_2024["EMA_Gold"]
        )
        data_2028["EMA_Total"] = (
            alpha_default * data_2024["Total"]
            + (1 - alpha_default) * data_2024["EMA_Total"]
        )
        data_2028["Host"] = (data_2028["Team"] == "USA").astype(int)

        features_gold_hybrid = [
            "Athletes",
            "G_ij",
            "EMA_Gold",
            "EventsTotal",
            "Host",
        ]
        features_total_hybrid = [
            "Athletes",
            "T_ij",
            "EMA_Total",
            "EventsTotal",
            "Host",
        ]

        X_2028_gold = data_2028[features_gold_hybrid].copy()
        X_2028_total = data_2028[features_total_hybrid].copy()

        mu_g = results_gold["hybrid"]["standardization"]["mu"]
        sd_g = results_gold["hybrid"]["standardization"]["sd"].replace(0, 1)
        mu_t = results_total["hybrid"]["standardization"]["mu"]
        sd_t = results_total["hybrid"]["standardization"]["sd"].replace(0, 1)

        X_2028_gold_std = (X_2028_gold - mu_g) / sd_g
        X_2028_total_std = (X_2028_total - mu_t) / sd_t

        pred_gold_2028 = results_gold["hybrid"]["model"].predict(X_2028_gold_std)
        pred_total_2028 = results_total["hybrid"]["model"].predict(
            X_2028_total_std
        )

        predictions_2028 = data_2028[["Team", "Athletes"]].copy()
        predictions_2028["PredictedGold2028"] = pred_gold_2028
        predictions_2028["PredictedTotal2028"] = pred_total_2028
        predictions_2028["Gold2024"] = data_2024["Gold"].values
        predictions_2028["Total2024"] = data_2024["Total"].values
        predictions_2028["DeltaGold"] = (
            pred_gold_2028 - data_2024["Gold"].values
        )
        predictions_2028["DeltaTotal"] = (
            pred_total_2028 - data_2024["Total"].values
        )

        predictions_2028_sorted = predictions_2028.sort_values(
            "PredictedGold2028", ascending=False
        )
        predictions_2028_sorted.to_csv(
            output_dir / "predictions_2028_hybrid.csv", index=False
        )

        print("\nTop 10 predicted gold medalists for 2028:")
        print(
            predictions_2028_sorted.head(10)[
                ["Team", "PredictedGold2028", "PredictedTotal2028", "DeltaGold"]
            ].to_string(index=False)
        )
        print(f"\n✓ Saved: {output_dir / 'predictions_2028_hybrid.csv'}")

    # FINAL SUMMARY
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nAll outputs saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - model_comparison.csv           (Cumulative vs EMA vs Hybrid)")
    print("  - alpha_sensitivity.csv          (Optimal alpha for EMA)")
    print("  - kfold_sensitivity.csv          (Optimal k for CV)")
    print("  - poisson_never_medal_predictions.csv  (Expected medal counts)")
    print("  - predictions_2028_hybrid.csv    (2028 predictions with best model)")
    print("  - feature_importance_*.csv       (Feature importance for each model)")

    return {
        "prep": prep,
        "model_comparison": comparison_df,
        "alpha_sensitivity": alpha_df,
        "kfold_sensitivity": kfold_df,
        "poisson_predictions": never_medal_pred,
        "predictions_2028": predictions_2028_sorted,
        "results_gold": results_gold,
        "results_total": results_total,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhanced Olympic medals prediction pipeline (v2.0)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing input data files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for output CSV files",
    )

    args = parser.parse_args()

    main(data_dir=args.data_dir, output_dir=args.output_dir)
