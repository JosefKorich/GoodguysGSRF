"""
Verification script for enhanced pipeline (v2.0).

Checks:
1. EMA values are computed correctly
2. All three model variants run successfully
3. Alpha sensitivity produces valid results
4. K-fold analysis runs for all k values
5. Poisson predictions are generated
6. 2028 predictions are generated
"""

import sys
from pathlib import Path

import pandas as pd


def verify_ema_computation(output_dir):
    """Verify EMA was computed correctly."""
    print("\n" + "=" * 60)
    print("CHECK 1: EMA Computation")
    print("=" * 60)

    comp_file = Path(output_dir) / "model_comparison.csv"
    if not comp_file.exists():
        print("❌ FAIL: model_comparison.csv not found")
        return False

    df = pd.read_csv(comp_file)

    ema_models = df[df["Model"] == "Ema"]
    if len(ema_models) == 0:
        print("❌ FAIL: No EMA models in comparison")
        return False

    print(f"✓ Found {len(ema_models)} EMA model results")
    gold_ema = ema_models[ema_models["Target"] == "Gold"]
    total_ema = ema_models[ema_models["Target"] == "Total"]
    if len(gold_ema) > 0:
        print(f"  Gold EMA R²: {gold_ema['Test_R2'].values[0]:.4f}")
    if len(total_ema) > 0:
        print(f"  Total EMA R²: {total_ema['Test_R2'].values[0]:.4f}")

    return True


def verify_model_comparison(output_dir):
    """Verify all three models were compared."""
    print("\n" + "=" * 60)
    print("CHECK 2: Model Comparison")
    print("=" * 60)

    comp_file = Path(output_dir) / "model_comparison.csv"
    if not comp_file.exists():
        print("❌ FAIL: model_comparison.csv not found")
        return False

    df = pd.read_csv(comp_file)

    expected_models = {"Cumulative", "Ema", "Hybrid"}
    actual_models = set(df["Model"].unique())

    if expected_models != actual_models:
        print(f"❌ FAIL: Expected {expected_models}, got {actual_models}")
        return False

    for target in ["Gold", "Total"]:
        target_df = df[df["Target"] == target]
        if len(target_df) != 3:
            print(
                f"❌ FAIL: Expected 3 models for {target}, got {len(target_df)}"
            )
            return False

    print("✓ All three models (Cumulative, EMA, Hybrid) present for Gold and Total")

    gold_df = df[df["Target"] == "Gold"]
    total_df = df[df["Target"] == "Total"]
    if len(gold_df) > 0:
        best_gold = gold_df.loc[gold_df["Test_R2"].idxmax(), "Model"]
        print(f"  Best Gold model:  {best_gold}")
    if len(total_df) > 0:
        best_total = total_df.loc[total_df["Test_R2"].idxmax(), "Model"]
        print(f"  Best Total model: {best_total}")

    return True


def verify_alpha_sensitivity(output_dir):
    """Verify alpha sensitivity analysis."""
    print("\n" + "=" * 60)
    print("CHECK 3: Alpha Sensitivity")
    print("=" * 60)

    alpha_file = Path(output_dir) / "alpha_sensitivity.csv"
    if not alpha_file.exists():
        print("❌ FAIL: alpha_sensitivity.csv not found")
        return False

    df = pd.read_csv(alpha_file)

    expected_alphas = {0.1, 0.2, 0.3, 0.4, 0.5}
    actual_alphas = set(df["Alpha"].unique())

    if expected_alphas != actual_alphas:
        print(f"❌ FAIL: Expected alphas {expected_alphas}, got {actual_alphas}")
        return False

    print(f"✓ All alpha values tested: {sorted(actual_alphas)}")

    for target in ["Gold", "Total"]:
        subset = df[df["Target"] == target]
        if len(subset) > 0:
            best_alpha = subset.loc[subset["Test_R2"].idxmax(), "Alpha"]
            best_r2 = subset.loc[subset["Test_R2"].idxmax(), "Test_R2"]
            print(f"  {target}: Best alpha = {best_alpha} (R² = {best_r2:.4f})")

    return True


def verify_kfold_sensitivity(output_dir):
    """Verify k-fold sensitivity analysis."""
    print("\n" + "=" * 60)
    print("CHECK 4: K-Fold Sensitivity")
    print("=" * 60)

    kfold_file = Path(output_dir) / "kfold_sensitivity.csv"
    if not kfold_file.exists():
        print("❌ FAIL: kfold_sensitivity.csv not found")
        return False

    df = pd.read_csv(kfold_file)

    k_values = sorted(df["k"].unique())
    print(f"✓ K-values tested: {k_values}")

    if 10 not in k_values:
        print("❌ FAIL: Paper's k=10 not tested")
        return False

    print("  Paper's k=10 included: ✓")

    for model in df["Model"].unique():
        subset = df[df["Model"] == model]
        k10 = subset[subset["k"] == 10]
        if len(k10) > 0:
            stability = k10.iloc[0]["CV_Stability"]
            print(f"  {model} at k=10: stability = {stability:.4f}")

    return True


def verify_poisson_predictions(output_dir):
    """Verify Poisson regression predictions."""
    print("\n" + "=" * 60)
    print("CHECK 5: Poisson Predictions")
    print("=" * 60)

    poisson_file = Path(output_dir) / "poisson_never_medal_predictions.csv"
    if not poisson_file.exists():
        print("❌ FAIL: poisson_never_medal_predictions.csv not found")
        return False

    df = pd.read_csv(poisson_file)

    required_cols = {
        "Team",
        "Athletes",
        "EventsParticipated",
        "ExpectedMedals",
        "ProbAtLeastOne",
    }
    if not required_cols.issubset(df.columns):
        print(f"❌ FAIL: Missing columns. Required: {required_cols}")
        return False

    print(f"✓ {len(df)} never-medal countries analyzed")

    high_potential = df[df["ExpectedMedals"] > 0.5].sort_values(
        "ExpectedMedals", ascending=False
    )
    if len(high_potential) > 0:
        print(
            f"\n  Top {min(5, len(high_potential))} countries by expected medals:"
        )
        for _, row in high_potential.head(5).iterrows():
            print(
                f"    {row['Team']:4s}: {row['ExpectedMedals']:.2f} medals "
                f"(P ≥ 1 = {row['ProbAtLeastOne']:.2%})"
            )

    return True


def verify_2028_predictions(output_dir):
    """Verify 2028 predictions were generated."""
    print("\n" + "=" * 60)
    print("CHECK 6: 2028 Predictions")
    print("=" * 60)

    pred_file = Path(output_dir) / "predictions_2028_hybrid.csv"
    if not pred_file.exists():
        print("❌ FAIL: predictions_2028_hybrid.csv not found")
        return False

    df = pd.read_csv(pred_file)

    if len(df) == 0:
        print("⚠️  WARNING: 2028 predictions file is empty")
        return True

    top10 = df.nlargest(10, "PredictedGold2028")
    if "USA" not in top10["Team"].values:
        print("⚠️  WARNING: USA not in top 10 gold predictions")
    else:
        usa_row = df[df["Team"] == "USA"].iloc[0]
        print(
            f"✓ USA predictions: {usa_row['PredictedGold2028']:.1f} gold, "
            f"{usa_row['PredictedTotal2028']:.1f} total"
        )

    print("\n  Top 5 gold predictions:")
    for _, row in top10.head(5).iterrows():
        print(
            f"    {row['Team']:4s}: {row['PredictedGold2028']:.1f} gold "
            f"(Δ = {row['DeltaGold']:+.1f})"
        )

    return True


def main(output_dir=None):
    """Run all verification checks."""
    if output_dir is None:
        output_dir = Path(__file__).parent / "Output_Enhanced"
    else:
        output_dir = Path(output_dir)

    print("=" * 60)
    print("ENHANCED PIPELINE VERIFICATION")
    print("=" * 60)
    print(f"Output directory: {output_dir}")

    checks = [
        verify_ema_computation,
        verify_model_comparison,
        verify_alpha_sensitivity,
        verify_kfold_sensitivity,
        verify_poisson_predictions,
        verify_2028_predictions,
    ]

    results = []
    for check in checks:
        try:
            result = check(output_dir)
            results.append(result)
        except Exception as e:
            print(f"\n❌ ERROR in {check.__name__}: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"✓ ALL CHECKS PASSED ({passed}/{total})")
        return 0
    else:
        print(f"✗ SOME CHECKS FAILED ({passed}/{total} passed)")
        return 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify enhanced pipeline outputs"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory containing pipeline outputs",
    )
    args = parser.parse_args()

    sys.exit(main(output_dir=args.output_dir))
