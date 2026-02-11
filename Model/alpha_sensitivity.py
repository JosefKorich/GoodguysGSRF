"""
Alpha Sensitivity Analysis for EMA Models

Tests different alpha values to find optimal balance between
recent performance and historical trends.
"""

import pandas as pd
import numpy as np

from model_comparison import train_and_evaluate_model, standardize
from preprocess_paper import compute_ema_medals


def run_alpha_sensitivity(
    data, medals_clean, alphas=(0.1, 0.2, 0.3, 0.4, 0.5)
):
    """
    Test different alpha values for EMA and compare model performance.

    Args:
        data: Base DataFrame with all features (must have G_ij, T_ij, etc.)
        medals_clean: Cleaned medals data for EMA computation
        alphas: List of alpha values to test

    Returns:
        Dictionary with results for each alpha
    """
    results = {"gold": {}, "total": {}}

    for alpha in alphas:
        print(f"\n{'='*60}")
        print(f"Testing alpha = {alpha}")
        print("=" * 60)

        ema_df = compute_ema_medals(medals_clean, alpha=alpha)
        suffix = f"a{int(alpha * 10)}"
        ema_renamed = ema_df.rename(columns={
            "EMA_Gold": f"EMA_Gold_{suffix}",
            "EMA_Total": f"EMA_Total_{suffix}",
        })

        data_alpha = data.copy()
        data_alpha = data_alpha.merge(
            ema_renamed, on=["Year", "Team"], how="left"
        )
        data_alpha[f"EMA_Gold_{suffix}"] = data_alpha[f"EMA_Gold_{suffix}"].fillna(0.0)
        data_alpha[f"EMA_Total_{suffix}"] = data_alpha[f"EMA_Total_{suffix}"].fillna(0.0)

        train = data_alpha[data_alpha["Year"] < 2024].copy()
        test = data_alpha[data_alpha["Year"] == 2024].copy()

        # Gold model
        features_gold = ["Athletes", f"EMA_Gold_{suffix}", "EventsTotal", "Host"]
        X_train_g = train[features_gold].copy()
        X_test_g = test[features_gold].copy()
        X_train_g_std, mu_g, sd_g = standardize(X_train_g)
        X_test_g_std, _, _ = standardize(X_test_g, X_train_g)

        results["gold"][alpha] = train_and_evaluate_model(
            X_train_g_std, train["Gold"],
            X_test_g_std, test["Gold"],
            f"EMA_Gold_alpha{alpha}",
        )

        # Total model
        features_total = ["Athletes", f"EMA_Total_{suffix}", "EventsTotal", "Host"]
        X_train_t = train[features_total].copy()
        X_test_t = test[features_total].copy()
        X_train_t_std, mu_t, sd_t = standardize(X_train_t)
        X_test_t_std, _, _ = standardize(X_test_t, X_train_t)

        results["total"][alpha] = train_and_evaluate_model(
            X_train_t_std, train["Total"],
            X_test_t_std, test["Total"],
            f"EMA_Total_alpha{alpha}",
        )

    return results


def create_alpha_comparison_table(alpha_results):
    """Create table comparing all alpha values."""
    rows = []

    for target in ["gold", "total"]:
        for alpha, result in alpha_results[target].items():
            metrics = result["metrics"]
            rows.append({
                "Target": target.title(),
                "Alpha": alpha,
                "Test_R2": metrics["test"]["R2"],
                "Test_MAE": metrics["test"]["MAE"],
                "Test_RMSE": metrics["test"]["RMSE"],
                "CV_R2": metrics["cv"]["R2"],
                "CV_MAE": metrics["cv"]["MAE"],
                "CV_RMSE": metrics["cv"]["RMSE"],
            })

    df = pd.DataFrame(rows)

    print("\n" + "=" * 80)
    print("OPTIMAL ALPHA VALUES")
    print("=" * 80)

    for target in ["Gold", "Total"]:
        subset = df[df["Target"] == target]
        if len(subset) == 0:
            continue
        best_r2 = subset.loc[subset["Test_R2"].idxmax()]
        best_mae = subset.loc[subset["Test_MAE"].idxmin()]

        print(f"\n{target}:")
        print(f"  Best R² :  alpha = {best_r2['Alpha']} (R² = {best_r2['Test_R2']:.4f})")
        print(f"  Best MAE:  alpha = {best_mae['Alpha']} (MAE = {best_mae['Test_MAE']:.4f})")

    return df
