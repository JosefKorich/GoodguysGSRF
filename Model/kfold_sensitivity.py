"""
K-Fold Cross-Validation Sensitivity Analysis

Tests different k values to optimize CV strategy and measure model stability.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from model_comparison import standardize


def evaluate_k_sensitivity(
    X_train,
    y_train,
    k_values=(3, 5, 7, 10, 15, 20),
    model_name="Model",
):
    """
    Test different k-fold values and return stability metrics.

    Args:
        X_train: Standardized training features
        y_train: Training target
        k_values: List of k values to test
        model_name: Name for reporting

    Returns:
        DataFrame with metrics for each k value
    """
    results = []

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
    )

    for k in k_values:
        if k > len(X_train):
            print(f"  Skipping k={k} (exceeds sample size {len(X_train)})")
            continue

        print(f"  Testing k={k}...")

        cv_scores = cross_val_score(rf, X_train, y_train, cv=k, scoring="r2")

        mean_r2 = cv_scores.mean()
        results.append({
            "Model": model_name,
            "k": k,
            "Mean_R2": mean_r2,
            "Std_R2": cv_scores.std(),
            "Min_R2": cv_scores.min(),
            "Max_R2": cv_scores.max(),
            "CV_Stability": (
                cv_scores.std() / abs(mean_r2) if mean_r2 != 0 else np.inf
            ),
            "Range_R2": cv_scores.max() - cv_scores.min(),
        })

    return pd.DataFrame(results)


def run_kfold_sensitivity_analysis(data):
    """
    Run k-fold sensitivity for all model variants.

    Returns:
        DataFrame with k-fold analysis for cumulative, EMA, and hybrid models
    """
    train = data[data["Year"] < 2024].copy()

    k_values = [3, 5, 7, 10, 15, 20]
    all_results = []

    # Cumulative gold
    print("\n" + "=" * 60)
    print("K-FOLD SENSITIVITY: Cumulative Gold Model")
    print("=" * 60)
    features = ["Athletes", "G_ij", "EventsTotal", "Host"]
    X_train = train[features].copy()
    X_train_std, _, _ = standardize(X_train)
    y_train = train["Gold"]
    results = evaluate_k_sensitivity(
        X_train_std, y_train, k_values, "Cumulative_Gold"
    )
    all_results.append(results)

    # EMA gold
    print("\n" + "=" * 60)
    print("K-FOLD SENSITIVITY: EMA Gold Model")
    print("=" * 60)
    features = ["Athletes", "EMA_Gold", "EventsTotal", "Host"]
    X_train = train[features].copy()
    X_train_std, _, _ = standardize(X_train)
    results = evaluate_k_sensitivity(
        X_train_std, y_train, k_values, "EMA_Gold"
    )
    all_results.append(results)

    # Cumulative total
    print("\n" + "=" * 60)
    print("K-FOLD SENSITIVITY: Cumulative Total Model")
    print("=" * 60)
    features = ["Athletes", "T_ij", "EventsTotal", "Host"]
    X_train = train[features].copy()
    X_train_std, _, _ = standardize(X_train)
    y_train = train["Total"]
    results = evaluate_k_sensitivity(
        X_train_std, y_train, k_values, "Cumulative_Total"
    )
    all_results.append(results)

    # EMA total
    print("\n" + "=" * 60)
    print("K-FOLD SENSITIVITY: EMA Total Model")
    print("=" * 60)
    features = ["Athletes", "EMA_Total", "EventsTotal", "Host"]
    X_train = train[features].copy()
    X_train_std, _, _ = standardize(X_train)
    results = evaluate_k_sensitivity(
        X_train_std, y_train, k_values, "EMA_Total"
    )
    all_results.append(results)

    combined = pd.concat(all_results, ignore_index=True)

    print("\n" + "=" * 80)
    print("K-FOLD SENSITIVITY SUMMARY")
    print("=" * 80)

    for model in combined["Model"].unique():
        subset = combined[combined["Model"] == model]
        best_k = subset.loc[subset["Mean_R2"].idxmax(), "k"]
        most_stable_k = subset.loc[subset["CV_Stability"].idxmin(), "k"]
        k10 = subset[subset["k"] == 10]
        r2_k10 = k10["Mean_R2"].values[0] if len(k10) > 0 else float("nan")
        print(f"\n{model}:")
        print(f"  Best R² at k={int(best_k)}")
        print(f"  Most stable at k={int(most_stable_k)}")
        print(f"  Paper uses k=10: R²={r2_k10:.4f}")

    return combined
