"""
Model Comparison: Cumulative vs EMA vs Hybrid

Compares three approaches to historical medal weighting:
1. Cumulative (paper baseline)
2. EMA (exponential moving average)
3. Hybrid (cumulative + EMA)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")


def standardize(X, fit_df=None):
    """Standardize features: x' = (x - mean) / std. Returns (X_std, mu, sd)."""
    if fit_df is None:
        fit_df = X
    mu = fit_df.mean()
    sd = fit_df.std()
    sd = sd.replace(0, 1)
    return (X - mu) / sd, mu, sd


def train_and_evaluate_model(
    X_train, y_train, X_test, y_test, model_name, param_grid=None
):
    """
    Train a random forest model with optional grid search and evaluate.

    Returns:
        Dictionary with model, predictions, and metrics
    """
    if param_grid is None:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        }

    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid, cv=10, scoring="r2", n_jobs=1, verbose=0
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_train_pred = best_model.predict(X_train)
    r2_train = r2_score(y_train, y_train_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

    cv_scores = cross_val_score(
        best_model, X_train, y_train, cv=10, scoring="r2"
    )
    y_cv_pred = cross_val_predict(best_model, X_train, y_train, cv=10)
    r2_cv = cv_scores.mean()
    mae_cv = mean_absolute_error(y_train, y_cv_pred)
    rmse_cv = np.sqrt(mean_squared_error(y_train, y_cv_pred))

    y_test_pred = best_model.predict(X_test)
    r2_test = r2_score(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

    feature_importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": best_model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    return {
        "model": best_model,
        "model_name": model_name,
        "best_params": grid_search.best_params_,
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
        "metrics": {
            "train": {"R2": r2_train, "MAE": mae_train, "RMSE": rmse_train},
            "cv": {"R2": r2_cv, "MAE": mae_cv, "RMSE": rmse_cv},
            "test": {"R2": r2_test, "MAE": mae_test, "RMSE": rmse_test},
        },
        "feature_importance": feature_importance,
    }


def compare_three_models(data_with_ema, target="Gold"):
    """
    Compare three models: Cumulative, EMA, and Hybrid.

    Args:
        data_with_ema: DataFrame with columns including:
            Athletes, G_ij, T_ij, EMA_Gold, EMA_Total, EventsTotal, Host,
            Gold, Total (targets)
        target: 'Gold' or 'Total'

    Returns:
        Dictionary with results for all three models
    """
    train = data_with_ema[data_with_ema["Year"] < 2024].copy()
    test = data_with_ema[data_with_ema["Year"] == 2024].copy()

    if target == "Gold":
        features_cumulative = ["Athletes", "G_ij", "EventsTotal", "Host"]
        features_ema = ["Athletes", "EMA_Gold", "EventsTotal", "Host"]
        features_hybrid = ["Athletes", "G_ij", "EMA_Gold", "EventsTotal", "Host"]
        target_col = "Gold"
    else:
        features_cumulative = ["Athletes", "T_ij", "EventsTotal", "Host"]
        features_ema = ["Athletes", "EMA_Total", "EventsTotal", "Host"]
        features_hybrid = ["Athletes", "T_ij", "EMA_Total", "EventsTotal", "Host"]
        target_col = "Total"

    results = {}

    # Model 1: Cumulative (baseline)
    print(f"\n{'='*60}")
    print(f"Training CUMULATIVE model for {target}...")
    X_train_c = train[features_cumulative].copy()
    X_test_c = test[features_cumulative].copy()
    X_train_c_std, mu_c, sd_c = standardize(X_train_c)
    X_test_c_std, _, _ = standardize(X_test_c, X_train_c)

    results["cumulative"] = train_and_evaluate_model(
        X_train_c_std, train[target_col],
        X_test_c_std, test[target_col],
        f"Cumulative_{target}",
    )
    results["cumulative"]["features"] = features_cumulative
    results["cumulative"]["standardization"] = {"mu": mu_c, "sd": sd_c}

    # Model 2: EMA
    print(f"\nTraining EMA model for {target}...")
    X_train_e = train[features_ema].copy()
    X_test_e = test[features_ema].copy()
    X_train_e_std, mu_e, sd_e = standardize(X_train_e)
    X_test_e_std, _, _ = standardize(X_test_e, X_train_e)

    results["ema"] = train_and_evaluate_model(
        X_train_e_std, train[target_col],
        X_test_e_std, test[target_col],
        f"EMA_{target}",
    )
    results["ema"]["features"] = features_ema
    results["ema"]["standardization"] = {"mu": mu_e, "sd": sd_e}

    # Model 3: Hybrid
    print(f"\nTraining HYBRID model for {target}...")
    X_train_h = train[features_hybrid].copy()
    X_test_h = test[features_hybrid].copy()
    X_train_h_std, mu_h, sd_h = standardize(X_train_h)
    X_test_h_std, _, _ = standardize(X_test_h, X_train_h)

    results["hybrid"] = train_and_evaluate_model(
        X_train_h_std, train[target_col],
        X_test_h_std, test[target_col],
        f"Hybrid_{target}",
    )
    results["hybrid"]["features"] = features_hybrid
    results["hybrid"]["standardization"] = {"mu": mu_h, "sd": sd_h}

    return results


def create_comparison_table(results_gold, results_total):
    """
    Create a comparison table of all models.

    Returns:
        DataFrame with rows for each model variant and metrics
    """
    rows = []

    for target, results in [("Gold", results_gold), ("Total", results_total)]:
        for model_type in ["cumulative", "ema", "hybrid"]:
            metrics = results[model_type]["metrics"]
            rows.append({
                "Target": target,
                "Model": model_type.title(),
                "Train_R2": metrics["train"]["R2"],
                "Train_MAE": metrics["train"]["MAE"],
                "Train_RMSE": metrics["train"]["RMSE"],
                "CV_R2": metrics["cv"]["R2"],
                "CV_MAE": metrics["cv"]["MAE"],
                "CV_RMSE": metrics["cv"]["RMSE"],
                "Test_R2": metrics["test"]["R2"],
                "Test_MAE": metrics["test"]["MAE"],
                "Test_RMSE": metrics["test"]["RMSE"],
            })

    return pd.DataFrame(rows)


def print_comparison_summary(comparison_df):
    """Print a formatted summary of model comparison."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)

    for target in ["Gold", "Total"]:
        print(f"\n{target.upper()} MEDALS:")
        print("-" * 80)
        subset = comparison_df[comparison_df["Target"] == target]

        best_test_r2 = subset.loc[subset["Test_R2"].idxmax(), "Model"]
        best_test_mae = subset.loc[subset["Test_MAE"].idxmin(), "Model"]

        for _, row in subset.iterrows():
            print(
                f"\n  {row['Model']:12s} | Test R²: {row['Test_R2']:.3f} | "
                f"MAE: {row['Test_MAE']:.3f} | RMSE: {row['Test_RMSE']:.3f}"
            )
            print(
                f"               | CV R²:   {row['CV_R2']:.3f} | "
                f"MAE: {row['CV_MAE']:.3f} | RMSE: {row['CV_RMSE']:.3f}"
            )

        print(f"\n  → Best Test R²:  {best_test_r2}")
        print(f"  → Best Test MAE: {best_test_mae}")
