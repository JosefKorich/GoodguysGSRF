"""
GSRF.py

End-to-end medal prediction script using:
- Athlete counts per (Year, Team)
- Total events per Olympic year (from programs sheet)
- Host indicator
- RandomForestRegressor w/ GridSearchCV
- Cross-validated metrics + 2028 predictions + simple 95% interval

Run (recommended):
    source .venv/bin/activate
    python -m pip install pandas openpyxl scikit-learn scipy numpy
    python GSRF.py
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, Tuple

# Use 1 job in restricted envs (Cursor/sandbox); use all cores otherwise
N_JOBS = int(os.environ.get("GSRF_N_JOBS", "1"))

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict


# -----------------------------
# Helpers
# -----------------------------
def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Data files live in Data/ relative to project root; use paths that work when run from project root
    data_dir = "Data"
    athletes_df = pd.read_excel(f"{data_dir}/summerOly_athletes.xlsx")
    medals_df = pd.read_excel(f"{data_dir}/summerOly_medal_counts.xlsx")
    programs_df = pd.read_excel(f"{data_dir}/summerOly_programs.xlsx")
    hosts_df = pd.read_csv(f"{data_dir}/summerOly_hosts.csv")
    return athletes_df, medals_df, programs_df, hosts_df


def extract_year_columns(programs_df: pd.DataFrame) -> list[int]:
    """
    Returns a sorted list of year columns from programs_df.
    Handles int columns like 1896 and string columns like "2024".
    Ignores columns like "1906*" automatically (not digits).
    """
    year_cols: list[int] = []
    for c in programs_df.columns:
        if isinstance(c, (int, np.integer)):
            year_cols.append(int(c))
        elif isinstance(c, str) and c.strip().isdigit():
            year_cols.append(int(c))
    return sorted(set(year_cols))


def build_events_per_year(programs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a DataFrame:
        Year, EventsTotal
    where EventsTotal is the sum across all sports/disciplines for that year.
    Converts mixed types safely (strings/notes -> NaN -> 0).
    Skips 1906 (unofficial) if present.
    """
    year_cols = extract_year_columns(programs_df)

    rows = []
    for y in year_cols:
        if y == 1906:
            continue

        col_label = y if y in programs_df.columns else str(y)
        col_numeric = pd.to_numeric(programs_df[col_label], errors="coerce").fillna(0)
        rows.append({"Year": y, "EventsTotal": float(col_numeric.sum())})

    return pd.DataFrame(rows)


def build_host_map(hosts_df: pd.DataFrame) -> Dict[int, str]:
    """
    Creates year -> host country/team string.

    Note:
    Your hosts file has a 'Host' column; this strips parentheses content.
    Then we do a small normalization step for common naming differences.
    """
    if "Year" not in hosts_df.columns or "Host" not in hosts_df.columns:
        raise ValueError("hosts_df must contain columns: 'Year' and 'Host'")

    hosts_df = hosts_df.copy()
    hosts_df["Country"] = hosts_df["Host"].astype(str).str.replace(r"\(.*\)", "", regex=True).str.strip()

    host_map: Dict[int, str] = hosts_df.set_index("Year")["Country"].to_dict()

    # Minimal normalization (adjust/add if your Team labels differ)
    for yr, country in list(host_map.items()):
        if country == "United Kingdom":
            host_map[yr] = "Great Britain"
        elif country == "USA":
            host_map[yr] = "United States"
        elif country == "United States":
            host_map[yr] = "United States"

    return host_map


def safe_drop_rank(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Rank" in df.columns:
        df.drop(columns=["Rank"], inplace=True)
    return df


# -----------------------------
# Main pipeline
# -----------------------------
def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)

    # Load datasets
    athletes_df, medals_df, programs_df, hosts_df = load_inputs()

    # 1) Athlete counts per (Year, Team)
    # Use nunique of athlete name as proxy for unique athletes.
    if not {"Year", "Team", "Name"}.issubset(set(athletes_df.columns)):
        raise ValueError("athletes_df must have columns: Year, Team, Name")

    athlete_counts = (
        athletes_df.groupby(["Year", "Team"], as_index=False)["Name"]
        .nunique()
        .rename(columns={"Name": "Athletes"})
    )

    # 2) Events total per year from programs_df (robust numeric conversion)
    events_per_year = build_events_per_year(programs_df)

    # 3) Merge medals (ensure consistent Team column)
    medals_df = medals_df.copy()
    if "NOC" in medals_df.columns and "Team" not in medals_df.columns:
        medals_df.rename(columns={"NOC": "Team"}, inplace=True)

    needed_medal_cols = {"Year", "Team", "Gold", "Silver", "Bronze", "Total"}
    if not needed_medal_cols.issubset(set(medals_df.columns)):
        raise ValueError(f"medals_df must have columns: {sorted(needed_medal_cols)}")

    medals_df = safe_drop_rank(medals_df)

    data = pd.merge(athlete_counts, medals_df, on=["Year", "Team"], how="left")
    data.fillna({"Gold": 0, "Silver": 0, "Bronze": 0, "Total": 0}, inplace=True)

    # 4) Host indicator
    host_map = build_host_map(hosts_df)
    data["Host"] = (data["Year"].map(host_map) == data["Team"]).astype(int)

    # 5) Merge events total
    data = pd.merge(data, events_per_year, on="Year", how="left")
    # If an Olympic year exists in athlete/medal data but not in programs_df, fill EventsTotal with 0
    data["EventsTotal"] = data["EventsTotal"].fillna(0)

    # Features/targets
    features = ["Athletes", "EventsTotal", "Host"]
    X = data[features].copy()
    y_gold = data["Gold"].astype(float)
    y_total = data["Total"].astype(float)

    # CV / grid search
    param_grid = {
        "n_estimators": [200, 500],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    rf = RandomForestRegressor(random_state=42, n_jobs=N_JOBS)
    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    print("\nFitting Gold model (GridSearchCV)...")
    grid_gold = GridSearchCV(
        rf,
        param_grid,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=N_JOBS,
        verbose=0,
    )
    grid_gold.fit(X, y_gold)
    best_gold_model = grid_gold.best_estimator_
    print("Best Gold params:", grid_gold.best_params_)

    print("\nFitting Total model (GridSearchCV)...")
    grid_total = GridSearchCV(
        rf,
        param_grid,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=N_JOBS,
        verbose=0,
    )
    grid_total.fit(X, y_total)
    best_total_model = grid_total.best_estimator_
    print("Best Total params:", grid_total.best_params_)

    # Cross-validated performance (out-of-fold predictions)
    print("\nCross-validated evaluation (out-of-fold predictions)...")
    pred_gold_cv = cross_val_predict(best_gold_model, X, y_gold, cv=cv, n_jobs=N_JOBS)
    pred_total_cv = cross_val_predict(best_total_model, X, y_total, cv=cv, n_jobs=N_JOBS)

    r2_gold = r2_score(y_gold, pred_gold_cv)
    mae_gold = mean_absolute_error(y_gold, pred_gold_cv)
    rmse_gold = float(np.sqrt(mean_squared_error(y_gold, pred_gold_cv)))

    r2_total = r2_score(y_total, pred_total_cv)
    mae_total = mean_absolute_error(y_total, pred_total_cv)
    rmse_total = float(np.sqrt(mean_squared_error(y_total, pred_total_cv)))

    print(f"Gold model:  R^2={r2_gold:.3f} | MAE={mae_gold:.3f} | RMSE={rmse_gold:.3f}")
    print(f"Total model: R^2={r2_total:.3f} | MAE={mae_total:.3f} | RMSE={rmse_total:.3f}")

    # -----------------------------
    # Predict 2028
    # -----------------------------
    # Use 2024 as proxy for Athletes & base structure.
    # EventsTotal: by default, reuse 2024 EventsTotal.
    year_base = 2024
    if year_base not in data["Year"].unique():
        raise ValueError(f"Cannot create 2028 features: Year {year_base} not found in merged data.")

    data_base = data[data["Year"] == year_base].copy()
    X_2028 = data_base[features].copy()

    # Host in 2028: Los Angeles, USA
    X_2028["Host"] = (data_base["Team"] == "United States").astype(int).values

    # EventsTotal in 2028: use 2024 EventsTotal as proxy (you can override if you have 2028 count)
    if len(X_2028) > 0:
        X_2028["EventsTotal"] = float(data_base["EventsTotal"].iloc[0])

    gold_pred_2028 = best_gold_model.predict(X_2028)
    total_pred_2028 = best_total_model.predict(X_2028)

    pred_2028_df = pd.DataFrame(
        {
            "Team": data_base["Team"].values,
            "PredictedGold2028": gold_pred_2028,
            "PredictedTotal2028": total_pred_2028,
        }
    )

    # Simple 95% interval using residual std from CV (rough; not a strict prediction interval)
    residuals = (y_total - pred_total_cv).astype(float)
    resid_std = float(residuals.std(ddof=1))
    dfree = max(int(len(residuals) - 1), 1)
    t_crit = float(t.ppf(0.975, dfree))

    pred_2028_df["TotalMedals_lower95"] = pred_2028_df["PredictedTotal2028"] - t_crit * resid_std
    pred_2028_df["TotalMedals_upper95"] = pred_2028_df["PredictedTotal2028"] + t_crit * resid_std

    # Sort by predicted totals and show top 20
    pred_2028_df.sort_values("PredictedTotal2028", ascending=False, inplace=True)

    print("\nTop 20 predicted Total medals for 2028 (with rough 95% interval):")
    print(pred_2028_df.head(20).to_string(index=False))

    # Save output
    pred_2028_df.to_csv("predictions_2028.csv", index=False)
    print("\nSaved: predictions_2028.csv")


if __name__ == "__main__":
    main()

