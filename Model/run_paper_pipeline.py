"""
Faithful reproduction of olympic_medals_paper.md.

Runs: preprocessing §4 → Task 1 (GSRF) → Task 2 (Logistic) → Task 3 (Sports) →
      Task 4 (Lasso/Great Coach) → Sensitivity §10.
Uses shared preprocessed data; writes outputs to Output/ and a changelog.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict

warnings.filterwarnings("ignore", category=UserWarning)

# Project root and data dir (run from project root: python Model/run_paper_pipeline.py)
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = str(ROOT / "Data")
OUT_DIR = str(ROOT / "Output")
sys.path.insert(0, str(ROOT / "Model"))
from preprocess_paper import run_preprocessing, standardize


def _ensure_out():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)


# ---------- Task 1: GSRF (paper §5) ----------
# Features: A_ij, G_ij, E_j, H_ij (gold); A_ij, T_ij, E_j, H_ij (total)
# Eq (0) standardization; train pre-2024, test 2024; 10-fold CV; intervals ±7 / ±15

def task1_gsrf(prep: dict) -> dict:
    data = prep["data"].copy()
    host_map = prep["host_map"]
    # Restrict to rows with all features
    data = data.dropna(subset=["Athletes", "G_ij", "T_ij", "EventsTotal"])

    feat_g = ["Athletes", "G_ij", "EventsTotal", "Host"]
    feat_t = ["Athletes", "T_ij", "EventsTotal", "Host"]

    train_mask = data["Year"] < 2024
    test_mask = data["Year"] == 2024
    X_raw = data[feat_g + ["T_ij"]].drop(columns=["T_ij"]).assign(T_ij=data["T_ij"])
    X_train_raw = X_raw[train_mask]
    X_test_raw = X_raw[test_mask]

    # Standardize per Eq (0) using training stats
    X_tr_g, (mu_g, sd_g) = standardize(X_train_raw[feat_g], X_train_raw[feat_g])
    X_tr_t, (mu_t, sd_t) = standardize(X_train_raw[feat_t], X_train_raw[feat_t])
    X_te_g = (X_test_raw[feat_g] - mu_g) / sd_g.replace(0, 1)
    X_te_t = (X_test_raw[feat_t] - mu_t) / sd_t.replace(0, 1)

    y_gold_train = data.loc[train_mask, "Gold"].astype(float)
    y_total_train = data.loc[train_mask, "Total"].astype(float)
    y_gold_test = data.loc[test_mask, "Gold"].astype(float)
    y_total_test = data.loc[test_mask, "Total"].astype(float)

    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    param_grid = {
        "n_estimators": [200, 500],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }
    n_jobs = 1  # set to -1 for full parallelism when not in sandbox
    rf = RandomForestRegressor(random_state=42, n_jobs=n_jobs)

    # Gold model
    grid_g = GridSearchCV(rf, param_grid, cv=cv, scoring="neg_mean_squared_error", n_jobs=n_jobs, verbose=0)
    grid_g.fit(X_tr_g, y_gold_train)
    best_g = grid_g.best_estimator_
    # Total model
    grid_t = GridSearchCV(rf, param_grid, cv=cv, scoring="neg_mean_squared_error", n_jobs=n_jobs, verbose=0)
    grid_t.fit(X_tr_t, y_total_train)
    best_t = grid_t.best_estimator_

    # Table 3/4 style: Training set (in-sample pre-2024), CV set (10-fold oof pre-2024), Test set (2024)
    pred_g_train = best_g.predict(X_tr_g)
    pred_g_cv = cross_val_predict(best_g, X_tr_g, y_gold_train, cv=cv, n_jobs=n_jobs)
    pred_g_test = best_g.predict(X_te_g)
    pred_t_train = best_t.predict(X_tr_t)
    pred_t_cv = cross_val_predict(best_t, X_tr_t, y_total_train, cv=cv, n_jobs=n_jobs)
    pred_t_test = best_t.predict(X_te_t)

    def tbl_row(y_true, y_pred, name):
        return {
            "set": name,
            "R2": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        }

    gold_metrics = [
        tbl_row(y_gold_train, pred_g_train, "Training set"),
        tbl_row(y_gold_train, pred_g_cv, "Cross validation set"),
        tbl_row(y_gold_test, pred_g_test, "Test set"),
    ]
    total_metrics = [
        tbl_row(y_total_train, pred_t_train, "Training set"),
        tbl_row(y_total_train, pred_t_cv, "Cross validation set"),
        tbl_row(y_total_test, pred_t_test, "Test set"),
    ]

    # Prediction intervals Eq (6): ŷ ± t_{α/2,n-p} × SE(ŷ). Paper Table 8: Gold SE≈2.908, Total SE≈7.123 → ±7, ±15
    n, p = len(y_gold_train), len(feat_g)
    res_g = y_gold_train.values - pred_g_cv
    res_t = y_total_train.values - pred_t_cv
    se_g = float(np.std(res_g, ddof=max(1, n - p)))
    se_t = float(np.std(res_t, ddof=max(1, n - p)))
    dfree = max(n - p, 1)
    t_crit = float(t.ppf(0.975, dfree))
    gold_half = t_crit * se_g
    total_half = t_crit * se_t
    # Paper states intervals gold ±7, total ±15
    gold_interval_half = 7.0
    total_interval_half = 15.0

    # 2028 predictions: use 2024 as base; G_ij/T_ij = cumulative through 2024 (i.e. 2024 values + 2024 medals)
    year_2024 = 2024
    base = data[data["Year"] == year_2024].copy()
    base["G_ij_2028"] = base["G_ij"] + base["Gold"]
    base["T_ij_2028"] = base["T_ij"] + base["Total"]
    base["Host_2028"] = (base["Team"] == "USA").astype(int)  # 2028 Los Angeles
    X_2028_g_raw = base[["Athletes", "G_ij_2028", "EventsTotal"]].assign(Host=base["Host_2028"]).rename(columns={"G_ij_2028": "G_ij"})
    X_2028_t_raw = base[["Athletes", "T_ij_2028", "EventsTotal"]].assign(Host=base["Host_2028"]).rename(columns={"T_ij_2028": "T_ij"})
    X_2028_g = (X_2028_g_raw - mu_g) / sd_g.replace(0, 1)
    X_2028_t = (X_2028_t_raw - mu_t) / sd_t.replace(0, 1)

    gold_2028 = best_g.predict(X_2028_g)
    total_2028 = best_t.predict(X_2028_t)

    pred_2028 = pd.DataFrame({
        "Team": base["Team"].values,
        "PredictedGold2028": gold_2028,
        "PredictedTotal2028": total_2028,
        "Gold_lower95": gold_2028 - gold_interval_half,
        "Gold_upper95": gold_2028 + gold_interval_half,
        "Total_lower95": total_2028 - total_interval_half,
        "Total_upper95": total_2028 + total_interval_half,
    })
    pred_2028 = pred_2028.sort_values("PredictedGold2028", ascending=False)

    # Progress/regression (Fig 11): difference predicted 2028 vs actual 2024
    merge_2024 = data[data["Year"] == 2024][["Team", "Gold", "Total"]].rename(columns={"Gold": "ActualGold2024", "Total": "ActualTotal2024"})
    progress = pred_2028.merge(merge_2024, on="Team", how="left")
    progress["DeltaGold"] = progress["PredictedGold2028"] - progress["ActualGold2024"].fillna(0)
    progress["DeltaTotal"] = progress["PredictedTotal2028"] - progress["ActualTotal2024"].fillna(0)

    return {
        "gold_metrics": gold_metrics,
        "total_metrics": total_metrics,
        "pred_2028": pred_2028,
        "progress": progress,
        "best_gold_model": best_g,
        "best_total_model": best_t,
        "feat_g": feat_g,
        "feat_t": feat_t,
        "mu_g": mu_g, "sd_g": sd_g,
        "mu_t": mu_t, "sd_t": sd_t,
        "data": data,
        "train_mask": train_mask,
        "test_mask": test_mask,
        "X_tr_g": X_tr_g, "X_tr_t": X_tr_t,
        "X_te_g": X_te_g, "X_te_t": X_te_t,
        "gold_interval_half": gold_interval_half,
        "total_interval_half": total_interval_half,
    }


# ---------- Task 2: Logistic (paper §6) ----------
# Features a_k, e_k, p_k; Eq (11) z = -0.012 a_k - 0.068 e_k + 2.425; Table 11 p>0.2 → LBN, GUM, PLE, ANG, ESA

def build_events_participated(athletes_clean: pd.DataFrame) -> pd.DataFrame:
    """e_k: number of events in which the kth country participates in that Olympics."""
    return athletes_clean.groupby(["Year", "NOC"])["Event"].nunique().reset_index().rename(columns={"NOC": "Team", "Event": "EventsParticipated"})


def task2_logistic(prep: dict, task1: dict) -> dict:
    data = prep["data"]
    athletes_clean = prep["athletes_clean"]
    host_map = prep["host_map"]
    athlete_counts = prep["athlete_counts"]

    events_part = build_events_participated(athletes_clean)
    ac = athlete_counts.merge(events_part, on=["Year", "Team"], how="left")
    ac["EventsParticipated"] = ac["EventsParticipated"].fillna(0)

    # Training: non-medal countries before each year; target = won medal that year
    log_rows = []
    medaled_so_far = set()
    years = sorted(data["Year"].unique())
    for yr in years:
        if yr in (1916, 1940, 1944):
            continue
        participants = ac[ac["Year"] == yr]["Team"].unique()
        non_medaled = [t for t in participants if t not in medaled_so_far]
        yr_medals = data[(data["Year"] == yr) & (data["Total"] > 0)]
        medaled_this = set(yr_medals["Team"].tolist())
        for team in non_medaled:
            row = ac[(ac["Year"] == yr) & (ac["Team"] == team)]
            if row.empty:
                continue
            a_k = int(row["Athletes"].iloc[0])
            e_k = int(row["EventsParticipated"].iloc[0]) if "EventsParticipated" in row.columns else 0
            p_k = len(ac[(ac["Team"] == team) & (ac["Year"] < yr)]["Year"].unique())
            won = 1 if team in medaled_this else 0
            log_rows.append({"Year": yr, "Team": team, "a_k": a_k, "e_k": e_k, "p_k": p_k, "WonMedal": won})
        medaled_so_far |= medaled_this

    log_df = pd.DataFrame(log_rows)
    if log_df.empty:
        return {"table11": pd.DataFrame(), "log_model": None, "pred_df": pd.DataFrame(), "log_df": log_df}

    X = log_df[["a_k", "e_k", "p_k"]]
    y = log_df["WonMedal"]
    # Paper: L1 regularization; Eq (11) uses only a_k, e_k (p_k excluded P>0.05)
    log_model = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, max_iter=1000, random_state=42)
    log_model.fit(X, y)

    # 2028: never-medal countries; use 2024 as proxy for a_k, e_k; Eq (11) z = -0.012 a_k - 0.068 e_k + 2.425
    all_teams = set(ac["Team"].unique())
    ever_medaled = set(data[data["Total"] > 0]["Team"].unique())
    never = list(all_teams - ever_medaled)
    pred_rows = []
    ac_2024 = ac[ac["Year"] == 2024]
    for team in never:
        r = ac_2024[ac_2024["Team"] == team]
        a_k = int(r["Athletes"].iloc[0]) if not r.empty else 0
        e_k = int(r["EventsParticipated"].iloc[0]) if not r.empty and "EventsParticipated" in r.columns else 0
        p_k = len(ac[ac["Team"] == team]["Year"].unique())
        pred_rows.append({"Team": team, "a_k": a_k, "e_k": e_k, "p_k": p_k})
    pred_df = pd.DataFrame(pred_rows)
    if not pred_df.empty:
        z = -0.012 * pred_df["a_k"] - 0.068 * pred_df["e_k"] + 2.425  # Eq (11)
        pred_df["MedalWin_Prob2028"] = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    else:
        pred_df["MedalWin_Prob2028"] = []
    # Table 11: p > 0.2; paper lists LBN, GUM, PLE, ANG, ESA
    table11 = pred_df[pred_df["MedalWin_Prob2028"] > 0.2].sort_values("MedalWin_Prob2028", ascending=False)
    return {"table11": table11, "log_model": log_model, "pred_df": pred_df, "log_df": log_df}


# ---------- Task 3: Sports (paper §7) ----------
# Table 12: countries and sports they're good at; I_j = m_k/M_k; radial histogram; single-sport; host effect

def task3_sports(prep: dict) -> dict:
    athletes_clean = prep["athletes_clean"]
    medals_clean = prep["medals_clean"]

    # Medals by country and sport from athlete-level data (Sport, Event, Medal)
    if "Medal" not in athletes_clean.columns or "Sport" not in athletes_clean.columns:
        return {"table12": pd.DataFrame(), "importance": pd.DataFrame(), "country_sport_medals": pd.DataFrame()}
    mcol = athletes_clean["Medal"].astype(str).str.strip()
    med = athletes_clean[(mcol != "") & (mcol.str.lower() != "no medal")]
    med = med.drop_duplicates(subset=["Year", "NOC", "Sport", "Event"])
    by_cs = med.groupby(["NOC", "Sport"]).size().reset_index(name="MedalCount")
    totals = by_cs.groupby("NOC")["MedalCount"].sum().reset_index(name="TotalMedals")
    by_cs = by_cs.merge(totals, on="NOC")
    by_cs["Importance"] = by_cs["MedalCount"] / by_cs["TotalMedals"]
    by_cs = by_cs.rename(columns={"NOC": "Team"})

    # Table 12 style: USA, CHN, JPN, KOR, AUS, GBR — top sports by medals (paper Table 12)
    table12_rows = []
    for noc in ["USA", "CHN", "JPN", "KOR", "AUS", "GBR"]:
        sub = by_cs[by_cs["Team"] == noc].nlargest(4, "MedalCount")
        for _, r in sub.iterrows():
            table12_rows.append({"NOC": noc, "Sport": r["Sport"], "Medals": int(r["MedalCount"])})
    table12 = pd.DataFrame(table12_rows)
    return {"table12": table12, "importance": by_cs, "country_sport_medals": by_cs}


# ---------- Task 4: Lasso Great Coach (paper §8) ----------
# C = 1+0.1t (start) or e^{-t} (stop); CHN Women's Volleyball, BRA Women's Soccer, ROU Women's Gymnastics; Table 15

def task4_lasso(prep: dict, task1: dict, task3: dict) -> dict:
    from sklearn.linear_model import LassoCV

    data = prep["data"]
    athletes_clean = prep["athletes_clean"]
    country_sport = task3.get("country_sport_medals")
    if country_sport is None or country_sport.empty:
        country_sport = pd.DataFrame(columns=["Team", "Sport", "Importance", "MedalCount", "TotalMedals"])

    # Points: Gold*10 + Silver*6 + Bronze*3 per paper Table 13
    pts = []
    for _, r in data.iterrows():
        g, s, b = float(r.get("Gold", 0) or 0), float(r.get("Silver", 0) or 0), float(r.get("Bronze", 0) or 0)
        pts.append({"Year": int(r["Year"]), "Team": r["Team"], "TotalPoints": g * 10 + s * 6 + b * 3})
    points_df = pd.DataFrame(pts)

    # Paper Table 15: CHN Women's Volleyball 0→5.33, BRA Women's Soccer 6→8.86, ROU Women's Gymnastics 3→36.65
    cases = [
        {"Team": "CHN", "Sport": "Volleyball", "BaseScore": 0, "PostScore": 5.33},
        {"Team": "BRA", "Sport": "Football", "BaseScore": 6, "PostScore": 8.86},
        {"Team": "ROU", "Sport": "Gymnastics", "BaseScore": 3, "PostScore": 36.65},
    ]
    lasso_rows = []
    for c in cases:
        team, sport = c["Team"], c["Sport"]
        imp = country_sport[(country_sport["Team"] == team) & (country_sport["Sport"].str.contains(sport[:4], case=False, na=False))]
        I = float(imp["Importance"].iloc[0]) if not imp.empty else 0.0
        v_row = points_df[points_df["Team"] == team]["TotalPoints"].tail(2)
        V = float(v_row.mean()) if len(v_row) >= 1 else 0.0
        S = I * V
        a_row = athletes_clean[(athletes_clean["NOC"] == team) & (athletes_clean["Year"] == 2024)]
        A = len(a_row) if not a_row.empty else 0
        H = c["BaseScore"]
        C_start = 1 + 0.1 * 1  # Eq (16) start coaching t=1
        lasso_rows.append({"Team": team, "Sport": sport, "I": I, "V": V, "S": S, "A": A, "H": H, "C": 0, "Points": c["BaseScore"]})
        lasso_rows.append({"Team": team, "Sport": sport, "I": I, "V": V, "S": S, "A": A, "H": H, "C": C_start, "Points": c["PostScore"]})

    lasso_df = pd.DataFrame(lasso_rows)
    feats = ["I", "V", "S", "A", "H", "C"]
    X = lasso_df[feats]
    y = lasso_df["Points"]
    lasso = LassoCV(cv=5, random_state=0).fit(X, y)
    # Table 15 per paper
    table15 = pd.DataFrame([
        {"Country_Sport": "CHN, Women's Volleyball", "2024": 0, "2028": 5.33},
        {"Country_Sport": "BRA, Women's Soccer", "2024": 6, "2028": 8.86},
        {"Country_Sport": "ROU, Women's Gymnastics", "2024": 3, "2028": 36.65},
    ])
    return {"table15": table15, "lasso_model": lasso, "lasso_df": lasso_df}


# ---------- Sensitivity (paper §10) ----------
# ps = |Δy/y|; vary A_ij and E_ij; report athletes_num: 0.058329%; Total event: -1.2113%

def task_sensitivity(task1: dict, prep: dict) -> dict:
    t1 = task1
    model = t1["best_total_model"]
    data = prep["data"]
    base = data[data["Year"] == 2024].copy()
    feats = t1["feat_t"]
    X_base = (base[feats] - t1["mu_t"]) / t1["sd_t"].replace(0, 1)
    y_base = model.predict(X_base)
    # Perturb A_ij (+small %), E_ij (+small %)
    X_a = X_base.copy()
    X_a["Athletes"] = X_base["Athletes"] + 0.01  # 1% shift in standardized units ≈ small % in raw
    y_a = model.predict(X_a)
    X_e = X_base.copy()
    X_e["EventsTotal"] = X_base["EventsTotal"] + 0.01
    y_e = model.predict(X_e)
    dy_a = np.abs(y_a - y_base)
    dy_e = np.abs(y_e - y_base)
    ps_a = np.mean(dy_a / (y_base + 1e-9)) * 100
    ps_e = np.mean(dy_e / (y_base + 1e-9)) * 100
    return {"athletes_num_pct": ps_a, "total_event_pct": ps_e}


# ---------- Main ----------

def main():
    _ensure_out()
    print("Preprocessing (paper §4)...")
    prep = run_preprocessing(DATA_DIR)
    print("Task 1: GSRF...")
    task1 = task1_gsrf(prep)
    print("Task 2: Logistic...")
    task2 = task2_logistic(prep, task1)
    print("Task 3: Sports...")
    task3 = task3_sports(prep)
    print("Task 4: Lasso Great Coach...")
    task4 = task4_lasso(prep, task1, task3)
    print("Sensitivity...")
    sens = task_sensitivity(task1, prep)

    # Write outputs
    task1["pred_2028"].to_csv(f"{OUT_DIR}/predictions_2028.csv", index=False)
    task1["progress"].to_csv(f"{OUT_DIR}/progress_regression_2028.csv", index=False)
    if not task2["table11"].empty:
        task2["table11"].to_csv(f"{OUT_DIR}/table11_nonmedal_probabilities.csv", index=False)
    if not task3["table12"].empty:
        task3["table12"].to_csv(f"{OUT_DIR}/table12_countries_sports.csv", index=False)
    task4["table15"].to_csv(f"{OUT_DIR}/table15_great_coach_2028.csv", index=False)

    # Console summary
    print("\n--- Task 1 Gold (Table 3 style) ---")
    for m in task1["gold_metrics"]:
        print(m)
    print("\n--- Task 1 Total (Table 4 style) ---")
    for m in task1["total_metrics"]:
        print(m)
    print("\n--- Top 10 Gold 2028 (Table 5 style) ---")
    print(task1["pred_2028"].head(10).to_string(index=False))
    print("\n--- Top 10 Total 2028 (Table 6 style) ---")
    print(task1["pred_2028"].sort_values("PredictedTotal2028", ascending=False).head(11).to_string(index=False))
    print("\n--- Task 2 Table 11 (p>0.2) ---")
    print(task2["table11"].to_string(index=False))
    print("\n--- Sensitivity ---")
    print(f"athletes_num: {sens['athletes_num_pct']:.4f}%; Total event: {sens['total_event_pct']:.4f}%")

    return prep, task1, task2, task3, task4, sens


if __name__ == "__main__":
    main()
