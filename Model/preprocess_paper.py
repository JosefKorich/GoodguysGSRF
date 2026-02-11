"""
Paper-faithful preprocessing per olympic_medals_paper.md §4.

- Outlier handling: clean team/country names (garbled, markers).
- Data standardization: x' = (x - x̄) / SD per Eq (0).
- ISO mapping: convert country names to NOC codes; exclude USSR/Russia; drop AIN.
- Athlete & event counts by country and year.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _data_dir() -> str:
    root = Path(__file__).resolve().parent.parent
    return str(root / "Data")


def load_raw(
    data_dir: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = data_dir or _data_dir()
    athletes_df = pd.read_excel(f"{data_dir}/summerOly_athletes.xlsx")
    medals_df = pd.read_excel(f"{data_dir}/summerOly_medal_counts.xlsx")
    programs_df = pd.read_excel(f"{data_dir}/summerOly_programs.xlsx")
    hosts_df = pd.read_csv(f"{data_dir}/summerOly_hosts.csv")
    return athletes_df, medals_df, programs_df, hosts_df


# Paper §4: "exclude USSR and Russia"; "AIN … should also be cleared"
EXCLUDE_NOC = {"AIN", "URS", "RUS"}

# Build full name -> NOC from athletes (prefer canonical country names).
def build_name_to_noc(athletes_df: pd.DataFrame) -> Dict[str, str]:
    a = athletes_df[["Team", "NOC"]].drop_duplicates()
    a = a[a["NOC"].astype(str).str.match(r"^[A-Z]{3}$", na=False)]
    # Prefer short/canonical names: United States, China, etc.
    name_to_noc = {}
    for _, row in a.iterrows():
        team = str(row["Team"]).strip().replace("\xa0", " ")
        noc = str(row["NOC"]).strip()
        if noc in EXCLUDE_NOC:
            continue
        # Prefer first occurrence for canonical names (often shorter)
        if team not in name_to_noc:
            name_to_noc[team] = noc
    # Paper Tables use USA, CHN, FRA, GBR, etc. Ensure common aliases.
    aliases = {
        "United States": "USA",
        "USA": "USA",
        "China": "CHN",
        "Great Britain": "GBR",
        "United Kingdom": "GBR",
        "France": "FRA",
        "Italy": "ITA",
        "Germany": "GER",
        "Japan": "JPN",
        "Australia": "AUS",
        "Canada": "CAN",
        "Netherlands": "NED",
        "Brazil": "BRA",
        "South Korea": "KOR",
        "Korea": "KOR",
        "Spain": "ESP",
        "Romania": "ROU",
        "Lebanon": "LBN",
        "Guam": "GUM",
        "Palestine": "PLE",
        "Angola": "ANG",
        "El Salvador": "ESA",
    }
    for k, v in aliases.items():
        if k not in name_to_noc:
            name_to_noc[k] = v
    return name_to_noc


def normalize_country(s: str) -> str:
    return str(s).strip().replace("\xa0", " ").strip()


def preprocess_athletes(
    athletes_df: pd.DataFrame, name_to_noc: Dict[str, str]
) -> pd.DataFrame:
    """Drop AIN, USSR, Russia; keep rows with valid NOC; use NOC as team key."""
    a = athletes_df.copy()
    a["NOC"] = a["NOC"].astype(str).str.strip()
    a = a[~a["NOC"].isin(EXCLUDE_NOC)]
    a = a[~a["Team"].astype(str).str.strip().str.upper().eq("AIN")]
    # Optional: restrict to rows where Team maps to same NOC (reduces boat-name noise).
    # For maximal fidelity we keep all rows with valid NOC and treat NOC as canonical.
    a = a[a["NOC"].str.match(r"^[A-Z]{3}$", na=False)]
    return a


def preprocess_medals(
    medals_df: pd.DataFrame, name_to_noc: Dict[str, str]
) -> pd.DataFrame:
    """Convert country names to NOC; exclude USSR/Russia per paper §4."""
    m = medals_df.copy()
    col = "NOC" if "NOC" in m.columns else "Team"
    m["Team"] = m[col].apply(lambda x: name_to_noc.get(normalize_country(x), None))
    m = m.dropna(subset=["Team"])
    m = m[~m["Team"].isin(EXCLUDE_NOC)]
    if col != "Team":
        m = m.drop(columns=[col], errors="ignore")
    if "Rank" in m.columns:
        m = m.drop(columns=["Rank"], errors="ignore")
    return m


def build_athlete_counts(athletes_clean: pd.DataFrame) -> pd.DataFrame:
    """Athlete counts per (Year, Team) with Team = NOC."""
    ac = (
        athletes_clean.groupby(["Year", "NOC"], as_index=False)["Name"]
        .nunique()
        .rename(columns={"NOC": "Team", "Name": "Athletes"})
    )
    return ac


def build_events_per_year(programs_df: pd.DataFrame) -> pd.DataFrame:
    """Total events per Olympic year. Skip 1906."""
    year_cols = []
    for c in programs_df.columns:
        if isinstance(c, (int, np.integer)):
            year_cols.append(int(c))
        elif isinstance(c, str) and c.strip().isdigit():
            year_cols.append(int(c))
    year_cols = sorted(set(y for y in year_cols if y != 1906))
    rows = []
    for y in year_cols:
        col = y if y in programs_df.columns else str(y)
        vals = pd.to_numeric(programs_df[col], errors="coerce").fillna(0)
        rows.append({"Year": y, "EventsTotal": float(vals.sum())})
    return pd.DataFrame(rows)


def build_host_map(hosts_df: pd.DataFrame, name_to_noc: Dict[str, str]) -> Dict[int, str]:
    """Year -> NOC of host. 2028 -> USA (Los Angeles)."""
    if "Year" not in hosts_df.columns or "Host" not in hosts_df.columns:
        raise ValueError("hosts_df must have Year and Host")
    host_map = {}
    for _, row in hosts_df.iterrows():
        try:
            yr = int(row["Year"])
        except Exception:
            continue
        h = str(row["Host"]).strip()
        if "Cancelled" in h or "cancelled" in h.lower():
            continue
        # " Los Angeles, United States" -> "United States"
        parts = [p.strip() for p in h.split(",")]
        country = parts[-1] if parts else ""
        country = country.replace("(postponed to 2021 due to the coronavirus pandemic)", "").strip()
        country = country.replace(" (postponed to 2021 due to the coronavirus pandemic)", "").strip()
        noc = name_to_noc.get(normalize_country(country)) or name_to_noc.get(country)
        if noc:
            host_map[yr] = noc
    return host_map


def build_cumulative_medals(medals_clean: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """G_ij and T_ij: cumulative gold and total medals *before* the j-th Olympics."""
    m = medals_clean.sort_values(["Team", "Year"])
    m = m[~m["Year"].isin([1916, 1940, 1944])]
    rows_g = []
    rows_t = []
    for team, g in m.groupby("Team"):
        cum_g = 0
        cum_t = 0
        for _, r in g.iterrows():
            yr = int(r["Year"])
            rows_g.append({"Year": yr, "Team": team, "G_ij": cum_g})
            rows_t.append({"Year": yr, "Team": team, "T_ij": cum_t})
            cum_g += int(r.get("Gold", 0) or 0)
            cum_t += int(r.get("Total", 0) or 0)
    return pd.DataFrame(rows_g), pd.DataFrame(rows_t)


def compute_ema_medals(medals_df: pd.DataFrame, alpha: float = 0.3) -> pd.DataFrame:
    """
    Compute exponential moving average of medal counts.

    Args:
        medals_df: DataFrame with columns [Year, Team, Gold, Silver, Bronze, Total]
        alpha: Smoothing factor (0 < alpha <= 1)
               Higher alpha = more weight on recent performance
               Common choices: 0.2 (slow), 0.3 (moderate), 0.4 (fast)

    Returns:
        DataFrame with columns [Year, Team, EMA_Gold, EMA_Total]

    Notes:
        EMA is computed in chronological order for each team.
        First year for each team uses actual medal count as initialization.
        Captures momentum/decline that cumulative metrics miss.
    """
    m = medals_df[~medals_df["Year"].isin([1916, 1940, 1944])].copy()
    ema_results = []

    for team in sorted(m["Team"].unique()):
        team_data = m[m["Team"] == team].sort_values("Year").copy()

        ema_gold = None
        ema_total = None

        for _, row in team_data.iterrows():
            year = int(row["Year"])
            gold = float(row.get("Gold", 0) or 0)
            total = float(row.get("Total", 0) or 0)

            if ema_gold is None:
                ema_gold = gold
                ema_total = total
            else:
                ema_gold = alpha * gold + (1 - alpha) * ema_gold
                ema_total = alpha * total + (1 - alpha) * ema_total

            ema_results.append({
                "Year": year,
                "Team": team,
                "EMA_Gold": ema_gold,
                "EMA_Total": ema_total,
            })

    return pd.DataFrame(ema_results)


def compute_ema_for_multiple_alphas(
    medals_df: pd.DataFrame,
    alphas: list = (0.1, 0.2, 0.3, 0.4, 0.5),
) -> Dict[float, pd.DataFrame]:
    """
    Compute EMA for multiple alpha values for sensitivity analysis.

    Args:
        medals_df: DataFrame with medal counts
        alphas: List of alpha values to test

    Returns:
        Dictionary mapping alpha -> DataFrame with EMA columns
    """
    ema_dict = {}
    for alpha in alphas:
        ema_df = compute_ema_medals(medals_df, alpha=alpha)
        ema_df = ema_df.rename(columns={
            "EMA_Gold": f"EMA_Gold_a{int(alpha * 10)}",
            "EMA_Total": f"EMA_Total_a{int(alpha * 10)}",
        })
        ema_dict[alpha] = ema_df
    return ema_dict


def standardize(X: pd.DataFrame, fit_df: pd.DataFrame | None = None) -> Tuple[pd.DataFrame, object]:
    """Eq (0): x' = (x - x̄) / SD. Returns (X_std, (means, stds)) for inverse."""
    if fit_df is None:
        fit_df = X
    means = fit_df.mean()
    stds = fit_df.std()
    stds = stds.replace(0, 1)
    X_std = (X - means) / stds
    return X_std, (means, stds)


def run_preprocessing(
    data_dir: str | None = None,
) -> dict:
    """Single entry: load, clean, build G_ij/T_ij, host_map, events, athlete_counts, medals_clean."""
    athletes_df, medals_df, programs_df, hosts_df = load_raw(data_dir)
    name_to_noc = build_name_to_noc(athletes_df)
    athletes_clean = preprocess_athletes(athletes_df, name_to_noc)
    medals_clean = preprocess_medals(medals_df, name_to_noc)
    athlete_counts = build_athlete_counts(athletes_clean)
    events_per_year = build_events_per_year(programs_df)
    host_map = build_host_map(hosts_df, name_to_noc)
    g_cum, t_cum = build_cumulative_medals(medals_clean)

    # Merge into one row per (Year, Team): Athletes, G_ij, T_ij, E_j, H_ij, Gold, Total
    data = athlete_counts.merge(medals_clean, on=["Year", "Team"], how="left")
    for c in ["Gold", "Silver", "Bronze", "Total"]:
        if c in data.columns:
            data[c] = data[c].fillna(0).astype(float)
    data = data.merge(g_cum, on=["Year", "Team"], how="left")
    data = data.merge(t_cum, on=["Year", "Team"], how="left")
    data["G_ij"] = data["G_ij"].fillna(0)
    data["T_ij"] = data["T_ij"].fillna(0)
    data["Host"] = (data["Year"].map(host_map) == data["Team"]).astype(int)
    data = data.merge(events_per_year, on="Year", how="left")
    data["EventsTotal"] = data["EventsTotal"].fillna(0)
    data = data[~data["Year"].isin([1916, 1940, 1944])]

    # Compute EMA for multiple alphas (for sensitivity analysis)
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5]
    ema_dict = compute_ema_for_multiple_alphas(medals_clean, alphas)

    # Merge default alpha=0.3 EMA into main data
    ema_default = compute_ema_medals(medals_clean, alpha=0.3)
    data = data.merge(ema_default, on=["Year", "Team"], how="left")
    data["EMA_Gold"] = data["EMA_Gold"].fillna(0.0)
    data["EMA_Total"] = data["EMA_Total"].fillna(0.0)

    return {
        "data": data,
        "ema_dict": ema_dict,
        "ema_default": ema_default,
        "athletes_clean": athletes_clean,
        "medals_clean": medals_clean,
        "athlete_counts": athlete_counts,
        "events_per_year": events_per_year,
        "host_map": host_map,
        "name_to_noc": name_to_noc,
        "programs_df": programs_df,
    }
