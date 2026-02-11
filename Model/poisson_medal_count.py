"""
Poisson Regression for First-Time Medal Winners

Predicts expected medal count (not just probability) for countries
that have never won Olympic medals.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _team_key(athletes_clean):
    """Column to use as team identifier (NOC or Team)."""
    if "NOC" in athletes_clean.columns:
        return "NOC"
    return "Team"


def _events_col(athletes_clean):
    """Column for events/sports count (Event or Sport)."""
    if "Event" in athletes_clean.columns:
        return "Event"
    if "Sport" in athletes_clean.columns:
        return "Sport"
    return None


def prepare_first_timer_data(data, athletes_clean):
    """
    Prepare training data from countries that won their first medal.

    Returns:
        DataFrame with features and medal counts for first-time winners
    """
    team_col = _team_key(athletes_clean)
    ev_col = _events_col(athletes_clean)

    first_medals = []

    for team in data["Team"].unique():
        team_data = data[data["Team"] == team].sort_values("Year")

        medal_years = team_data[team_data["Total"] > 0]
        if len(medal_years) == 0:
            continue

        first_year = int(medal_years.iloc[0]["Year"])
        first_record = medal_years.iloc[0]

        team_athletes = athletes_clean[
            (athletes_clean[team_col] == team)
            & (athletes_clean["Year"] == first_year)
        ]

        if len(team_athletes) == 0:
            continue

        athletes_count = len(team_athletes)
        if ev_col:
            events_participated = team_athletes[ev_col].nunique()
        else:
            events_participated = 0

        first_medals.append({
            "Team": team,
            "Year": first_year,
            "Athletes": athletes_count,
            "EventsParticipated": events_participated,
            "TotalMedals": first_record["Total"],
            "Gold": first_record.get("Gold", 0),
            "Silver": first_record.get("Silver", 0),
            "Bronze": first_record.get("Bronze", 0),
        })

    return pd.DataFrame(first_medals)


def train_poisson_model(first_timer_data):
    """
    Train Poisson regression to predict medal count for first-timers.

    Returns:
        Fitted model and evaluation metrics
    """
    if len(first_timer_data) == 0:
        raise ValueError("No first-timer data to train on")

    X = first_timer_data[["Athletes", "EventsParticipated"]]
    y = first_timer_data["TotalMedals"]

    poisson = PoissonRegressor(max_iter=1000, alpha=0.1)
    poisson.fit(X, y)

    y_pred = poisson.predict(X)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    print("\n" + "=" * 60)
    print("POISSON REGRESSION: First-Time Medal Winners")
    print("=" * 60)
    print(f"Training samples: {len(first_timer_data)}")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print("\nCoefficients:")
    print(f"  Athletes: {poisson.coef_[0]:.4f}")
    print(f"  Events Participated: {poisson.coef_[1]:.4f}")
    print(f"  Intercept: {poisson.intercept_:.4f}")

    return poisson, {"MAE": mae, "RMSE": rmse}


def predict_never_medal_countries(
    poisson_model, data, athletes_clean, year=2024
):
    """
    Predict expected medal counts for never-medal countries.

    Returns:
        DataFrame with predictions
    """
    team_col = _team_key(athletes_clean)
    ev_col = _events_col(athletes_clean)

    teams_with_medals = set(data[data["Total"] > 0]["Team"].unique())
    all_teams = set(athletes_clean[team_col].unique())
    never_medal = sorted(all_teams - teams_with_medals)

    print(f"\nFound {len(never_medal)} countries that have never won medals")

    predictions = []

    for team in never_medal:
        team_yr = athletes_clean[
            (athletes_clean[team_col] == team)
            & (athletes_clean["Year"] == year)
        ]

        if len(team_yr) == 0:
            continue

        athletes = len(team_yr)
        events = team_yr[ev_col].nunique() if ev_col else 0

        X_pred = pd.DataFrame({
            "Athletes": [athletes],
            "EventsParticipated": [events],
        })
        expected_medals = poisson_model.predict(X_pred)[0]
        prob_at_least_one = 1 - np.exp(-expected_medals)

        predictions.append({
            "Team": team,
            "Athletes": athletes,
            "EventsParticipated": events,
            "ExpectedMedals": expected_medals,
            "ProbAtLeastOne": prob_at_least_one,
        })

    pred_df = pd.DataFrame(predictions).sort_values(
        "ExpectedMedals", ascending=False
    )
    return pred_df
