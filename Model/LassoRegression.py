from sklearn.linear_model import Lasso

# Compute historical medal "points" for countries by sport per year (Gold=10, Silver=6, Bronze=3).
# We'll build a dataset of country-sport performance over time to train the model.
points_data = []
for yr in sorted(medals_df['Year'].unique()):
    if yr in [1916, 1940, 1944]:
        continue  # skip canceled games
    # Calculate points for each country in this year
    year_medals = medals_df[medals_df['Year']==yr]
    for _, row in year_medals.iterrows():
        country = row['Team']
        # Medal points for this country in each sport can be derived from athlete data if needed.
        # Here, compute total points across all sports (for simplicity, as the model uses overall performance V as well).
        total_points = row['Gold']*10 + row['Silver']*6 + row['Bronze']*3
        points_data.append({'Year': yr, 'Team': country, 'TotalPoints': total_points})
points_df = pd.DataFrame(points_data)

# Identify known "Great Coach" introduction cases (historical):
# Example: Women's Gymnastics for Romania (coach introduced ~1976) and USA (coach from 1984):contentReference[oaicite:29]{index=29}.
# We will create features:
# I = importance of that sport to the country (from Task 3)
# V = average total points of last two Olympics (country's overall performance):contentReference[oaicite:30]{index=30}
# S = I * V (theoretical points in that sport):contentReference[oaicite:31]{index=31}
# A = number of athletes from that country in current Olympics (from athlete_counts)
# H = historical average points in that sport for the country
# C = coach indicator (1 if a great coach is assumed introduced for that sport, else 0)
# We'll construct a small dataset for selected countries and sports.
selected_cases = [
    {'Team':'Romania', 'Sport':'Gymnastics', 'CoachIntro':1, 'BaseYear':2024, 'BaseScore':3, 'PostScore':36.65},
    {'Team':'India', 'Sport':'Athletics', 'CoachIntro':1, 'BaseYear':2024, 'BaseScore':6, 'PostScore':15},
    {'Team':'Nigeria', 'Sport':'Boxing', 'CoachIntro':1, 'BaseYear':2024, 'BaseScore':3, 'PostScore':12}
]
# Note: In a full analysis, we'd derive BaseScore and PostScore from model predictions or scenarios.
# Here we use hypothetical before/after scores for demonstration.

lasso_rows = []
for case in selected_cases:
    team = case['Team']; sport = case['Sport']
    # Calculate I (importance of sport to country from Task 3 data)
    imp_record = country_sport_medals[(country_sport_medals['Team']==team)&(country_sport_medals['Sport']==sport)]
    I = imp_record['Importance'].values[0] if not imp_record.empty else 0.0
    # Calculate V (average total points of last two games for the country)
    last_two = points_df[points_df['Team']==team]['TotalPoints'].tail(2).values
    V = np.mean(last_two) if len(last_two)==2 else (last_two[0] if len(last_two)==1 else 0.0)
    # S = I * V
    S = I * V
    # A = number of athletes country sent in last Olympics (2024 data as proxy)
    A_val = int(athlete_counts[(athlete_counts['Year']==2024) & (athlete_counts['Team']==team)]['Athletes']) if \
            ((athlete_counts['Year']==2024) & (athlete_counts['Team']==team)).any() else 0
    # H = historical average points in that sport (approximate by average points when coach not present)
    # For simplicity, use BaseScore as a proxy for historical performance in that sport.
    H_val = case['BaseScore']
    # Baseline (no coach)
    lasso_rows.append({'Team':team, 'Sport':sport, 'I':I, 'V':V, 'S':S, 'A':A_val, 'H':H_val, 'C':0, 'Points': case['BaseScore']})
    # After introducing coach
    lasso_rows.append({'Team':team, 'Sport':sport, 'I':I, 'V':V, 'S':S, 'A':A_val, 'H':H_val, 'C':1, 'Points': case['PostScore']})

lasso_df = pd.DataFrame(lasso_rows)
X_lasso = lasso_df[['I','V','S','A','H','C']]
y_lasso = lasso_df['Points']

# Train Lasso regression to identify significant predictors (alpha tuned for demonstration)
lasso_model = Lasso(alpha=0.1, fit_intercept=True, random_state=0)
lasso_model.fit(X_lasso, y_lasso)
print("Lasso coefficients:", dict(zip(['I','V','S','A','H','C'], lasso_model.coef_)))

# Simulate the effect of introducing a great coach:
for case in selected_cases:
    team, sport = case['Team'], case['Sport']
    # Take the baseline feature values from lasso_df for this case
    base_features = lasso_df[(lasso_df['Team']==team)&(lasso_df['Sport']==sport)&(lasso_df['C']==0)][['I','V','S','A','H','C']].values
    if base_features.size == 0:
        continue
    base_pred = lasso_model.predict(base_features)[0]
    # Set coach indicator C=1
    coach_features = base_features.copy()
    coach_features[0,-1] = 1  # set C=1
    coach_pred = lasso_model.predict(coach_features)[0]
    improvement = coach_pred - base_pred
    print(f"{team} - {sport}: baseline score={base_pred:.2f}, with Great Coach score={coach_pred:.2f} (Δ={improvement:.2f})")
