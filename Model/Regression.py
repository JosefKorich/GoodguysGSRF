from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Prepare training data for logistic model:
# We will generate training examples from past Olympics: for each Olympics year, 
# consider countries that (up to the previous Olympics) had never won a medal.
log_rows = []
medaled_countries = set()  # track countries that have won at least one medal so far
years = sorted(data['Year'].unique())
for yr in years:
    if yr in [1916, 1940, 1944]:  # skip canceled games
        continue
    # Countries participating in year 'yr'
    participants = athlete_counts[athlete_counts['Year']==yr]['Team'].unique()
    # Countries that have never medaled before 'yr'
    non_medaled_participants = [team for team in participants if team not in medaled_countries]
    # Determine which of these won a medal in year 'yr'
    medaled_this_year = medals_df[(medals_df['Year']==yr)]['Team'].unique()
    for team in non_medaled_participants:
        athletes_num = int(athlete_counts[(athlete_counts['Year']==yr) & (athlete_counts['Team']==team)]['Athletes'])
        prev_participations = athlete_counts[(athlete_counts['Year']<yr) & (athlete_counts['Team']==team)]['Year'].nunique()
        host_flag = 1 if host_map.get(yr) == team else 0
        won_medal = 1 if team in medaled_this_year else 0
        log_rows.append({
            'Year': yr, 'Team': team,
            'Athletes': athletes_num,
            'PrevGames': prev_participations,
            'Host': host_flag,
            'WonMedal': won_medal
        })
    # Update set of countries that have medaled by end of this year
    medaled_countries.update(medaled_this_year)

log_df = pd.DataFrame(log_rows)
X_log = log_df[['Athletes','PrevGames','Host']]
y_log = log_df['WonMedal']

# Train logistic regression with L1 regularization (Lasso):contentReference[oaicite:12]{index=12}
log_model = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, max_iter=1000)
log_model.fit(X_log, y_log)

# Evaluate on training set (could also do cross-validation for more robust estimate)
y_pred = log_model.predict(X_log)
y_proba = log_model.predict_proba(X_log)[:,1]
acc = accuracy_score(y_log, y_pred)
prec = precision_score(y_log, y_pred)
rec = recall_score(y_log, y_pred)
f1 = f1_score(y_log, y_pred)
auc = roc_auc_score(y_log, y_proba)
print(f"Training metrics: Accuracy={acc:.3f}, Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}, AUC={auc:.3f}")

# Identify countries with no prior medals (up to 2024) to predict their probabilities for 2028
all_countries = athlete_counts['Team'].unique()
medaled_ever = medals_df['Team'].unique()
never_medaled_countries = [team for team in all_countries if team not in medaled_ever]

# Prepare features for these countries for 2028 prediction:
# Use data from 2024 as a proxy for their athlete count in 2028 (if not present in 2024, assume small participation)
pred_rows = []
for team in never_medaled_countries:
    # Use latest known athlete count (max over years participated) as estimate
    if team in athlete_counts['Team'].values:
        latest_year = athlete_counts[athlete_counts['Team']==team]['Year'].max()
        athletes_num = int(athlete_counts[(athlete_counts['Team']==team)&(athlete_counts['Year']==latest_year)]['Athletes'])
        prev_participations = athlete_counts[athlete_counts['Team']==team]['Year'].nunique()
    else:
        athletes_num = 0
        prev_participations = 0
    host_flag = 1 if host_map.get(2028)==team else 0
    pred_rows.append({'Team': team, 'Athletes': athletes_num, 'PrevGames': prev_participations, 'Host': host_flag})

pred_log_df = pd.DataFrame(pred_rows)
X_pred_log = pred_log_df[['Athletes','PrevGames','Host']]
pred_log_df['MedalWin_Prob2028'] = log_model.predict_proba(X_pred_log)[:,1]
pred_log_df = pred_log_df.sort_values('MedalWin_Prob2028', ascending=False)
print("Top 10 countries (never medaled) by predicted probability of winning a medal in 2028:")
print(pred_log_df.head(10))
