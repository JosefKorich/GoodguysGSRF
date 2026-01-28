# Using the trained Random Forest model (from Task 1) for total medals to assess sensitivity.
rf_model = best_total_model  # total-medal RF model from Task 1

# Baseline predictions for all countries (using 2024 features as 2028 proxy)
X_base = data_2024[features].copy()
y_pred_base = rf_model.predict(X_base)

# Perturb 1: increase number of athletes by +5% for all countries
X_ath_plus5 = X_base.copy()
X_ath_plus5['Athletes'] = X_ath_plus5['Athletes'] * 1.05
y_pred_ath_plus5 = rf_model.predict(X_ath_plus5)

# Perturb 2: increase total events by +5%
X_evt_plus5 = X_base.copy()
X_evt_plus5['EventsTotal'] = X_evt_plus5['EventsTotal'] * 1.05
y_pred_evt_plus5 = rf_model.predict(X_evt_plus5)

# Compute sensitivity: average percent change in predictions
delta_y_ath = (y_pred_ath_plus5 - y_pred_base) / (y_pred_base + 1e-9) * 100  # % change per country
delta_y_evt = (y_pred_evt_plus5 - y_pred_base) / (y_pred_base + 1e-9) * 100
avg_sens_ath = np.mean(delta_y_ath)
avg_sens_evt = np.mean(delta_y_evt)
print(f"Average sensitivity to +5% athletes: {avg_sens_ath:.4f}%")
print(f"Average sensitivity to +5% events:   {avg_sens_evt:.4f}%")
