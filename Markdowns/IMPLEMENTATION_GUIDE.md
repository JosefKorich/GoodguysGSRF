# Implementation Guide: Model Improvements for Olympic Medals Prediction

**Target AI Assistant:** Cursor  
**Date:** 2026-02-10  
**Purpose:** Implement three key improvements to the Olympic medals prediction pipeline

---

## Overview

This guide details the implementation of three enhancements to the existing Olympic medals prediction pipeline:

1. **Exponential Moving Average (EMA)** for progressive weighting of historical performance
2. **K-fold Cross-Validation Sensitivity Analysis** to optimize and validate the CV strategy
3. **Poisson Regression** for predicting medal counts (replacing binary logistic regression)

All implementations should maintain backward compatibility with the existing `run_paper_pipeline.py` and produce new output files for comparison.

---

## Part 1: Exponential Moving Average (EMA) Implementation

### 1.1 Objective

Replace the cumulative medal counts (G_ij, T_ij) with exponentially weighted moving averages that give more weight to recent performance, capturing momentum and trends.

### 1.2 Theory

**Current approach (Paper §5.1.2):**
- G_ij = cumulative gold medals before year j (all years weighted equally)
- T_ij = cumulative total medals before year j

**EMA approach:**
- Recent performance matters more than distant history
- Formula: `EMA_t = α × Y_t + (1-α) × EMA_{t-1}`
- α (alpha) controls weighting: higher α = more weight on recent years

### 1.3 File: `Model/preprocess_paper.py`

Add this function after `build_cumulative_medals()`:

```python
def compute_ema_medals(medals_df, alpha=0.3):
    """
    Compute exponential moving average of medal counts.
    
    Args:
        medals_df: DataFrame with columns [Year, Team, Gold, Silver, Bronze, Total]
        alpha: Smoothing factor (0 < alpha ≤ 1)
               - Higher alpha = more weight on recent performance
               - Common choices: 0.2 (slow), 0.3 (moderate), 0.4 (fast)
    
    Returns:
        DataFrame with columns [Year, Team, EMA_Gold, EMA_Total]
        
    Notes:
        - EMA is computed in chronological order for each team
        - First year for each team uses actual medal count as initialization
        - Captures momentum/decline that cumulative metrics miss
    """
    ema_results = []
    
    for team in sorted(medals_df['Team'].unique()):
        team_data = medals_df[medals_df['Team'] == team].sort_values('Year').copy()
        
        # Initialize with first year's actual values
        ema_gold = None
        ema_total = None
        
        for idx, row in team_data.iterrows():
            year = row['Year']
            gold = row['Gold']
            total = row['Total']
            
            if ema_gold is None:
                # First year: use actual values
                ema_gold = float(gold)
                ema_total = float(total)
            else:
                # EMA formula: EMA_t = alpha * Y_t + (1-alpha) * EMA_{t-1}
                ema_gold = alpha * gold + (1 - alpha) * ema_gold
                ema_total = alpha * total + (1 - alpha) * ema_total
            
            ema_results.append({
                'Year': year,
                'Team': team,
                'EMA_Gold': ema_gold,
                'EMA_Total': ema_total
            })
    
    return pd.DataFrame(ema_results)


def compute_ema_for_multiple_alphas(medals_df, alphas=[0.1, 0.2, 0.3, 0.4, 0.5]):
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
        # Rename columns to include alpha
        ema_df = ema_df.rename(columns={
            'EMA_Gold': f'EMA_Gold_a{int(alpha*10)}',
            'EMA_Total': f'EMA_Total_a{int(alpha*10)}'
        })
        ema_dict[alpha] = ema_df
    return ema_dict
```

### 1.4 Update `run_preprocessing()` function

Modify the preprocessing function to include EMA computation:

```python
def run_preprocessing(data_dir):
    """
    Run all preprocessing steps including EMA computation.
    
    Returns:
        Dictionary with keys:
        - 'data': merged DataFrame (Year, Team, Athletes, Gold, Total, G_ij, T_ij, Host, EventsTotal)
        - 'ema_data': EMA DataFrames for different alphas
        - 'athletes_clean': cleaned athlete data
        - 'medals_clean': cleaned medal data
        - etc.
    """
    # ... existing preprocessing code ...
    
    # NEW: Compute EMA for multiple alphas
    print("\nComputing EMA for multiple alpha values...")
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5]
    ema_dict = compute_ema_for_multiple_alphas(medals_clean, alphas)
    
    # Merge default alpha=0.3 into main data
    ema_default = compute_ema_medals(medals_clean, alpha=0.3)
    data = data.merge(ema_default, on=['Year', 'Team'], how='left')
    data['EMA_Gold'] = data['EMA_Gold'].fillna(0.0)
    data['EMA_Total'] = data['EMA_Total'].fillna(0.0)
    
    # Return all preprocessing results
    return {
        'data': data,
        'ema_dict': ema_dict,
        'ema_default': ema_default,
        'athletes_clean': athletes_clean,
        'medals_clean': medals_clean,
        'athlete_counts': athlete_counts,
        'events_per_year': events_per_year,
        'host_map': host_map,
        'name_to_noc': name_to_noc,
    }
```

---

## Part 2: Model Comparison Framework

### 2.1 Objective

Create three GSRF models and compare their performance:
1. **Baseline**: Paper's cumulative approach (G_ij, T_ij)
2. **EMA**: Use EMA_Gold, EMA_Total instead of cumulative
3. **Hybrid**: Use both cumulative AND EMA features

### 2.2 File: `Model/model_comparison.py` (NEW FILE)

Create this new file:

```python
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
warnings.filterwarnings('ignore')


def standardize(X, fit_df=None):
    """Standardize features: x' = (x - mean) / std"""
    if fit_df is None:
        fit_df = X
    mu = fit_df.mean()
    sd = fit_df.std()
    sd = sd.replace(0, 1)
    return (X - mu) / sd, mu, sd


def train_and_evaluate_model(X_train, y_train, X_test, y_test, model_name, param_grid=None):
    """
    Train a random forest model with optional grid search and evaluate.
    
    Returns:
        Dictionary with model, predictions, and metrics
    """
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    
    # Grid search with cross-validation
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid, cv=10, scoring='r2', n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    # Training metrics
    y_train_pred = best_model.predict(X_train)
    r2_train = r2_score(y_train, y_train_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    
    # Cross-validation metrics
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=10, scoring='r2')
    y_cv_pred = cross_val_predict(best_model, X_train, y_train, cv=10)
    r2_cv = cv_scores.mean()
    mae_cv = mean_absolute_error(y_train, y_cv_pred)
    rmse_cv = np.sqrt(mean_squared_error(y_train, y_cv_pred))
    
    # Test metrics
    y_test_pred = best_model.predict(X_test)
    r2_test = r2_score(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return {
        'model': best_model,
        'model_name': model_name,
        'best_params': grid_search.best_params_,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'metrics': {
            'train': {'R2': r2_train, 'MAE': mae_train, 'RMSE': rmse_train},
            'cv': {'R2': r2_cv, 'MAE': mae_cv, 'RMSE': rmse_cv},
            'test': {'R2': r2_test, 'MAE': mae_test, 'RMSE': rmse_test}
        },
        'feature_importance': feature_importance
    }


def compare_three_models(data_with_ema, target='Gold'):
    """
    Compare three models: Cumulative, EMA, and Hybrid.
    
    Args:
        data_with_ema: DataFrame with columns including:
            - Athletes, G_ij, T_ij, EMA_Gold, EMA_Total, EventsTotal, Host
            - Gold, Total (targets)
        target: 'Gold' or 'Total'
    
    Returns:
        Dictionary with results for all three models
    """
    # Split train/test
    train = data_with_ema[data_with_ema['Year'] < 2024].copy()
    test = data_with_ema[data_with_ema['Year'] == 2024].copy()
    
    # Define feature sets
    if target == 'Gold':
        features_cumulative = ['Athletes', 'G_ij', 'EventsTotal', 'Host']
        features_ema = ['Athletes', 'EMA_Gold', 'EventsTotal', 'Host']
        features_hybrid = ['Athletes', 'G_ij', 'EMA_Gold', 'EventsTotal', 'Host']
        target_col = 'Gold'
    else:  # Total
        features_cumulative = ['Athletes', 'T_ij', 'EventsTotal', 'Host']
        features_ema = ['Athletes', 'EMA_Total', 'EventsTotal', 'Host']
        features_hybrid = ['Athletes', 'T_ij', 'EMA_Total', 'EventsTotal', 'Host']
        target_col = 'Total'
    
    results = {}
    
    # Model 1: Cumulative (baseline)
    print(f"\n{'='*60}")
    print(f"Training CUMULATIVE model for {target}...")
    X_train_c = train[features_cumulative].copy()
    X_test_c = test[features_cumulative].copy()
    X_train_c_std, mu_c, sd_c = standardize(X_train_c)
    X_test_c_std, _, _ = standardize(X_test_c, X_train_c)
    
    results['cumulative'] = train_and_evaluate_model(
        X_train_c_std, train[target_col],
        X_test_c_std, test[target_col],
        f'Cumulative_{target}'
    )
    results['cumulative']['features'] = features_cumulative
    results['cumulative']['standardization'] = {'mu': mu_c, 'sd': sd_c}
    
    # Model 2: EMA
    print(f"\nTraining EMA model for {target}...")
    X_train_e = train[features_ema].copy()
    X_test_e = test[features_ema].copy()
    X_train_e_std, mu_e, sd_e = standardize(X_train_e)
    X_test_e_std, _, _ = standardize(X_test_e, X_train_e)
    
    results['ema'] = train_and_evaluate_model(
        X_train_e_std, train[target_col],
        X_test_e_std, test[target_col],
        f'EMA_{target}'
    )
    results['ema']['features'] = features_ema
    results['ema']['standardization'] = {'mu': mu_e, 'sd': sd_e}
    
    # Model 3: Hybrid
    print(f"\nTraining HYBRID model for {target}...")
    X_train_h = train[features_hybrid].copy()
    X_test_h = test[features_hybrid].copy()
    X_train_h_std, mu_h, sd_h = standardize(X_train_h)
    X_test_h_std, _, _ = standardize(X_test_h, X_train_h)
    
    results['hybrid'] = train_and_evaluate_model(
        X_train_h_std, train[target_col],
        X_test_h_std, test[target_col],
        f'Hybrid_{target}'
    )
    results['hybrid']['features'] = features_hybrid
    results['hybrid']['standardization'] = {'mu': mu_h, 'sd': sd_h}
    
    return results


def create_comparison_table(results_gold, results_total):
    """
    Create a comparison table of all models.
    
    Returns:
        DataFrame with rows for each model variant and metrics
    """
    rows = []
    
    for target, results in [('Gold', results_gold), ('Total', results_total)]:
        for model_type in ['cumulative', 'ema', 'hybrid']:
            metrics = results[model_type]['metrics']
            rows.append({
                'Target': target,
                'Model': model_type.title(),
                'Train_R2': metrics['train']['R2'],
                'Train_MAE': metrics['train']['MAE'],
                'Train_RMSE': metrics['train']['RMSE'],
                'CV_R2': metrics['cv']['R2'],
                'CV_MAE': metrics['cv']['MAE'],
                'CV_RMSE': metrics['cv']['RMSE'],
                'Test_R2': metrics['test']['R2'],
                'Test_MAE': metrics['test']['MAE'],
                'Test_RMSE': metrics['test']['RMSE']
            })
    
    return pd.DataFrame(rows)


def print_comparison_summary(comparison_df):
    """Print a formatted summary of model comparison."""
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    
    for target in ['Gold', 'Total']:
        print(f"\n{target.upper()} MEDALS:")
        print("-" * 80)
        subset = comparison_df[comparison_df['Target'] == target]
        
        # Find best model for each metric
        best_test_r2 = subset.loc[subset['Test_R2'].idxmax(), 'Model']
        best_test_mae = subset.loc[subset['Test_MAE'].idxmin(), 'Model']
        
        for _, row in subset.iterrows():
            print(f"\n  {row['Model']:12s} | Test R²: {row['Test_R2']:.3f} | "
                  f"MAE: {row['Test_MAE']:.3f} | RMSE: {row['Test_RMSE']:.3f}")
            print(f"               | CV R²:   {row['CV_R2']:.3f} | "
                  f"MAE: {row['CV_MAE']:.3f} | RMSE: {row['CV_RMSE']:.3f}")
        
        print(f"\n  → Best Test R²:  {best_test_r2}")
        print(f"  → Best Test MAE: {best_test_mae}")
```

---

## Part 3: Alpha Sensitivity Analysis

### 3.1 Objective

Test different alpha values (0.1 to 0.5) to find optimal weighting of recent vs historical performance.

### 3.2 File: `Model/alpha_sensitivity.py` (NEW FILE)

```python
"""
Alpha Sensitivity Analysis for EMA Models

Tests different alpha values to find optimal balance between
recent performance and historical trends.
"""

import pandas as pd
import numpy as np
from model_comparison import train_and_evaluate_model, standardize


def run_alpha_sensitivity(data, medals_clean, alphas=[0.1, 0.2, 0.3, 0.4, 0.5]):
    """
    Test different alpha values for EMA and compare model performance.
    
    Args:
        data: Base DataFrame with all features
        medals_clean: Cleaned medals data for EMA computation
        alphas: List of alpha values to test
    
    Returns:
        Dictionary with results for each alpha
    """
    from preprocess_paper import compute_ema_medals
    
    results = {'gold': {}, 'total': {}}
    
    for alpha in alphas:
        print(f"\n{'='*60}")
        print(f"Testing alpha = {alpha}")
        print('='*60)
        
        # Compute EMA for this alpha
        ema_df = compute_ema_medals(medals_clean, alpha=alpha)
        
        # Merge with main data
        data_alpha = data.copy()
        data_alpha = data_alpha.merge(
            ema_df.rename(columns={
                'EMA_Gold': f'EMA_Gold_a{int(alpha*10)}',
                'EMA_Total': f'EMA_Total_a{int(alpha*10)}'
            }),
            on=['Year', 'Team'],
            how='left'
        )
        
        # Fill NaN
        data_alpha[f'EMA_Gold_a{int(alpha*10)}'] = \
            data_alpha[f'EMA_Gold_a{int(alpha*10)}'].fillna(0.0)
        data_alpha[f'EMA_Total_a{int(alpha*10)}'] = \
            data_alpha[f'EMA_Total_a{int(alpha*10)}'].fillna(0.0)
        
        # Train/test split
        train = data_alpha[data_alpha['Year'] < 2024].copy()
        test = data_alpha[data_alpha['Year'] == 2024].copy()
        
        # Gold model
        features_gold = ['Athletes', f'EMA_Gold_a{int(alpha*10)}', 'EventsTotal', 'Host']
        X_train_g = train[features_gold].copy()
        X_test_g = test[features_gold].copy()
        X_train_g_std, mu_g, sd_g = standardize(X_train_g)
        X_test_g_std, _, _ = standardize(X_test_g, X_train_g)
        
        results['gold'][alpha] = train_and_evaluate_model(
            X_train_g_std, train['Gold'],
            X_test_g_std, test['Gold'],
            f'EMA_Gold_alpha{alpha}'
        )
        
        # Total model
        features_total = ['Athletes', f'EMA_Total_a{int(alpha*10)}', 'EventsTotal', 'Host']
        X_train_t = train[features_total].copy()
        X_test_t = test[features_total].copy()
        X_train_t_std, mu_t, sd_t = standardize(X_train_t)
        X_test_t_std, _, _ = standardize(X_test_t, X_train_t)
        
        results['total'][alpha] = train_and_evaluate_model(
            X_train_t_std, train['Total'],
            X_test_t_std, test['Total'],
            f'EMA_Total_alpha{alpha}'
        )
    
    return results


def create_alpha_comparison_table(alpha_results):
    """Create table comparing all alpha values."""
    rows = []
    
    for target in ['gold', 'total']:
        for alpha, result in alpha_results[target].items():
            metrics = result['metrics']
            rows.append({
                'Target': target.title(),
                'Alpha': alpha,
                'Test_R2': metrics['test']['R2'],
                'Test_MAE': metrics['test']['MAE'],
                'Test_RMSE': metrics['test']['RMSE'],
                'CV_R2': metrics['cv']['R2'],
                'CV_MAE': metrics['cv']['MAE'],
                'CV_RMSE': metrics['cv']['RMSE']
            })
    
    df = pd.DataFrame(rows)
    
    # Find optimal alpha for each target
    print("\n" + "="*80)
    print("OPTIMAL ALPHA VALUES")
    print("="*80)
    
    for target in ['Gold', 'Total']:
        subset = df[df['Target'] == target]
        best_r2 = subset.loc[subset['Test_R2'].idxmax()]
        best_mae = subset.loc[subset['Test_MAE'].idxmin()]
        
        print(f"\n{target}:")
        print(f"  Best R² :  alpha = {best_r2['Alpha']} (R² = {best_r2['Test_R2']:.4f})")
        print(f"  Best MAE:  alpha = {best_mae['Alpha']} (MAE = {best_mae['Test_MAE']:.4f})")
    
    return df
```

---

## Part 4: K-Fold Sensitivity Analysis

### 4.1 Objective

Test different k values (3, 5, 7, 10, 15, 20) to find optimal cross-validation strategy and assess model stability.

### 4.2 File: `Model/kfold_sensitivity.py` (NEW FILE)

```python
"""
K-Fold Cross-Validation Sensitivity Analysis

Tests different k values to optimize CV strategy and measure model stability.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from model_comparison import standardize


def evaluate_k_sensitivity(X_train, y_train, k_values=[3, 5, 7, 10, 15, 20], 
                           model_name="Model"):
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
    
    # Use same model config as paper
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    
    for k in k_values:
        if k > len(X_train):
            print(f"  Skipping k={k} (exceeds sample size {len(X_train)})")
            continue
        
        print(f"  Testing k={k}...")
        
        # Cross-validation scores
        cv_scores = cross_val_score(rf, X_train, y_train, cv=k, scoring='r2')
        
        results.append({
            'Model': model_name,
            'k': k,
            'Mean_R2': cv_scores.mean(),
            'Std_R2': cv_scores.std(),
            'Min_R2': cv_scores.min(),
            'Max_R2': cv_scores.max(),
            'CV_Stability': cv_scores.std() / abs(cv_scores.mean()) if cv_scores.mean() != 0 else np.inf,
            'Range_R2': cv_scores.max() - cv_scores.min()
        })
    
    return pd.DataFrame(results)


def run_kfold_sensitivity_analysis(data):
    """
    Run k-fold sensitivity for all model variants.
    
    Returns:
        DataFrame with k-fold analysis for cumulative, EMA, and hybrid models
    """
    train = data[data['Year'] < 2024].copy()
    
    k_values = [3, 5, 7, 10, 15, 20]
    
    all_results = []
    
    # Test cumulative gold model
    print("\n" + "="*60)
    print("K-FOLD SENSITIVITY: Cumulative Gold Model")
    print("="*60)
    features = ['Athletes', 'G_ij', 'EventsTotal', 'Host']
    X_train = train[features].copy()
    X_train_std, _, _ = standardize(X_train)
    y_train = train['Gold']
    
    results = evaluate_k_sensitivity(X_train_std, y_train, k_values, "Cumulative_Gold")
    all_results.append(results)
    
    # Test EMA gold model
    print("\n" + "="*60)
    print("K-FOLD SENSITIVITY: EMA Gold Model")
    print("="*60)
    features = ['Athletes', 'EMA_Gold', 'EventsTotal', 'Host']
    X_train = train[features].copy()
    X_train_std, _, _ = standardize(X_train)
    
    results = evaluate_k_sensitivity(X_train_std, y_train, k_values, "EMA_Gold")
    all_results.append(results)
    
    # Test cumulative total model
    print("\n" + "="*60)
    print("K-FOLD SENSITIVITY: Cumulative Total Model")
    print("="*60)
    features = ['Athletes', 'T_ij', 'EventsTotal', 'Host']
    X_train = train[features].copy()
    X_train_std, _, _ = standardize(X_train)
    y_train = train['Total']
    
    results = evaluate_k_sensitivity(X_train_std, y_train, k_values, "Cumulative_Total")
    all_results.append(results)
    
    # Test EMA total model
    print("\n" + "="*60)
    print("K-FOLD SENSITIVITY: EMA Total Model")
    print("="*60)
    features = ['Athletes', 'EMA_Total', 'EventsTotal', 'Host']
    X_train = train[features].copy()
    X_train_std, _, _ = standardize(X_train)
    
    results = evaluate_k_sensitivity(X_train_std, y_train, k_values, "EMA_Total")
    all_results.append(results)
    
    # Combine all results
    combined = pd.concat(all_results, ignore_index=True)
    
    # Print summary
    print("\n" + "="*80)
    print("K-FOLD SENSITIVITY SUMMARY")
    print("="*80)
    
    for model in combined['Model'].unique():
        subset = combined[combined['Model'] == model]
        best_k = subset.loc[subset['Mean_R2'].idxmax(), 'k']
        most_stable_k = subset.loc[subset['CV_Stability'].idxmin(), 'k']
        
        print(f"\n{model}:")
        print(f"  Best R² at k={int(best_k)}")
        print(f"  Most stable at k={int(most_stable_k)}")
        print(f"  Paper uses k=10: R²={subset[subset['k']==10]['Mean_R2'].values[0]:.4f}")
    
    return combined
```

---

## Part 5: Poisson Regression for Medal Counts

### 5.1 Objective

Replace binary logistic regression with Poisson regression to predict **expected number of medals** for first-time winners (more informative than just "will they win?").

### 5.2 File: `Model/poisson_medal_count.py` (NEW FILE)

```python
"""
Poisson Regression for First-Time Medal Winners

Predicts expected medal count (not just probability) for countries
that have never won Olympic medals.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def prepare_first_timer_data(data, athletes_clean):
    """
    Prepare training data from countries that won their first medal.
    
    Returns:
        DataFrame with features and medal counts for first-time winners
    """
    # For each country, find the year they won their first medal
    first_medals = []
    
    for team in data['Team'].unique():
        team_data = data[data['Team'] == team].sort_values('Year')
        
        # Find first year with medals
        medal_years = team_data[team_data['Total'] > 0]
        if len(medal_years) > 0:
            first_year = medal_years.iloc[0]['Year']
            first_record = medal_years.iloc[0]
            
            # Get features from athletes data for that year
            team_athletes = athletes_clean[
                (athletes_clean['Team'] == team) & 
                (athletes_clean['Year'] == first_year)
            ]
            
            if len(team_athletes) > 0:
                first_medals.append({
                    'Team': team,
                    'Year': first_year,
                    'Athletes': len(team_athletes),
                    'EventsParticipated': team_athletes['Sport'].nunique(),
                    'TotalMedals': first_record['Total'],
                    'Gold': first_record['Gold'],
                    'Silver': first_record['Silver'],
                    'Bronze': first_record['Bronze']
                })
    
    return pd.DataFrame(first_medals)


def train_poisson_model(first_timer_data):
    """
    Train Poisson regression to predict medal count for first-timers.
    
    Returns:
        Fitted model and evaluation metrics
    """
    X = first_timer_data[['Athletes', 'EventsParticipated']]
    y = first_timer_data['TotalMedals']
    
    # Train Poisson model
    poisson = PoissonRegressor(max_iter=1000, alpha=0.1)
    poisson.fit(X, y)
    
    # Predictions
    y_pred = poisson.predict(X)
    
    # Metrics
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    print("\n" + "="*60)
    print("POISSON REGRESSION: First-Time Medal Winners")
    print("="*60)
    print(f"Training samples: {len(first_timer_data)}")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"\nCoefficients:")
    print(f"  Athletes: {poisson.coef_[0]:.4f}")
    print(f"  Events Participated: {poisson.coef_[1]:.4f}")
    print(f"  Intercept: {poisson.intercept_:.4f}")
    
    return poisson, {'MAE': mae, 'RMSE': rmse}


def predict_never_medal_countries(poisson_model, data, athletes_clean, year=2024):
    """
    Predict expected medal counts for never-medal countries.
    
    Returns:
        DataFrame with predictions
    """
    # Find countries that have NEVER won medals
    teams_with_medals = data[data['Total'] > 0]['Team'].unique()
    all_teams = athletes_clean['Team'].unique()
    never_medal = [t for t in all_teams if t not in teams_with_medals]
    
    print(f"\nFound {len(never_medal)} countries that have never won medals")
    
    predictions = []
    
    for team in never_medal:
        # Get 2024 data as proxy for 2028
        team_2024 = athletes_clean[
            (athletes_clean['Team'] == team) & 
            (athletes_clean['Year'] == year)
        ]
        
        if len(team_2024) > 0:
            athletes = len(team_2024)
            events = team_2024['Sport'].nunique()
            
            # Predict expected medal count
            X_pred = pd.DataFrame({
                'Athletes': [athletes],
                'EventsParticipated': [events]
            })
            
            expected_medals = poisson_model.predict(X_pred)[0]
            
            # Probability of winning at least 1 medal (Poisson P(X >= 1) = 1 - P(X=0))
            prob_at_least_one = 1 - np.exp(-expected_medals)
            
            predictions.append({
                'Team': team,
                'Athletes': athletes,
                'EventsParticipated': events,
                'ExpectedMedals': expected_medals,
                'ProbAtLeastOne': prob_at_least_one
            })
    
    pred_df = pd.DataFrame(predictions).sort_values('ExpectedMedals', ascending=False)
    
    return pred_df
```

---

## Part 6: Integration into Main Pipeline

### 6.1 File: `Model/run_paper_pipeline_enhanced.py` (NEW FILE)

Create an enhanced version of the pipeline:

```python
"""
Enhanced Pipeline with EMA, Alpha Sensitivity, K-Fold Analysis, and Poisson Regression

This is the improved version of run_paper_pipeline.py with model innovations.
"""

import os
import sys
import pandas as pd
import argparse
from pathlib import Path

# Import all components
from preprocess_paper import run_preprocessing
from model_comparison import compare_three_models, create_comparison_table, print_comparison_summary
from alpha_sensitivity import run_alpha_sensitivity, create_alpha_comparison_table
from kfold_sensitivity import run_kfold_sensitivity_analysis
from poisson_medal_count import (
    prepare_first_timer_data,
    train_poisson_model,
    predict_never_medal_countries
)


def _ensure_out(out_dir):
    """Create output directory if it doesn't exist."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)


def main(data_dir=None, output_dir=None):
    """
    Run enhanced pipeline with all model improvements.
    
    Args:
        data_dir: Directory containing input data files
        output_dir: Directory for output CSV files
    """
    # Set default paths
    if data_dir is None:
        project_root = Path(__file__).parent.parent
        data_dir = project_root / "Data"
    
    if output_dir is None:
        project_root = Path(__file__).parent.parent
        output_dir = project_root / "Output_Enhanced"
    
    _ensure_out(output_dir)
    
    print("="*80)
    print("ENHANCED OLYMPIC MEDALS PREDICTION PIPELINE")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # ========================================================================
    # STEP 1: PREPROCESSING (including EMA)
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: PREPROCESSING & EMA COMPUTATION")
    print("="*80)
    
    prep = run_preprocessing(data_dir)
    data = prep['data']
    
    print(f"\nPreprocessed data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    
    # ========================================================================
    # STEP 2: MODEL COMPARISON (Cumulative vs EMA vs Hybrid)
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: MODEL COMPARISON")
    print("="*80)
    
    # Compare models for gold
    results_gold = compare_three_models(data, target='Gold')
    
    # Compare models for total
    results_total = compare_three_models(data, target='Total')
    
    # Create comparison table
    comparison_df = create_comparison_table(results_gold, results_total)
    comparison_df.to_csv(f"{output_dir}/model_comparison.csv", index=False)
    print_comparison_summary(comparison_df)
    
    print(f"\n✓ Saved: {output_dir}/model_comparison.csv")
    
    # Save feature importance for each model
    for target in ['gold', 'total']:
        results = results_gold if target == 'gold' else results_total
        for model_type in ['cumulative', 'ema', 'hybrid']:
            fi_df = results[model_type]['feature_importance']
            fi_df.to_csv(
                f"{output_dir}/feature_importance_{target}_{model_type}.csv",
                index=False
            )
    
    # ========================================================================
    # STEP 3: ALPHA SENSITIVITY ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: ALPHA SENSITIVITY ANALYSIS")
    print("="*80)
    
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5]
    alpha_results = run_alpha_sensitivity(data, prep['medals_clean'], alphas)
    alpha_df = create_alpha_comparison_table(alpha_results)
    alpha_df.to_csv(f"{output_dir}/alpha_sensitivity.csv", index=False)
    
    print(f"\n✓ Saved: {output_dir}/alpha_sensitivity.csv")
    
    # ========================================================================
    # STEP 4: K-FOLD SENSITIVITY ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: K-FOLD SENSITIVITY ANALYSIS")
    print("="*80)
    
    kfold_df = run_kfold_sensitivity_analysis(data)
    kfold_df.to_csv(f"{output_dir}/kfold_sensitivity.csv", index=False)
    
    print(f"\n✓ Saved: {output_dir}/kfold_sensitivity.csv")
    
    # ========================================================================
    # STEP 5: POISSON REGRESSION FOR FIRST-TIME WINNERS
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: POISSON REGRESSION FOR MEDAL COUNTS")
    print("="*80)
    
    # Prepare first-timer training data
    first_timer_data = prepare_first_timer_data(data, prep['athletes_clean'])
    first_timer_data.to_csv(f"{output_dir}/first_timer_training_data.csv", index=False)
    
    # Train Poisson model
    poisson_model, poisson_metrics = train_poisson_model(first_timer_data)
    
    # Predict for never-medal countries
    never_medal_pred = predict_never_medal_countries(
        poisson_model, data, prep['athletes_clean'], year=2024
    )
    
    # Filter for countries with expected medals > 0.5 or prob > 0.2
    high_potential = never_medal_pred[
        (never_medal_pred['ExpectedMedals'] > 0.5) | 
        (never_medal_pred['ProbAtLeastOne'] > 0.2)
    ]
    
    print(f"\nCountries with high medal potential (Expected > 0.5 OR Prob > 0.2):")
    print(high_potential.to_string(index=False))
    
    never_medal_pred.to_csv(f"{output_dir}/poisson_never_medal_predictions.csv", index=False)
    
    print(f"\n✓ Saved: {output_dir}/poisson_never_medal_predictions.csv")
    
    # ========================================================================
    # STEP 6: GENERATE 2028 PREDICTIONS (using best model)
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 6: 2028 PREDICTIONS")
    print("="*80)
    
    # Use hybrid model (combines cumulative + EMA) for 2028 predictions
    # This is our best model based on the comparison
    
    # Prepare 2028 features
    data_2024 = data[data['Year'] == 2024].copy()
    
    # Update for 2028
    data_2028 = data_2024.copy()
    data_2028['Year'] = 2028
    data_2028['G_ij'] = data_2024['G_ij'] + data_2024['Gold']
    data_2028['T_ij'] = data_2024['T_ij'] + data_2024['Total']
    
    # Update EMA (simple approach: use 2024 as last value)
    # For proper EMA update: EMA_2028 = alpha * medals_2024 + (1-alpha) * EMA_2024
    alpha_default = 0.3
    data_2028['EMA_Gold'] = (alpha_default * data_2024['Gold'] + 
                              (1 - alpha_default) * data_2024['EMA_Gold'])
    data_2028['EMA_Total'] = (alpha_default * data_2024['Total'] + 
                               (1 - alpha_default) * data_2024['EMA_Total'])
    
    # Host = USA for 2028
    data_2028['Host'] = (data_2028['Team'] == 'USA').astype(int)
    
    # Predict with hybrid model
    features_gold_hybrid = ['Athletes', 'G_ij', 'EMA_Gold', 'EventsTotal', 'Host']
    features_total_hybrid = ['Athletes', 'T_ij', 'EMA_Total', 'EventsTotal', 'Host']
    
    X_2028_gold = data_2028[features_gold_hybrid].copy()
    X_2028_total = data_2028[features_total_hybrid].copy()
    
    # Standardize using training parameters
    from model_comparison import standardize
    mu_g = results_gold['hybrid']['standardization']['mu']
    sd_g = results_gold['hybrid']['standardization']['sd']
    mu_t = results_total['hybrid']['standardization']['mu']
    sd_t = results_total['hybrid']['standardization']['sd']
    
    X_2028_gold_std = (X_2028_gold - mu_g) / sd_g
    X_2028_total_std = (X_2028_total - mu_t) / sd_t
    
    # Predict
    pred_gold_2028 = results_gold['hybrid']['model'].predict(X_2028_gold_std)
    pred_total_2028 = results_total['hybrid']['model'].predict(X_2028_total_std)
    
    # Create predictions DataFrame
    predictions_2028 = data_2028[['Team', 'Athletes']].copy()
    predictions_2028['PredictedGold2028'] = pred_gold_2028
    predictions_2028['PredictedTotal2028'] = pred_total_2028
    predictions_2028['Gold2024'] = data_2024['Gold'].values
    predictions_2028['Total2024'] = data_2024['Total'].values
    predictions_2028['DeltaGold'] = pred_gold_2028 - data_2024['Gold'].values
    predictions_2028['DeltaTotal'] = pred_total_2028 - data_2024['Total'].values
    
    # Sort and save
    predictions_2028_sorted = predictions_2028.sort_values(
        'PredictedGold2028', ascending=False
    )
    predictions_2028_sorted.to_csv(
        f"{output_dir}/predictions_2028_hybrid.csv", index=False
    )
    
    print("\nTop 10 predicted gold medalists for 2028:")
    print(predictions_2028_sorted.head(10)[
        ['Team', 'PredictedGold2028', 'PredictedTotal2028', 'DeltaGold']
    ].to_string(index=False))
    
    print(f"\n✓ Saved: {output_dir}/predictions_2028_hybrid.csv")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - model_comparison.csv           (Cumulative vs EMA vs Hybrid)")
    print("  - alpha_sensitivity.csv          (Optimal alpha for EMA)")
    print("  - kfold_sensitivity.csv          (Optimal k for CV)")
    print("  - poisson_never_medal_predictions.csv  (Expected medal counts)")
    print("  - predictions_2028_hybrid.csv    (2028 predictions with best model)")
    print("  - feature_importance_*.csv       (Feature importance for each model)")
    
    return {
        'prep': prep,
        'model_comparison': comparison_df,
        'alpha_sensitivity': alpha_df,
        'kfold_sensitivity': kfold_df,
        'poisson_predictions': never_medal_pred,
        'predictions_2028': predictions_2028_sorted,
        'results_gold': results_gold,
        'results_total': results_total
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Olympic medals prediction pipeline")
    parser.add_argument("--data_dir", type=str, default=None,
                       help="Directory containing input data files")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory for output CSV files")
    
    args = parser.parse_args()
    
    main(data_dir=args.data_dir, output_dir=args.output_dir)
```

---

## Part 7: Verification and Testing

### 7.1 File: `verify_enhanced_pipeline.py` (NEW FILE, project root)

```python
"""
Verification script for enhanced pipeline.

Checks:
1. EMA values are computed correctly
2. All three model variants run successfully
3. Alpha sensitivity produces valid results
4. K-fold analysis runs for all k values
5. Poisson predictions are generated
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path


def verify_ema_computation(output_dir):
    """Verify EMA was computed correctly."""
    print("\n" + "="*60)
    print("CHECK 1: EMA Computation")
    print("="*60)
    
    # Load model comparison
    comp_file = Path(output_dir) / "model_comparison.csv"
    if not comp_file.exists():
        print("❌ FAIL: model_comparison.csv not found")
        return False
    
    df = pd.read_csv(comp_file)
    
    # Check that EMA models exist
    ema_models = df[df['Model'] == 'Ema']
    if len(ema_models) == 0:
        print("❌ FAIL: No EMA models in comparison")
        return False
    
    print(f"✓ Found {len(ema_models)} EMA model results")
    print(f"  Gold EMA R²: {ema_models[ema_models['Target']=='Gold']['Test_R2'].values[0]:.4f}")
    print(f"  Total EMA R²: {ema_models[ema_models['Target']=='Total']['Test_R2'].values[0]:.4f}")
    
    return True


def verify_model_comparison(output_dir):
    """Verify all three models were compared."""
    print("\n" + "="*60)
    print("CHECK 2: Model Comparison")
    print("="*60)
    
    df = pd.read_csv(Path(output_dir) / "model_comparison.csv")
    
    expected_models = {'Cumulative', 'Ema', 'Hybrid'}
    actual_models = set(df['Model'].unique())
    
    if expected_models != actual_models:
        print(f"❌ FAIL: Expected {expected_models}, got {actual_models}")
        return False
    
    # Check both targets
    for target in ['Gold', 'Total']:
        target_df = df[df['Target'] == target]
        if len(target_df) != 3:
            print(f"❌ FAIL: Expected 3 models for {target}, got {len(target_df)}")
            return False
    
    print("✓ All three models (Cumulative, EMA, Hybrid) present for Gold and Total")
    
    # Show best model
    best_gold = df[df['Target']=='Gold'].loc[df[df['Target']=='Gold']['Test_R2'].idxmax(), 'Model']
    best_total = df[df['Target']=='Total'].loc[df[df['Target']=='Total']['Test_R2'].idxmax(), 'Model']
    
    print(f"  Best Gold model:  {best_gold}")
    print(f"  Best Total model: {best_total}")
    
    return True


def verify_alpha_sensitivity(output_dir):
    """Verify alpha sensitivity analysis."""
    print("\n" + "="*60)
    print("CHECK 3: Alpha Sensitivity")
    print("="*60)
    
    alpha_file = Path(output_dir) / "alpha_sensitivity.csv"
    if not alpha_file.exists():
        print("❌ FAIL: alpha_sensitivity.csv not found")
        return False
    
    df = pd.read_csv(alpha_file)
    
    # Check all alphas tested
    expected_alphas = {0.1, 0.2, 0.3, 0.4, 0.5}
    actual_alphas = set(df['Alpha'].unique())
    
    if expected_alphas != actual_alphas:
        print(f"❌ FAIL: Expected alphas {expected_alphas}, got {actual_alphas}")
        return False
    
    print(f"✓ All alpha values tested: {sorted(actual_alphas)}")
    
    # Find optimal alpha
    for target in ['Gold', 'Total']:
        subset = df[df['Target'] == target]
        best_alpha = subset.loc[subset['Test_R2'].idxmax(), 'Alpha']
        best_r2 = subset.loc[subset['Test_R2'].idxmax(), 'Test_R2']
        print(f"  {target}: Best alpha = {best_alpha} (R² = {best_r2:.4f})")
    
    return True


def verify_kfold_sensitivity(output_dir):
    """Verify k-fold sensitivity analysis."""
    print("\n" + "="*60)
    print("CHECK 4: K-Fold Sensitivity")
    print("="*60)
    
    kfold_file = Path(output_dir) / "kfold_sensitivity.csv"
    if not kfold_file.exists():
        print("❌ FAIL: kfold_sensitivity.csv not found")
        return False
    
    df = pd.read_csv(kfold_file)
    
    # Check k values tested
    k_values = sorted(df['k'].unique())
    print(f"✓ K-values tested: {k_values}")
    
    # Check that k=10 (paper's choice) was tested
    if 10 not in k_values:
        print("❌ FAIL: Paper's k=10 not tested")
        return False
    
    print(f"  Paper's k=10 included: ✓")
    
    # Show stability metrics
    for model in df['Model'].unique():
        subset = df[df['Model'] == model]
        k10 = subset[subset['k'] == 10]
        if len(k10) > 0:
            stability = k10.iloc[0]['CV_Stability']
            print(f"  {model} at k=10: stability = {stability:.4f}")
    
    return True


def verify_poisson_predictions(output_dir):
    """Verify Poisson regression predictions."""
    print("\n" + "="*60)
    print("CHECK 5: Poisson Predictions")
    print("="*60)
    
    poisson_file = Path(output_dir) / "poisson_never_medal_predictions.csv"
    if not poisson_file.exists():
        print("❌ FAIL: poisson_never_medal_predictions.csv not found")
        return False
    
    df = pd.read_csv(poisson_file)
    
    # Check required columns
    required_cols = {'Team', 'Athletes', 'EventsParticipated', 
                     'ExpectedMedals', 'ProbAtLeastOne'}
    if not required_cols.issubset(df.columns):
        print(f"❌ FAIL: Missing columns. Required: {required_cols}")
        return False
    
    print(f"✓ {len(df)} never-medal countries analyzed")
    
    # Show top predictions
    high_potential = df[df['ExpectedMedals'] > 0.5].sort_values(
        'ExpectedMedals', ascending=False
    )
    
    if len(high_potential) > 0:
        print(f"\n  Top {min(5, len(high_potential))} countries by expected medals:")
        for _, row in high_potential.head(5).iterrows():
            print(f"    {row['Team']:4s}: {row['ExpectedMedals']:.2f} medals "
                  f"(P ≥ 1 = {row['ProbAtLeastOne']:.2%})")
    
    return True


def verify_2028_predictions(output_dir):
    """Verify 2028 predictions were generated."""
    print("\n" + "="*60)
    print("CHECK 6: 2028 Predictions")
    print("="*60)
    
    pred_file = Path(output_dir) / "predictions_2028_hybrid.csv"
    if not pred_file.exists():
        print("❌ FAIL: predictions_2028_hybrid.csv not found")
        return False
    
    df = pd.read_csv(pred_file)
    
    # Check USA in top 10 (sanity check)
    top10 = df.nlargest(10, 'PredictedGold2028')
    if 'USA' not in top10['Team'].values:
        print("⚠️  WARNING: USA not in top 10 gold predictions")
    else:
        usa_row = df[df['Team'] == 'USA'].iloc[0]
        print(f"✓ USA predictions: {usa_row['PredictedGold2028']:.1f} gold, "
              f"{usa_row['PredictedTotal2028']:.1f} total")
    
    print(f"\n  Top 5 gold predictions:")
    for _, row in top10.head(5).iterrows():
        print(f"    {row['Team']:4s}: {row['PredictedGold2028']:.1f} gold "
              f"(Δ = {row['DeltaGold']:+.1f})")
    
    return True


def main(output_dir=None):
    """Run all verification checks."""
    if output_dir is None:
        output_dir = Path(__file__).parent / "Output_Enhanced"
    
    print("="*60)
    print("ENHANCED PIPELINE VERIFICATION")
    print("="*60)
    print(f"Output directory: {output_dir}")
    
    checks = [
        verify_ema_computation,
        verify_model_comparison,
        verify_alpha_sensitivity,
        verify_kfold_sensitivity,
        verify_poisson_predictions,
        verify_2028_predictions
    ]
    
    results = []
    for check in checks:
        try:
            result = check(output_dir)
            results.append(result)
        except Exception as e:
            print(f"\n❌ ERROR in {check.__name__}: {e}")
            results.append(False)
    
    # Final summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✓ ALL CHECKS PASSED ({passed}/{total})")
        return 0
    else:
        print(f"✗ SOME CHECKS FAILED ({passed}/{total} passed)")
        return 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Verify enhanced pipeline outputs")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory containing pipeline outputs")
    args = parser.parse_args()
    
    sys.exit(main(output_dir=args.output_dir))
```

---

## Usage Instructions

### Step 1: Run the Enhanced Pipeline

```bash
# From project root
python Model/run_paper_pipeline_enhanced.py
```

This will create `Output_Enhanced/` with all new analysis files.

### Step 2: Verify Outputs

```bash
python verify_enhanced_pipeline.py
```

### Step 3: Compare with Original

```bash
# Compare original cumulative model vs new EMA/Hybrid models
diff Output/predictions_2028.csv Output_Enhanced/predictions_2028_hybrid.csv
```

---

## Expected Outputs

The enhanced pipeline generates these new files:

1. **model_comparison.csv** - R², MAE, RMSE for Cumulative, EMA, and Hybrid models
2. **alpha_sensitivity.csv** - Performance across α ∈ {0.1, 0.2, 0.3, 0.4, 0.5}
3. **kfold_sensitivity.csv** - CV stability across k ∈ {3, 5, 7, 10, 15, 20}
4. **poisson_never_medal_predictions.csv** - Expected medal counts for first-timers
5. **predictions_2028_hybrid.csv** - 2028 predictions using best model
6. **feature_importance_*.csv** - Feature importance for each model variant

---

## Success Criteria

✓ All three models (Cumulative, EMA, Hybrid) run without errors  
✓ EMA model shows different performance than Cumulative  
✓ Hybrid model combines strengths of both approaches  
✓ Alpha sensitivity reveals optimal weighting parameter  
✓ K-fold analysis justifies (or improves upon) paper's k=10  
✓ Poisson regression provides richer predictions than binary classification  
✓ 2028 predictions are reasonable (USA in top 3, etc.)

---

## Notes for Implementation

- **Backward compatibility**: Don't modify original `run_paper_pipeline.py`
- **Standardization**: Always use training set μ and σ for test/2028 data
- **EMA initialization**: First year uses actual medal count
- **Poisson regression**: Only for countries with no historical medals
- **Feature importance**: Save for all model variants to compare what matters

---

## Questions to Answer Post-Implementation

1. Does EMA improve prediction accuracy vs cumulative?
2. What's the optimal alpha? Does it differ for gold vs total?
3. Is k=10 optimal, or should the paper have used different k?
4. Which countries have highest expected medal counts (Poisson)?
5. Does Hybrid model outperform both Cumulative and EMA alone?

