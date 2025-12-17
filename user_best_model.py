
"""
USER BEST MODEL (OPTIMIZED + SMART INJECTION)
=============================================
This script runs the State-of-the-Art inflation forecasting model developed in this project.
It combines:
1. An Optimized Random Forest (300 estimators, Depth 20)
2. A "Smart Regime Injection" mechanism to capture volatility

Usage:
    python user_best_model.py

Output:
    Prints RMSE results and saves 'regime_experiment_results.csv'
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics.pairwise import euclidean_distances
import warnings
import sys
import os

# Import user's feature engineering
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from feature_engineering import StationaryFeatureEngineer
except ImportError:
    # If not found, try adding current directory
    sys.path.append(os.getcwd())
    from feature_engineering import StationaryFeatureEngineer

warnings.filterwarnings('ignore')

# Constants
DATA_PATH = "2025-11-MD.csv"
TARGET_VAR = "CPIAUCSL" # V106

# Improved Params for Base Model
User_RF_Params = {'n_estimators': 300, 'max_depth': 20, 'random_state': 42, 'n_jobs': -1}

def load_and_transform_paper_style(file_path):
    """
    Standard Paper Pipeline (Leakage Free version)
    """
    try:
        transforms = pd.read_csv(file_path, header=None, nrows=1, skiprows=[0]).iloc[0, 1:]
        transforms = {f'V{i+1}': int(code) for i, code in enumerate(transforms)}
    except Exception as e:
        print(f"Error reading transforms: {e}")
        return None

    df = pd.read_csv(file_path, header=0, skiprows=[1])
    df['sasdate'] = pd.to_datetime(df['sasdate'], format='%m/%d/%Y')
    df = df.set_index('sasdate')
    df.columns = [f'V{i+1}' for i in range(len(df.columns))]
    
    df_transformed = pd.DataFrame(index=df.index)
    
    for col, code in transforms.items():
        if col not in df.columns: continue
        series = df[col]
        # Standard transformations 1-7 (omitted for brevity, assume standard)
        if code == 1: trans = series
        elif code == 2: trans = series.diff()
        elif code == 3: trans = series.diff().diff()
        elif code == 4: trans = np.log(series)
        elif code == 5: trans = np.log(series).diff()
        elif code == 6: trans = np.log(series).diff().diff()
        elif code == 7: trans = series.pct_change()
        else: trans = series
        df_transformed[col] = trans

    df_transformed = df_transformed.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Target
    raw_target = df['V106']
    df_transformed['target_inflation'] = raw_target.pct_change() * 100
    
    # Lags
    for lag in range(1, 5):
        df_transformed[f'lag_{lag}'] = df_transformed['target_inflation'].shift(lag)
        
    df_final = df_transformed.replace([np.inf, -np.inf], np.nan).dropna()
    return df_final


def load_and_transform_user_style(file_path):
    fe = StationaryFeatureEngineer()
    df_raw = fe.load_and_prepare_data(file_path)
    df_features = fe.get_all_stationary_features(df_raw)
    df_features['target_inflation'] = df_raw['inflation_rate']
    df_final = df_features.replace([np.inf, -np.inf], np.nan).dropna()
    return df_final

def find_regime_shock(current_features, historical_features_db, historical_target_db, feature_cols, top_k=7):
    """
    Improved "Regime-Aware" Injection
    1. Selects only relevant volatility/stress features for matching.
    2. Uses Consensus Gating: Only inject if neighbors agree on direction.
    """
    
    # KEY FEATURE SELECTION FOR MATCHING
    # V106: CPI, V104: Oil, V127: VIX, V95: Dollar Index, V78: FedFunds
    # We look for columns related to these in the transformed feature set
    
    match_candidates = []
    
    # Identify relevant columns in user's transformed dataframe
    # User's features have names like 'V106_pct_change', 'V106_rolling_std_6', etc.
    
    keywords = ['V106', 'inflation', 'V104', 'oil', 'V127', 'vix', 'V95', 'dollar', 'V78', 'fed']
    
    for col in feature_cols:
        col_lower = col.lower()
        if any(k in col_lower for k in keywords):
            match_candidates.append(col)
            
    if not match_candidates:
        match_candidates = feature_cols # Fallback
        
    # Extract vectors
    current_vec = current_features[match_candidates].values.reshape(1, -1)
    hist_vecs = historical_features_db[match_candidates].values
    
    if len(hist_vecs) == 0: return 0.0
        
    dists = euclidean_distances(current_vec, hist_vecs).flatten()
    closest_indices = np.argsort(dists)[:top_k]
    
    shocks = []
    similarities = []
    
    sigma = np.mean(dists) * 0.5 + 1e-6
    
    for idx in closest_indices:
        if idx + 1 >= len(historical_target_db): continue
            
        val_t = historical_target_db.iloc[idx]
        val_next = historical_target_db.iloc[idx+1]
        
        # Shock = Realized - Previous (What actually happened next)
        shock = val_next - val_t
        
        sim = np.exp(-(dists[idx]**2) / (2 * sigma**2))
        
        shocks.append(shock)
        similarities.append(sim)
    
    if not shocks: return 0.0
    
    shocks = np.array(shocks)
    similarities = np.array(similarities)
    
    # CONSENSUS GATING
    n_up = np.sum(shocks > 0)
    n_down = np.sum(shocks < 0)
    total = len(shocks)
    
    consensus_threshold = 0.65 # Needs 65% agreement
    
    final_shock = 0.0
    
    if (n_up / total) >= consensus_threshold:
        # Strong UP signal
        # Weighted mean of positive shocks only? Or all shocks? 
        # Safest: Weighted mean of ALL shocks (since consensus exists, mean will be positive)
        final_shock = np.average(shocks, weights=similarities)
        
    elif (n_down / total) >= consensus_threshold:
        # Strong DOWN signal
        final_shock = np.average(shocks, weights=similarities)
        
    else:
        # Mixed signal - Ambiguous regime
        # Inject NOTHING (0.0) or very small dampener
        final_shock = 0.0
        
    # Volatility Scaler (Don't inject massive shocks if similarity is low)
    avg_sim = np.mean(similarities)
    confidence = np.clip(avg_sim * 2.0, 0.0, 1.0)
    
    return final_shock * confidence

def run_improved_experiment():
    print("="*60)
    print("IMPROVED REGIME-AWARE INJECTION EXPERIMENT")
    print("="*60)
    
    # 1. Load Data
    try:
        df_user = load_and_transform_user_style(DATA_PATH)
        # We also load paper data just to align index correctly to 1959-2025 common set
        df_paper_temp = load_and_transform_paper_style(DATA_PATH)
        common_index = df_user.index.intersection(df_paper_temp.index)
        df_user = df_user.loc[common_index]
    except Exception as e:
        print(f"Error: {e}")
        return

    print(f"Data validated. Size: {len(df_user)}")
    
    # 2. Recursive Validation
    N_TEST_MONTHS = 180
    test_start_idx = len(df_user) - N_TEST_MONTHS
    
    results = {'Actual': [], 'User_Base': [], 'User_Regime': []}
    
    print("Running Recursive Walk-Forward...")
    
    for i in range(N_TEST_MONTHS):
        curr_train_end_idx = test_start_idx + i
        if i % 20 == 0: print(f"  Step {i}/{N_TEST_MONTHS}...")
        
        train_dates = df_user.index[:curr_train_end_idx]
        test_date = df_user.index[curr_train_end_idx]
        
        # Split
        X_train = df_user.loc[train_dates].drop('target_inflation', axis=1)
        y_train = df_user.loc[train_dates]['target_inflation']
        X_test = df_user.loc[[test_date]].drop('target_inflation', axis=1)
        y_test = df_user.loc[[test_date]]['target_inflation']
        
        actual = y_test.values[0]
        
        # Scaling
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
        
        # 1. Base Model (Improved Params)
        rf = RandomForestRegressor(**User_RF_Params)
        rf.fit(X_train_scaled, y_train)
        pred_base = rf.predict(X_test_scaled)[0]
        
        # 2. Regime Injection
        shock = find_regime_shock(X_test_scaled, X_train_scaled, y_train, X_train.columns)
        pred_regime = pred_base + shock
        
        results['Actual'].append(actual)
        results['User_Base'].append(pred_base)
        results['User_Regime'].append(pred_regime)
        
    # Evaluation
    res_df = pd.DataFrame(results)
    
    rmse_base = np.sqrt(mean_squared_error(res_df['Actual'], res_df['User_Base']))
    rmse_regime = np.sqrt(mean_squared_error(res_df['Actual'], res_df['User_Regime']))
    
    print("\n" + "="*40)
    print(f"RESULTS (N={N_TEST_MONTHS})")
    print("="*40)
    print(f"User Base (Improved) RMSE: {rmse_base:.5f}")
    print(f"User Regime (Smart)  RMSE: {rmse_regime:.5f}")
    
    if rmse_regime < rmse_base:
        print("\nSUCCESS: Smart Injection improved accuracy!")
    else:
        print(f"\nAnalysis: Injection cost = {rmse_regime - rmse_base:.5f}")
        
    res_df.to_csv("regime_experiment_results.csv", index=False)

if __name__ == "__main__":
    run_improved_experiment()
