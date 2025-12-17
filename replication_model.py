
"""
STRICT REPLICATION MODEL (PAPER VERIFICATION)
=============================================
This script runs the original Medeiros et al. (2021) methodology on their specific
test period (1990-2015) using a Strict Recursive Walk-Forward approach.

Purpose:
    To prove that our code base can reproduce the paper's reported RMSE (~0.42%)
    when using identical data and time horizons, confirming correctness.

Usage:
    python replication_model.py

Output:
    Prints validation metrics (RMSE, Ratio to Random Walk, etc.)
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
import sys
import os

warnings.filterwarnings('ignore')

# Constants
DATA_PATH = "2025-11-MD.csv"
Paper_RF_Params = {'n_estimators': 500, 'random_state': 42, 'n_jobs': -1}

def load_and_transform_paper_style(file_path):
    # Same as before
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
    
    raw_target = df['V106']
    df_transformed['target_inflation'] = raw_target.pct_change() * 100
    
    for lag in range(1, 5):
        df_transformed[f'lag_{lag}'] = df_transformed['target_inflation'].shift(lag)
        
    df_final = df_transformed.replace([np.inf, -np.inf], np.nan).dropna()
    return df_final

def run_strict_verification():
    print("STRICT REPLICATION CHECK (1990-2015)")
    print("Methodology: Recursive Walk-Forward | PCA: Within-Fold (No Leakage)")
    print("="*50)
    
    df = load_and_transform_paper_style(DATA_PATH)
    
    # Paper Period: 1990-2015
    test_start_date = '1990-01-01'
    test_end_date = '2015-12-31'
    
    test_indices = df.index[(df.index >= test_start_date) & (df.index <= test_end_date)]
    
    print(f"Test Start: {test_indices[0].date()}")
    print(f"Test End  : {test_indices[-1].date()}")
    print(f"Steps     : {len(test_indices)}")
    
    results = {'Actual': [], 'Paper_RF': [], 'Paper_LASSO': [], 'Paper_RW': []}
    
    for date in test_indices:
        # Train on all data BEFORE this date
        train_data = df[df.index < date]
        test_row = df.loc[[date]]
        
        # --- LOCAL PCA (NO LEAKAGE) ---
        scaler = StandardScaler()
        cols_drop = ['target_inflation'] + [c for c in train_data.columns if 'lag_' in c]
        cols_drop = [c for c in cols_drop if c in train_data.columns]
        
        X_train_raw = train_data.drop(cols_drop, axis=1)
        X_test_raw = test_row.drop(cols_drop, axis=1)
        
        # Robust Cleaning
        X_train_raw = X_train_raw.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_test_raw = X_test_raw.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        if X_train_raw.isna().any().any():
            print("NaN found in Train Raw AFTER fillna!")
        if np.isinf(X_train_raw.values).any():
            print("Inf found in Train Raw!")
            
        try:
            X_train_scaled = scaler.fit_transform(X_train_raw)
            X_test_scaled = scaler.transform(X_test_raw)
            
            # Post-scaling cleanup (e.g. if constant filtering caused NaNs)
            X_train_scaled = np.nan_to_num(X_train_scaled)
            X_test_scaled = np.nan_to_num(X_test_scaled)
            
        except Exception as e:
            print(f"Error Scaling at date {date}: {e}")
            print(f"Train Shape: {X_train_raw.shape}")
            continue
        
        try:
            pca = PCA(n_components=4)
            X_train_factors = pca.fit_transform(X_train_scaled)
            X_test_factors = pca.transform(X_test_scaled)
        except Exception as e:
            print(f"Error PCA at date {date}: {e}")
            print(f"Train Scaled Shape: {X_train_scaled.shape}")
            continue
        
        # Construct Features
        X_train_final = pd.DataFrame(X_train_factors, index=train_data.index, columns=[f'F{k}' for k in range(4)])
        X_test_final = pd.DataFrame(X_test_factors, index=test_row.index, columns=[f'F{k}' for k in range(4)])
        
        for c in train_data.columns:
            if 'lag_' in c:
                X_train_final[c] = train_data[c].values
                X_test_final[c] = test_row[c].values
                
        y_train = train_data['target_inflation']
        actual = test_row['target_inflation'].values[0]
        
        # Models
        rf = RandomForestRegressor(**Paper_RF_Params)
        rf.fit(X_train_final, y_train)
        pred_rf = rf.predict(X_test_final)[0]
        
        lasso = Lasso(alpha=0.001, random_state=42)
        lasso.fit(X_train_final, y_train)
        pred_lasso = lasso.predict(X_test_final)[0]
        
        # RW (Naive)
        pred_rw = y_train.iloc[-1]
        
        results['Actual'].append(actual)
        results['Paper_RF'].append(pred_rf)
        results['Paper_LASSO'].append(pred_lasso)
        results['Paper_RW'].append(pred_rw)
        
        if len(results['Actual']) % 20 == 0:
            print(f"Step {len(results['Actual'])}/{len(test_indices)}...")
            
    # Calculate Metrics
    rmse_rf = np.sqrt(mean_squared_error(results['Actual'], results['Paper_RF']))
    rmse_lasso = np.sqrt(mean_squared_error(results['Actual'], results['Paper_LASSO']))
    rmse_rw = np.sqrt(mean_squared_error(results['Actual'], results['Paper_RW']))
    
    ratio_rf = rmse_rf / rmse_rw
    ratio_lasso = rmse_lasso / rmse_rw
    
    print("\n" + "="*50)
    print("FINAL VERIFICATION RESULTS (1990-2015)")
    print("="*50)
    print(f"Random Walk (RW) RMSE: {rmse_rw:.4f}%")
    print(f"Our Replicated RF RMSE: {rmse_rf:.4f}%  (Ratio: {ratio_rf:.4f})")
    print(f"Our Replicated LASSO RMSE: {rmse_lasso:.4f}% (Ratio: {ratio_lasso:.4f})")
    
    # Check Verification
    if abs(rmse_rf - 0.42) < 0.05:
         print("SUCCESS: RF matches Paper (~0.42).")
    else:
         print(f"WARNING: RF ({rmse_rf:.4f}) differs from Paper (0.42).")

if __name__ == "__main__":
    run_strict_verification()

if __name__ == "__main__":
    run_strict_verification()
