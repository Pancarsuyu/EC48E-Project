#!/usr/bin/env python3
"""
Medeiros et al. (2021) Replication - Python Implementation
Forecasting Inflation with Machine Learning Methods

Minimum kod ile makale replikasyonu
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pyreadr
import requests
import os

class InflationReplication:
    """Makale replikasyonu için minimum sınıf"""

    def __init__(self):
        self.target = 'CPIAUCSL'
        self.window_size = 359  # Makaledeki eğitim penceresi
        self.test_start = '1990-01-01'

    def download_data(self):
        """FRED-MD verisini indir"""
        print("Downloading FRED-MD data...")

        # Veri URL'i (alternatif)
        # Asıl veri second-sample'dan gelecek
        try:
            # GitHub'dan second-sample verisini simüle et
            url = "https://github.com/EoghanONeill/ForecastingInflation/raw/master/second-sample/rawdata.RData"

            # Alternatif: local dosya kullan
            local_file = "data/fred_md_data.csv"

            if os.path.exists(local_file):
                print("Using local data file")
                return pd.read_csv(local_file, index_col=0, parse_dates=True)
            else:
                print("Please download data from GitHub manually")
                print("Run: git clone https://github.com/EoghanONeill/ForecastingInflation.git")
                return None

        except Exception as e:
            print(f"Data download failed: {e}")
            return None

    def load_data(self):
        """Veriyi yükle"""
        print("Loading replication data...")

        # Önce local dosyayı dene
        data_files = [
            "data/fred_md_data.csv",
            "../ForecastingInflation/second-sample/rawdata.RData"
        ]

        for file_path in data_files:
            if os.path.exists(file_path):
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    print(f"Loaded CSV data: {df.shape}")
                    return df
                elif file_path.endswith('.RData'):
                    result = pyreadr.read_r(file_path)
                    df = result['dados']
                    print(f"Loaded RData: {df.shape}")
                    return df

        print("No data file found. Please:")
        print("1. Download from GitHub: git clone https://github.com/EoghanONeill/ForecastingInflation.git")
        print("2. Or place CSV file in data/ folder")
        return None

    def prepare_data(self, df):
        """Veriyi makaledeki formata hazırla"""
        # Test period: 1990-2015
        test_mask = (df.index >= self.test_start) & (df.index <= '2015-12-31')
        test_data = df[test_mask].copy()

        print(f"Test period: {test_data.index.min()} to {test_data.index.max()}")
        print(f"Test observations: {len(test_data)}")

        return test_data

    def run_rw(self, test_data):
        """Random Walk forecasting"""
        predictions = []
        real_values = []

        for i in range(len(test_data) - 1):
            # RW: next value = current value
            current_value = test_data[self.target].iloc[i]
            next_real = test_data[self.target].iloc[i + 1]

            predictions.append(current_value)
            real_values.append(next_real)

        rmse = np.sqrt(mean_squared_error(real_values, predictions))
        mae = mean_absolute_error(real_values, predictions)

        return {'rmse': rmse, 'mae': mae, 'predictions': predictions, 'real': real_values}

    def run_ar(self, test_data):
        """AR(4) forecasting"""
        predictions = []
        real_values = []

        for i in range(4, len(test_data) - 1):
            # AR(4) training
            y_train = test_data[self.target].iloc[i-4:i]
            X_train = np.column_stack([y_train.shift(lag) for lag in range(1, 5)])[4:]
            y_ar = y_train.iloc[4:]

            if len(X_train) < 5:
                continue

            ar_model = LinearRegression()
            ar_model.fit(X_train, y_ar)

            # Predict next
            current_4 = test_data[self.target].iloc[i-3:i+1].values
            pred = ar_model.predict([current_4[::-1]])[0]
            real_next = test_data[self.target].iloc[i + 1]

            predictions.append(pred)
            real_values.append(real_next)

        if len(predictions) == 0:
            return {'rmse': np.nan, 'mae': np.nan, 'predictions': [], 'real': []}

        rmse = np.sqrt(mean_squared_error(real_values, predictions))
        mae = mean_absolute_error(real_values, predictions)

        return {'rmse': rmse, 'mae': mae, 'predictions': predictions, 'real': real_values}

    def run_lasso(self, test_data):
        """LASSO forecasting"""
        predictions = []
        real_values = []

        for i in range(len(test_data) - 1):
            # Training window
            train_end = test_data.index[i]
            train_data = test_data.loc[:train_end]

            if len(train_data) < 50:  # Minimum training size
                predictions.append(np.nan)
                real_values.append(test_data[self.target].iloc[i + 1])
                continue

            train_window = train_data.iloc[-self.window_size:]

            y_train = train_window[self.target]
            X_train = train_window.drop(columns=[self.target])

            # LASSO
            lasso_model = Lasso(alpha=0.001, max_iter=10000)
            lasso_model.fit(X_train, y_train)

            # Predict
            current_X = test_data.iloc[i:i+1].drop(columns=[self.target])
            pred = lasso_model.predict(current_X)[0]
            real_next = test_data[self.target].iloc[i + 1]

            predictions.append(pred)
            real_values.append(real_next)

        # NaN'ları temizle
        valid_idx = ~np.isnan(predictions)
        predictions_clean = np.array(predictions)[valid_idx]
        real_clean = np.array(real_values)[valid_idx]

        if len(predictions_clean) == 0:
            return {'rmse': np.nan, 'mae': np.nan, 'predictions': [], 'real': []}

        rmse = np.sqrt(mean_squared_error(real_clean, predictions_clean))
        mae = mean_absolute_error(real_clean, predictions_clean)

        return {'rmse': rmse, 'mae': mae, 'predictions': predictions_clean, 'real': real_clean}

    def run_rf(self, test_data):
        """Random Forest forecasting"""
        predictions = []
        real_values = []

        for i in range(len(test_data) - 1):
            # Training window
            train_end = test_data.index[i]
            train_data = test_data.loc[:train_end]

            if len(train_data) < 50:
                predictions.append(np.nan)
                real_values.append(test_data[self.target].iloc[i + 1])
                continue

            train_window = train_data.iloc[-self.window_size:]

            y_train = train_window[self.target]
            X_train = train_window.drop(columns=[self.target])

            # RF
            rf_model = RandomForestRegressor(n_estimators=500, random_state=42)
            rf_model.fit(X_train, y_train)

            # Predict
            current_X = test_data.iloc[i:i+1].drop(columns=[self.target])
            pred = rf_model.predict(current_X)[0]
            real_next = test_data[self.target].iloc[i + 1]

            predictions.append(pred)
            real_values.append(real_next)

        # NaN'ları temizle
        valid_idx = ~np.isnan(predictions)
        predictions_clean = np.array(predictions)[valid_idx]
        real_clean = np.array(real_values)[valid_idx]

        if len(predictions_clean) == 0:
            return {'rmse': np.nan, 'mae': np.nan, 'predictions': [], 'real': []}

        rmse = np.sqrt(mean_squared_error(real_clean, predictions_clean))
        mae = mean_absolute_error(real_clean, predictions_clean)

        return {'rmse': rmse, 'mae': mae, 'predictions': predictions_clean, 'real': real_clean}

    def run_all_models(self):
        """Tüm modelleri çalıştır"""
        print("\n=== RUNNING ALL MODELS ===")

        # Veri yükle
        df = self.load_data()
        if df is None:
            return None

        test_data = self.prepare_data(df)

        # Modelleri çalıştır
        results = {}

        print("\nRunning RW...")
        results['RW'] = self.run_rw(test_data)

        print("Running AR...")
        results['AR'] = self.run_ar(test_data)

        print("Running LASSO...")
        results['LASSO'] = self.run_lasso(test_data)

        print("Running RF...")
        results['RF'] = self.run_rf(test_data)

        return results

def main():
    """Ana fonksiyon"""
    print("🔬 MEDEIROS ET AL. (2021) REPLICATION")
    print("=" * 50)

    # Replication başlat
    replication = InflationReplication()
    results = replication.run_all_models()

    if results is None:
        print("❌ Veri yüklenemedi. Lütfen veri dosyasını kontrol edin.")
        return

    # Sonuçları göster
    print("\n" + "=" * 50)
    print("📊 REPLICATION RESULTS")
    print("=" * 50)

    print("<10")
    print("-" * 50)

    for model_name, result in results.items():
        rmse = result['rmse']
        mae = result['mae']
        n_pred = len(result['predictions'])

        if not np.isnan(rmse):
            print("8")

    print("\n" + "=" * 50)
    print("📋 EXPECTED RESULTS FROM PAPER")
    print("=" * 50)
    print("RW:     RMSE ~0.0043, MAE ~0.0031")
    print("AR:     RMSE ~0.0027, MAE ~0.0018")
    print("LASSO:  RMSE ~0.0039, MAE ~0.0028")
    print("RF:     RMSE ~0.0042, MAE ~0.0030")

    print("\n✅ REPLICATION COMPLETED!")
    print("Compare your results with the paper's Table 1.")

if __name__ == "__main__":
    main()

