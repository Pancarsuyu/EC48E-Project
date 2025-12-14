#!/usr/bin/env python3
"""
2025-11-MD.csv ile Makale Replication Testi
- Makaledeki zaman aralığı (2015'e kadar) ile test
- Tam veri (2025'e kadar) ile test
- RF, AR, LASSO modelleri
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Görselleştirme kütüphaneleri
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
plt.style.use('seaborn-v0_8')

def load_and_transform_fred_md(file_path="2025-11-MD.csv"):
    """FRED-MD verisini yükle ve transform et"""
    print("📥 FRED-MD verisi yükleniyor...")

    # Transformasyon kodlarını oku
    transforms = pd.read_csv(file_path, header=None, nrows=1, skiprows=[0]).iloc[0, 1:]
    transforms = {f'V{i+1}': int(code) for i, code in enumerate(transforms)}

    # Veriyi yükle
    df = pd.read_csv(file_path, header=0, skiprows=[1])
    df['sasdate'] = pd.to_datetime(df['sasdate'], format='%m/%d/%Y')
    df = df.set_index('sasdate')

    # Değişken isimlerini V1, V2, ... olarak değiştir (makaledeki gibi)
    df.columns = [f'V{i+1}' for i in range(len(df.columns))]
    df['CPIAUCSL'] = df['V112']  # CPI değişkeni

    # Transformasyonları uygula
    print("🔄 Transformasyonlar uygulanıyor...")
    df_transformed = df.copy()

    for col, code in transforms.items():
        if col not in df.columns:
            continue

        if code == 1:
            # No transformation
            pass
        elif code == 2:
            # Δ (first difference)
            df_transformed[col] = df[col].diff()
        elif code == 3:
            # Δ² (second difference)
            df_transformed[col] = df[col].diff().diff()
        elif code == 4:
            # Log
            df_transformed[col] = np.log(df[col])
        elif code == 5:
            # Δ log
            df_transformed[col] = np.log(df[col]).diff()
        elif code == 6:
            # Δ² log
            df_transformed[col] = np.log(df[col]).diff().diff()
        elif code == 7:
            # Δ % (percentage change)
            df_transformed[col] = df[col].pct_change()

    # NaN değerleri temizle
    df_transformed = df_transformed.dropna()

    # PCA ekle (4 bileşen)
    print("📊 PCA hesaplanıyor...")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_transformed.drop('CPIAUCSL', axis=1))

    pca = PCA(n_components=4)
    pca_components = pca.fit_transform(data_scaled)

    for i in range(4):
        df_transformed[f'PC{i+1}'] = pca_components[:, i]

    # Aylık enflasyon oranını hesapla (MoM inflation rate)
    print("📈 Aylık enflasyon oranları hesaplanıyor...")
    df_transformed['inflation_rate'] = df_transformed['CPIAUCSL'].pct_change() * 100

    # Lag değişkenleri ekle (4 lag) - enflasyon oranı için
    print("🔄 Lag değişkenleri ekleniyor...")
    target_col = 'inflation_rate'
    for lag in range(1, 5):
        df_transformed[f'{target_col}_lag{lag}'] = df_transformed[target_col].shift(lag)

    # AR terimleri ekle (4 lag) - enflasyon oranı için
    for lag in range(1, 5):
        df_transformed[f'AR{lag}'] = df_transformed[target_col].shift(lag)

    # Son temizlik
    df_final = df_transformed.dropna()

    print(f"✅ Transform edilmiş veri: {df_final.shape}")
    print(f"📅 Tarih aralığı: {df_final.index.min()} - {df_final.index.max()}")

    return df_final

def prepare_rolling_window_data(df, nprev=180):
    """Rolling window için veriyi hazırla"""
    # Test dönemi: son nprev ay
    test_start_idx = len(df) - nprev
    test_data = df.iloc[test_start_idx:]

    print(f"🔍 Test dönemi: {test_data.index.min()} - {test_data.index.max()}")
    print(f"📊 Test gözlemleri: {len(test_data)}")

    return test_data

def run_rf_model(train_data, test_point, target_col='inflation_rate'):
    """RF modeli çalıştır"""
    # Özellikleri hazırla
    feature_cols = [col for col in train_data.columns if col != target_col]
    X_train = train_data[feature_cols]
    y_train = train_data[target_col]

    # Test noktası
    X_test = test_point[feature_cols].values.reshape(1, -1)

    # RF modeli
    rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # Tahmin
    pred = rf.predict(X_test)[0]

    return pred

def run_ar_model(train_data, target_col='inflation_rate', p=4):
    """AR(p) modeli - sadece lag değişkenleri kullan"""
    # AR terimleri
    ar_cols = [f'AR{i}' for i in range(1, p+1)]
    X_train = train_data[ar_cols]
    y_train = train_data[target_col]

    # Test noktası
    X_test = train_data[ar_cols].iloc[-1:].values

    # OLS (Linear Regression)
    from sklearn.linear_model import LinearRegression
    ar_model = LinearRegression()
    ar_model.fit(X_train, y_train)

    pred = ar_model.predict(X_test)[0]

    return pred

def run_lasso_model(train_data, test_point, target_col='inflation_rate', alpha=0.001):
    """LASSO modeli"""
    # Özellikleri hazırla
    feature_cols = [col for col in train_data.columns if col != target_col]
    X_train = train_data[feature_cols]
    y_train = train_data[target_col]

    # Test noktası
    X_test = test_point[feature_cols].values.reshape(1, -1)

    # LASSO
    lasso = Lasso(alpha=alpha, random_state=42)
    lasso.fit(X_train, y_train)

    pred = lasso.predict(X_test)[0]

    return pred

def rolling_window_forecast(df, model_type='RF', nprev=180, target_col='CPIAUCSL'):
    """Rolling window forecasting"""
    print(f"\n🏃 {model_type} modeli - Rolling Window Testi")

    predictions = []
    real_values = []

    test_data = prepare_rolling_window_data(df, nprev)

    for i in range(len(test_data) - 1):
        if i % 20 == 0:
            print(f"Progress: {i+1}/{len(test_data)-1}")

        # Eğitim verisi: test noktasına kadar
        train_end_idx = len(df) - nprev + i
        train_data = df.iloc[:train_end_idx + 1]

        # Test noktası
        test_point = df.iloc[train_end_idx + 1:train_end_idx + 2]

        if len(train_data) < 50:  # Minimum eğitim verisi
            continue

        try:
            if model_type == 'RF':
                pred = run_rf_model(train_data, test_point, target_col)
            elif model_type == 'AR':
                pred = run_ar_model(train_data, target_col)
            elif model_type == 'LASSO':
                pred = run_lasso_model(train_data, test_point, target_col)

            # Gerçek değer (bir sonraki ay)
            real_val = df.iloc[train_end_idx + 2][target_col] if train_end_idx + 2 < len(df) else np.nan

            if not np.isnan(real_val):
                predictions.append(pred)
                real_values.append(real_val)

        except Exception as e:
            continue

    # Sonuçlar
    if len(predictions) > 0:
        rmse = np.sqrt(mean_squared_error(real_values, predictions))
        mae = mean_absolute_error(real_values, predictions)

        print(f"✅ {model_type} tamamlandı")
        print(".6f")
        print(".6f")
        print(f"📈 Geçerli tahmin sayısı: {len(predictions)}")

        return rmse, mae, len(predictions)
    else:
        print(f"❌ {model_type} için yeterli veri bulunamadı")
        return np.nan, np.nan, 0

def rolling_window_forecast_with_predictions(df, model_type='RF', nprev=180, target_col='inflation_rate'):
    """Rolling window forecasting - tahminleri de döndürür"""
    print(f"\n🏃 {model_type} modeli - Rolling Window Testi")

    predictions = []
    real_values = []
    pred_dates = []

    test_data = prepare_rolling_window_data(df, nprev)

    for i in range(len(test_data) - 1):
        if i % 20 == 0:
            print(f"Progress: {i+1}/{len(test_data)-1}")

        # Eğitim verisi: test noktasına kadar
        train_end_idx = len(df) - nprev + i
        train_data = df.iloc[:train_end_idx + 1]

        # Test noktası
        test_point = df.iloc[train_end_idx + 1:train_end_idx + 2]

        if len(train_data) < 50:  # Minimum eğitim verisi
            continue

        try:
            if model_type == 'RF':
                pred = run_rf_model(train_data, test_point, target_col)
            elif model_type == 'AR':
                pred = run_ar_model(train_data, target_col)
            elif model_type == 'LASSO':
                pred = run_lasso_model(train_data, test_point, target_col)

            # Gerçek değer (bir sonraki ay)
            real_val = df.iloc[train_end_idx + 2][target_col] if train_end_idx + 2 < len(df) else np.nan
            pred_date = df.index[train_end_idx + 1]

            if not np.isnan(real_val):
                predictions.append(pred)
                real_values.append(real_val)
                pred_dates.append(pred_date)

        except Exception as e:
            continue

    # Sonuçlar
    if len(predictions) > 0:
        rmse = np.sqrt(mean_squared_error(real_values, predictions))
        mae = mean_absolute_error(real_values, predictions)

        print(f"✅ {model_type} tamamlandı")
        print(".6f")
        print(".6f")
        print(f"📈 Geçerli tahmin sayısı: {len(predictions)}")

        return rmse, mae, len(predictions), predictions, real_values, pred_dates
    else:
        print(f"❌ {model_type} için yeterli veri bulunamadı")
        return np.nan, np.nan, 0, [], [], []

def plot_forecasts(results, period_name, save_path=None):
    """Tahminleri görselleştir"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle(f'Aylık Enflasyon Oranı (%) Forecasting: {period_name}', fontsize=16, fontweight='bold')

    models = ['RF', 'AR', 'LASSO']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for idx, model in enumerate(models):
        ax = axes[idx]

        data = results[model]
        dates = data['dates']
        predictions = data['predictions']
        real_values = data['real_values']

        if len(predictions) > 0:
            # Gerçek enflasyon (CPI değişimi olarak hesapla)
            real_inflation = np.array(real_values)

            # Tahminler
            pred_inflation = np.array(predictions)

            # Plot
            ax.plot(dates, real_inflation, 'k-', linewidth=2, label='Gerçek Enflasyon (%)', alpha=0.8)
            ax.plot(dates, pred_inflation, color=colors[idx], linewidth=2, label=f'{model} Tahmini (%)', alpha=0.8)

            # Hata gölgelendirmesi
            errors = pred_inflation - real_inflation
            ax.fill_between(dates, pred_inflation, real_inflation,
                          where=(errors > 0), color='red', alpha=0.3, label='Fazla Tahmin')
            ax.fill_between(dates, pred_inflation, real_inflation,
                          where=(errors < 0), color='blue', alpha=0.3, label='Az Tahmin')

            # Metrikler
            rmse = data['rmse']
            mae = data['mae']
            ax.text(0.02, 0.98, f'RMSE: {rmse:.4f}%\nMAE: {mae:.4f}%',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Y-ekseni etiketi
            ax.set_ylabel('Enflasyon Oranı (%)')
            ax.grid(True, alpha=0.3)

            # Y-ekseni limitlerini ayarla (daha iyi görünüm için)
            y_min = min(min(real_inflation), min(pred_inflation)) - 0.5
            y_max = max(max(real_inflation), max(pred_inflation)) + 0.5
            ax.set_ylim(y_min, y_max)

            # Özel olayları işaretle (COVID, yüksek enflasyon dönemleri)
            if '2016-2025' in period_name:
                # COVID-19 başlangıcı
                covid_start = pd.to_datetime('2020-03-01')
                if dates[0] <= covid_start <= dates[-1]:
                    ax.axvline(x=covid_start, color='red', linestyle='--', alpha=0.7, label='COVID-19 Başlangıç')

                # Yüksek enflasyon dönemi
                inflation_peak = pd.to_datetime('2022-06-01')
                if dates[0] <= inflation_peak <= dates[-1]:
                    ax.axvline(x=inflation_peak, color='orange', linestyle='--', alpha=0.7, label='Enflasyon Zirvesi')

        ax.set_title(f'{model} Modeli', fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Tarih formatı
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Grafik kaydedildi: {save_path}")

    plt.show()

def plot_model_comparison(results, period_name, save_path=None):
    """Modelleri karşılaştıran grafik"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Model Karşılaştırması: {period_name}', fontsize=16, fontweight='bold')

    models = ['RF', 'AR', 'LASSO']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    rmse_values = []
    mae_values = []

    for model in models:
        data = results[model]
        rmse_values.append(data['rmse'])
        mae_values.append(data['mae'])

    # RMSE karşılaştırması
    bars1 = ax1.bar(models, rmse_values, color=colors, alpha=0.7)
    ax1.set_title('RMSE Karşılaştırması', fontweight='bold')
    ax1.set_ylabel('RMSE')
    ax1.grid(True, alpha=0.3)

    # Değer etiketleri
    for bar, val in zip(bars1, rmse_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}%', ha='center', va='bottom', fontweight='bold')

    # MAE karşılaştırması
    bars2 = ax2.bar(models, mae_values, color=colors, alpha=0.7)
    ax2.set_title('MAE Karşılaştırması (%)', fontweight='bold')
    ax2.set_ylabel('MAE (%)')
    ax2.grid(True, alpha=0.3)

    # Değer etiketleri
    for bar, val in zip(bars2, mae_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Karşılaştırma grafiği kaydedildi: {save_path}")

    plt.show()

def main():
    print("🔬 2025-11-MD.csv ile MAKALE REPLICATION TESTİ")
    print("=" * 60)

    # Veri yükleme ve transform
    df_full = load_and_transform_fred_md("2025-11-MD.csv")

    # Makaledeki zaman aralığı (2015'e kadar)
    df_2015 = df_full[df_full.index <= '2015-12-31']

    # 2016-2025 arası veri
    df_2016_2025 = df_full[(df_full.index >= '2016-01-01') & (df_full.index <= '2025-12-31')]

    print(f"\n📊 TAM VERI: {len(df_full)} gözlem")
    print(f"📅 {df_full.index.min()} - {df_full.index.max()}")

    print(f"\n📊 2015 VERISI: {len(df_2015)} gözlem")
    print(f"📅 {df_2015.index.min()} - {df_2015.index.max()}")

    print(f"\n📊 2016-2025 VERISI: {len(df_2016_2025)} gözlem")
    print(f"📅 {df_2016_2025.index.min()} - {df_2016_2025.index.max()}")

    results = {}

    # Test 1: Makaledeki zaman aralığı (2015'e kadar)
    print("\n" + "="*60)
    print("🧪 TEST 1: MAKALEDEKI ZAMAN ARALIĞI (1959-2015)")
    print("="*60)

    models = ['RF', 'AR', 'LASSO']
    results_2015 = {}

    for model in models:
        rmse, mae, n_preds, preds, reals, dates = rolling_window_forecast_with_predictions(df_2015, model, nprev=180)
        results_2015[model] = {
            'rmse': rmse, 'mae': mae, 'n_preds': n_preds,
            'predictions': preds, 'real_values': reals, 'dates': dates
        }

    # Test 2: Tam veri (2025'e kadar)
    print("\n" + "="*60)
    print("🧪 TEST 2: TAM VERI (1959-2025)")
    print("="*60)

    results_2025 = {}

    for model in models:
        rmse, mae, n_preds, preds, reals, dates = rolling_window_forecast_with_predictions(df_full, model, nprev=180)
        results_2025[model] = {
            'rmse': rmse, 'mae': mae, 'n_preds': n_preds,
            'predictions': preds, 'real_values': reals, 'dates': dates
        }

    # Test 3: 2016-2025 arası veri
    print("\n" + "="*60)
    print("🧪 TEST 3: 2016-2025 ARASI VERI (PANDEMİ DÖNEMİ)")
    print("="*60)

    # 2016-2025 için uygun test dönemi uzunluğu (verinin %70'i)
    nprev_2016_2025 = max(60, int(len(df_2016_2025) * 0.7))
    print(f"2016-2025 için test dönemi: {nprev_2016_2025} ay")

    results_2016_2025 = {}

    for model in models:
        rmse, mae, n_preds, preds, reals, dates = rolling_window_forecast_with_predictions(df_2016_2025, model, nprev=nprev_2016_2025)
        results_2016_2025[model] = {
            'rmse': rmse, 'mae': mae, 'n_preds': n_preds,
            'predictions': preds, 'real_values': reals, 'dates': dates
        }

    # Karşılaştırma
    print("\n" + "="*60)
    print("📊 SONUÇLAR KARŞILAŞTIRMASI")
    print("="*60)

    print("<15")
    print("-" * 90)
    print("<15")
    print("-" * 90)

    for model in models:
        rmse_2015 = results_2015[model]['rmse']
        mae_2015 = results_2015[model]['mae']
        rmse_2025 = results_2025[model]['rmse']
        mae_2025 = results_2025[model]['mae']
        rmse_2016_2025 = results_2016_2025[model]['rmse']
        mae_2016_2025 = results_2016_2025[model]['mae']

        print("<15")

    print("\n📋 MAKALEDEKI BEKLENEN DEĞERLER (h=1, CPI):")
    print("RF    RMSE: 0.0042, MAE: 0.0030")
    print("AR    RMSE: 0.0027, MAE: 0.0018")
    print("LASSO RMSE: 0.0039, MAE: 0.0028")

    # Görselleştirmeler
    print("\n" + "="*60)
    print("📊 GÖRSELLEŞTİRMELER")
    print("="*60)

    # 2016-2025 tahminleri görselleştirme
    print("\n📈 2016-2025 Dönemi Tahmin Grafikleri:")
    try:
        plot_forecasts(results_2016_2025, "2016-2025 Dönemi (Pandemi & Yüksek Enflasyon)",
                      save_path="inflation_forecasts_2016_2025.png")
    except Exception as e:
        print(f"Görselleştirme hatası: {e}")

    # Model karşılaştırması
    print("\n📊 2016-2025 Model Karşılaştırması:")
    try:
        plot_model_comparison(results_2016_2025, "2016-2025 Dönemi",
                             save_path="model_comparison_2016_2025.png")
    except Exception as e:
        print(f"Karşılaştırma hatası: {e}")

    # Sonuçları kaydet
    results['2015_data'] = results_2015
    results['2025_data'] = results_2025
    results['2016_2025_data'] = results_2016_2025

    print("\n✅ Testler ve görselleştirmeler tamamlandı!")
    return results

if __name__ == "__main__":
    results = main()
