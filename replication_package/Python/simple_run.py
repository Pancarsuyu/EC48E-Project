#!/usr/bin/env python3
"""
Basit Replication Kodu - Sadece RF ile Test
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("🔬 BASİT RF REPLICATION TESTİ")
print("=" * 40)

# 1. Veri yükleme (manuel)
print("Veri yükleniyor...")
try:
    # GitHub reposundan second-sample verisini yükle
    # Bu kodu çalıştırmadan önce:
    # 1. git clone https://github.com/EoghanONeill/ForecastingInflation.git
    # 2. Aşağıdaki yolu kendi bilgisayarınızdaki yola göre değiştirin

    import pyreadr

    # VERİ YOLUNU KENDİ BİLGISAYARINIZDAKİ YOLA GÖRE DEĞIŞTIRIN:
    # Örnek: Eğer masaüstünde ise
    data_path = r"C:\Users\Pancar\Desktop\arşiv\boun\EC\TermProject\ForecastingInflation\second-sample\rawdata.RData"

    # Alternatif: İndirdiğiniz yere göre ayarlayın
    # data_path = r"C:\Downloads\ForecastingInflation\second-sample\rawdata.RData"
    # data_path = r"C:\Users\YOUR_USERNAME\Documents\ForecastingInflation\second-sample\rawdata.RData"

    result = pyreadr.read_r(data_path)
    df = result['dados']

    print(f"✅ Veri yüklendi: {df.shape}")

except Exception as e:
    print(f"❌ Veri yükleme hatası: {e}")
    print("Lütfen:")
    print("1. GitHub reposunu indirin: git clone https://github.com/EoghanONeill/ForecastingInflation.git")
    print(f"2. data_path değişkenini kendi yolunuzla değiştirin")
    exit()

# 2. Veri hazırlama
print("Veri hazırlanıyor...")
test_mask = (df.index >= '1990-01-01') & (df.index <= '2015-12-31')
test_data = df[test_mask]

print(f"Test dönemi: {test_data.index.min()} - {test_data.index.max()}")
print(f"Test gözlemleri: {len(test_data)}")

target = 'CPI'

# 3. Basit RF testi
print("RF modeli çalışıyor...")
predictions = []
real_values = []
window_size = 359

for i in range(len(test_data) - 1):
    if i % 20 == 0:
        print(f"Progress: {i+1}/{len(test_data)}")

    # Training data
    train_end = test_data.index[i]
    train_data = df.loc[:train_end]

    if len(train_data) < window_size:
        continue

    train_window = train_data.iloc[-window_size:]
    y_train = train_window[target]
    X_train = train_window.drop(columns=[target])

    # RF
    rf = RandomForestRegressor(n_estimators=100, random_state=42)  # Daha hızlı için
    rf.fit(X_train, y_train)

    # Predict
    current_X = test_data.iloc[i:i+1].drop(columns=[target])
    pred = rf.predict(current_X)[0]
    real_next = test_data[target].iloc[i + 1]

    predictions.append(pred)
    real_values.append(real_next)

# 4. Sonuçlar
print("\n" + "=" * 40)
print("📊 SONUÇLAR")
print("=" * 40)

if len(predictions) > 0:
    rmse = np.sqrt(mean_squared_error(real_values, predictions))
    mae = mean_absolute_error(real_values, predictions)

    print(".6f")
    print(".6f")
    print(f"Geçerli tahmin sayısı: {len(predictions)}")

    print("\nMakaledeki beklenen değerler:")
    print("RF RMSE: ~0.0042")
    print("RF MAE: ~0.0030")

    print("\n✅ Test tamamlandı!")
else:
    print("❌ Yeterli veri bulunamadı")

