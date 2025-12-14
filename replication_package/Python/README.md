# Python ile Replication

Bu klasör, Python ile yazılmış replication kodlarını içerir.

## 📋 Gereksinimler

- **Python** 3.8+
- **Gerekli Paketler**:
  ```bash
  pip install -r requirements.txt
  ```

## 📦 Kurulum

```bash
# Gerekli paketleri yükle
pip install pandas numpy scikit-learn pyreadr

# Veya requirements.txt ile
pip install -r requirements.txt
```

## 🚀 Çalıştırma

```bash
python run_replication.py
```

## 📊 Veri

Program otomatik olarak veriyi indirir ve hazırlar. Alternatif olarak:

1. Veri dosyasını `data/` klasörüne koyun
2. Kod otomatik olarak tanır

## 📈 Beklenen Çıktı

```
=== REPLICATION RESULTS ===
Random Forest h=1 Results:
RMSE: 0.0045
MAE: 0.0034

Expected from paper:
RF RMSE: ~0.0042
RF MAE: ~0.0030
```

## 🔧 Sorun Giderme

- **Paket hatası**: `pip install` ile eksik paketleri kurun
- **Memory hatası**: Daha küçük `window_size` kullanın
- **Data hatası**: İnternet bağlantınızı kontrol edin

## 📝 Özellikler

- ✅ **Otomatik veri indirme**
- ✅ **Tüm modeller**: RW, AR, LASSO, RF
- ✅ **Rolling window forecasting**
- ✅ **Detaylı sonuçlar**
- ✅ **Grafikler ve analiz**

## 🎯 Minimum Kod

Eğer sadece temel RF replikasyonu yapmak istiyorsanız:

```python
from replication_package import run_basic_rf

# Sadece RF çalıştır
results = run_basic_rf()
print(f"RF RMSE: {results['rmse']:.4f}")
```

