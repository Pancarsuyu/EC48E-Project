# 🔬 Inflation Forecasting Replication Study

**Makale:** Medeiros et al. (2018) - "Forecasting Inflation in a data-rich environment: the benefits of machine learning methods"

**Amaç:** Orijinal makaledeki enflasyon forecasting modellerini (RF, AR, LASSO) 2025'e kadar güncel verilerle test etmek ve karşılaştırmak.

---

## 📁 Proje İçeriği

### 📊 **Veri Dosyaları**
- `2025-11-MD.csv` - Güncel FRED-MD aylık makroekonomik göstergeler (1959-2025)
- `ForecastingInflation/` - Orijinal makale kodları ve verileri

### 🐍 **Python Kodları**
- `test_2025_data.py` - Ana test scripti (RF, AR, LASSO modelleri)
- `replication_package/Python/` - Temiz replication kodları

### 📈 **Grafikler ve Görselleştirmeler**
- `inflation_forecasts_2016_2025.png` - 2016-2025 enflasyon tahminleri
- `model_comparison_2016_2025.png` - Model performans karşılaştırması

### 📄 **Raporlar**
- `replication_test_results.md` - Detaylı analiz raporu
- `replication_package/README.md` - Kod kullanım kılavuzu

---

## 🚀 Nasıl Çalıştırılır

### Gereksinimler
```bash
pip install pandas numpy scikit-learn matplotlib pyreadr
```

### Temel Test Çalıştırma
```bash
python test_2025_data.py
```

Bu komut:
- 2025-11-MD.csv'yi yükler ve işler
- FRED-MD transformasyonlarını uygular
- 3 farklı dönemde (2015'e kadar, 2015-2025, 2016-2025) modelleri test eder
- Grafikler oluşturur

---

## 📊 Ana Bulgular

### 🔍 **Tahmin Edilen Değişken**
**Aylık enflasyon oranı (%)** - CPI'nin aylık yüzde değişimi

### 📈 **Model Performansları (RMSE %)**

| Dönem | RF | AR | LASSO | En İyi Model |
|-------|----|----|--------|--------------|
| **2015'e kadar** | 0.58 | 0.57 | 0.58 | AR |
| **2015-2025** | 0.21 | 0.20 | 0.21 | AR |
| **2016-2025 (Pandemi)** | 0.27 | 0.30 | 0.26 | RF/LASSO |

### 🎯 **Ana Sonuçlar**
- **AR modeli** normal dönemlerde en başarılı
- **ML yöntemleri (RF/LASSO)** kriz dönemlerinde öne çıkıyor
- **2025 verisi** ile modeller %63 daha iyi performans gösteriyor
- **Pandemi sonrası** AR zayıfladı, ML yöntemleri avantajlı

---

## 🛠 Teknik Detaylar

### Kullanılan Modeller
- **RF (Random Forest):** 500 ağaç, tüm ekonomik göstergeler
- **AR (Autoregressive):** 4. dereceden AR modeli
- **LASSO:** α=0.001, regularization ile özellik seçimi

### Özellikler
- 126+ ekonomik gösterge (FRED-MD)
- PCA bileşenleri (4 adet)
- Lag değişkenleri (4 dönem)
- AR terimleri (4 dönem)

### Test Metodolojisi
- **Rolling Window Forecasting** (180 aylık test dönemi)
- **Horizon:** h=1 (1-aylık öndeyi)
- **Validasyon:** Rolling window cross-validation

---

## 📈 Grafik Açıklamaları

### `inflation_forecasts_2016_2025.png`
- **3 panel:** RF, AR, LASSO modelleri için ayrı grafikler
- **Siyah çizgi:** Gerçek enflasyon oranı (%)
- **Renkli çizgiler:** Model tahminleri (%)
- **Renklendirme:** Kırmızı=mazla tahmin, Mavi=az tahmin
- **Dikey çizgiler:** COVID-19 başlangıcı ve enflasyon zirvesi

### `model_comparison_2016_2025.png`
- **İki bar grafik:** RMSE ve MAE karşılaştırması
- **Yüzde değerleri** ile performans kıyaslaması

---

## 📚 Kaynaklar

- **Orijinal Makale:** Medeiros, M. C., Vasconcelos, G., Veiga, A., & Zilberman, E. (2018). Forecasting Inflation in a data-rich environment: the benefits of machine learning methods. *Journal of Applied Econometrics*.

- **Veri Kaynağı:** FRED-MD (Federal Reserve Economic Data - Monthly Database)

- **GitHub Repo:** [ForecastingInflation](https://github.com/EoghanONeill/ForecastingInflation)

---

## 👥 İletişim

Bu çalışma eğitim amaçlı hazırlanmıştır. Sorularınız için:

- **GitHub Issues:** Bu repo üzerinden soru sorabilirsiniz
- **E-posta:** [sizineposta@domain.com]

---

## 📄 Lisans

Bu çalışma akademik araştırma amaçlıdır. Orijinal makalenin lisans koşulları geçerlidir.

---

*Son güncelleme: 14 Aralık 2025* 🚀
