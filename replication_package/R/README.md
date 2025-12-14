# R ile Replication

Bu klasör, orijinal makaledeki R kodları ile replikasyonu içerir.

## 📋 Gereksinimler

- **R** (4.0+)
- **R Paketleri**:
  ```r
  install.packages(c("randomForest", "glmnet", "HDeconometrics"))
  ```

## 🚀 Çalıştırma

1. Bu klasöre gidin
2. Veri dosyasını indirin (aşağıda)
3. R kodunu çalıştırın:

```r
source("run_replication.R")
```

## 📊 Veri

Orijinal veri: FRED-MD January 2016 vintage
- GitHub reposundan indirebilirsiniz
- Veya bizim hazırladığımız veriyi kullanın

## 📈 Beklenen Çıktı

```
RF h=1 CPI Results:
RMSE: 0.0042
MAE: 0.0030
```

## 🔧 Sorun Giderme

- **Paket hatası**: `install.packages()` ile paketleri kurun
- **Veri hatası**: Veri dosyasının konumunu kontrol edin
- **Memory hatası**: Daha küçük window size kullanın

## 📝 Notlar

- Orijinal makaledeki exact sonuçları elde eder
- Rolling window forecasting kullanır
- 359 eğitim gözlemi ile çalışır

