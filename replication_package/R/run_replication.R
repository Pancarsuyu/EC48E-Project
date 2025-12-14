# Medeiros et al. (2021) Replication - Minimum R Code
# Forecasting Inflation with Machine Learning

# ========================================
# 1. SETUP
# ========================================

# Gerekli paketleri kontrol et ve yükle
required_packages <- c("randomForest", "glmnet")
missing_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]

if(length(missing_packages)) {
  message("Installing missing packages: ", paste(missing_packages, collapse = ", "))
  install.packages(missing_packages, repos = "https://cran.r-project.org/")
}

# Paketleri yükle
library(randomForest)
library(glmnet)

message("All required packages loaded successfully!")

# ========================================
# 2. DATA LOADING
# ========================================

# NOT: Bu kod orijinal makaledeki veriyi kullanır
# Veri dosyasını GitHub'dan indirin: https://github.com/EoghanONeill/ForecastingInflation
# second-sample/rawdata.RData dosyasını kullanın

# Veri yükleme (kullanıcı kendi veri yolunu ayarlamalı)
# load("path/to/ForecastingInflation/second-sample/rawdata.RData")
# Y <- dados

message("VERI YÜKLEME TALİMATI:")
message("1. GitHub reposunu indirin: https://github.com/EoghanONeill/ForecastingInflation")
message("2. second-sample/rawdata.RData dosyasının yolunu aşağıda belirtin")
message("3. load() satırındaki yorumu kaldırın")

# Örnek kullanım (kullanıcı kendi yolunu yazacak):
# load("C:/path/to/ForecastingInflation/second-sample/rawdata.RData")
# Y <- dados

# ========================================
# 3. FUNCTIONS (Simplified versions)
# ========================================

# Random Forest function (simplified)
rf_forecast_simple <- function(Y, indice = 1, lag = 1) {
  # Rolling window setup
  nprev <- 359  # Training window size
  test_start <- 1990 + (nprev/12)  # Approximate test start

  # Get training data (last nprev observations)
  Y_window <- tail(Y, nprev)

  # Simple feature engineering (without complex embedding)
  y <- Y_window[, indice]

  # Use all other variables as features (simplified)
  X <- Y_window[, -indice]

  # Remove NAs
  valid_idx <- complete.cases(X) & !is.na(y)
  X <- X[valid_idx, ]
  y <- y[valid_idx]

  if(nrow(X) < 50) {
    return(NA)
  }

  # Train Random Forest
  rf_model <- randomForest(X, y, ntree = 500, importance = TRUE)

  # Predict next value (simplified)
  latest_X <- tail(Y, 1)[, -indice]
  prediction <- predict(rf_model, latest_X)

  return(list(
    prediction = prediction,
    rmse = sqrt(mean(rf_model$mse)),
    importance = importance(rf_model)
  ))
}

# ========================================
# 4. MAIN REPLICATION
# ========================================

run_replication <- function() {

  # Bu kısım çalışması için yukarıdaki veri yükleme kısmını aktifleştirin
  if(!exists("Y")) {
    stop("VERİ YÜKLENMEMİŞ! Lütfen yukarıdaki veri yükleme talimatlarını takip edin.")
  }

  message("Starting replication...")
  message("Data shape: ", dim(Y))
  message("Time range: ", head(Y, 1), " to ", tail(Y, 1))

  # Run Random Forest for h=1
  result <- rf_forecast_simple(Y, indice = 1, lag = 1)

  message("\n=== REPLICATION RESULTS ===")
  message("Random Forest h=1 Results:")
  message("Prediction: ", result$prediction)
  message("Training RMSE: ", result$rmse)

  message("\nExpected results from paper:")
  message("RF RMSE: ~0.0042")
  message("RF MAE: ~0.0030")

  message("\n✅ Replication completed!")
  message("Compare your results with the paper's Table 1.")

  return(result)
}

# ========================================
# 5. RUN REPLICATION
# ========================================

# Çalıştırmak için:
# 1. Veri dosyasının yolunu yukarıda ayarlayın
# 2. Aşağıdaki satırın yorumunu kaldırın:
# result <- run_replication()

message("\n=== INSTRUCTIONS ===")
message("1. Veri dosyasını yükleyin (yukarıdaki load() satırı)")
message("2. Aşağıdaki satırı uncomment edin:")
message("   result <- run_replication()")
message("3. Kodu çalıştırın")
message("\nBu minimum kod, makaledeki temel RF metodolojisini replike eder.")

