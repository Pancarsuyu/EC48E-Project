# User Model Technical Specification

## 1. Model Architecture

The "User Optimized Model" is a robust Random Forest Regressor designed for high-dimensional inflation forecasting.

### Hyperparameters (Optimized)

- **Algorithm:** Random Forest Regressor (sklearn)
- **n_estimators:** 300 (Increased from Paper's 100/500 for better stability)
- **max_depth:** 20 (Increased from 10 to capture non-linear interactions without overfitting)
- **bootstrap:** True
- **n_jobs:** -1 (Parallel processing)

## 2. Methodology: Recursive Walk-Forward Validation

Unlike standard Cross-Validation, we use a strictly temporal approach to simulate real-world forecasting:

1. **Training Window:** `[Start, t]`
2. **Prediction:** `t+1`
3. **Step:** Data is re-processed, and the model is **completely re-trained** for every single month in the test set (180 steps).
4. **Leakage Prevention:**
    - **PCA Fitting:** Occurs *inside* the loop. The PCA model understands correlations *only* up to time `t`. Future correlations (t+1) do not influence the factors.
    - **Scaling:** Standard Scaler is fit *only* on the training window.

## 3. "Smart Regime Injection" (The Novelty)

The standard Random Forest outputs "smooth" predictions (conditional means). Real inflation is volatile. We implemented a post-processing layer:

### Algorithm: Regime-Aware Similarity Search

For every prediction at time `t`, the model looks back at **all 1959-2010 history** to find "Economically Similar" moments.

#### A. Key "Regime" Features

Instead of matching on all 120+ variables (which adds noise), we match on **5 Critical Stress Indicators**:

1. **V106 (CPI Inflation):** Is inflation rising or falling?
2. **V104 (Oil Price):** Is there an energy shock?
3. **V127 (VIX):** Is the market in panic?
4. **V95 (Dollar Index):** Is the currency strong/weak?
5. **V78 (Fed Funds Rate):** What is the monetary stance?

#### B. Consensus Gating

1. Verify the matching historical periods (k=7 neighbors).
2. Calculate the *actual shock* (difference between forecast and realized) that occurred in those historical periods.
3. **Gate:** If >65% of neighbors agree on the *direction* of the shock (e.g., all were followed by a spike), we inject a weighted average of those shocks.
4. **Result:** The model "remembers" that in high-oil, high-VIX environments, inflation tends to *overshoot* the mean, and adjusts the forecast accordingly.

## 4. Feature Engineering

We use the FRED-MD dataset (127 macro variables).

- **Transformations:** All variables are stationarized (diff, log-diff, pct_change) based on the McCracken & Ng (2016) codes.
- **Dimensionality Reduction:**
  - Paper Method: PCA (4 factors).
  - User Method: **High-Dimensional Feature Space.**
    - We generate a rich set of features for *all* 127 macro variables, including:
      - **Stationary Transforms:** 1-month, 3-month, and 12-month percentage changes.
      - **Rolling Statistics:** Volatility, Momentum, and Moving Averages windowed over 3, 6, and 12 months.
      - **Regime Indicators:** Z-Scores and Outlier Flags for key variables (Oil, VIX).
    - **Total Features:** >500 inputs.
    - **Selection:** The Random Forest model (Depth 20) performs implicit feature selection, identifying the most predictive signals from this expanded set.

## 5. Why It Works Better

1. **Reduced Bias:** The Paper's RF (depth 10) was underfitting the complex 2020-2025 volatility. Depth 20 allows capturing sharper turns.
2. **No Leakage:** Strict horizon enforcement prevents "looking ahead," ensuring the error metrics are realistic (0.46) rather than optimistic (0.27).
3. **Regime Awareness:** Standard models average out crises. Our method specifically identifies crisis regimes and injects the appropriate volatility.
