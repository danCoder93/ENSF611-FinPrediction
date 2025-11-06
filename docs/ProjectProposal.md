# AAPL Stock Forecasting

**Hybrid CNN→TFT Model for Multi-Horizon Forecasting of AAPL with Optional ARIMA/XGBoost**

---

## 1. Research Motivation and Background

### What

This project develops a **hybrid deep learning framework** to forecast **Apple Inc. (AAPL)**’s stock price dynamics by combining **fundamental**, **technical**, and **sentiment-based** indicators.
It draws structural inspiration from the “Attention-based CNN–LSTM and XGBoost hybrid model for stock prediction” [1] but **replaces the LSTM component** with a **Temporal Fusion Transformer (TFT)** for richer temporal modeling and interpretability.

Both **ARIMA pre-processing** and **XGBoost residual correction** are included as **optional modules**, enabling modular experimentation within a unified architecture.

### Why

Stock price movements are influenced by interactions among **company fundamentals**, **market technicals**, and **investor sentiment**.
Hybrid neural architectures—especially those combining **convolutions for local trend extraction** and **transformers for long-term dependencies**—have recently achieved state-of-the-art performance on financial time-series data [1][2][3].

This project applies that concept to AAPL, seeking to answer:
*Can a CNN→TFT hybrid capture multi-scale dependencies across heterogeneous financial features to produce accurate, interpretable stock forecasts?*

### How (Research Lineage)

| Concept                                                            | Prior Evidence             | Adaptation in This Project                                       |
| ------------------------------------------------------------------ | -------------------------- | ---------------------------------------------------------------- |
| Attention-CNN captures short-term temporal signals                 | AttCLX (2024) [1]          | Retain CNN encoder to model localized movements in AAPL’s series |
| Transformer fusion improves long-term and multi-covariate modeling | CNN–TFT (2025) [3]         | Replace LSTM with TFT backbone for multi-horizon forecasting     |
| Hybrid CNN-sequence models achieve superior forecasting stability  | CNN–LSTM Hybrid (2025) [2] | Preserve CNN front-end and temporal gating ideas in TFT          |
| Residual correction improves generalization                        | AttCLX (2024) [1]          | Keep XGBoost as optional residual corrector only                 |

---

## 2. Research Objectives

1. **Design** and implement a CNN→TFT hybrid forecasting model for AAPL.
2. **Integrate** heterogeneous features: fundamentals, technical indicators, and sentiment.
3. **Enable** optional ARIMA pre-processing and XGBoost fine-tuning modules.
4. **Quantify** optional interpretability via SHAP.
5. **Evaluate** predictive power through statistical metrics and optional linear baseline.

---

## 3. System Architecture

```
┌──────────────────────────────────────────┐
│ [Optional] ARIMA Pre-Processing          │
│ - Detrend / Smooth AAPL time series      │
└──────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────┐
│ Attention-Based CNN Encoder              │
│ - Multi-scale convolution layers         │
│ - Extract local temporal patterns        │
└──────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────┐
│ Temporal Fusion Transformer (TFT)        │
│ - Gated residual connections             │
│ - Multi-horizon forecasting              │
│ - Variable- and time-level attention     │
└──────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────┐
│ [Optional] XGBoost Residual Correction   │
│ - Fine-tune residuals from CNN→TFT       │
└──────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────┐
│ [Optional] SHAP Explainability           │
│ - Feature-level importance visualization │
└──────────────────────────────────────────┘
```

---

## 4. Data Preparation and Feature Engineering

### 4.1 Data Ingestion

* **Scope:** AAPL (Apple Inc.)
* **Sources:**

  * Historical OHLCV data via Yahoo Finance or AlphaVantage.
  * Fundamental financials (quarterly) for EBITDA, EV/EBITDA.
  * Technical indicators derived from OHLCV: MACD, RSI.
  * Market-level feature: Volatility Index (VIX or realized volatility).
  * Sentiment feature: Daily *sentiment score* — a normalized scalar derived from FinBERT-based financial news analysis (averaged daily).

All series are aligned into a unified, daily-frequency panel, forward-filled for missing days, and normalized for scale compatibility.

---

### 4.2 Feature Set

| Category             | Feature                   | Description                              | Rationale                                                        |
| -------------------- | ------------------------- | ---------------------------------------- | ---------------------------------------------------------------- |
| **Fundamental**      | **EBITDA**                | Core measure of operating profitability  | Reflects business fundamentals                                   |
| **Fundamental**      | **EV/EBITDA**             | Valuation multiple                       | Complements EBITDA by embedding market expectations              |
| **Technical**        | **MACD**                  | Momentum-based moving average divergence | Detects momentum shifts                                          |
| **Technical**        | **RSI**                   | Relative Strength Index                  | Independent oscillator capturing overbought/oversold dynamics    |
| **Macro/Volatility** | **VIX (proxy)**           | Market volatility index                  | Captures systemic uncertainty                                    |
| **Sentiment**        | **Daily Sentiment Score** | Mean daily polarity from FinBERT outputs | Represents investor sentiment intensity impacting price movement |

---

## 5. Model Training and Evaluation

### Model Setup

* **Primary architecture:** CNN→TFT hybrid
* **Optional modules:** ARIMA preprocessing, XGBoost residual correction
* **Optional comparison:** Linear regression baseline (for interpretability benchmarking)

### Metrics

* MAE, RMSE, MAPE, R²
* Directional accuracy (percentage of correct up/down predictions)

### Interpretability (Optional)

Global SHAP Analysis — Calculate mean absolute SHAP values across features to estimate their relative contribution to prediction accuracy.
This provides a compact and interpretable feature-importance view consistent with the interpretability framework of the CNN–TFT paper [3].

---

## 6. Implementation Roadmap

| Phase                                                | Duration  | Deliverable                                        |
| ---------------------------------------------------- | --------- | -------------------------------------------------- |
| **1. Data Acquisition & Cleaning**                   | 1 week    | Unified AAPL dataset with 6 aligned features       |
| **2. ARIMA Pre-Processing (Optional)**               | 0.5 week  | Differenced or smoothed price series               |
| **3. CNN Encoder Implementation**                    | 1 week    | Attention-CNN module extracting local dependencies |
| **4. TFT Backbone Development**                      | 1.5 weeks | Transformer with multi-horizon output              |
| **5. XGBoost Residual Stage (Optional)**             | 0.5 week  | Optional residual correction                       |
| **6. Explainability Integration (SHAP & Attention)** | 1 week    | Feature importance and interpretability reports    |
| **7. Evaluation & Reporting**                        | 0.5 week  | Metrics, plots, and ablation results               |


---

## 7. Expected Contributions

### Academic

* Demonstrates how **fundamental, technical, macro, and sentiment features** interact in a deep hybrid model.
* Extends AttCLX architecture with a **TFT backbone** to enable interpretability without recurrent components.

### Engineering

* End-to-end reproducible pipeline in PyTorch/TensorFlow.
* Optional toggles for ARIMA/XGBoost to modularize experimentation.
* Optional Integrated SHAP explainability layer for feature-wise attribution.

### Educational

* Illustrates practical integration of **deep learning + finance** with explainable methods.
* Offers an accessible baseline for multi-feature forecasting tasks on cloud or local environments.

---

## 8. Reference Mapping

| Component                 | Source Paper                 | Contribution                                              |
| ------------------------- | ---------------------------- | --------------------------------------------------------- |
| Attention-CNN Encoder     | Shi *et al.* (2024) [1]      | Efficient feature extraction from time series             |
| Hybrid Model Design       | Hu *et al.* (2025) [2]       | CNN + sequence hybrid structure principles                |
| TFT & SHAP Explainability | Stefenon *et al.* (2025) [3] | Multi-horizon forecasting and interpretability mechanisms |

---

## References

[1] Z. Shi, Y. Hu, G. Mo, and J. Wu, “*Attention-based CNN-LSTM and XGBoost hybrid model for stock prediction*,” *arXiv preprint* arXiv:2204.02623, 2024.

[2] Z.-x. Hu, B. Shen, Y. W. Hu, and C. Zhao, “*Research on stock price forecast of General Electric based on mixed CNN-LSTM model*,” *arXiv preprint* arXiv:2501.08539, 2025.

[3] S. F. Stefenon, J. P. Matos-Carvalho, V. R. Q. Leithardt, Senior Member, IEEE, and K.-C. Yow, Senior Member, IEEE, “*CNN-TFT explained by SHAP with multi-head attention weights for time-series forecasting*,” *arXiv preprint* arXiv:2510.06840, 2025.

---