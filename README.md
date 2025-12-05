# ENSF611-FinPrediction

Hybrid time-series forecasting for AAPL using classical, deep, and boosting models.

## Abstract

We forecast 5-day AAPL returns by mixing technical indicators, fundamentals, and macro volatility features. Models include ARIMA, Temporal Convolutional Network (TCN), Temporal Fusion Transformer (TFT), an XGBoost residual booster, and a linear baseline. Everything runs in a single notebook for fast iteration on GPU.

## Background

Stock returns have weak, nonlinear signals that live at different time scales. Classical models like ARIMA capture simple autocorrelation but miss nonlinear structure. Sequence models (TCN, TFT) learn local and long-range patterns. XGBoost on residuals can correct systematic errors. This repo is an exploratory adaptation of hybrid ideas from recent CNN and transformer-based forecasting papers, applied to AAPL.

## Data and Features

- Horizon: predict 5-day forward percent change of close.
- Span: 2016-03-31 to 2025-11-05 (2,416 business-day rows).
- Sources: price/volume (`data/aapl_price.json`), fundamentals (`data/aapl_fundamentals.csv`), VIX (`data/vix.csv`).
- Data references: Price/Volume Data - Link; VIX - Link; Fundamental financials (quarterly) for EBITDA and EV (web scraped) - Link.
- Features (daily, aligned, forward-filled where needed):
  - Normalized close = close / 20-day SMA.
  - MACD line, MACD diff, MACD signal.
  - RSI.
  - Normalized OBV diff = OBV delta vs 20-day SMA of OBV.
  - EV/EBITDA from quarterly EV and EBITDA, forward-filled to daily.
  - Normalized VIX change = 5-day pct change scaled by 20-day SMA.
- Target: 5-day forward percent change of close.

## Models and Parameters

- ARIMA: order (1,0,1), no intercept.
- TCN: 60-step lookback, kernel 3, 25 filters, dropout 0.3, ~40 epochs, GPU trainer in Darts.
- TFT: 60-step lookback, hidden size 40, 2 LSTM layers, 4 attention heads, dropout 0.3, GPU trainer in Darts.
- XGBoost residual booster: 500 trees, depth 5, learning rate 0.05, subsample 0.8, colsample 0.8, hist GPU trainer, random_state 42. Trained on TFT residuals and added back to TFT predictions.
- Linear regression: scaled features, standard least squares baseline.

## Pipeline

1. Load price, fundamentals, VIX. Parse dates, set daily index, forward-fill gaps.
2. Build features listed above. Create target = 5-day forward percent change. Drop rows with NA. Keep business-day frequency.
3. Chronological split: 90 percent train, 10 percent test.
4. Train model branches: ARIMA, TCN, TFT, linear regression. Train XGBoost on TFT residuals and add residual predictions to TFT output.
5. Evaluate train and test R² and plot predicted vs actual plus residuals.

## Results (R² ×100, from `data/aapl_model_metrics.csv`)

- ARIMA: 61.16 train / -0.26 test.
- TCN: 84.74 train / 69.15 test (best generalization).
- TFT: 56.72 train / 48.90 test.
- XGBoost booster: 97.79 train / 66.62 test (strong but larger gap).
- Linear: 0.57 train / -0.48 test.

Interpretation: Nonlinear sequence models beat classical and linear baselines on this horizon. XGBoost fits tightly on train but gives up some ground on test. ARIMA and linear are not competitive on out-of-sample data.

## Limitations

- Single asset (AAPL) and single horizon (5-day). No sentiment features yet, though planned in proposal.
- Only one 90/10 split; walk-forward CV would be more robust.
- Quarterly EV/EBITDA is coarse and forward-filled.
- GPU dependencies (RAPIDS, Darts) can be finicky across environments.

## How to Run (notebook-first)

1. Open `src/FinPrediction.ipynb` in Google Colab with GPU runtime.
2. Upload required data files from `data/` into Colab file browser.
3. Run cells in order. GPU is needed for cudf/cuml ARIMA and Darts GPU trainers.
4. Metrics and derived CSVs are written in the notebook flow.

## Repo Map

- `src/FinPrediction.ipynb` - main notebook with feature engineering, models, and metrics.
- `docs/ProjectProposal.md` - original project proposal.
- `data/*.csv` and `data/aapl_price.json` - raw and derived datasets (see Data and Features).

## References

- ARIMA via RAPIDS cuML.
- Darts library for TCN and TFT.
- XGBoost documentation for hist GPU training.
- Z. Shi, Y. Hu, G. Mo, and J. Wu, “Attention-based CNN-LSTM and XGBoost hybrid model for stock prediction,” arXiv preprint arXiv:2204.02623, 2024.
- Z.-x. Hu, B. Shen, Y. W. Hu, and C. Zhao, “Research on stock price forecast of General Electric based on mixed CNN-LSTM model,” arXiv preprint arXiv:2501.08539, 2025.
- S. F. Stefenon, J. P. Matos-Carvalho, V. R. Q. Leithardt, Senior Member, IEEE, and K.-C. Yow, Senior Member, IEEE, “CNN-TFT explained by SHAP with multi-head attention weights for time-series forecasting,” arXiv preprint arXiv:2510.06840, 2025.
