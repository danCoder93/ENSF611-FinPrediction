## Project Title

**Hybrid CNN → TFT with Residual XGBoost for Multi-Horizon Stock Forecasting: A Case Study on AAPL (10-year horizon)**

---

## 1. Motivation & Prior Work

Financial time-series forecasting is hard: signals are noisy, regimes shift, and the task mixes short-term motifs with long-range dependencies. Hybrid approaches help CNNs catch local patterns, while sequence models (RNNs/Transformers) capture longer arcs.

Recent work shows that pairing CNNs with transformer-style backbones can outperform standalone CNN/RNN setups. ([arxiv.org][1]) One study applies a CNN–Transformer hybrid to finance with strong results. ([arxiv.org][2]) Separately, residual correction with tree models (e.g., XGBoost) often picks up nonlinear leftovers the deep model misses.

This project combines those ideas: a CNN front end for local features → a Temporal Fusion Transformer (TFT) in PyTorch Forecasting / Lightning for multi-horizon forecasts and interpretability → a residual XGBoost head for error correction (in the spirit of AttCLX).

Using 10 years of AAPL data with fundamentals, technicals, macro risk (VIX), and sentiment (FinBERT), we forecast log-returns at 1, 5, and 10 days. We’ll evaluate predictive metrics (RMSE, IC) and trading-style metrics (Sharpe, max drawdown, turnover).

We build on prior work (e.g., Ge 2025, “Enhancing stock market forecasting: A hybrid model…,” [sciencedirect.com][3]) and add a specific blend: CNN→TFT for multi-horizon forecasting, residual XGBoost, FinBERT sentiment features, and a strict walk-forward CV + backtest framework.

---

## 2. Data Scope

**Asset**: AAPL (Apple Inc.) covering 10 years of history.

**Data sources and type**:

- Daily, adjusted OHLCV prices for AAPL from yfinance.
- Daily macro risk proxy: VIX (lagged by 1 business‐day) from yfinance.
- Quarterly fundamental data: EBITDA (TTM), EV, derived EV/EBITDA (TTM) from yfinance or FinancialModelingPrep (free tier acceptable).
- Calendar flags (earnings date windows, month/quarter end, U.S. holidays) — optional but recommended.
- **Sentiment features**: Outputs from a separate FinBERT‐based pipeline, applied to news articles, earnings calls transcripts, and press releases for AAPL. (Details: see Section 4.)

**Time‐indexing & alignment**:

- Business‐day index aligned to U.S. trading days.
- Price data adjusted for splits and dividends.
- Fundamental signals stamped at (filing date + 1 business day), forward‐filled until next report.
- VIX lagged by 1 trading day to avoid contemporaneous leakage.

---

## 3. Core Feature Set (9 signals)

**Fundamentals (2, quarterly → daily alignment with safe lags)**

1. EBITDA (TTM) — operating profitability.
2. EV/EBITDA (TTM, log‐transformed) — valuation/mean-reversion signal.

**Technicals (6, daily)** 3. SMA-5 (5-day simple moving average) 4. SMA-20 5. SMA-50 6. SMA-200 7. RSI-14 (14‐day relative strength index) 8. MACD (12,26,9) — use the MACD line as the core signal for the “budget” of 9 features.

**Macro (1, daily)** 9. VIX (lag 1 day) — a risk‐appetite proxy.

**Engineering & alignment rules**:

- Fundamentals: stamp at filing date + 1 business day, then forward‐fill until next filing.
- VIX: lag by 1 trading day.
- All technicals computed on adjusted close (adjusted for splits/dividends).
- Features normalized: z-score per feature; for EV/EBITDA first apply log transform then z-score.
- Optional engineered variants (if time permits): distance from moving averages, crossovers, slopes of MAs, RSI regimes, ΔVIX & VIX z‐score.

**Sentiment features (placeholder)**:

- Additional feature block: e.g., FinBERT sentiment score for past n days, sentiment momentum, sentiment volatility.
- To be treated as a time‐varying observed feature (or known‐future if using upcoming press release date).
- Alignments: sentiment features stamped at publication date + safe lag (e.g., 1 business day) to avoid leak.
- Expect separate preprocessing (text ingestion → FinBERT → numerical score(s)) pipeline, producing outputs that become additional columns in the dataset.

---

## 4. Prediction Targets

**Primary (Required):** Multi‐horizon forward log‐returns:

- ( \text{ret_fwd_1} = \ln\big(\text{Close}[t+1]/\text{Close}[t]\big) )
- ( \text{ret_fwd_5} = \ln\big(\text{Close}[t+5]/\text{Close}[t]\big) )
- ( \text{ret_fwd_10} = \ln\big(\text{Close}[t+10]/\text{Close}[t]\big) )
  Rows with NaNs (due to horizon look-ahead) will be removed.

**Auxiliary (Optional):**

- 3‐class direction over 5 days (−1 / 0 / +1).
- 10-day realized volatility (as risk target) — if time allows.

**Note:** The sentiment features from the FinBERT‐based pipeline will feed into the upstream model; we treat them as input features, **not** as targets.

---

## 5. Model Stack

**Required components:**

1. **CNN encoder** (1D convolutions) on a 60‐day lookback window of the 9 (or more, including sentiment) features to extract local motifs.
2. **TFT backbone** (via PyTorch Forecasting) for multi‐horizon sequence modelling and built‐in attention for interpretability.
3. **Walk‐forward cross‐validation** with yearly folds + embargo, early stopping & dropout regularization.

**Optional (if time allows / improves metrics):** 4. ARIMA‐based residualization (pre-filter) on price or returns to stabilize the series (borrowed from AttCLX). 5. **Residual XGBoost head**: train an XGBoost regressor on the residuals ( (y - \hat y\_{TFT}) ) using current features (and/or TFT latent features) to capture leftover nonlinearities. 6. Probability‐calibration (Platt/Isotonic) if the classification head (auxiliary) is active. 7. Explainability pack: SHAP on the XGBoost residual head + TFT attention maps for per‐horizon driver analysis.

---

## 6. Pipeline: End-to-End Workflow

**1. Ingest & Clean (Required):**

- Pull AAPL 10 yr OHLCV, VIX series, fundamentals (EBITDA, EV → compute EV/EBITDA).
- Construct business‐day index; adjust for splits/dividends.

**2. Safe Alignment (Required):**

- Stamp fundamentals at filing date +1; forward‐fill until next.
- Lag VIX by one day.
- Compute SMA5/20/50/200, RSI14, MACD line.
- Load sentiment feature outputs (FinBERT pipeline) with safe lag (e.g., publication +1 day); align to business‐day index.

**3. Target Construction (Required):**

- Compute ret_fwd_1, ret_fwd_5, ret_fwd_10 and drop rows with NaN look‐ahead.

**4. (Optional) ARIMA residualization:**

- Fit ARIMA (or similar) on Close (or returns) on training period; compute residuals; pass residuals as extra input channel or pre‐filter target.

**5. Dataset & Dataloaders (Required):**

- Encoder length = 60 days; prediction length = {1,5,10}.
- Time-varying unknown reals = the feature set (including sentiment).
- Known‐future covariates = calendar flags (if included).
- Group id = “AAPL” (in case of panel extension later).
- Normalize features (z-score) based on training folds.

**6. Model Training (Required):**

- CNN→TFT using PyTorch Lightning, mixed precision if GPU available.
- Loss: e.g., sum of MSEs across horizons. Monitor RMSE/MAE/IC/hit‐rate.
- Early stopping based on validation set; dropout regularization; seed fixed for reproducibility.

**7. (Optional) Residual XGBoost:**

- Take residuals ( y - \hat y\_{TFT} ); train XGBoost regressor on current features (and/or TFT latent features).
- Evaluate uplift in RMSE/IC vs TFT alone.

**8. Backtest (Required):**

- Define a simple vectorised strategy using predicted 5-day return (or direction): long if above threshold, short/flat otherwise.
- Include transaction cost & slippage assumptions.
- Report CAGR, Sharpe, max draw-down, turnover; compare against SMA-benchmark (e.g., SMA-50) or buy‐and‐hold.

**9. Explainability (Optional but recommended):**

- Use SHAP on residual head to assess feature importances.
- Use TFT attention maps to extract per‐horizon, per‐feature/time weights; summarise drivers by regime (e.g. low VIX vs high VIX, valuation quartiles).
- Provide narrative of what features “matter” when forecast horizon = 10 days vs 1 day.

**10. Packaging (Required):**

- Provide fully reproducible environment: `requirements.txt`, deterministic seeds, one notebook (Colab or AWS) that trains/evaluates the model end‐to‐end.
- Code modularised (data ingestion, feature engineering, model training, backtest, explainability).

---

## 7. Validation Design (Guardrails)

- **Time‐ordered splits only**: no shuffling of time series; use walk‐forward splits with embargo to avoid leakage.
- **Leakage prevention**: fundamentals stamped at filing date +1; sentiment features lagged; VIX lagged.
- **Ablation experiments**:
   • Base model: 9 features only (no engineered variants, no residual XGBoost).
   • Then add engineered variants (MA distances, cross-overs) and/or residual XGBoost only if they improve out-of-sample IC/Sharpe.
- **Robustness checks**: Train model on first 7 years, test on last 3 years; swap anchors (train 5y/test2y) to verify stability of results.

---

## 8. Metrics

**Prediction metrics:**

- RMSE and MAE per horizon (1d, 5d, 10d).
- Spearman Rank‐Correlation (Information Coefficient, IC) between predicted & realized return.
- Hit‐rate (percentage of correct sign predictions) or Brier score if classification head enabled.

**Portfolio/backtest metrics:**

- CAGR (compound annual growth rate).
- Sharpe ratio (excess return / volatility).
- Maximum draw‐down (maxDD).
- Turnover (percentage of portfolio changed each period).
- Precision@K if using a top‐confidence threshold for signals.

---

## 9. Feasibility & Timeline (6-week, team of 3)

**Resources required:**

- GPU: Colab/Kaggle free or AWS g5.xlarge / Paperspace (paid) as needed.
- Libraries: PyTorch, PyTorch Lightning, PyTorch Forecasting, XGBoost, SHAP, yfinance, ta, pandas, numpy, matplotlib.

**Workplan by week:**

- **Week 1:** Data ingestion, cleaning, feature engineering (9 signals + sentiment pipeline integration) + baseline (e.g., ARIMA or simple linear).
- **Week 2:** Build CNN→TFT model, train initial version, validate multi‐horizon forecasts.
- **Week 3:** Add residual XGBoost head; begin explainability modules (SHAP, TFT attention maps).
- **Week 4:** Backtest strategy using 5-day predictions; compute portfolio metrics; perform ablation experiments (engineered features on/off).
- **Week 5:** Robustness tests (various train/test splits), refine hyperparameters, document results; integrate sentiment feature analysis.
- **Week 6:** Final polish: code cleanup, reproducible notebook + requirements.txt (or Docker container), write report (paper-style) covering methodology, experiments, ablations, results, limitations, ethics statement.

---

## 10. Risks & Mitigations

- **Over‐fitting risk**: Deep models may overfit noisy financial data. _Mitigations_: early stopping, dropout, regularisation, walk-forward CV, hold‐out period.
- **Leakage risk**: Use of future info inadvertently in features (fundamentals, VIX, sentiment). _Mitigation_: rigorous stamping/lagging protocols for fundamentals, VIX, sentiment features.
- **Data gaps / missing filings**: Some quarters may have delayed or missing fundamental data. _Mitigation_: forward‐fill fundamentals only until next report; do not extrapolate beyond.
- **Sentiment feature complexity**: Integrating a FinBERT pipeline may introduce delays or noise. _Mitigation_: treat sentiment features as optional or secondary; evaluate incremental value via ablation.
- **Operational complexity for multi‐horizon + backtest**: Many moving parts. _Mitigation_: stick to minimal viable model first (9 features, CNN→TFT) then layer in complexity.

---

## 11. Deliverables

1. **Code**: Full end‐to‐end scripts/modules for data ingestion, feature engineering (including sentiment integration), model training (CNN→TFT ± residual XGBoost), backtesting, and explainability.
2. **Reproducible run**: Notebook (Colab or AWS) + `requirements.txt` (or Dockerfile) that allows one‐click training/evaluation of the model.
3. **Report**: ~8-12 page document (or ~2-4 page supervisor summary plus appendix) covering: introduction, literature grounding, methodology, experiments, ablation results, backtest metrics, interpretability findings, limitations & ethics (with caution: not a “signal to trade” guarantee).

---

## 12. Minimal Environment Setup

```bash
pip install torch pytorch-lightning pytorch-forecasting torchmetrics \
            yfinance ta xgboost shap numpy pandas matplotlib
```

Add additional libraries (e.g., transformers for FinBERT) if you include the sentiment pipeline.

---

## 13. Use of Sentiment (FinBERT) Pipeline – Placeholder

- A separate pipeline will ingest textual sources: e.g., news articles (about AAPL), earnings call transcripts, company press releases.
- The pipeline uses FinBERT (or similar finance‐domain BERT) to produce numerical sentiment scores (e.g., positive/neutral/negative, magnitude, sentiment momentum).
- These sentiment outputs become additional features in the main model dataset (time‐aligned, lagged appropriately).
- In the feature engineering stage, you should evaluate: (a) raw sentiment score, (b) sentiment change (Δ), (c) sentiment volatility over rolling window.
- In ablation studies, evaluate the incremental predictive value of the sentiment features (with/without sentiment).
- Document the alignment, lagging, and treatment of sentiment features to avoid leakage (e.g., use publication date +1 business day).

---

## 14. References & Grounding

- Stefenon S. et al., “CNN-TFT explained by SHAP with multi-head attention” (2025) — hybrid CNN + TFT architecture for multivariate time series. ([arxiv.org][1])
- Tu T., “Bridging Short- and Long-Term Dependencies: A CNN-Transformer Hybrid for Financial Time Series Forecasting” (2025) — financial time series domain hybrid. ([arxiv.org][2])
- Ge Q., “Enhancing stock market forecasting: A hybrid model for…” (2025) — hybrid deep models in stock forecasting. ([sciencedirect.com][3])
- Tiwari D., “Attention-augmented hybrid CNN-LSTM model for social… cryptocurrency” (2025) — illustrates sentiment/attention in finance forecasting. ([Nature][4])
