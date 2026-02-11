# 🗺️ Delta Anti — Profitability Roadmap

A prioritized roadmap for improving the trading system's profitability, grouped by effort and impact.

---

## ⚡ Tier 1: Quick Wins (High Impact, Low Effort)

### 1. Retrain LSTM on More Symbols & Timeframes
The LSTM is only trained on 21 epochs with limited data. Training with more data significantly improves accuracy.

- [ ] Run `train_lstm.py` with 2–3 years of data, 50+ epochs
- [ ] Train separate models per symbol + timeframe

### 2. ✅ Tune ML Confidence Thresholds
~~Currently ML needs > 50 confidence to act. Sentiment needs |score| > 0.2. These are arbitrary.~~

**DONE** — `ML_VALIDATION_CONFIG` in `config.py`:
- ML confidence threshold raised from 50 → **60**
- Sentiment score threshold raised from 0.2 → **0.3**

### 3. Weight ML Models by Accuracy
Right now Lag-Llama and sentiment get fixed boosts regardless of historical accuracy. Track each model's hit rate and weight their influence dynamically.

- [ ] Track per-model win rate over rolling window
- [ ] Dynamically adjust `ml_confirm_boost` / `sentiment_confirm_boost` based on accuracy

---

## 🚀 Tier 2: Medium Effort, High Reward

### 4. ✅ Multi-Timeframe Confirmation
~~Currently only one timeframe is analyzed. Checking if higher TF trends agree before entering dramatically improves win rate.~~

**DONE** — `analysis/multi_timeframe.py`:
- 1h candle trend analysis (EMA 9/21/50 + ADX + RSI)
- Blocks trades against strong HTF trend (strength ≥ 50)
- Integrated into `_validate_with_ml()` as first filter

### 5. ✅ Integrate the DQN Agent
~~`ml/agents/dqn_trader.py` exists but is not connected to the signal pipeline.~~

**DONE** — DQN fully integrated:
- Signal validation layer: agrees = +10 boost, HOLD = -5, opposes = -15
- 50-dim state vector built from indicators
- Online learning: feeds trade outcomes back via experience replay
- Auto-saves model every 10 completed trades

### 6. ✅ Dynamic Position Sizing Based on Confluence
~~When ML + Sentiment + SMC + Technical all agree, use larger position sizes.~~

**DONE** — `DYNAMIC_SIZING_CONFIG` in `config.py`:
| Confidence | Multiplier | Effective Risk |
|-----------|------------|---------------|
| 80–85     | 1.0x       | 2%            |
| 86–90     | 1.5x       | 3%            |
| 91–95     | 2.0x       | 4%            |
| 96–100    | 2.5x       | 5%            |

### 7. Backtest Before Live
No backtesting engine exists. Building one to test signal generation against historical data would validate changes before risking real money.

- [ ] Build backtesting framework using historical candle data
- [ ] Replay signals through `generate_signal()` with historical DataFrames
- [ ] Track simulated PnL, win rate, max drawdown

---

## 💎 Tier 3: High Effort, Transformative

### 8. ✅ Ensemble Model Voting
~~Instead of Lag-Llama → LSTM fallback, run both in parallel and require consensus.~~

**DONE** — `_get_ml_prediction()` in `signals.py`:
- Both models run in parallel
- **Consensus**: average confidence + **20% bonus**
- **Split**: use stronger model with confidence penalty
- Single model available: use as-is (backward compatible)

### 9. Adaptive Strategy Selection
`range_strategy.py` and main momentum strategy serve different market conditions. Automatically switch between them based on ADX/volatility regime detection.

- [ ] Implement regime classifier (trending vs ranging vs volatile)
- [ ] Auto-select strategy based on detected regime
- [ ] Smooth transitions to avoid whipsawing between strategies

### 10. Order Flow / Orderbook Imbalance
WebSocket orderbook subscriptions exist but bid/ask imbalances aren't analyzed. Large buy walls or aggressive selling at key levels can confirm entries.

- [ ] Analyze orderbook depth for bid/ask imbalance ratios
- [ ] Detect large walls and aggressive market orders
- [ ] Use imbalance as additional signal confirmation layer

---

## Progress Summary

| # | Feature | Status |
|---|---------|--------|
| 1 | Retrain LSTM | ⬜ Todo |
| 2 | Tune ML Thresholds | ✅ Done |
| 3 | Weight Models by Accuracy | ⬜ Todo |
| 4 | Multi-Timeframe Confirmation | ✅ Done |
| 5 | DQN Agent Integration | ✅ Done |
| 6 | Dynamic Position Sizing | ✅ Done |
| 7 | Backtesting Engine | ⬜ Todo |
| 8 | Ensemble Model Voting | ✅ Done |
| 9 | Adaptive Strategy Selection | ⬜ Todo |
| 10 | Order Flow Analysis | ⬜ Todo |

**Completed: 5/10** — All Tier 2 items done, 1 of 3 Tier 1, 1 of 3 Tier 3.
