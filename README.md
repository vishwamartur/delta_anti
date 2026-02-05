# Delta Anti - AI-Powered Trading System

A **production-ready** quantitative trading system for Delta Exchange with **AI/ML predictions**, **adaptive learning**, and **200x leverage support**.

## ✨ Features

### 🤖 AI/ML Trading Intelligence
- **Lag-Llama Forecaster**: Time series foundation model for price predictions
- **LSTM Predictor**: Bidirectional LSTM with attention mechanism
- **FinBERT Sentiment**: Financial news sentiment analysis (local & cloud)
- **Hugging Face Inference API**: Cloud-based AI models for sentiment and classification
- **DQN Trading Agent**: Deep Q-Network reinforcement learning for optimal actions
- **Adaptive Learning**: Learns from trade history to improve decisions

### 📈 Advanced Trading Strategies
- **Momentum Trading**: 200x leverage with risk management
- **Low Volatility Strategy**: Range trading for sideways markets (mean reversion)
- **Limit Orders**: Uses maker orders (0.02% fees vs 0.05% taker)
- **Auto-Topup**: Prevents liquidation by adding margin automatically
- **Trailing Stops**: Activates after 0.5% profit to lock in gains
- **Trade Cooldown**: 60-second minimum between trades

### 📊 Technical Analysis
- **Indicators**: RSI, MACD, Bollinger Bands, ATR, ADX, EMA/SMA
- **Market Regime Detection**: Auto-detects trending vs ranging markets
- **Signal Validation**: ML confirms technical signals before entry
- **Confidence Scores**: 0-100% confidence on every signal

### 🖥️ Real-Time Dashboard
- **Rich Terminal UI**: Beautiful console dashboard with live updates
- **Market Prices**: Real-time price feeds and changes
- **Signal Monitoring**: Active signals with confidence levels
- **Position Tracking**: Open positions with P&L display
- **System Messages**: Trade alerts and status updates

### 🛡️ Pre-Trade Market Analysis (NEW)
- **Market Regime Detection**: AI-driven identification of trending, ranging, or volatile markets
- **Noise Filtering**: Advanced noise cancellation (volatility spikes, false breakouts, whipsaws)
- **Structure Analysis**: Automatic support/resistance detection and structure quality scoring
- **Trade Quality Score**: Composite 0-100 score for every trade opportunity
- **Risk-Reward Validation**: Ensures optimal risk/reward ratios before entry

### 🌐 REST API Server
- **FastAPI Server**: Production-ready endpoints
- **WebSocket Streaming**: Real-time predictions
- **Webhooks**: TradingView & Telegram integration

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/vishwamartur/delta_anti.git
cd delta_anti

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API credentials
cp .env.example .env
# Edit .env with your Delta Exchange API keys

# 4. (Optional) Install Lag-Llama for AI forecasting
python scripts/install_lag_llama.py

# 5. Start trading
python run_system.py
```

---

## ⚙️ Configuration

### Environment Variables (`.env`)

```env
# Delta Exchange API
DELTA_API_KEY=your_api_key
DELTA_API_SECRET=your_api_secret
DELTA_REST_URL=https://api.india.delta.exchange
DELTA_WS_URL=wss://socket.india.delta.exchange

# Hugging Face (for cloud AI inference)
HF_TOKEN=your_huggingface_token

# Trading
TRADING_SYMBOLS=BTCUSD,ETHUSD
DEFAULT_TIMEFRAME=5m
RISK_PER_TRADE=0.10          # 10% risk per trade
RISK_AMOUNT_USD=300          # Fixed $300 risk per trade
DEFAULT_LEVERAGE=200
AUTO_TOPUP=true
AUTO_EXECUTION=true

# Trade Frequency
MIN_TRADE_INTERVAL=60        # Seconds between trades
```

### Key Settings Explained

| Setting | Default | Description |
|---------|---------|-------------|
| `RISK_PER_TRADE` | 0.10 | 10% of balance risked per trade |
| `RISK_AMOUNT_USD` | 300 | Fixed USD risk per trade |
| `DEFAULT_LEVERAGE` | 200 | Leverage multiplier |
| `AUTO_TOPUP` | true | Adds margin to prevent liquidation |
| `MIN_TRADE_INTERVAL` | 60 | Cooldown between trades (seconds) |
| `HF_TOKEN` | - | Hugging Face API token for cloud inference |
| `MARKET_ANALYSIS_CONFIG` | Enabled | Pre-trade validation settings |

---

## 🤖 AI/ML Models

### 1. Lag-Llama (Foundation Model)
Time series transformer trained on 100+ datasets.

```bash
# Install (requires GPU, 4GB+ VRAM)
python scripts/install_lag_llama.py
```

**Logs:**
```
[LAG-LLAMA] BTCUSD: bullish (72%, +1.25%)
```

### 2. LSTM Predictor
Bidirectional LSTM with attention for price forecasting.

### 3. FinBERT Sentiment
Analyzes crypto news sentiment before trades.

```
[SENTIMENT] BTCUSD: bullish (score: 0.45)
```

### 4. Hugging Face Inference API
Cloud-based AI models when local GPU unavailable.

```bash
# Set your HF token in .env
HF_TOKEN=hf_your_token_here
```

**Features:**
- FinBERT sentiment via API
- Zero-shot market regime classification
- News summarization

### 5. DQN Trading Agent
Deep Q-Network for reinforcement learning-based trading.

**Features:**
- State: 50 features (indicators + price data)
- Actions: Buy, Sell, Hold
- Experience replay buffer
- Epsilon-greedy exploration

### Model Fallback Chain
```
Lag-Llama → LSTM → HF Inference → Momentum → Technical Only
```

---

## 📁 Project Structure

```
delta_anti/
├── run_system.py              # Main entry point
├── main.py                    # Alternative entry with UI
├── config.py                  # Configuration
├── .env                       # API credentials
│
├── api/
│   ├── delta_rest.py          # REST API client
│   ├── delta_websocket.py     # WebSocket client
│   └── server/                # FastAPI server
│
├── ml/                        # Machine Learning
│   ├── hf_inference.py            # Hugging Face Inference API client
│   ├── models/
│   │   ├── lstm_predictor.py      # LSTM model
│   │   └── lag_llama_predictor.py # Lag-Llama model
│   ├── agents/
│   │   └── dqn_trader.py          # DQN reinforcement learning
│   ├── features/
│   │   └── feature_engineer.py    # 100+ features
│   └── sentiment/
│       └── market_sentiment.py    # FinBERT sentiment
│
├── analysis/
│   ├── indicators.py          # Technical indicators
│   └── signals.py             # Signal generation + ML validation
│
├── strategy/
│   ├── advanced_trade_manager.py  # Trade execution
│   ├── range_strategy.py          # Low volatility strategy
│   └── trade_analyzer.py          # Adaptive learning
│
├── ui/
│   └── dashboard.py           # Rich terminal dashboard
│
├── scripts/
│   └── install_lag_llama.py   # ML setup script
│
└── data/
    └── trades.json            # Trade history
```

---

## 📉 Low Volatility Strategy

Automatically switches to range trading when markets are sideways.

### How It Works
1. **Regime Detection**: ADX < 25 indicates ranging market
2. **Mean Reversion**: Buy at lower BB, sell at upper BB
3. **Tight Targets**: 0.3% take profit, 0.4% stop loss
4. **Scalping Mode**: Quick trades for small, consistent gains

```
[RANGE] BTCUSD: LONG @ $78,450 (BB position: 0.12)
  Regime: ranging | ADX: 18.5
  TP: $78,685 (0.3%) | SL: $78,136 (0.4%)
```

---

## 📊 Trading Dashboard

Rich terminal dashboard using the `rich` library:

```
╔════════════════════════════════════════════════════════════╗
║         📈 DELTA ANTI TRADING SYSTEM 📈              ║
╠════════════════════════════════════════════════════════════╣
║ MARKET PRICES                                      ║
║ BTCUSD  $78,500.00  ▲+1.25%   ETHUSD  $2,150.00    ║
╠════════════════════════════════════════════════════════════╣
║ SIGNALS          | POSITIONS                        ║
║ BTCUSD: LONG 72% | BTCUSD SHORT @ $78,600 [-0.12%]  ║
║ ETHUSD: HOLD 45% | ---                              ║
╚════════════════════════════════════════════════════════════╝
```

### Trade Entry Alerts
```
============================================================
[LIVE ENTRY] BTCUSD SHORT @ $78,500.00
  Order ID: 12345678
----------------------------------------
  Size:      1 contracts
  Notional:  $78,500.00
  Stop Loss: $79,128.00 (0.80% away)
  Take Profit: $77,558.00 (1.20% away)
  Risk/Reward: 1.5x
  Est. Fees:   $31.40 (round-trip @ 0.04%)
============================================================
```

---

## 🔬 Pre-Trade Market Analysis

The system performs a comprehensive health check before every trade:

### 1. Market Analyzer
Evaluates 4 key dimensions to generate a **Trade Quality Score (0-100)**:
- **Trend Confidence**: Multi-indicator agreement (EMA, MACD, RSI, ADX)
- **Structure Quality**: Clarity of support/resistance levels
- **Risk/Reward**: Potential profit vs. risk logic
- **Regime Alignment**: Is the strategy suitable for current conditions?

### 2. Smart Noise Filter
Prevents entering trades during chaotic market conditions by detecting:
- **Volatility Spikes**: Abnormal ATR expansion
- **False Breakouts**: Price failing to hold new levels
- **Volume Anomalies**: Suspicious volume without price movement
- **Choppy Action**: High wick-to-body ratio candles

```
[MARKET] BTCUSD: trending_up (Confidence: 85%)
  Structure: Strong Support @ $78,200
  Quality Score: 92/100
  Action: GO (Long)
```

---

## 💰 Fee Optimization

Uses **limit orders** to reduce fees:

| Fee Type | Market Order | Limit Order | Savings |
|----------|--------------|-------------|---------|
| Entry | 0.05% | 0.02% | 60% |
| Exit | 0.05% | 0.02% | 60% |
| **Round-trip** | 0.10% | **0.04%** | **60%** |

---

## 🛡️ Risk Management

1. **Daily Loss Limit**: 50% (configurable)
2. **Max Drawdown**: 70%
3. **Auto-Topup**: Uses wallet balance to prevent liquidation
4. **Trailing Stop**: Activates after 0.5% profit
5. **Trade Cooldown**: Prevents overtrading

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/trades/active` | GET | Get open trades |
| `/api/trades/stats` | GET | Get trading statistics |
| `/api/v2/predict` | POST | ML price predictions |
| `/api/v2/signals/{symbol}` | GET | Trading signals |
| `/api/v2/sentiment/{symbol}` | GET | Sentiment analysis |

**API Docs**: `http://localhost:8000/docs`

---

## 📈 Performance Tracking

Trade history saved to `data/trades.json`:

```json
{
  "stats": {
    "total_trades": 27,
    "win_rate": 40.0,
    "total_pnl": 125.50,
    "account_balance": 505.50
  }
}
```

---

## 🔧 Troubleshooting

### "Daily loss limit reached"
```bash
# Reset by restarting the bot
python run_system.py
```

### GPU not detected for Lag-Llama
```bash
# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### High fees
- Ensure `order_type` is `"maker"` in config
- Check that limit orders are being placed (see logs)

---

## 📄 License

MIT

---

## 🔗 Links

- [Delta Exchange API Docs](https://docs.delta.exchange/)
- [Lag-Llama Paper](https://arxiv.org/abs/2310.08278)
- [FinBERT](https://huggingface.co/ProsusAI/finbert)
