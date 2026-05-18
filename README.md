# 📊 BTC Futures Trading Bot (Testnet) - Ledger Based System

A production-style cryptocurrency futures trading bot built for Binance Testnet environment.  
This system integrates **machine learning models, technical indicators, and a ledger-based PnL tracking system** to simulate professional trading workflows used in quant and fintech companies.

---

## 🏢 Project Background

This project was originally developed as part of a **company-level trading system prototype**, where a ledger-based architecture was used to track all trading activities, profit/loss (PnL), and signal executions in real-time.

All trades and calculations are stored in a structured **ledger system**, ensuring transparency, traceability, and auditability of trading performance.

---

## ⚙️ Core Features

### 📈 Trading Engine
- Binance Futures Testnet integration
- Real-time market data fetching
- Automated trade signal generation
- Entry, TP (Take Profit), SL (Stop Loss) logic

---

### 📊 Ledger-Based System (Key Feature)
- All trades stored in a centralized **ledger database**
- Real-time PnL calculation
- Trade history tracking
- Hit/Miss tracking for TP & SL
- Performance analytics per strategy

---

### 🤖 Machine Learning Models
- LSTM-based price prediction
- GRU-based forecasting model
- Linear regression model for trend estimation
- Prophet model for time-series prediction

---

### 📉 Technical Indicators
- EMA (Exponential Moving Average)
- Supertrend indicator
- Stochastic oscillator
- Hybrid indicator strategies
- Backtesting modules

---

### 📊 Backtesting System
- Strategy testing on historical data
- Performance evaluation
- Win/Loss ratio tracking
- Risk/reward analysis

---

## 🗂️ Project Structure



---

## 📊 Ledger System Explained

The ledger is the **core financial tracking engine** of this project.

It records:
- Entry price
- Exit price
- PnL (Profit/Loss)
- TP/SL hit status
- Strategy used
- Timestamp of trade

👉 This makes the system similar to **institutional trading logs** used in hedge funds.

---

## 🚀 How to Run

```bash
# Clone repository
git clone https://github.com/username/repo-name.git

# Install dependencies
pip install -r requirements.txt

# Run main bot
python main.py

📌 Key Highlights
📊 Ledger-based trading architecture (industry style)
🤖 Multiple AI/ML prediction models
📈 Technical + statistical hybrid strategy system
🔁 Fully modular structure (scalable design)
🧠 Research + production hybrid project
📸 Future Improvements
Live web dashboard (Flask/Django)
Real-time WebSocket integration
Portfolio risk management module
Auto strategy optimization (AI tuning)
Cloud deployment (AWS/GCP)

👨‍💻 Author

Sabi Ul Hassan
Python Developer | Machine Learning & Quant Trading Enthusiast

⚠️ Disclaimer

This project is developed for educational and research purposes using Binance Testnet only.
No real financial trading is performed.

⭐ Support

If you like this project, consider giving it a ⭐ on GitHub