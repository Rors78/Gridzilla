# GEMINI.md

## Project Overview
Gridzilla VIP PRO is a high-performance, single-file paper-trading bot designed for **Binance.US**. It uses a multi-threaded architecture to monitor market data, calculate complex indicators, and execute simulated trades based on an 11-signal confluence engine. The bot features a rich terminal-based dashboard (TUI) for real-time monitoring of PnL, open positions, and market "radar."

### Core Technologies
- **Language:** Python 3.10+
- **Data Source:** Binance.US Public REST API & WebSockets
- **UI Framework:** [Rich](https://github.com/Textualize/rich) for terminal dashboard
- **Concurrency:** Multi-threaded (5 daemon threads for execution, market pulse, pair scanning, WebSockets, and heartbeats)
- **Math:** `Decimal` for high-precision financial calculations

### Architecture
- **`gridzilla_vip_pro.py`**: The main script containing all logic, configuration, and TUI code (~1,800 lines).
- **EMC Confluence Engine**: Requires 6 of 11 technical signals (RSI, EMA, MACD, Stochastic, SuperTrend, Volume Oscillator, Chandelier Exit, etc.) to trigger an entry.
- **8-Level Exit Cascade**: Manages exits through partial take-profits, trailing stops, and breakeven protections.
- **State Management**: Persists to `gridzilla_vip_pro_state.json`.

---

## Building and Running

### Prerequisites
- Python 3.10 or newer.
- Dependencies: `requests`, `websocket-client`, `rich`.

### Installation
```powershell
pip install requests websocket-client rich
```

### Running the Bot
```powershell
# Must be run in a real terminal (PowerShell, Windows Terminal, etc.)
python gridzilla_vip_pro.py
```

### Testing / Validation
- **Syntax Check:** `python -c "import ast; ast.parse(open('gridzilla_vip_pro.py',encoding='utf-8').read()); print('OK')"`
- **Paper Trading:** The bot is inherently "safe" as it does not use real API keys or funds.

---

## Development Conventions

### Coding Style
- **Single-File Logic:** Keep the core bot logic within `gridzilla_vip_pro.py` for ease of deployment.
- **Configuration:** All settings (risk, indicators, cooldowns) are defined in the `Config` class at the top of the file. Edit this class directly to tune the bot.
- **Precision:** Always use `Decimal` for prices, quantities, and indicators to avoid floating-point errors.
- **Thread Safety:** Use `state.lock` (an RLock) when accessing or modifying shared state in `BotState`.

### Operational Guidelines
- **TUI Compatibility:** Do not use `print()` or redirect `stdout`. Use `add_log()` to push messages to the dashboard log feed.
- **Thread Stability:** All daemon threads must be wrapped in `try/except` blocks to prevent silent death. Errors should be logged to `add_log()` or `gridzilla_stderr.txt`.
- **State Persistence:** Ensure `state.save()` is called after important events (trade entry/exit) and periodically via the heartbeat thread.
- **Encoding:** Always open files (state, logs) with `encoding='utf-8'` to avoid `UnicodeEncodeError` on Windows.

### Signal Engine (EMC)
- Entry signals are calculated in `get_market_context`.
- New signals should be added to the 11-signal list in the `trade_executioner` loop.
- The threshold for entry is typically `sigs >= 6`.
