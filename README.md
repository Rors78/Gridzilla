# Gridzilla VIP PRO

Paper-trading bot for Binance.US. Runs a multi-signal confluence entry engine (EMC) with an 8-level cascade exit system and a Rich live TUI.

> **Paper trading only.** No real funds are touched. All positions and balance live in a local JSON state file.

---

## Requirements

- Python 3.10+
- Binance.US account (public REST + WebSocket, no API key required for paper mode)

**Windows:**
```powershell
pip install requests websocket-client rich
```

**Linux/macOS:**
```bash
pip3 install requests websocket-client rich
```

---

## Running

**Windows:**
```powershell
python gridzilla_vip_pro.py
```

**Linux/macOS:**
```bash
python3 gridzilla_vip_pro.py
```

Run in a real TTY (Windows Terminal, GNOME Terminal, etc.) — the Rich live display requires one. Stop with `Ctrl+C`.

The bot prevents duplicate instances via a PID lock file (`gridzilla_vip_pro.lock`). If a second launch is attempted while one is already running, it exits with an error message.

---

## State

All state persists to `gridzilla_vip_pro_state.json` — balance, open positions, win/loss record, cooldowns. The file is saved on every trade and every 30 seconds by a heartbeat thread.

**Reset to clean slate:**
```python
import json
json.dump({"balance":"1000.0","pnl":"0.0","wins":0,"losses":0,"streak":0,
  "signal_counter":0,"gross_wins":"0","gross_losses":"0","total_r_sum":"0",
  "daily_wins":0,"daily_trades":0,"daily_date":"2026-02-18",
  "peak_equity":"1000.0","cooldowns":{},"positions":{}},
  open("gridzilla_vip_pro_state.json","w"))
```

---

### TUI Layout

```
┌─────────────────── header: equity / PnL / WR / streak / uptime ───────────────────┐
│  RADAR TABLE (30 pairs)         │  SIGNAL CARDS (one per open position)            │
│  PAIR  PRICE  SUPP  RES  SCORE  │  #001 BTCUSDT  entry/price/PnL/stop/TP1-4       │
│  ...                            │  ...                                              │
├─────────────────────────────────┴──────────────────────────────────────────────────┤
│  LOG FEED (last 15 events)                        │  Melania (ASCII dancer, 24 moves) │
└───────────────────────────────────────────────────┴────────────────────────────────┘
```

Signal cards show TP levels as dollar prices (+1R, +2R, +3.5R, +5.5R where 1R = entry × 4.5%).

---
