# Gridzilla VIP PRO

Paper-trading bot for Binance.US. Runs a multi-signal confluence entry engine (EMC) with an 8-level cascade exit system and a Rich live TUI.

> **Paper trading only.** No real funds are touched. All positions and balance live in a local JSON state file.

---

## Requirements

- Python 3.10+
- Binance.US account (public REST + WebSocket, no API key required for paper mode)

```bash
pip install requests websocket-client rich
```

---

## Running

```bash
python gridzilla_vip_pro.py
```

Run in **Windows Terminal** or any real TTY — the Rich live display requires one. Stop with `Ctrl+C`.

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

## How It Works

### Entry — EMC Confluence Engine

Fires a LONG when **5 of 11 signals** agree:

| # | Signal |
|---|---|
| a | EMC: 3 of 4 indicator velocity vectors positive AND RSI < 55 |
| b | EMA(13) > EMA(33) |
| c | RSI < 42 and rising vs 3 bars ago |
| d | MACD bullish (line > signal, histogram > 0) |
| e | Stoch %K > %D crossover from below 35 |
| f | SuperTrend bullish, no flip |
| g | Volume Oscillator > +10% |
| h | 1h close > SMA(20) |
| i | 5m 3-candle higher lows |
| j | MACD histogram expanding and positive |
| k | 1m volume surge (WebSocket data, optional bonus) |

Pre-flight gates block entries regardless of signals: cooldown active, equity below 85% of peak, more than 6 positions already open.

### Exit — 8-Level Cascade

1. **Stoch partials** — K>76 → sell 35%, K>84 → 30%, K>91 → 25%, K>96 → 10%
2. **SuperTrend flip** → sell 20% of remaining
3. **VO zero-cross** (+3% → -3%) with Stoch>80 → sell 50% of remaining
4. **Chandelier exit** — 1h close below level → full exit
5. **RSI bearish divergence** — RSI falling while price rising and RSI>65 → full exit
6. **VO breakeven arm** — VO>10% arms BE stop, but only after price reaches +2.25% above entry
7. **ATR trailing stop** — clamped 0.8%–2.0% below session high
8. **Hard stop** — 4.5% below entry
9. **Breakeven stop** — exits at entry if armed and price retreats
10. **Stale exit** — loss>2% after 30m, or loss>0.5% after 60m

### TUI Layout

```
┌─────────────────── header: equity / PnL / WR / streak / uptime ───────────────────┐
│  RADAR TABLE (30 pairs)         │  SIGNAL CARDS (one per open position)            │
│  PAIR  PRICE  SUPP  RES  SCORE  │  #001 BTCUSDT  entry/price/PnL/stop/TP1-4       │
│  ...                            │  ...                                              │
├─────────────────────────────────┴──────────────────────────────────────────────────┤
│  LOG FEED (last 15 events)                        │  Melania (ASCII pole dancer)   │
└───────────────────────────────────────────────────┴────────────────────────────────┘
```

Signal cards show TP levels as dollar prices (+1R, +2R, +3.5R, +5.5R where 1R = entry × 4.5%).

---

## Configuration

All tuning in the `Config` class at the top of `gridzilla_vip_pro.py`:

| Key | Default | Effect |
|---|---|---|
| `alloc_base` | 0.18 | Base position size (18% of balance) |
| `alloc_heater` | 0.30 | Max size on win streak |
| `stop_loss_pct` | 0.045 | Hard stop distance + 1R unit |
| `max_positions` | 6 | Max concurrent open trades |
| `rsi_oversold` | 42 | Signal #c threshold |
| `drawdown_threshold` | 0.85 | Pause entries below 85% of peak equity |
| `partial_levels` | (76,35%)(84,30%)(91,25%)(96,10%) | Stoch K exit thresholds |
| `cooldown_light/medium/heavy` | 15m/30m/60m | Post-loss cooldown |
| `max_pairs` | 30 | Pairs tracked simultaneously |

Entry threshold is `sigs < 5` — lower it to trade more, raise it to trade less.
