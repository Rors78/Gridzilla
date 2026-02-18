# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running

```bash
# Primary bot — run in Windows Terminal (PowerShell). Rich TUI requires a real TTY.
python D:/gridzilla_vip_pro.py

# Stop with Ctrl+C. Do NOT redirect stdout (> file.txt) — kills the live display.
# Never run multiple instances simultaneously — they corrupt the shared state JSON.
```

State persists to `gridzilla_vip_pro_state.json`. Reset to clean slate:
```python
import json
json.dump({"balance":"1000.0","pnl":"0.0","wins":0,"losses":0,"streak":0,
  "signal_counter":0,"gross_wins":"0","gross_losses":"0","total_r_sum":"0",
  "daily_wins":0,"daily_trades":0,"daily_date":"2026-02-18",
  "peak_equity":"1000.0","cooldowns":{},"positions":{}},
  open("D:/gridzilla_vip_pro_state.json","w"))
```

Syntax check before running:
```bash
python -c "import ast; ast.parse(open('D:/gridzilla_vip_pro.py',encoding='utf-8').read()); print('OK')"
```

---

## Architecture — `gridzilla_vip_pro.py`

Single-file bot (~1,350 lines). No external config files.

### Core objects

- **`Config`** (class, top of file) — all constants. Edit directly to tune: `alloc_base`, `stop_loss_pct`, `partial_levels`, `rsi_oversold`, `drawdown_threshold`, etc.
- **`BotState`** — singleton `state`. Holds live prices, positions, grid data, cooldowns, WS ref, lock. Auto-loads JSON on init, saves on every trade + heartbeat every 30s.

### Threading model

Five daemon threads, all wrapped in `try/except` so a crash logs the error and sleeps rather than dying silently:

| Thread | Role | Interval |
|---|---|---|
| `trade_executioner` | Checks all pairs for entries/exits | 1s |
| `market_pulse` | Fetches 1h+5m klines, recomputes indicators | ~60s |
| `pair_scanner` | Re-ranks pairs by 24h volume, reconnects WS | 3600s |
| `ws_loop` | Binance.US WebSocket (miniTicker + kline_1m) | continuous |
| `heartbeat` | Saves state JSON | 30s |

All access to `state` goes through `state.lock` (RLock). `trade_executioner` holds the lock for its full cycle.

### Indicator pipeline (`get_market_context`)

Called by `market_pulse` for each symbol. Fetches 1h klines (75 bars) + 5m klines (40 bars) via REST. Computes:
- RSI(14) — Wilder smoothing
- EMA(13), EMA(33)
- MACD(12,26,9) — line, signal, histogram
- Stochastic(14,3,3) — %K and %D with prev-bar values for crossover detection
- SuperTrend(10, 3.0) — with prev-bar direction for flip detection
- Volume Oscillator (EMA5 vs EMA14 of volume)
- Chandelier Exit(14, 2.75) — uses confirmed 1h close, NOT live tick (prevents wick false exits)
- ATR(14)
- SMA(20) for trend bias
- 5m higher-lows pattern

Sentinel values: on first bar, `stoch_prev_k/d` default to `Decimal('50')`, `supertrend_prev_dir` defaults to `False`. Never default to the current-bar value (creates dead zones).

### Entry logic — EMC Confluence Engine

Requires **5 of 11 signals** to fire a LONG:

| # | Signal | Notes |
|---|---|---|
| a | EMC: 3 of 4 velocity vectors positive AND RSI < 55 | Claude-original; rsi_vel, hist_vel, stoch_vel, vo_vel |
| b | EMA(13) > EMA(33) | |
| c | RSI < 42 AND rising vs 3 bars ago | Oversold bounce |
| d | MACD bullish (line > signal AND histogram > 0) | |
| e | Stoch K > D crossover from below 35 | prev-bar crossover check |
| f | SuperTrend bullish, no flip | |
| g | VO > +10% | |
| h | 1h close > SMA(20) | |
| i | 5m 3-candle higher lows | |
| j | MACD histogram expanding and positive | |
| k | 1m volume surge (WS kline data, optional bonus) | |

Pre-flight gates (not signals): cooldown active, drawdown below 85% of peak equity.

### Exit cascade (in order, inside `trade_executioner`)

1. **Stoch partials** — when in profit: K>76→sell 35%, K>84→30%, K>91→25%, K>96→10%
2. **SuperTrend flip** (bearish) → sell 20% of remaining
3. **VO zero-cross** (+3% → -3%) with Stoch>80 → sell 50% of remaining
4. **Chandelier** — 1h close below level → full exit
5. **RSI bearish divergence** — RSI falling while price rising and RSI>65 → full exit
6. **VO-BE activation** — VO>10% activates breakeven stop, BUT only if `p['high'] > p['orig_entry']` (critical: VO is also an entry signal; without this guard, every VO entry self-exits instantly)
7. **ATR trailing stop** — clamped 0.8%–2.0% below session high; only fires if `price > orig_entry`
8. **Hard stop** — 4.5% below entry
9. **Breakeven stop** — only fires if `p['high'] > p['orig_entry']` (same guard as activation)
10. **Stale graduated exit** — loss>2% after 30m, or loss>0.5% after 60m

### Display (Rich Live)

Layout: header (1 row) / [radar table | signal cards] / [logs | Melania dancer]

- **Radar**: 30 pairs, SCORE column shows `GO 7/11` / `4/11` (yellow) / `2/11` (dim) + EMC sub-score
- **Signal cards**: one panel per open position — entry, current price, PnL, stop, ATR trail, TP1-4 as dollar prices (R-multiple targets: +1R, +2R, +3.5R, +5.5R)
- **`_fmt(v)`**: auto-scales decimal precision for tiny-price coins (SHIB, PEPE → 8dp)

---

## Known Gotchas

**Log file encoding**: `_log_file` must be opened with `encoding='utf-8'`. Windows cp1252 cannot write emoji characters → `UnicodeEncodeError` inside `add_log()` → `trade_executioner` thread dies silently → zero trades/exits forever.

**VO-BE instant exit loop**: VO > threshold is entry signal #g. If CASCADE 5 activates BE immediately at entry (before price moves), and price == entry, BE fires instantly. Both the activation and the exit condition require `p['high'] > p['orig_entry']`.

**Silent thread death**: Daemon threads that raise an uncaught exception die permanently with no restart. Always wrap the main loop body in `try/except Exception as e: add_log(...)`.

**Multiple instances**: Two bots writing the same state JSON interleave saves and produce duplicate signal IDs and corrupted balance. Kill all `python.exe` processes before restarting.

**Chandelier on live tick**: Use `grid.get('price_cur', price)` (last confirmed 1h close) for the Chandelier trigger check. Using the live tick price causes false exits on intracandle wicks.

---

## Configuration Knobs

All in the `Config` class:

| Key | Default | Effect |
|---|---|---|
| `alloc_base` | 0.18 | Base position size (18% of balance) |
| `alloc_heater` | 0.30 | Max size on streak |
| `stop_loss_pct` | 0.045 | Hard stop + 1R unit for TP price targets |
| `rsi_oversold` | 42 | Signal #c threshold |
| `drawdown_threshold` | 0.85 | Block entries if equity < 85% of peak |
| `partial_levels` | (76,35%),(84,30%),(91,25%),(96,10%) | Stoch K thresholds for partials |
| `cooldown_light/medium/heavy` | 15m/30m/60m | Post-loss cooldown by severity |
| `max_pairs` | 30 | Pairs tracked simultaneously |

Entry fires at **6/11 signals** (`if sigs < 6: continue`). Adjust this threshold to control trade frequency.

