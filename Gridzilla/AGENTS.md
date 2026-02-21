# AGENTS.md

This file provides guidance to AI agents operating on the Gridzilla VIP PRO codebase.

---

## Build/Test Commands

### Syntax Validation
```bash
python -c "import ast; ast.parse(open('gridzilla_vip_pro.py',encoding='utf-8').read()); print('OK')"
```

### Dependencies
```bash
pip install requests websocket-client rich
```

### Running the Bot
```bash
python gridzilla_vip_pro.py
```
- Must run in a real terminal (PowerShell, Windows Terminal, etc.)
- Do NOT redirect stdout (> file.txt) - breaks the Rich Live display

### Reset State to Clean Slate
```python
import json
from datetime import date
json.dump({
    "balance": "1000.0", "pnl": "0.0", "wins": 0, "losses": 0, "streak": 0,
    "signal_counter": 0, "gross_wins": "0", "gross_losses": "0", "total_r_sum": "0",
    "daily_wins": 0, "daily_trades": 0, "daily_date": str(date.today()),
    "peak_equity": "1000.0", "cooldowns": {}, "positions": {}
}, open('data/gridzilla_vip_pro_state.json', 'w'))
```

---

## Code Style Guidelines

### Decimal Arithmetic (MANDATORY)
- All financial math MUST use `Decimal` - never `float`
- Always use string notation: `Decimal('0.18')` not `Decimal(0.18)`
- When mixing Decimal with float, wrap float in `str()`: `Decimal(str(brain.size_multiplier))`

### Decimal Comparisons
- NEVER use raw integers: `> 0`, `== 0`
- ALWAYS use Decimal: `> Decimal('0')`, `>= Decimal('0')`

### Dictionary Access
- NEVER use `dict['key']` - use `.get()` with defaults
- CORRECT: `grid.get('rsi', Decimal('50'))`
- WRONG: `grid['rsi']`

### Error Handling
- NEVER use silent `pass` in except blocks
- ALWAYS log errors: `except Exception as e: add_log(f"error: {e}", "red")`

### Thread Safety
- All access to shared `state` must go through `state.lock` (RLock)
- Use `with state.lock:` for any read/write to positions, prices, etc.

### Comments
- NONE unless explicitly requested by user

### Type Ignore Comments
- Use `# type: ignore` for Decimal/float math warnings from LSP
- Example: `volatility / sma  # type: ignore`

### Imports
- Grouped: stdlib, external, local
- No unused imports

---

## Architecture Overview

Single-file bot (~2000 lines). No external config files.

### Core Classes
- **`Config`**: All constants at top of file (lines 17-90)
- **`AdaptiveBrain`**: Learning system for signal weights/thresholds
- **`BotState`**: Singleton holding live prices, positions, cooldowns

### Threading Model (5 daemon threads)
| Thread | Role | Interval |
|--------|------|----------|
| `trade_executioner` | Entry/exit logic | 1s |
| `market_pulse` | Fetch 1h+5m klines, compute indicators | ~60s |
| `pair_scanner` | Re-rank pairs by volume, blacklist stale | 3600s |
| `ws_loop` | WebSocket for live prices | continuous |
| `heartbeat` | Save state to JSON | 30s |

---

## Key Features (Not in CLAUDE.md)

### Slippage Simulation
- `CONFIG.slippage_pct = Decimal('0.0008')` (0.08%)
- Applied to entry and exit prices for paper-trade realism

### 4h Macro Trend Filter
- Fetches 4h klines, calculates EMA(20) vs EMA(50)
- Entry blocked if `macro_bullish = False`

### Bollinger Band Width (BBW) Squeeze
- Calculates standard deviation-based BBW
- BBW expansion detection for "institutional squeeze" entry filter

### Golden Pocket Fibonacci
- Fibonacci 0.5 and 0.618 retracement levels
- Used for re-entry zone display

### Institutional Squeeze Gate
- Entry only fires when BBW > BBW_avg (volatility expanding)
- Filters out "squeeze" or contracting volatility

### Confidence Scoring (Grade A)
- Scale: 60% base + scaled momentum (up to 100%)
- Labels: STANDARD (60-74), HIGH (75-89), ULTRA (90-100)
- Displayed in entry log and trade cards

### Step-Up Stop Escalation
- After TP1 hit: stop moves to breakeven
- After TP2 hit: stop moves to TP1 level (+1R)
- After TP3 hit: stop moves to TP2 level (+2R)
- After TP4 hit: MOON MODE - ATR trailing only

### Moon Mode
- Activated after TP4 partial exit
- All fixed targets removed, only ATR trailing stop remains

### Dynamic Grid Advisor (3 Panels)
- Rotating panels: L1 SCALP, MEME VOL, INFRA TREND
- Each shows: price, 10% range, 50 grid levels, profit per grid

### Headless Mode
- Auto-detects non-TTY environment
- Runs without TUI, just maintains state

---

## Data Directory Structure

```
data/
├── gridzilla_vip_pro_state.json   # Bot state (balance, positions)
├── gridzilla_brain.json          # Brain learning data
├── gridzilla_vip_pro.log         # Log file
├── bot_stderr.txt               # stderr capture
└── bot_stdout.txt              # stdout capture
```

---

## Known Gotchas

1. **BBW sqrt()**: The Bollinger Band calculation uses `bb_var.sqrt()` which fails on Decimal. Consider using float conversion.

2. **best_win not persisted**: `state.best_win` is tracked but not saved to JSON.

3. **Socket lock**: Port 21487 prevents multiple instances. If port stuck after force-kill, wait 60s for release.

4. **VO-BE instant exit loop**: Breakeven only activates when `p['high'] > orig_entry` - prevents instant BE fire on entry.

5. **Chandelier pre-filter**: Entry skipped if price already below Chandelier level - prevents instant exit.

---

## Files Reference

- **Full Architecture**: See `CLAUDE.md`
- **Development Conventions**: See `GEMINI.md`
- **User Documentation**: See `README.md`
