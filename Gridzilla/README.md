<h1 align="center">Gridzilla VIP PRO</h1>

<p align="center">
  A paper-trading bot for <strong>Binance.US</strong> with a live terminal dashboard.<br>
  No real money. No API key. Just watch it trade.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Exchange-Binance.US-yellow?logo=binance&logoColor=white" alt="Binance.US">
  <img src="https://img.shields.io/badge/Mode-Paper%20Trading%20Only-green" alt="Paper Trading">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="MIT License">
</p>

---

## What Is This?

Gridzilla VIP PRO watches the top 30 crypto pairs on Binance.US in real time.
It runs a multi-signal entry engine that requires **6 of 11 technical conditions** to align before placing a simulated trade.
Exits are managed by an 8-level cascade system (partial take-profits, trailing stops, breakeven protection, and more).

Everything happens on paper — your real funds are never touched.

---

## What You Need

Before you start, make sure you have:

<table>
  <tr>
    <th>Requirement</th>
    <th>Minimum Version</th>
    <th>Where to Get It</th>
  </tr>
  <tr>
    <td><strong>Python</strong></td>
    <td>3.10 or newer</td>
    <td><a href="https://www.python.org/downloads/">python.org/downloads</a></td>
  </tr>
  <tr>
    <td><strong>Internet connection</strong></td>
    <td>—</td>
    <td>Required to connect to Binance.US market data</td>
  </tr>
  <tr>
    <td><strong>A terminal</strong></td>
    <td>Any real terminal window</td>
    <td>Windows Terminal, PowerShell, GNOME Terminal, etc.</td>
  </tr>
</table>

> **No Binance.US account or API key is needed.** The bot reads public market data only.

---

## Installation

<details>
<summary><strong>🪟 Windows — Step-by-Step</strong></summary>

<br>

**Step 1 — Check your Python version**

Open **Windows Terminal** or **PowerShell** and type:

```powershell
python --version
```

You should see something like `Python 3.11.x`. If you see `3.9.x` or lower, or get an error, download Python from [python.org/downloads](https://www.python.org/downloads/) and check **"Add Python to PATH"** during install.

---

**Step 2 — Download the bot**

Option A — If you have Git installed:

```powershell
git clone https://github.com/Rors78/Gridzilla.git
cd Gridzilla
```

Option B — No Git: Click the green **Code** button at the top of this page → **Download ZIP** → extract the folder somewhere easy to find (e.g. `C:\Users\YourName\Gridzilla`).

---

**Step 3 — Install the required libraries**

```powershell
pip install requests websocket-client rich
```

Wait for it to finish. You should see `Successfully installed ...` at the end.

---

**Step 4 — Done.** Jump to [Running the Bot](#running-the-bot) below.

</details>

---

<details>
<summary><strong>🐧 Linux — Step-by-Step</strong></summary>

<br>

**Step 1 — Check your Python version**

Open a terminal and type:

```bash
python3 --version
```

You need **3.10 or newer**. If your version is too old, update it:

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install python3.11 python3.11-pip -y

# Fedora / RHEL
sudo dnf install python3.11 -y
```

---

**Step 2 — Download the bot**

```bash
git clone https://github.com/Rors78/Gridzilla.git
cd Gridzilla
```

No Git? Download with curl:

```bash
curl -L https://github.com/Rors78/Gridzilla/archive/refs/heads/master.zip -o gridzilla.zip
unzip gridzilla.zip
cd Gridzilla-master
```

---

**Step 3 — Install the required libraries**

```bash
pip3 install requests websocket-client rich
```

If you get a permissions error, add `--user`:

```bash
pip3 install --user requests websocket-client rich
```

---

**Step 4 — Done.** Jump to [Running the Bot](#running-the-bot) below.

</details>

---

## Running the Bot

> **Important:** Always run the bot inside a real terminal window (Windows Terminal, PowerShell, GNOME Terminal, iTerm2, etc.).
> Do **not** redirect output to a file (`> output.txt`) — this breaks the live display.

<details>
<summary><strong>🪟 Windows</strong></summary>

<br>

Open **Windows Terminal** or **PowerShell**, navigate to the folder where you put the bot, then run:

```powershell
python gridzilla_vip_pro.py
```

Example (if you cloned to your Downloads folder):

```powershell
cd C:\Users\YourName\Downloads\Gridzilla
python gridzilla_vip_pro.py
```

</details>

<details>
<summary><strong>🐧 Linux</strong></summary>

<br>

Open a terminal, navigate to the bot folder, then run:

```bash
python3 gridzilla_vip_pro.py
```

Example:

```bash
cd ~/Gridzilla
python3 gridzilla_vip_pro.py
```

</details>

<br>

The bot will spend about 30–60 seconds loading pair data before the live dashboard appears. This is normal.

**To stop the bot:** Press <kbd>Ctrl</kbd>+<kbd>C</kbd>.

> **Already running?** The bot uses a socket lock on port 21487 to prevent two copies from running at the same time. If you try to start a second instance, it will print `[ERROR] Gridzilla is already running` and exit safely.

---

## Understanding the Screen

Once running, you will see a four-panel live dashboard:

```
┌─────────── HEADER: balance · PnL · win rate · streak · uptime ───────────┐
│                                                                            │
│  RADAR (30 pairs)                  │  OPEN TRADES (one card per position) │
│  PAIR   PRICE   SUPP   RES   SCORE │  #001 BTCUSDT                        │
│  ...                               │   Entry / Current Price / PnL        │
│                                    │   Stop · Trail · TP1 · TP2 · TP3 · TP4 │
│                                    │  ...                                  │
├────────────────────────────────────┴───────────────────────────────────────┤
│  LOG FEED (latest 15 events)              │  ♀ Melania (ASCII dancer)      │
└───────────────────────────────────────────┴────────────────────────────────┘
```

<table>
  <tr>
    <th>Panel</th>
    <th>What it shows</th>
  </tr>
  <tr>
    <td><strong>Header</strong></td>
    <td>Live equity, realized PnL, win/loss record, streak, and uptime</td>
  </tr>
  <tr>
    <td><strong>Radar</strong></td>
    <td>All 30 watched pairs with live price, support/resistance, RSI, Stochastic %K, Volume Oscillator, and confluence score. <code>GO 6/11</code> means an entry is triggering.</td>
  </tr>
  <tr>
    <td><strong>Open Trades</strong></td>
    <td>One card per open position showing entry price, current price, PnL%, stop loss, ATR trail stop, and the four take-profit targets (+1R, +2R, +3.5R, +5.5R)</td>
  </tr>
  <tr>
    <td><strong>Log Feed</strong></td>
    <td>Real-time trade events: entries, partial exits, stop hits, errors</td>
  </tr>
  <tr>
    <td><strong>Melania</strong></td>
    <td>24-frame ASCII dancer. Cycles every 2 seconds.</td>
  </tr>
</table>

---

## Resetting to a Fresh Start

The bot saves all state (balance, positions, win/loss record) to `gridzilla_vip_pro_state.json` in the same folder as the script. It auto-saves on every trade and every 30 seconds.

To wipe everything and start over with a $1,000 paper balance, **stop the bot first**, then run:

<details>
<summary><strong>🪟 Windows</strong></summary>

```powershell
python -c "
import json, datetime
json.dump({
  'balance':'1000.0','pnl':'0.0','wins':0,'losses':0,'streak':0,
  'signal_counter':0,'gross_wins':'0','gross_losses':'0','total_r_sum':'0',
  'daily_wins':0,'daily_trades':0,'daily_date':str(datetime.date.today()),
  'peak_equity':'1000.0','cooldowns':{},'positions':{}
}, open('gridzilla_vip_pro_state.json','w'))
print('State reset to \$1000.')
"
```

</details>

<details>
<summary><strong>🐧 Linux</strong></summary>

```bash
python3 -c "
import json, datetime
json.dump({
  'balance':'1000.0','pnl':'0.0','wins':0,'losses':0,'streak':0,
  'signal_counter':0,'gross_wins':'0','gross_losses':'0','total_r_sum':'0',
  'daily_wins':0,'daily_trades':0,'daily_date':str(datetime.date.today()),
  'peak_equity':'1000.0','cooldowns':{},'positions':{}
}, open('gridzilla_vip_pro_state.json','w'))
print('State reset to \$1000.')
"
```

</details>

---

## Configuration

All settings live in the `Config` class at the top of `gridzilla_vip_pro.py`. Open the file in any text editor and change values there. No other files to edit.

<table>
  <tr>
    <th>Setting</th>
    <th>Default</th>
    <th>What it controls</th>
  </tr>
  <tr>
    <td><code>alloc_base</code></td>
    <td><code>0.18</code></td>
    <td>Base position size — 18% of current balance per trade</td>
  </tr>
  <tr>
    <td><code>alloc_heater</code></td>
    <td><code>0.30</code></td>
    <td>Max position size on a winning streak — 30% of balance</td>
  </tr>
  <tr>
    <td><code>stop_loss_pct</code></td>
    <td><code>0.045</code></td>
    <td>Hard stop loss — 4.5% below entry. Also sets the 1R unit for TP targets.</td>
  </tr>
  <tr>
    <td><code>max_positions</code></td>
    <td><code>6</code></td>
    <td>Maximum number of open trades at one time</td>
  </tr>
  <tr>
    <td><code>rsi_oversold</code></td>
    <td><code>42</code></td>
    <td>RSI threshold for the oversold-bounce signal</td>
  </tr>
  <tr>
    <td><code>drawdown_threshold</code></td>
    <td><code>0.85</code></td>
    <td>Stops new entries if equity falls below 85% of peak balance</td>
  </tr>
  <tr>
    <td><code>partial_levels</code></td>
    <td><code>(76,35%) (84,30%) (91,25%) (96,10%)</code></td>
    <td>Stochastic %K thresholds that trigger partial exits (take-profits)</td>
  </tr>
  <tr>
    <td><code>cooldown_light/medium/heavy</code></td>
    <td><code>15m / 30m / 60m</code></td>
    <td>How long a pair is blocked from re-entry after a loss, by severity</td>
  </tr>
  <tr>
    <td><code>max_pairs</code></td>
    <td><code>30</code></td>
    <td>Number of pairs tracked simultaneously (ranked by 24h volume)</td>
  </tr>
</table>

Entry fires when **6 or more of 11 signals** align (`if sigs < 6: skip`). Raise this number to trade less frequently; lower it to trade more.

---

## How It Works

<details>
<summary><strong>Entry — EMC Confluence Engine (6 of 11 signals required)</strong></summary>

<br>

<table>
  <tr>
    <th>#</th>
    <th>Signal</th>
    <th>Notes</th>
  </tr>
  <tr><td>a</td><td>EMC velocity: 3 of 4 momentum vectors positive AND RSI &lt; 55</td><td>RSI velocity, MACD histogram velocity, Stoch velocity, VO velocity</td></tr>
  <tr><td>b</td><td>EMA(13) &gt; EMA(33)</td><td>Short-term trend above long-term trend</td></tr>
  <tr><td>c</td><td>RSI &lt; 42 and rising vs. 3 bars ago</td><td>Oversold bounce</td></tr>
  <tr><td>d</td><td>MACD line &gt; signal AND histogram &gt; 0</td><td>Full MACD bullish alignment</td></tr>
  <tr><td>e</td><td>Stochastic %K crosses above %D from below 35</td><td>Oversold crossover</td></tr>
  <tr><td>f</td><td>SuperTrend is bullish, no flip on current bar</td><td></td></tr>
  <tr><td>g</td><td>Volume Oscillator &gt; +10%</td><td>Above-average buying volume</td></tr>
  <tr><td>h</td><td>1h close &gt; SMA(20)</td><td>Price above 20-period moving average</td></tr>
  <tr><td>i</td><td>3-candle higher lows on the 5m chart</td><td>Short-term momentum confirmation</td></tr>
  <tr><td>j</td><td>MACD histogram expanding and positive</td><td>Strengthening momentum</td></tr>
  <tr><td>k</td><td>1-minute volume surge (WebSocket data)</td><td>Optional bonus signal</td></tr>
</table>

Pre-flight gates (not counted as signals): active cooldown on that pair, equity below 85% of peak, confirmed 1h close already below the Chandelier level.

</details>

<details>
<summary><strong>Exit — 8-Level Cascade (checked every second)</strong></summary>

<br>

<table>
  <tr>
    <th>#</th>
    <th>Exit Type</th>
    <th>Action</th>
  </tr>
  <tr><td>1</td><td><strong>Stochastic partials</strong></td><td>K &gt; 76 → sell 35% · K &gt; 84 → sell 30% · K &gt; 91 → sell 25% · K &gt; 96 → sell 10%</td></tr>
  <tr><td>2</td><td><strong>SuperTrend flip</strong> (bearish)</td><td>Sell 20% of remaining position</td></tr>
  <tr><td>3</td><td><strong>Volume Oscillator zero-cross</strong></td><td>+3% → −3% with Stoch &gt; 80 → sell 50% of remaining</td></tr>
  <tr><td>4</td><td><strong>Chandelier exit</strong></td><td>Live price falls below Chandelier level → full exit</td></tr>
  <tr><td>5</td><td><strong>RSI bearish divergence</strong></td><td>RSI falling while price rising and RSI &gt; 65 → full exit</td></tr>
  <tr><td>6</td><td><strong>VO breakeven activation</strong></td><td>VO &gt; 10% arms breakeven stop (only if price moved above entry this session)</td></tr>
  <tr><td>7</td><td><strong>ATR trailing stop</strong></td><td>0.8%–2.0% below session high; only fires if price is above original entry</td></tr>
  <tr><td>8</td><td><strong>Hard stop</strong></td><td>4.5% below entry — full exit</td></tr>
  <tr><td>9</td><td><strong>Breakeven stop</strong></td><td>Fires only if price moved above entry this session</td></tr>
  <tr><td>10</td><td><strong>Stale graduated exit</strong></td><td>Loss &gt; 2% after 30 minutes, or loss &gt; 1.5% after 60 minutes → full exit</td></tr>
</table>

</details>

---

## Troubleshooting

<details>
<summary><strong>The screen is blank or garbled</strong></summary>

<br>

The live display requires a proper terminal. Make sure you are running inside **Windows Terminal**, **PowerShell**, or a full Linux terminal — not a plain `cmd.exe` window or an IDE's embedded console.

Do not pipe or redirect output:

```powershell
# Wrong — breaks the display
python gridzilla_vip_pro.py > log.txt

# Correct
python gridzilla_vip_pro.py
```

</details>

<details>
<summary><strong>"Gridzilla is already running" error</strong></summary>

<br>

Another instance is already running. Find your terminal window and press <kbd>Ctrl</kbd>+<kbd>C</kbd> to stop it, then start fresh.

If the bot crashed hard and you are sure nothing is running, the socket lock releases automatically — just try starting again.

</details>

<details>
<summary><strong>pip install fails with "not found" or permission errors</strong></summary>

<br>

**Windows:** Make sure Python was added to PATH during install. Reinstall Python from [python.org](https://www.python.org/downloads/) and check the **"Add Python to PATH"** box.

**Linux:** Try with `--user` flag:

```bash
pip3 install --user requests websocket-client rich
```

Or use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install requests websocket-client rich
python3 gridzilla_vip_pro.py
```

</details>

<details>
<summary><strong>Pairs show 0.0000 prices at startup</strong></summary>

<br>

Normal. The bot subscribes to a WebSocket stream on startup. It takes 30–90 seconds for live prices to arrive for all 30 pairs. Wait a moment and they will fill in.

</details>

<details>
<summary><strong>The bot trades too often / too rarely</strong></summary>

<br>

Open `gridzilla_vip_pro.py` in a text editor. Near the top, find the `Config` class. Adjust the signal threshold in the entry logic:

- Default is `if sigs < 6` — requires 6 signals out of 11.
- Change to `if sigs < 7` for fewer, higher-quality trades.
- Change to `if sigs < 5` for more frequent trades.

You can also adjust `max_positions` (default 6) to limit how many trades run concurrently.

</details>

---

<p align="center">
  <sub>Paper trading only. This project does not constitute financial advice.<br>
  Built with <a href="https://github.com/Textualize/rich">Rich</a> · Data from <a href="https://www.binance.us">Binance.US</a> public API</sub>
</p>
