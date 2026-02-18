import os, json, threading, time, requests
from decimal import Decimal, ROUND_DOWN
from collections import deque
from datetime import datetime, date
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.align import Align
from rich import box
from rich.console import Console, Group as RichGroup
from websocket import WebSocketApp

# ==========================================
# GRIDZILLA VIP PRO — 10X QUANTUM ENGINE
# ==========================================
class Config:
    version = "VIP-PRO-10X"
    initial_balance = Decimal('1000.0')
    max_pairs = 30
    # --- VIP RISK ---
    alloc_base         = Decimal('0.18')
    alloc_heater       = Decimal('0.30')
    stop_loss_pct      = Decimal('0.045')
    max_risk_per_trade = Decimal('0.015')
    max_hold_seconds   = 3600
    # --- INDICATORS (1h primary) ---
    kline_interval     = "1h"
    kline_limit        = 75
    kline_5m_interval  = "5m"
    kline_5m_limit     = 40
    rsi_period         = 14
    rsi_oversold       = Decimal('42')
    ema_fast           = 13
    ema_slow           = 33
    macd_fast          = 12
    macd_slow          = 26
    macd_signal        = 9
    stoch_k_period     = 14
    stoch_d_period     = 3
    stoch_smooth       = 3
    supertrend_period  = 10
    supertrend_mult    = Decimal('3.0')
    vo_fast            = 5
    vo_slow            = 14
    vo_threshold       = Decimal('10')
    chandelier_period  = 14
    chandelier_mult    = Decimal('2.75')
    atr_trail_mult     = Decimal('2.0')
    atr_trail_min_pct  = Decimal('0.008')   # 0.8% floor
    atr_trail_max_pct  = Decimal('0.020')   # 2.0% ceiling
    # --- PARTIAL EXIT (Stoch K thresholds) ---
    partial_levels     = [(76, Decimal('0.35')), (84, Decimal('0.30')),
                          (91, Decimal('0.25')), (96, Decimal('0.10'))]
    # --- COOLDOWNS ---
    cooldown_light   = 900
    cooldown_medium  = 1800
    cooldown_heavy   = 3600
    drawdown_threshold = Decimal('0.85')
    max_positions      = 6          # hard cap on concurrent open trades
    # --- PAIR SELECTION ---
    rescan_interval = 3600
    EXCLUDED_BASES  = {"USDC", "BUSD", "TUSD", "DAI", "FDUSD", "UST", "USDP"}
    STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gridzilla_vip_pro_state.json")

CONFIG = Config()
console = Console(highlight=False)

class BotState:
    def __init__(self):
        self.balance = CONFIG.initial_balance
        self.positions, self.realized_pnl = {}, Decimal('0.0')
        self.wins, self.losses, self.streak = 0, 0, 0
        self.live_prices, self.grid_data, self.symbols, self.filters = {}, {}, [], {}
        self.price_dir = {}
        self.kline_data = {}
        self.cooldowns = {}
        self.logs = deque(maxlen=15)
        self.lock = threading.RLock()
        self.shutdown_flag = threading.Event()
        self.connection_ok = True
        self.boot_time = time.time()
        self.ws_tick = 0
        self.ws_ref = None
        self.peak_equity = CONFIG.initial_balance
        # VIP PRO performance tracking
        self.signal_counter = 0
        self.gross_wins   = Decimal('0')
        self.gross_losses = Decimal('0')
        self.total_r_sum  = Decimal('0')
        self.daily_wins   = 0
        self.daily_trades = 0
        self.daily_date   = str(date.today())
        self.load_state()

    def load_state(self):
        if os.path.exists(CONFIG.STATE_FILE):
            try:
                with open(CONFIG.STATE_FILE, "r") as f:
                    d = json.load(f)
                    self.balance        = Decimal(d.get('balance', '1000.0'))
                    self.realized_pnl   = Decimal(d.get('pnl', '0.0'))
                    self.wins           = d.get('wins', 0)
                    self.losses         = d.get('losses', 0)
                    self.streak         = d.get('streak', 0)
                    self.signal_counter = d.get('signal_counter', 0)
                    self.gross_wins     = Decimal(d.get('gross_wins', '0'))
                    self.gross_losses   = Decimal(d.get('gross_losses', '0'))
                    self.total_r_sum    = Decimal(d.get('total_r_sum', '0'))
                    self.daily_wins     = d.get('daily_wins', 0)
                    self.daily_trades   = d.get('daily_trades', 0)
                    self.daily_date     = d.get('daily_date', str(date.today()))
                    self.peak_equity    = Decimal(d.get('peak_equity', str(CONFIG.initial_balance)))
                    self.cooldowns      = {s: float(t) for s, t in d.get('cooldowns', {}).items()}
                    self.positions      = {
                        s: {
                            "qty":              Decimal(p['qty']),
                            "entry":            Decimal(p['entry']),
                            "high":             Decimal(p['high']),
                            "ts":               p['ts'],
                            "orig_qty":         Decimal(p.get('orig_qty', p['qty'])),
                            "orig_entry":       Decimal(p.get('orig_entry', p['entry'])),
                            "partial_exits":    p.get('partial_exits', []),
                            "partial_proceeds": Decimal(p.get('partial_proceeds', '0')),
                            "breakeven_active": p.get('breakeven_active', False),
                            "vol_confirmed_be": p.get('vol_confirmed_be', False),
                            "st_flip_acted":    p.get('st_flip_acted', False),
                            "signal_id":        p.get('signal_id', '000'),
                        }
                        for s, p in d.get('positions', {}).items()
                    }
            except Exception:
                pass

    def save_state(self):
        with self.lock:
            data = {
                "balance":        str(self.balance),
                "pnl":            str(self.realized_pnl),
                "wins":           self.wins,
                "losses":         self.losses,
                "streak":         self.streak,
                "signal_counter": self.signal_counter,
                "gross_wins":     str(self.gross_wins),
                "gross_losses":   str(self.gross_losses),
                "total_r_sum":    str(self.total_r_sum),
                "daily_wins":     self.daily_wins,
                "daily_trades":   self.daily_trades,
                "daily_date":     self.daily_date,
                "peak_equity":    str(self.peak_equity),
                "cooldowns":      self.cooldowns,
                "positions": {
                    s: {
                        "qty":              str(p['qty']),
                        "entry":            str(p['entry']),
                        "high":             str(p['high']),
                        "ts":               p['ts'],
                        "orig_qty":         str(p.get('orig_qty', p['qty'])),
                        "orig_entry":       str(p.get('orig_entry', p['entry'])),
                        "partial_exits":    p.get('partial_exits', []),
                        "partial_proceeds": str(p.get('partial_proceeds', Decimal('0'))),
                        "breakeven_active": p.get('breakeven_active', False),
                        "vol_confirmed_be": p.get('vol_confirmed_be', False),
                        "st_flip_acted":    p.get('st_flip_acted', False),
                        "signal_id":        p.get('signal_id', '000'),
                    }
                    for s, p in self.positions.items()
                }
            }
        tmp = CONFIG.STATE_FILE + ".tmp"
        with open(tmp, "w") as f: json.dump(data, f)
        os.replace(tmp, CONFIG.STATE_FILE)

state = BotState()
_base_dir = os.path.dirname(os.path.abspath(__file__))
_log_file = open(os.path.join(_base_dir, "gridzilla_vip_pro.log"), "a", buffering=1, encoding='utf-8')

def add_log(msg: str, style="white"):
    state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] [{style}]{msg}[/]")
    try:
        _log_file.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
    except Exception:
        pass

def _perf_tag():
    """Live performance stats string."""
    tt = state.wins + state.losses
    if tt == 0:
        return "0% WR | 0.0 PF | +0.0R avg"
    wr = state.wins / tt * 100
    pf = float(state.gross_wins / state.gross_losses) if state.gross_losses > 0 else 99.9
    avg_r = float(state.total_r_sum / tt)
    return f"{wr:.0f}% WR | {pf:.1f} PF | {avg_r:+.1f}R avg"

def _daily_check():
    """Reset daily counters if date changed."""
    today = str(date.today())
    if state.daily_date != today:
        state.daily_wins   = 0
        state.daily_trades = 0
        state.daily_date   = today

# ------------------------------------------
# SIGNAL ENGINE — 10X INDICATORS
# ------------------------------------------
def calc_rsi(closes):
    period = CONFIG.rsi_period
    if len(closes) < period + 1:
        return Decimal('50')
    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, Decimal('0')))
        losses.append(max(-diff, Decimal('0')))
    # Wilder's smoothing: seed with simple average, then exponentially smooth
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return Decimal('100')
    rs = avg_gain / avg_loss
    return Decimal('100') - (Decimal('100') / (1 + rs))

def calc_ema_series(closes, period):
    if len(closes) < period:
        return closes[:]
    k = Decimal('2') / (Decimal(str(period)) + 1)
    sma = sum(closes[:period]) / period
    series = [sma]
    for c in closes[period:]:
        series.append(c * k + series[-1] * (1 - k))
    return series

def calc_macd(closes, fast=12, slow=26, sig=9):
    ema_f = calc_ema_series(closes, fast)
    ema_s = calc_ema_series(closes, slow)
    n = min(len(ema_f), len(ema_s))
    if n == 0:
        return Decimal('0'), Decimal('0'), Decimal('0'), False
    macd_line = [ema_f[-(n - i)] - ema_s[-(n - i)] for i in range(n)]
    signal_line = calc_ema_series(macd_line, sig)
    macd_val   = macd_line[-1]
    signal_val = signal_line[-1] if signal_line else macd_val
    hist       = macd_val - signal_val
    return macd_val, signal_val, hist, macd_val > signal_val and hist > 0

def calc_stochastic(highs, lows, closes, k_period=14, d_period=3, smooth=3):
    default = Decimal('50')
    if len(closes) < k_period:
        return default, default, default, default
    raw_k = []
    for i in range(k_period - 1, len(closes)):
        h = max(highs[i - k_period + 1:i + 1])
        lo = min(lows[i - k_period + 1:i + 1])
        raw_k.append((closes[i] - lo) / (h - lo) * 100 if h != lo else default)
    if len(raw_k) < smooth:
        return raw_k[-1], raw_k[-1], raw_k[-1], raw_k[-1]
    k_series = []
    for i in range(smooth - 1, len(raw_k)):
        k_series.append(sum(raw_k[i - smooth + 1:i + 1]) / smooth)
    if len(k_series) < d_period:
        return k_series[-1], k_series[-1], k_series[-1], k_series[-1]
    d_series = []
    for i in range(d_period - 1, len(k_series)):
        d_series.append(sum(k_series[i - d_period + 1:i + 1]) / d_period)
    return (k_series[-1], d_series[-1],
            k_series[-2] if len(k_series) >= 2 else k_series[-1],
            d_series[-2] if len(d_series) >= 2 else d_series[-1])

def calc_supertrend(highs, lows, closes, period=10, mult=Decimal('3')):
    n = len(closes)
    if n < period + 1:
        return closes[-1] if closes else Decimal('0'), True
    tr_list = [highs[0] - lows[0]]
    for i in range(1, n):
        tr_list.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]),
                           abs(lows[i] - closes[i - 1])))
    # Wilder's ATR series: atr_series[k] = ATR at bar (period-1+k)
    atr_val = sum(tr_list[:period]) / period
    atr_series = [atr_val]
    for i in range(period, n):
        atr_val = (atr_val * (period - 1) + tr_list[i]) / period
        atr_series.append(atr_val)
    hl2 = [(h + lo) / 2 for h, lo in zip(highs, lows)]
    upper_band = hl2[period - 1] + mult * atr_series[0]
    lower_band = hl2[period - 1] - mult * atr_series[0]
    direction = True
    supertrend = lower_band
    for i in range(period, n):
        atr = atr_series[i - period + 1]
        basic_upper = hl2[i] + mult * atr
        basic_lower = hl2[i] - mult * atr
        upper_band = basic_upper if basic_upper < upper_band or closes[i - 1] > upper_band else upper_band
        lower_band = basic_lower if basic_lower > lower_band or closes[i - 1] < lower_band else lower_band
        if direction:
            if closes[i] < lower_band:
                direction = False
                supertrend = upper_band
            else:
                supertrend = lower_band
        else:
            if closes[i] > upper_band:
                direction = True
                supertrend = lower_band
            else:
                supertrend = upper_band
    return supertrend, direction

def calc_volume_oscillator(volumes, fast=5, slow=14):
    if len(volumes) < slow:
        return Decimal('0')
    fast_ma = sum(volumes[-fast:]) / fast
    slow_ma = sum(volumes[-slow:]) / slow
    if slow_ma == 0:
        return Decimal('0')
    return (fast_ma - slow_ma) / slow_ma * 100

def calc_chandelier(highs, lows, closes, period=14, mult=Decimal('2.75')):
    if len(closes) < period + 1:
        return Decimal('0')
    highest_high = max(highs[-period:])
    atr = calc_atr(highs, lows, closes, period)
    return highest_high - atr * mult

def calc_atr(highs, lows, closes, period=14):
    if len(closes) < period + 1:
        return Decimal('0')
    tr_list = []
    for i in range(1, len(closes)):
        tr_list.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]),
                           abs(lows[i] - closes[i - 1])))
    # Wilder's smoothing: seed with simple average, then exponentially smooth
    atr = sum(tr_list[:period]) / period
    for i in range(period, len(tr_list)):
        atr = (atr * (period - 1) + tr_list[i]) / period
    return atr

# ------------------------------------------
# MARKET DATA
# ------------------------------------------
def get_market_context(symbol, prev=None):
    """Full 10X indicator suite from 1h + 5m klines."""
    if prev is None:
        prev = {}
    try:
        klines = requests.get(
            "https://api.binance.us/api/v3/klines",
            params={"symbol": symbol, "interval": CONFIG.kline_interval, "limit": CONFIG.kline_limit},
            timeout=5
        ).json()
        if not isinstance(klines, list) or len(klines) < CONFIG.kline_limit // 2:
            return None
        highs   = [Decimal(c[2]) for c in klines]
        lows    = [Decimal(c[3]) for c in klines]
        closes  = [Decimal(c[4]) for c in klines]
        volumes = [Decimal(c[5]) for c in klines]
        n = len(closes)
        if n == 0:
            return None

        # Bollinger bands (MAD-based, adaptive)
        sma = sum(closes) / n
        volatility = sum(abs(x - sma) for x in closes) / n
        vol_pct = volatility / sma if sma > 0 else Decimal('0.01')
        if vol_pct > Decimal('0.02'):
            band_mult = Decimal('2.5')
        elif vol_pct < Decimal('0.005'):
            band_mult = Decimal('1.8')
        else:
            band_mult = Decimal('2.1')

        sma20 = sum(closes[-20:]) / min(20, len(closes[-20:]))
        bullish = closes[-1] > sma20

        rsi = calc_rsi(closes)
        rsi_3ago = calc_rsi(closes[:-3]) if len(closes) > 3 else rsi
        rsi_prev = calc_rsi(closes[:-1]) if len(closes) > CONFIG.rsi_period else rsi

        ema_f = calc_ema_series(closes, CONFIG.ema_fast)
        ema_s = calc_ema_series(closes, CONFIG.ema_slow)
        macd_val, signal_val, hist, macd_bull = calc_macd(closes, CONFIG.macd_fast, CONFIG.macd_slow, CONFIG.macd_signal)
        sk, sd, spk, spd = calc_stochastic(highs, lows, closes,
                                            CONFIG.stoch_k_period, CONFIG.stoch_d_period, CONFIG.stoch_smooth)
        st_val, st_dir = calc_supertrend(highs, lows, closes,
                                          CONFIG.supertrend_period, CONFIG.supertrend_mult)
        vo = calc_volume_oscillator(volumes, CONFIG.vo_fast, CONFIG.vo_slow)
        chandelier = calc_chandelier(highs, lows, closes,
                                      CONFIG.chandelier_period, CONFIG.chandelier_mult)
        atr = calc_atr(highs, lows, closes, 14)

        price_3ago = closes[-4] if len(closes) >= 4 else closes[-1]

        # 5m klines for higher-lows
        higher_lows = False
        try:
            r5m = requests.get(
                "https://api.binance.us/api/v3/klines",
                params={"symbol": symbol, "interval": CONFIG.kline_5m_interval, "limit": CONFIG.kline_5m_limit},
                timeout=5
            ).json()
            if isinstance(r5m, list) and len(r5m) >= 3:
                lows_5m = [Decimal(c[3]) for c in r5m]
                higher_lows = lows_5m[-1] > lows_5m[-2] > lows_5m[-3]
        except Exception:
            pass

        return {
            "floor":              sma - volatility * band_mult,
            "ceil":               sma + volatility * band_mult,
            "bullish":            bullish,
            "rsi":                rsi,
            "rsi_3ago":           rsi_3ago,
            "rsi_prev":           rsi_prev,
            "price_3ago":         price_3ago,
            "price_cur":          closes[-1],
            "ema_fast":           ema_f[-1] if ema_f else sma,
            "ema_slow":           ema_s[-1] if ema_s else sma,
            "macd_bullish":       macd_bull,
            "macd_hist":          hist,
            "macd_hist_prev":     prev.get('macd_hist', Decimal('0')),  # FIX: use 0 as sentinel instead of hist
            "stoch_k":            sk,
            "stoch_d":            sd,
            "stoch_prev_k":       prev.get('stoch_k', Decimal('50')),  # FIX: use 50 as neutral sentinel; prev.get('stoch_k', spk) made crossover impossible on first cycle
            "stoch_prev_d":       prev.get('stoch_d', Decimal('50')),  # FIX: same sentinel fix as stoch_prev_k
            "supertrend_dir":     st_dir,
            "supertrend_prev_dir": prev.get('supertrend_dir', False),  # FIX: use False as neutral sentinel; st_dir default made prev == current on first cycle, masking fresh trend flips
            "supertrend_val":     st_val,
            "vo":                 vo,
            "vo_prev":            prev.get('vo', Decimal('0')),  # FIX: use 0 as sentinel instead of vo
            "chandelier":         chandelier,
            "atr":                atr,
            "higher_lows":        higher_lows,
        }
    except Exception:
        return None

def market_pulse():
    while not state.shutdown_flag.is_set():
        try:
            cycle_start = time.time()
            temp_grids = {}
            with state.lock:
                symbols_snapshot = list(state.symbols)
            for s in symbols_snapshot:
                prev = state.grid_data.get(s, {})
                ctx = get_market_context(s, prev)
                if ctx: temp_grids[s] = ctx
            with state.lock:
                state.grid_data.update(temp_grids)
            elapsed = time.time() - cycle_start
            time.sleep(max(0.0, 60.0 - elapsed))
        except Exception as e:
            add_log(f"market_pulse error: {e}", "red")
            time.sleep(5)

def pair_scanner():
    time.sleep(CONFIG.rescan_interval)
    while not state.shutdown_flag.is_set():
        try:
            ticker = requests.get("https://api.binance.us/api/v3/ticker/24hr", timeout=10).json()
            ticker.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
            candidates = list(dict.fromkeys(
                t['symbol'] for t in ticker
                if t['symbol'].endswith("USDT")
                and t['symbol'].replace("USDT", "") not in CONFIG.EXCLUDED_BASES
            ))
            info = requests.get("https://api.binance.us/api/v3/exchangeInfo", timeout=10).json()
            valid_symbols = {s['symbol'] for s in info['symbols'] if s['status'] == 'TRADING'}
            candidates = [s for s in candidates if s in valid_symbols]
            new_symbols = []
            for sym in candidates:
                if len(new_symbols) >= CONFIG.max_pairs:
                    break
                if validate_pair(sym):
                    new_symbols.append(sym)
                time.sleep(0.1)  # FIX: throttle validate_pair REST calls to avoid Binance.US 429 rate limit
            new_filters = {}
            for s_info in info['symbols']:
                if s_info['symbol'] in new_symbols:
                    flts = {filt['filterType']: filt for filt in s_info['filters']}
                    new_filters[s_info['symbol']] = {
                        'stepSize':    Decimal(flts['LOT_SIZE']['stepSize']),
                        'minNotional': Decimal(flts.get('NOTIONAL', flts.get('MIN_NOTIONAL', {'minNotional': '10.0'}))['minNotional'])
                    }
            with state.lock:
                added   = [s for s in new_symbols if s not in state.symbols]
                removed = [s for s in state.symbols if s not in new_symbols]
                for pos_sym in state.positions:
                    if pos_sym not in new_symbols:
                        new_symbols.append(pos_sym)
                state.symbols = new_symbols
                state.filters.update(new_filters)
                for s in removed:  # FIX: purge stale kline_data and grid_data for dropped pairs; stale prev_close can trigger false floor-bounce on re-add
                    state.kline_data.pop(s, None)
                    state.grid_data.pop(s, None)
                if added:   add_log(f"SCAN +{','.join(added)}", "dim cyan")
                if removed: add_log(f"SCAN -{','.join(removed)}", "dim yellow")
                if (added or removed) and state.ws_ref:
                    try: state.ws_ref.close()
                    except: pass
        except Exception as e:
            add_log(f"Rescan error: {e}", "red")
        time.sleep(CONFIG.rescan_interval)

# ------------------------------------------
# TRADE ENGINE — VIP PRO 10X SYSTEM
# ------------------------------------------
def _full_exit(sym, price, reason):
    """Full exit with R-multiple and performance tracking."""
    p = state.positions[sym]
    val = p['qty'] * price
    remaining_profit = val - (p['qty'] * p['orig_entry'])
    state.balance      += val
    state.realized_pnl += remaining_profit
    total_proceeds = p.get('partial_proceeds', Decimal('0')) + val
    total_cost     = p['orig_qty'] * p['orig_entry']
    total_pnl      = total_proceeds - total_cost
    # R-multiple: 1R = risk at entry
    one_r = p['orig_entry'] * CONFIG.stop_loss_pct * p['orig_qty']
    r_mult = total_pnl / one_r if one_r > 0 else Decimal('0')
    state.total_r_sum += r_mult
    _daily_check()
    state.daily_trades += 1
    sid = p.get('signal_id', '???')
    if total_pnl > 0:
        state.wins   += 1
        state.streak += 1
        state.gross_wins += total_pnl
        state.daily_wins += 1
        add_log(f"WIN #{sid} +{float(r_mult):.1f}R | Streak:{state.streak}x | [{_perf_tag()}] | {sym}", "bold green")
    else:
        state.losses += 1
        state.streak  = state.streak // 2
        state.gross_losses += abs(total_pnl)
        loss_pct = abs(total_pnl) / total_cost if total_cost else Decimal('0')
        if loss_pct >= CONFIG.stop_loss_pct:
            cd = CONFIG.cooldown_heavy
        elif loss_pct >= Decimal('0.02'):
            cd = CONFIG.cooldown_medium
        else:
            cd = CONFIG.cooldown_light
        state.cooldowns[sym] = time.time() + cd
        add_log(f"LOSS [{reason}] #{sid} {float(r_mult):+.1f}R CD:{cd//60}m | [{_perf_tag()}] | {sym}", "bold red")
    state.positions.pop(sym)
    state.save_state()

def _partial_exit(sym, pct_of_orig, reason, price, stoch_k):
    """Execute partial position exit."""
    p = state.positions[sym]
    f = state.filters.get(sym)
    sell_qty = p['orig_qty'] * pct_of_orig
    if f:
        sell_qty = (sell_qty / f['stepSize']).quantize(Decimal('1'), rounding=ROUND_DOWN) * f['stepSize']
    sell_qty = min(sell_qty, p['qty'])
    if sell_qty <= 0:
        return
    proceeds = sell_qty * price
    cost     = sell_qty * p['orig_entry']
    state.balance        += proceeds
    state.realized_pnl   += proceeds - cost
    p['partial_proceeds'] = p.get('partial_proceeds', Decimal('0')) + proceeds
    p['qty']             -= sell_qty
    be_str = "+BE" if not p.get('breakeven_active', False) else ""
    if not p.get('breakeven_active', False):
        p['breakeven_active'] = True
    pct_int = int(pct_of_orig * 100)
    sid = p.get('signal_id', '???')
    add_log(f"PART {pct_int}%{be_str} {sym} @{price:.4f} ST:{int(stoch_k)}", "yellow")
    if p['qty'] <= 0:
        total_cost = p['orig_qty'] * p['orig_entry']
        total_pnl  = p['partial_proceeds'] - total_cost
        one_r = p['orig_entry'] * CONFIG.stop_loss_pct * p['orig_qty']
        r_mult = total_pnl / one_r if one_r > 0 else Decimal('0')
        state.total_r_sum += r_mult
        _daily_check()
        state.daily_trades += 1
        sid = p.get('signal_id', '???')
        state.positions.pop(sym)
        if total_pnl > 0:  # FIX: was unconditional WIN; cascades 1+2 can close position at a loss
            state.wins       += 1
            state.streak     += 1
            state.gross_wins += total_pnl
            state.daily_wins += 1
            add_log(f"WIN #{sid} +{float(r_mult):.1f}R | Streak:{state.streak}x | [{_perf_tag()}] | {sym}", "bold green")
        else:
            state.losses       += 1
            state.streak        = state.streak // 2
            state.gross_losses += abs(total_pnl)
            state.cooldowns[sym] = time.time() + CONFIG.cooldown_light
            add_log(f"LOSS [PARTIAL] #{sid} {float(r_mult):+.1f}R CD:{CONFIG.cooldown_light//60}m | [{_perf_tag()}] | {sym}", "bold red")
    state.save_state()

def trade_executioner():
    while not state.shutdown_flag.is_set():
        try:
            if not state.connection_ok:
                time.sleep(1); continue
            with state.lock:
                streak_bonus = min(state.streak, 6) * Decimal('0.02')
                current_bet  = min(CONFIG.alloc_base + streak_bonus, CONFIG.alloc_heater)
                risk_cap     = CONFIG.max_risk_per_trade / CONFIG.stop_loss_pct
                current_bet  = min(current_bet, risk_cap)
    
                # Track peak equity (realized + open positions at live prices)
                _open_val = sum(p['qty'] * state.live_prices.get(s2, p['entry'])
                                for s2, p in state.positions.items())
                _total_equity = state.balance + _open_val
                if _total_equity > state.peak_equity:
                    state.peak_equity = _total_equity
    
                for s in list(state.symbols):
                    price = state.live_prices.get(s)
                    grid  = state.grid_data.get(s)
                    f     = state.filters.get(s)
                    if not price or not grid or not f: continue
    
                    # ======= EXIT LOGIC =======
                    if s in state.positions:
                        p = state.positions[s]
                        if price > p['high']: p['high'] = price
                        age = time.time() - p['ts']
                        stoch_k = grid.get('stoch_k', Decimal('50'))
    
                        # --- STOCHASTIC PARTIALS (when in profit) ---
                        if price > p['orig_entry']:
                            for threshold, pct in CONFIG.partial_levels:
                                if stoch_k > Decimal(str(threshold)) and threshold not in p.get('partial_exits', []):
                                    p.setdefault('partial_exits', []).append(threshold)
                                    _partial_exit(s, pct, f"ST>{threshold}", price, stoch_k)
                                    break
                            if s not in state.positions:
                                continue
    
                        # CASCADE 1: SuperTrend FLIP (bearish) -> 20% partial
                        st_dir  = grid.get('supertrend_dir', True)
                        st_prev = grid.get('supertrend_prev_dir', True)
                        _cascade_fired = False
                        if not st_dir and st_prev and not p.get('st_flip_acted', False):
                            p['st_flip_acted'] = True
                            rem_frac = p['qty'] / p['orig_qty'] if p['orig_qty'] > 0 else Decimal('0')
                            _partial_exit(s, rem_frac * Decimal('0.20'), "ST-FLP", price, stoch_k)
                            _cascade_fired = True
                            if s not in state.positions:
                                continue
                        if st_dir:
                            p['st_flip_acted'] = False

                        # CASCADE 2: VO ZeroCross (+3 -> -3%) + Stoch>80 -> 50% partial
                        vo_cur  = grid.get('vo', Decimal('0'))
                        vo_prev = grid.get('vo_prev', Decimal('0'))
                        if not _cascade_fired and vo_cur < Decimal('-3') and vo_prev > Decimal('3') and stoch_k > Decimal('80'):
                            rem_frac = p['qty'] / p['orig_qty'] if p['orig_qty'] > 0 else Decimal('0')
                            _partial_exit(s, rem_frac * Decimal('0.50'), "VO-CROSS", price, stoch_k)
                            if s not in state.positions:
                                continue
    
                        # CASCADE 3: Chandelier exit
                        chandelier = grid.get('chandelier', Decimal('0'))
                        price_1h = grid.get('price_cur', price)  # FIX: use last confirmed 1h close for Chandelier trigger; live tick price causes false exits on intracandle wicks
                        if chandelier > 0 and price_1h < chandelier:
                            _full_exit(s, price, "CHANDELIER")  # exit still executes at live price
                            continue
    
                        # CASCADE 4: RSI bearish divergence (RSI down, price up, overbought)
                        rsi_cur  = grid.get('rsi', Decimal('50'))
                        rsi_3ago = grid.get('rsi_3ago', rsi_cur)
                        pr_cur   = grid.get('price_cur', price)
                        pr_3ago  = grid.get('price_3ago', pr_cur)
                        if rsi_cur < rsi_3ago and pr_cur > pr_3ago and rsi_cur > Decimal('65'):
                            _full_exit(s, price, "RSI-DIV")
                            continue
    
                        # CASCADE 5: Volume BE activation (only after meaningful profit)
                        # VO > threshold is also entry signal #g, so it is always true the cycle
                        # after entry. Require price to have reached 50% of one stop-loss distance
                        # above orig_entry before VO can arm BE — this prevents the pattern where
                        # price ticks up 1 cent, high > orig_entry passes, then price drops back
                        # to entry and BE fires instantly (VO-driven instant-exit loop).
                        _be_arm_level = p['orig_entry'] * (1 + CONFIG.stop_loss_pct * Decimal('0.5'))
                        if vo_cur > CONFIG.vo_threshold and not p.get('vol_confirmed_be', False) and p['high'] >= _be_arm_level:
                            p['vol_confirmed_be'] = True
                            if not p.get('breakeven_active', False):
                                p['breakeven_active'] = True
                                add_log(f"BE ACTIVE [VO] {s}", "dim yellow")
    
                        # CASCADE 6: ATR trailing (clamped 0.8%-2%)
                        atr = grid.get('atr', Decimal('0'))
                        if atr > 0 and p['high'] > p['orig_entry']:
                            trail_dist = atr * CONFIG.atr_trail_mult
                            min_dist = p['orig_entry'] * CONFIG.atr_trail_min_pct
                            max_dist = p['orig_entry'] * CONFIG.atr_trail_max_pct
                            trail_dist = max(min_dist, min(max_dist, trail_dist))
                            atr_trail = p['high'] - trail_dist
                            if price < atr_trail and price > p['orig_entry']:
                                _full_exit(s, price, "ATR-TRAIL")
                                continue
    
                        # CASCADE 7: Hard stop (4.5%)
                        if (price - p['orig_entry']) / p['orig_entry'] <= -CONFIG.stop_loss_pct:
                            _full_exit(s, price, "STOP")
                            continue
    
                        # Breakeven stop (only after price has first risen above entry)
                        # FIX: require high > orig_entry so BE never fires on a fresh entry
                        # where price hasn't moved yet (prevents VO-driven instant exit loop).
                        if p.get('breakeven_active', False) and p['high'] > p['orig_entry'] and price <= p.get('orig_entry', p['entry']):
                            _full_exit(s, price, "BE")
                            continue
    
                        # CASCADE 8: Graduated stale
                        losing   = price < p['orig_entry']
                        loss_pct = (p['orig_entry'] - price) / p['orig_entry'] if losing else Decimal('0')
                        stale    = (loss_pct > Decimal('0.02') and age > 1800) or \
                                   (loss_pct > Decimal('0.005') and age > CONFIG.max_hold_seconds)
                        if stale:
                            _full_exit(s, price, "STALE")
                            continue
    
                    # ======= ENTRY LOGIC: EMC CONFLUENCE ENGINE =======
                    else:
                        # --- Pre-flight gates (not signals) ---
                        if time.time() < state.cooldowns.get(s, 0): continue
                        if len(state.positions) >= CONFIG.max_positions: continue
                        open_val = sum(p2['qty'] * state.live_prices.get(s2, p2['entry'])
                                       for s2, p2 in state.positions.items())
                        if state.balance + open_val < state.peak_equity * CONFIG.drawdown_threshold: continue
    
                        # --- Signal counting ---
                        sigs = 0
    
                        # #a Entropic Momentum Convergence (Claude-original)
                        # Measures simultaneous first-derivative acceleration across 4 independent
                        # indicator velocity vectors. Coherent bullish acceleration across RSI,
                        # MACD momentum, Stochastic, and Volume is a rare high-probability state.
                        rsi_vel   = grid.get('rsi', Decimal('50')) - grid.get('rsi_prev', Decimal('50'))
                        hist_vel  = grid.get('macd_hist', Decimal('0'))  - grid.get('macd_hist_prev',  Decimal('0'))
                        stoch_vel = grid.get('stoch_k',   Decimal('50')) - grid.get('stoch_prev_k',    Decimal('50'))
                        vo_vel    = grid.get('vo',         Decimal('0'))  - grid.get('vo_prev',         Decimal('0'))
                        emc = sum(1 for v in [rsi_vel, hist_vel, stoch_vel, vo_vel] if v > 0)
                        if emc >= 3 and grid.get('rsi', Decimal('50')) < Decimal('55'):
                            sigs += 1
    
                        # #b EMA(13) > EMA(33)
                        if grid.get('ema_fast', Decimal('0')) > grid.get('ema_slow', Decimal('0')):
                            sigs += 1
                        # #c RSI < 42 AND rsi > rsi_3ago (rising)
                        rsi_v = grid['rsi']
                        rsi_3a = grid.get('rsi_3ago', rsi_v)
                        if rsi_v < CONFIG.rsi_oversold and rsi_v > rsi_3a:
                            sigs += 1
                        # #d MACD: line > signal AND histogram > 0
                        if grid.get('macd_bullish', False):
                            sigs += 1
                        # #e Stoch %K > %D from <35
                        sk = grid.get('stoch_k', Decimal('50'))
                        sd = grid.get('stoch_d', Decimal('50'))
                        spk = grid.get('stoch_prev_k', Decimal('50'))
                        spd = grid.get('stoch_prev_d', Decimal('50'))
                        if sk > sd and spk < spd and sk < Decimal('35'):
                            sigs += 1
                        # #f SuperTrend bullish NO flip
                        if grid.get('supertrend_dir', False) and grid.get('supertrend_prev_dir', False):
                            sigs += 1
                        # #g VO > +10%
                        vo_v = grid.get('vo', Decimal('0'))
                        if vo_v > CONFIG.vo_threshold:
                            sigs += 1
                        # #h 1h close > SMA(20)
                        if grid.get('bullish', False):
                            sigs += 1
                        # #i 5m 3-candle higher lows
                        if grid.get('higher_lows', False):
                            sigs += 1
                        # #j MACD histogram expanding (momentum accelerating)
                        hist_cur  = grid.get('macd_hist', Decimal('0'))
                        hist_prev = grid.get('macd_hist_prev', Decimal('0'))
                        if hist_cur > hist_prev and hist_cur > 0:
                            sigs += 1
                        # #k Volume surge bonus (WS kline data when available)
                        kd = state.kline_data.get(s)
                        if kd:
                            vol_hist = kd.get('vol_history', [])
                            if len(vol_hist) >= 5:
                                avg_prior = sum(vol_hist[:-1]) / (len(vol_hist) - 1)
                                if kd.get('last_vol', Decimal('0')) >= avg_prior:
                                    sigs += 1
    
                        if sigs < 5:
                            continue
    
                        target = state.balance * current_bet
                        if target >= f['minNotional']:
                            qty = (target / price / f['stepSize']).quantize(Decimal('1'), rounding=ROUND_DOWN) * f['stepSize']
                            if qty > 0:
                                if qty * price > state.balance:
                                    continue
                                state.signal_counter += 1
                                sid = f"{state.signal_counter:03d}"
                                state.balance -= qty * price
                                state.positions[s] = {
                                    "qty": qty, "entry": price, "high": price, "ts": time.time(),
                                    "orig_qty": qty, "orig_entry": price,
                                    "partial_exits": [], "partial_proceeds": Decimal('0'),
                                    "breakeven_active": False, "vol_confirmed_be": False,
                                    "st_flip_acted": False, "signal_id": sid,
                                }
                                state.kline_data.pop(s, None)
                                pct = int(current_bet * 100)
                                add_log(f"LONG #{sid} {s} @{price:.8g} [{pct}%] [{_perf_tag()}]", "cyan")
                                state.save_state()
                        else:
                            if state.balance < f['minNotional'] * 2:
                                add_log(f"LOW BAL {s}: ${float(state.balance):.2f} below entry floor", "dim red")
    
            time.sleep(1)
        except Exception as e:
            add_log(f"trade_executioner error: {e}", "bold red")
            time.sleep(2)

# ------------------------------------------
# MAIN
# ------------------------------------------
def validate_pair(symbol):
    """Check if pair has valid kline data on Binance.US."""
    try:
        r = requests.get(
            "https://api.binance.us/api/v3/klines",
            params={"symbol": symbol, "interval": "1h", "limit": 5},
            timeout=5
        )
        data = r.json()
        if data and len(data) >= 3:
            closes = [Decimal(c[4]) for c in data]
            return all(c > Decimal('0') for c in closes)
        return False
    except Exception:
        return False

def main():
    # ---- Single-instance guard (PID lock file) ----
    _lock_path = CONFIG.STATE_FILE.replace('_state.json', '.lock')
    if os.path.exists(_lock_path):
        try:
            with open(_lock_path) as _lf:
                _old_pid = int(_lf.read().strip())
            import subprocess
            _r = subprocess.run(
                ['tasklist', '/FI', f'PID eq {_old_pid}', '/NH'],
                capture_output=True, text=True
            )
            if str(_old_pid) in _r.stdout:
                print(f"[ERROR] Gridzilla is already running (PID {_old_pid}). Kill it first.")
                return
        except Exception:
            pass  # stale lock — proceed
    with open(_lock_path, 'w') as _lf:
        _lf.write(str(os.getpid()))
    import atexit
    atexit.register(lambda: os.remove(_lock_path) if os.path.exists(_lock_path) else None)
    # -----------------------------------------------

    add_log("Fetching top pairs by volume...", "yellow")
    ticker = requests.get("https://api.binance.us/api/v3/ticker/24hr", timeout=10).json()
    ticker.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
    candidates = list(dict.fromkeys(
        t['symbol'] for t in ticker
        if t['symbol'].endswith("USDT")
        and t['symbol'].replace("USDT", "") not in CONFIG.EXCLUDED_BASES
    ))
    info = requests.get("https://api.binance.us/api/v3/exchangeInfo", timeout=10).json()
    valid_symbols = {s['symbol'] for s in info['symbols'] if s['status'] == 'TRADING'}
    candidates = [s for s in candidates if s in valid_symbols][:CONFIG.max_pairs]
    add_log("Validating pairs for active data...", "yellow")
    validated = []
    for sym in candidates:
        if validate_pair(sym):
            validated.append(sym)
        if len(validated) >= CONFIG.max_pairs:
            break
        time.sleep(0.1)  # FIX: throttle validate_pair REST calls to avoid Binance.US 429 rate limit (same fix as pair_scanner)
    state.symbols = validated
    for pos_sym in list(state.positions.keys()):
        if pos_sym not in state.symbols:
            state.symbols.append(pos_sym)
            add_log(f"RESCUE {pos_sym} (open position)", "yellow")
    skipped = len(candidates) - len(validated)
    if skipped > 0:
        add_log(f"Filtered {skipped} dead pairs", "dim yellow")

    add_log(f"Loaded {len(state.symbols)} active pairs. Fetching exchange info...", "yellow")
    needed = set(state.symbols)
    for s_info in info['symbols']:
        if s_info['symbol'] in needed:
            flts = {filt['filterType']: filt for filt in s_info['filters']}
            state.filters[s_info['symbol']] = {
                'stepSize':    Decimal(flts['LOT_SIZE']['stepSize']),
                'minNotional': Decimal(flts.get('NOTIONAL', flts.get('MIN_NOTIONAL', {'minNotional': '10.0'}))['minNotional'])
            }
            ctx = get_market_context(s_info['symbol'])
            if ctx:
                state.grid_data[s_info['symbol']] = ctx

    state.save_state()
    add_log("Starting threads...", "yellow")
    threading.Thread(target=trade_executioner, daemon=True).start()
    threading.Thread(target=market_pulse,      daemon=True).start()
    threading.Thread(target=pair_scanner,      daemon=True).start()

    def heartbeat():
        while not state.shutdown_flag.is_set():
            try:
                state.save_state()
            except Exception as e:
                add_log(f"heartbeat error: {e}", "red")
            time.sleep(30)
    threading.Thread(target=heartbeat, daemon=True).start()

    def ws_loop():
        while not state.shutdown_flag.is_set():
            try:
                streams = "/".join([
                    f"{s.lower()}@miniTicker/{s.lower()}@kline_1m"
                    for s in state.symbols
                ])
                def on_message(ws, m):
                    state.connection_ok = True
                    state.ws_tick += 1
                    d = json.loads(m)
                    ev = d.get('e', '')
                    if ev == '24hrMiniTicker':
                        sym, new_p = d['s'], Decimal(d['c'])
                        with state.lock:
                            old_p = state.live_prices.get(sym)
                            state.live_prices[sym] = new_p
                            if old_p is not None and new_p != old_p:
                                state.price_dir[sym] = ('up', time.time()) if new_p > old_p else ('dn', time.time())
                    elif ev == 'kline' and d['k']['x']:
                        sym = d['s']
                        new_close = Decimal(d['k']['c'])
                        new_vol   = Decimal(d['k']['v'])
                        with state.lock:
                            existing   = state.kline_data.get(sym, {})
                            prev_close = existing.get('last_close')
                            vol_hist   = existing.get('vol_history', [])
                            state.kline_data[sym] = {
                                'last_close':  new_close,
                                'prev_close':  prev_close,
                                'last_vol':    new_vol,
                                'vol_history': (vol_hist + [new_vol])[-20:]
                            }
                def on_error(ws, err):
                    state.connection_ok = False
                    add_log(f"WS error: {err}", "red")
                def on_close(ws, code, msg):
                    state.connection_ok = False
                    add_log("WS closed, reconnecting...", "yellow")
                ws = WebSocketApp(
                    f"wss://stream.binance.us:9443/ws/{streams}",
                    on_message=on_message,
                    on_error=on_error,
                    on_close=on_close
                )
                state.ws_ref = ws
                ws.run_forever()
            except Exception as e:
                add_log(f"WS exception: {e}", "red")
                time.sleep(5)
    threading.Thread(target=ws_loop, daemon=True).start()

    MELANIA = [
        ["   \u2640    \u2551",
         "   |>   \u2551",
         "   |    \u2551",
         "  / \\   \u2551",
         " '   `  \u2551",
         "new routine who dis"],
        ["  ~\u2640~   \u2551",
         "  /|\\   \u2551",
         "   |    \u2551",
         "  / \\   \u2551",
         "        \u2551",
         "proof of twerk"],
        ["   \u2640~~~~\u2551",
         "  /|    \u2551",
         "   |    \u2551",
         "  / \\   \u2551",
         "        \u2551",
         "found my support lvl"],
        [" \u2640      \u2551",
         "  \\_____\u2551",
         "   \\    \u2551",
         "    \\   \u2551",
         "     \\  \u2551",
         "my body is a candle"],
        ["   \u2640    \u2551",
         "  /|____\u2551",
         "  /     \u2551",
         " /      \u2551",
         "        \u2551",
         "loading spin.exe"],
        ["  \u2640     \u2551",
         "   |~~~~\u2551",
         "  /     \u2551",
         " //     \u2551",
         "        \u2551",
         "WEEEEEE *brrrrr*"],
        ["        \u2551",
         " \u2640~~~~~~\u2551",
         "   \\    \u2551",
         "    \\   \u2551",
         "        \u2551",
         "fully horizontal ser"],
        ["   \\    \u2551",
         "    \\   \u2551",
         " \u2640------\u2551",
         "        \u2551",
         "        \u2551",
         "defying gravity & SEC"],
        ["  \\     \u2551",
         "   \\    \u2551",
         "    \u2640---\u2551",
         "    |   \u2551",
         "        \u2551",
         "rug pull? i AM the rug"],
        ["  /   \\ \u2551",
         "   \\ /  \u2551",
         "    \u2640---\u2551",
         "    |   \u2551",
         "        \u2551",
         "the chart is upside dn"],
        [" /      \u2551",
         "  \\     \u2551",
         "   \u2640~~~~\u2551",
         "    ~   \u2551",
         "        \u2551",
         "bearish on gravity"],
        ["   |    \u2551",
         "   /    \u2551",
         "   \u2640----\u2551",
         "    \\   \u2551",
         "     \\  \u2551",
         "inverted yield curve"],
        ["        \u2551",
         "  \u2640~~~~~\u2551",
         "  |\\    \u2551",
         "  | \\   \u2551",
         "        \u2551",
         "reversal confirmed"],
        ["   \u2640==--\u2551",
         "  /| // \u2551",
         "   |//  \u2551",
         "        \u2551",
         "        \u2551",
         "climbing to new ATH"],
        ["  \\\u2640  --\u2551",
         "   |  / \u2551",
         "   | /  \u2551",
         "        \u2551",
         "        \u2551",
         "green candles only"],
        ["___\u2640____\u2551",
         "   |    \u2551",
         "        \u2551",
         "        \u2551",
         "        \u2551",
         "i AM the resistance"],
        ["   \u2640    \u2551",
         "  /|~~~~\u2551",
         " //     \u2551",
         "        \u2551",
         "        \u2551",
         "taking profits slowly"],
        ["        \u2551",
         "   \u2640    \u2551",
         "   |\\___\u2551",
         "  / \\   \u2551",
         "        \u2551",
         "sliding into ur DMs"],
        ["        \u2551",
         "        \u2551",
         "  \u2640_____\u2551",
         "  |     \u2551",
         " / \\    \u2551",
         "floor is lava & support"],
        ["        \u2551",
         "        \u2551",
         "  _~\u2640___\u2551",
         " /      \u2551",
         "/       \u2551",
         "this IS the bottom"],
        ["    //  \u2551",
         "   / /  \u2551",
         "  \u2640~/   \u2551",
         " /      \u2551",
         "        \u2551",
         "V shaped recovery"],
        ["        \u2551",
         "  \\\u2640/   \u2551",
         "   |    \u2551",
         "   |    \u2551",
         "  / \\   \u2551",
         "resurrection candle"],
        ["  \\\u2640    \u2551",
         "   |~~~~\u2551",
         "   |    \u2551",
         "  / \\   \u2551",
         " /      \u2551",
         "short me. i dare u"],
        [" $ * $ *\u2551",
         "  \\\u2640/   \u2551",
         "   |    \u2551",
         "  /|\\   \u2551",
         " /$ $\\  \u2551",
         "tyvm for the liquidity"],
    ]

    layout = Layout()
    layout.split_column(Layout(name="h", size=3), Layout(name="m", ratio=1), Layout(name="f", size=10))
    layout["m"].split_row(Layout(name="mkt"), Layout(name="pos"))
    layout["f"].split_row(Layout(name="logs", ratio=1), Layout(name="dancer", size=22))

    add_log("GRIDZILLA VIP PRO online. Paper trading active.", "bold green")

    PULSE = ['⣾','⣽','⣻','⢿','⡿','⣟','⣯','⣷']

    with Live(layout, console=console, refresh_per_second=10, screen=console.is_terminal):
        while not state.shutdown_flag.is_set():

            with state.lock:
                snap_prices    = dict(state.live_prices)
                snap_positions = {s: dict(p) for s, p in state.positions.items()}
                snap_grid      = dict(state.grid_data)
                snap_symbols   = list(state.symbols)
                snap_cooldowns = dict(state.cooldowns)
                snap_balance   = state.balance
                snap_pnl       = state.realized_pnl
                snap_wins      = state.wins
                snap_losses    = state.losses
                snap_streak    = state.streak
                snap_conn      = state.connection_ok
                snap_tick      = state.ws_tick
                snap_boot      = state.boot_time
                snap_dirs      = dict(state.price_dir)
                snap_kline     = {s: dict(v) for s, v in state.kline_data.items()}
                snap_logs      = list(state.logs)
                snap_gw        = state.gross_wins
                snap_gl        = state.gross_losses
                snap_tr        = state.total_r_sum
                snap_dw        = state.daily_wins
                snap_dt        = state.daily_trades
                snap_dd        = state.daily_date
                snap_peak      = state.peak_equity

            open_val   = sum(p['qty'] * snap_prices.get(s, p['entry']) for s, p in snap_positions.items())
            cost_basis = sum(p['qty'] * p['entry'] for p in snap_positions.values())
            unrealized = open_val - cost_basis
            total_val  = snap_balance + open_val
            p_style    = "green" if snap_pnl    >= 0 else "red"
            u_style    = "green" if unrealized  >= 0 else "red"
            total_trades = snap_wins + snap_losses
            wr_pct     = int(snap_wins / total_trades * 100) if total_trades else 0
            pf         = float(snap_gw / snap_gl) if snap_gl > 0 else 0.0
            avg_r      = float(snap_tr / total_trades) if total_trades else 0.0
            streak_bonus = min(snap_streak, 6) * Decimal('0.02')
            risk_cap   = CONFIG.max_risk_per_trade / CONFIG.stop_loss_pct
            cur_alloc  = int(min(CONFIG.alloc_base + streak_bonus, CONFIG.alloc_heater, risk_cap) * 100)
            pulse      = PULSE[snap_tick % len(PULSE)]
            up         = int(time.time() - snap_boot)
            up_str     = f"{up//3600}h{(up%3600)//60:02d}m" if up >= 3600 else f"{up//60}m{up%60:02d}s"
            bulls      = sum(1 for g in snap_grid.values() if g and g['bullish'])
            bears      = len(snap_grid) - bulls
            dd_active  = total_val < snap_peak * CONFIG.drawdown_threshold
            dd_str     = " | [bold red blink]DD PAUSE[/]" if dd_active else ""
            # Daily stats
            today = str(date.today())
            d_wins = snap_dw if snap_dd == today else 0
            d_trades = snap_dt if snap_dd == today else 0
            d_wr = f"({d_wins/d_trades*100:.0f}%)" if d_trades else ""
            today_str = f"Today:{d_wins}/{d_trades}{d_wr}"

            layout["h"].update(Panel(
                Align.center(
                    f"[bold yellow]GRIDZILLA VIP PRO[/]  [dim]|[/]  "
                    f"[bold white]${total_val:,.2f}[/]  "
                    f"PnL:[{p_style}]{snap_pnl:+.2f}[/]  [dim]|[/]  "
                    f"[bold green]{wr_pct}%[/] WR  "
                    f"[green]{snap_wins}[/]W [red]{snap_losses}[/]L  [dim]|[/]  "
                    f"[magenta]{snap_streak}x[/] [dim]({cur_alloc}%)[/]  [dim]|[/]  "
                    f"Pos:[bold]{len(snap_positions)}[/]{dd_str}  [dim]|[/]  "
                    f"[dim]{up_str}[/]  "
                    f"[{'green' if snap_conn else 'red'}]{pulse}[/]"
                ),
                border_style="dim blue"
            ))

            # --- RADAR TABLE ---
            m_tab = Table(box=box.SIMPLE_HEAD, expand=True, show_edge=False, border_style="dim blue",
                          header_style="bold dim")
            m_tab.add_column("PAIR",  style="cyan",   no_wrap=True, min_width=10)
            m_tab.add_column("PRICE", style="white",  no_wrap=True, justify="right")
            m_tab.add_column("SUPP",  style="dim",    no_wrap=True, justify="right")
            m_tab.add_column("RES",   style="dim",    no_wrap=True, justify="right")
            m_tab.add_column("RSI",   no_wrap=True,   justify="right")
            m_tab.add_column("SK",    no_wrap=True,   justify="right")
            m_tab.add_column("VO%",   no_wrap=True,   justify="right")
            m_tab.add_column("SCORE", no_wrap=True,   justify="right")
            def _fmt(v):
                f = float(v)
                if f == 0:       return "0.0000"
                if f < 0.0001:   return f"{f:.8f}"
                if f < 0.01:     return f"{f:.6f}"
                return f"{f:.4f}"
            for s in snap_symbols:
                grid  = snap_grid.get(s)
                price = snap_prices.get(s, Decimal('0'))
                if grid:
                    rsi_val  = grid['rsi']
                    rsi_str  = f"{rsi_val:.1f}"
                    rsi_disp = (f"[green]{rsi_str}[/]"  if rsi_val < CONFIG.rsi_oversold else
                                f"[yellow]{rsi_str}[/]" if rsi_val < CONFIG.rsi_oversold + 10 else
                                f"[dim]{rsi_str}[/]")
                    stk = grid.get('stoch_k', Decimal('50'))
                    stk_str = f"{stk:.0f}"
                    st_disp = (f"[green]{stk_str}[/]"  if stk < 30 else
                               f"[yellow]{stk_str}[/]" if stk < 50 else
                               f"[dim]{stk_str}[/]")
                    vo_val = grid.get('vo', Decimal('0'))
                    vo_str = f"{vo_val:+.0f}"
                    vo_disp = f"[green]{vo_str}[/]" if vo_val > CONFIG.vo_threshold else f"[dim]{vo_str}[/]"
                    # Signal count (EMC engine, mirrors entry logic)
                    sc = 0
                    r_rsi_vel   = grid.get('rsi', Decimal('50')) - grid.get('rsi_prev', Decimal('50'))
                    r_hist_vel  = grid.get('macd_hist', Decimal('0'))   - grid.get('macd_hist_prev', Decimal('0'))
                    r_stoch_vel = grid.get('stoch_k', Decimal('50'))    - grid.get('stoch_prev_k', Decimal('50'))
                    r_vo_vel    = grid.get('vo', Decimal('0'))          - grid.get('vo_prev', Decimal('0'))
                    r_emc = sum(1 for v in [r_rsi_vel, r_hist_vel, r_stoch_vel, r_vo_vel] if v > 0)
                    if r_emc >= 3 and grid.get('rsi', Decimal('50')) < Decimal('55'): sc += 1
                    if grid.get('ema_fast', Decimal('0')) > grid.get('ema_slow', Decimal('0')): sc += 1
                    rsi_3a = grid.get('rsi_3ago', rsi_val)
                    if rsi_val < CONFIG.rsi_oversold and rsi_val > rsi_3a: sc += 1
                    if grid.get('macd_bullish', False): sc += 1
                    g_sk = grid.get('stoch_k', Decimal('50'))
                    g_sd = grid.get('stoch_d', Decimal('50'))
                    g_spk = grid.get('stoch_prev_k', Decimal('50'))
                    g_spd = grid.get('stoch_prev_d', Decimal('50'))
                    if g_sk > g_sd and g_spk < g_spd and g_sk < Decimal('35'): sc += 1
                    if grid.get('supertrend_dir', False) and grid.get('supertrend_prev_dir', False): sc += 1
                    if vo_val > CONFIG.vo_threshold: sc += 1
                    if grid.get('bullish', False): sc += 1
                    if grid.get('higher_lows', False): sc += 1
                    hist_cur  = grid.get('macd_hist', Decimal('0'))
                    hist_prev = grid.get('macd_hist_prev', Decimal('0'))
                    if hist_cur > hist_prev and hist_cur > 0: sc += 1
                    kd_s = snap_kline.get(s, {})
                    if kd_s:
                        r_vol_hist = kd_s.get('vol_history', [])
                        if len(r_vol_hist) >= 5:
                            r_avg = sum(r_vol_hist[:-1]) / (len(r_vol_hist) - 1)
                            if kd_s.get('last_vol', Decimal('0')) >= r_avg: sc += 1
                    in_cd = time.time() < snap_cooldowns.get(s, 0)
                    cd_str = " [red]CD[/]" if in_cd else ""
                    emc_str = f"[cyan]E{r_emc}[/]" if r_emc >= 3 else f"[dim]e{r_emc}[/]"
                    if sc >= 5:
                        sig = f"[bold green]GO {sc}/11[/] {emc_str}{cd_str}"
                    elif sc >= 3:
                        sig = f"[yellow]{sc}/11[/] {emc_str}{cd_str}"
                    else:
                        sig = f"[dim]{sc}/11 {emc_str}[/]{cd_str}"
                    floor_str = f"[bold yellow]{_fmt(grid['floor'])}[/]" if price <= grid['floor'] else _fmt(grid['floor'])
                    ceil_str  = _fmt(grid['ceil'])
                else:
                    rsi_disp = floor_str = ceil_str = sig = st_disp = vo_disp = "[dim]...[/]"
                d_info = snap_dirs.get(s)
                now = time.time()
                if d_info and now - d_info[1] < 1.5:
                    arrow = "[bold green]^[/]" if d_info[0] == 'up' else "[bold red]v[/]"
                    p_str = f"{arrow}{_fmt(price)}"
                else:
                    p_str = f" {_fmt(price)}"
                m_tab.add_row(s, p_str, floor_str, ceil_str, rsi_disp, st_disp, vo_disp, sig)
            layout["mkt"].update(Panel(m_tab, title="[dim cyan]Radar[/]", border_style="dim blue"))

            # --- SIGNAL CARDS ---
            card_lines = []
            now = time.time()
            for s, p in snap_positions.items():
                cur     = snap_prices.get(s, p['entry'])
                diff    = (cur - p['entry']) / p['entry']
                usd_pnl = (cur - p['entry']) * p['qty']
                age     = int(now - p['ts'])
                c       = 'green' if diff >= 0 else 'red'
                sid     = p.get('signal_id', '???')
                rem_pct = int(p['qty'] / p.get('orig_qty', p['qty']) * 100) if p.get('orig_qty', p['qty']) > 0 else 100
                # direction indicator
                d_info = snap_dirs.get(s)
                arr = ("^" if d_info[0] == 'up' else "v") if (d_info and now - d_info[1] < 1.5) else " "
                # ATR trail stop level
                g = snap_grid.get(s, {})
                atr = g.get('atr', Decimal('0'))
                trail_str = ""
                if atr > 0 and p['high'] > p['orig_entry']:
                    trail_dist = atr * CONFIG.atr_trail_mult
                    min_d = p['orig_entry'] * CONFIG.atr_trail_min_pct
                    max_d = p['orig_entry'] * CONFIG.atr_trail_max_pct
                    trail_dist = max(min_d, min(max_d, trail_dist))
                    trail_lvl  = p['high'] - trail_dist
                    trail_str  = f"  Trail:{_fmt(trail_lvl)}"
                # TP price targets (1R = entry * stop_loss_pct = 4.5%)
                fired   = p.get('partial_exits', [])
                stoch_k = g.get('stoch_k', Decimal('50'))
                one_r   = p['orig_entry'] * CONFIG.stop_loss_pct
                # R-mult per level: 1R, 2R, 3.5R, 5.5R — escalating targets
                r_mults = [Decimal('1'), Decimal('2'), Decimal('3.5'), Decimal('5.5')]
                tp_prices = [p['orig_entry'] + one_r * m for m in r_mults]
                tp_parts = []
                found_next = False
                for i, ((thr, _), tp_px) in enumerate(zip(CONFIG.partial_levels, tp_prices)):
                    px_str = _fmt(tp_px)
                    if thr in fired:
                        tp_parts.append(f"[dim]TP{i+1} ${px_str} DONE[/]")
                    elif not found_next:
                        tp_parts.append(f"[bold yellow]TP{i+1} ${px_str}[/]")
                        found_next = True
                    else:
                        tp_parts.append(f"[dim]TP{i+1} ${px_str}[/]")
                tp_line = "  ".join(tp_parts)
                sk_str  = f"  [dim]SK:{int(stoch_k)}[/]"
                be_str   = "  [yellow]BE:ACTIVE[/]" if p.get('breakeven_active') else ""
                stop_lvl = _fmt(p['orig_entry'] * (1 - CONFIG.stop_loss_pct))
                card_body = (
                    f"[bold cyan]#{sid}[/]  [bold white]{s}[/]  [dim]{rem_pct}% rem[/]\n"
                    f" LONG   Entry:[white]{_fmt(p['entry'])}[/]   {arr}[{c}]{_fmt(cur)}  {diff:+.2%}  ${float(usd_pnl):+.2f}[/]\n"
                    f" Stop:[dim]{stop_lvl}[/]{trail_str}   Age:[dim]{age//60}m{age%60:02d}s[/]{be_str}\n"
                    f" {tp_line}{sk_str}"
                )
                card_lines.append(Panel(card_body, border_style="dim green", padding=(0, 1)))
            if card_lines:
                layout["pos"].update(Panel(RichGroup(*card_lines), title="[dim green]Open Trades[/]", border_style="dim blue"))
            else:
                layout["pos"].update(Panel("[dim]No open trades[/]", title="[dim green]Open Trades[/]", border_style="dim blue"))
            layout["logs"].update(Panel("\n".join(snap_logs), title="[dim cyan]Logs[/]", border_style="dim cyan"))

            # --- MELANIA ---
            frame      = MELANIA[int(time.time() / 2.0) % len(MELANIA)]
            dancer_art = "\n".join(f"[magenta]{line}[/]" for line in frame[:-1])
            dancer_art += f"\n[dim italic magenta]{frame[-1]}[/]"
            layout["dancer"].update(Panel(dancer_art, title="[magenta]\u2640 MELANIA[/]", border_style="magenta"))

            time.sleep(0.1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        state.shutdown_flag.set()
        console.print("\n[yellow]Shutting down GRIDZILLA VIP PRO...[/]")
        state.save_state()
        console.print("[green]State saved. Goodbye.[/]")
