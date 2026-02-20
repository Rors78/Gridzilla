import os, json, threading, time, requests, sys
from decimal import Decimal, ROUND_DOWN
from collections import deque
from datetime import datetime, date, timedelta
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
    version = "VIP-PRO-QUANTUM"
    initial_balance = Decimal('1000.0')
    max_pairs = 30
    # --- VIP RISK ---
    alloc_base         = Decimal('0.18')
    alloc_heater       = Decimal('0.30')
    stop_loss_pct      = Decimal('0.045')
    max_risk_per_trade = Decimal('0.015')
    max_hold_seconds   = 3600
    slippage_pct       = Decimal('0.0008')  # 0.08% realistic slippage penalty
    # --- INDICATORS (1h primary) ---
    kline_interval     = "1h"
    kline_limit        = 75
    kline_5m_interval  = "5m"
    kline_5m_limit     = 40
    kline_4h_interval  = "4h"
    kline_4h_limit     = 60
    rsi_period         = 14
    rsi_oversold       = Decimal('42')
    ema_fast           = 13
    ema_slow           = 33
    ema_macro_f        = 20
    ema_macro_s        = 50
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
    # --- VOLATILITY & FIB ---
    bb_period          = 20
    fib_mid            = Decimal('0.50')
    fib_golden         = Decimal('0.618')
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
    
    # --- CORRELATION GROUPS ---
    GROUPS = {
        'L1': {"BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "BNBUSDT", "AVAXUSDT", "LTCUSDT", "XRPUSDT", "DOTUSDT", "MATICUSDT", "ATOMUSDT", "NEARUSDT", "SUIUSDT", "APTUSDT"},
        'MEME': {"DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "PUMPUSDT", "PENGUUSDT", "BONKUSDT", "FLOKIUSDT", "APEUSDT"},
        'AI_INFRA': {"RENDERUSDT", "GRTUSDT", "FETUSDT", "LINKUSDT", "UNIUSDT", "ARUSDT", "IMXUSDT", "INJUSDT", "OPUSDT", "ARBUSDT", "THETAUSDT"}
    }

    # --- PATHS ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
    
    STATE_FILE = os.path.join(DATA_DIR, "gridzilla_vip_pro_state.json")
    BRAIN_FILE = os.path.join(DATA_DIR, "gridzilla_brain.json")
    LOG_FILE   = os.path.join(DATA_DIR, "gridzilla_vip_pro.log")

    # --- API CONFIG ---
    API_BASE = "https://api.binance.us"
    WS_BASE  = "wss://stream.binance.us:9443"
    TUI_FPS  = 5  # Reduced from 10 for better performance

CONFIG = Config()
console = Console(highlight=False)

class AdaptiveBrain:
    SIGNAL_NAMES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']
    MIN_SAMPLES_SIGNAL = 5
    MIN_SAMPLES_THRESHOLD = 10
    MIN_SAMPLES_SIZE = 20
    MIN_SAMPLES_STALE = 8
    MIN_SAMPLES_PAIR = 5

    def __init__(self):
        self.signal_stats = {s: {'fired': 0, 'wins': 0, 'r_sum': Decimal('0')} for s in self.SIGNAL_NAMES}
        self.exit_stats = {}
        self.trade_history = deque(maxlen=200)
        self.signal_weights = {s: Decimal('1.0') for s in self.SIGNAL_NAMES}
        self.optimal_threshold = Decimal('6.0')
        self.size_multiplier = Decimal('1.0')
        self.stale_loss_pct_30m = Decimal('0.020')
        self.stale_loss_pct_60m = Decimal('0.015')
        self.pair_scores = {}
        self._pair_history = {}
        self.greylisted_pairs = set()
        self.toxic_clusters = {} # {cluster_tuple: count_of_losses}
        self.total_trades_seen = 0
        self.load()

    def load(self):
        default_stats = {s: {'fired': 0, 'wins': 0, 'r_sum': Decimal('0')} for s in self.SIGNAL_NAMES}
        if os.path.exists(CONFIG.BRAIN_FILE):
            try:
                with open(CONFIG.BRAIN_FILE, "r") as f:
                    d = json.load(f)
                    loaded = d.get('signal_stats', {})
                    if loaded and all(sig in loaded for sig in self.SIGNAL_NAMES):
                        self.signal_stats = loaded
                    else:
                        self.signal_stats = default_stats
                    for s in self.signal_stats:
                        r_sum_val = self.signal_stats[s].get('r_sum', '0')
                        if isinstance(r_sum_val, str):
                            self.signal_stats[s]['r_sum'] = Decimal(r_sum_val)
                        else:
                            self.signal_stats[s]['r_sum'] = Decimal(str(r_sum_val))
                    loaded_exit = d.get('exit_stats', {})
                    self.exit_stats = {}
                    for k, v in loaded_exit.items():
                        self.exit_stats[k] = {
                            'count': v.get('count', 0),
                            'r_sum': Decimal(str(v.get('r_sum', '0')))
                        }
                    history = d.get('trade_history', [])
                    self.trade_history = deque(history, maxlen=200)
                    loaded_weights = d.get('signal_weights', {})
                    if loaded_weights and all(sig in loaded_weights for sig in self.SIGNAL_NAMES):
                        self.signal_weights = {s: Decimal(str(v)) for s, v in loaded_weights.items()}
                    else:
                        self.signal_weights = {s: Decimal('1.0') for s in self.SIGNAL_NAMES}
                    self.optimal_threshold = Decimal(str(d.get('optimal_threshold', '6.0')))
                    self.size_multiplier = Decimal(str(d.get('size_multiplier', '1.0')))
                    self.stale_loss_pct_30m = Decimal(d.get('stale_loss_pct_30m', '0.020'))
                    self.stale_loss_pct_60m = Decimal(d.get('stale_loss_pct_60m', '0.015'))
                    self.pair_scores = {k: float(v) for k, v in d.get('pair_scores', {}).items()}
                    self.greylisted_pairs = set(d.get('greylisted_pairs', []))
                    self.toxic_clusters = {tuple(k.split(',')): v for k, v in d.get('toxic_clusters', {}).items()}
                    self.total_trades_seen = d.get('total_trades_seen', 0)
            except Exception:
                self.signal_stats = default_stats
                self.signal_weights = {s: Decimal('1.0') for s in self.SIGNAL_NAMES}

    def save(self):
        data = {
            'signal_stats': {s: {'fired': v['fired'], 'wins': v['wins'], 'r_sum': str(v['r_sum'])} for s, v in self.signal_stats.items()},
            'exit_stats': {k: {'count': v['count'], 'r_sum': str(v['r_sum'])} for k, v in self.exit_stats.items()},
            'trade_history': list(self.trade_history),
            'signal_weights': {k: float(v) for k, v in self.signal_weights.items()},
            'optimal_threshold': float(self.optimal_threshold),
            'size_multiplier': float(self.size_multiplier),
            'stale_loss_pct_30m': str(self.stale_loss_pct_30m),
            'stale_loss_pct_60m': str(self.stale_loss_pct_60m),
            'pair_scores': self.pair_scores,
            'greylisted_pairs': list(self.greylisted_pairs),
            'toxic_clusters': {",".join(k): v for k, v in self.toxic_clusters.items()},
            'total_trades_seen': self.total_trades_seen,
        }
        tmp = CONFIG.BRAIN_FILE + ".tmp"
        with open(tmp, "w") as f: json.dump(data, f)
        os.replace(tmp, CONFIG.BRAIN_FILE)

    def record_trade(self, fingerprint, r_multiple, exit_reason, hold_seconds):
        self.total_trades_seen += 1
        record = {
            'signals_fired': fingerprint.get('signals_fired', []),
            'signal_score': fingerprint.get('signal_score', 0),
            'r_multiple': r_multiple,
            'exit_reason': exit_reason,
            'hold_seconds': hold_seconds,
            'pair': fingerprint.get('pair', ''),
            'entry_time': fingerprint.get('entry_time', 0),
        }
        self.trade_history.append(record)
        
        # Record Toxic Clusters (combinations that lead to significant losses)
        if r_multiple <= -0.8:
            cluster = tuple(sorted(fingerprint.get('signals_fired', [])))
            if cluster:
                self.toxic_clusters[cluster] = self.toxic_clusters.get(cluster, 0) + 1
                if self.toxic_clusters[cluster] >= 2:
                    add_log(f"BRAIN: Learned Toxic Cluster detected {cluster}", "bold red")

        for sig in fingerprint.get('signals_fired', []):
            if sig in self.signal_stats:
                self.signal_stats[sig]['fired'] += 1
                if r_multiple > 0.0:
                    self.signal_stats[sig]['wins'] += 1
                self.signal_stats[sig]['r_sum'] += Decimal(str(r_multiple))
        if exit_reason not in self.exit_stats:
            self.exit_stats[exit_reason] = {'count': 0, 'r_sum': Decimal('0')}
        self.exit_stats[exit_reason]['count'] += 1
        self.exit_stats[exit_reason]['r_sum'] += Decimal(str(r_multiple))
        pair = fingerprint.get('pair', '')
        if pair:
            if pair not in self._pair_history:
                self._pair_history[pair] = []
            self._pair_history[pair].append(r_multiple)
            if len(self._pair_history[pair]) > 10:
                self._pair_history[pair] = self._pair_history[pair][-10:]
            self.pair_scores[pair] = sum(self._pair_history[pair]) / len(self._pair_history[pair])

    def update(self):
        changes = []
        for sig in self.SIGNAL_NAMES:
            stats = self.signal_stats[sig]
            if stats['fired'] >= self.MIN_SAMPLES_SIGNAL:
                win_rate = Decimal(str(stats['wins'])) / Decimal(str(stats['fired'])) if stats['fired'] > 0 else Decimal('0')
                old_weight = self.signal_weights[sig]
                self.signal_weights[sig] = Decimal('0.5') + (win_rate * Decimal('1.0'))
                self.signal_weights[sig] = max(Decimal('0.5'), min(Decimal('1.5'), self.signal_weights[sig]))
                if old_weight != self.signal_weights[sig]:
                    changes.append(f"W{sig}:{float(old_weight):.2f}→{float(self.signal_weights[sig]):.2f}")
        threshold_counts = {5: [], 6: [], 7: [], 8: []}
        for rec in self.trade_history:
            score = rec.get('signal_score', 0)
            if score in threshold_counts:
                threshold_counts[score].append(rec.get('r_multiple', 0))
        best_threshold = Decimal('6.0')
        best_r = -999.0
        for thresh, r_values in threshold_counts.items():
            if len(r_values) >= self.MIN_SAMPLES_THRESHOLD:
                avg_r = sum(r_values) / len(r_values)
                if avg_r > best_r:
                    best_r = avg_r
                    best_threshold = Decimal(str(thresh))
        if best_r > -999.0:
            old_thresh = self.optimal_threshold
            self.optimal_threshold = max(Decimal('5.0'), min(Decimal('8.0'), best_threshold))
            if old_thresh != self.optimal_threshold:
                changes.append(f"TH:{float(old_thresh):.1f}→{float(self.optimal_threshold):.1f}")
        recent_trades = list(self.trade_history)[-self.MIN_SAMPLES_SIZE:]
        if len(recent_trades) >= self.MIN_SAMPLES_SIZE:
            wins = sum(1 for t in recent_trades if t.get('r_multiple', 0) > 0)
            wr = Decimal(str(wins)) / Decimal(str(len(recent_trades)))
            old_mult = self.size_multiplier
            if wr > Decimal('0.60'):
                self.size_multiplier = min(Decimal('1.30'), self.size_multiplier * Decimal('1.05'))
            elif wr < Decimal('0.40'):
                self.size_multiplier = max(Decimal('0.70'), self.size_multiplier * Decimal('0.90'))
            if old_mult != self.size_multiplier:
                changes.append(f"SZ:{float(old_mult):.2f}→{float(self.size_multiplier):.2f}")
        stale_exits = [t for t in self.trade_history if t.get('exit_reason') == 'STALE']
        stop_exits = [t for t in self.trade_history if t.get('exit_reason') == 'STOP']
        if len(stale_exits) >= self.MIN_SAMPLES_STALE:
            stale_avg_r = sum(t.get('r_multiple', 0) for t in stale_exits) / len(stale_exits)
            stop_avg_r = sum(t.get('r_multiple', 0) for t in stop_exits) / len(stop_exits) if stop_exits else 0
            old_30 = self.stale_loss_pct_30m
            old_60 = self.stale_loss_pct_60m
            if stale_avg_r < stop_avg_r - 0.3:
                self.stale_loss_pct_30m = min(Decimal('0.030'), self.stale_loss_pct_30m + Decimal('0.002'))
                self.stale_loss_pct_60m = min(Decimal('0.025'), self.stale_loss_pct_60m + Decimal('0.002'))
            elif stale_avg_r > stop_avg_r + 0.2:
                self.stale_loss_pct_30m = max(Decimal('0.010'), self.stale_loss_pct_30m - Decimal('0.002'))
                self.stale_loss_pct_60m = max(Decimal('0.010'), self.stale_loss_pct_60m - Decimal('0.002'))
            if old_30 != self.stale_loss_pct_30m or old_60 != self.stale_loss_pct_60m:
                changes.append(f"STALE:{float(old_30):.1%}→{float(self.stale_loss_pct_30m):.1%}")
                changes.append(f"STALE:{float(old_30):.1%}→{float(self.stale_loss_pct_30m):.1%}")
        for pair, score in list(self.pair_scores.items()):
            trades_on_pair = [t for t in self.trade_history if t.get('pair') == pair]
            if len(trades_on_pair) >= self.MIN_SAMPLES_PAIR:
                if score < -0.8 and pair not in self.greylisted_pairs:
                    self.greylisted_pairs.add(pair)
                    changes.append(f"GREY:{pair}")
                elif score > 0.0 and pair in self.greylisted_pairs:
                    recent_pair = [t for t in trades_on_pair[-3:] if t.get('r_multiple', 0) > 0]
                    if len(recent_pair) >= 2:
                        self.greylisted_pairs.discard(pair)
                        changes.append(f"CLEAR:{pair}")
        if changes:
            add_log(f"BRAIN: {', '.join(changes)}", "dim white")

    def get_threshold(self, symbol):
        base = max(Decimal('3.0'), self.optimal_threshold)
        if symbol in self.greylisted_pairs:
            return base + Decimal('1.0')
        return base

    def get_weighted_score(self, signals_fired):
        return sum(self.signal_weights.get(sig, Decimal('1.0')) for sig in signals_fired)

    def is_cluster_toxic(self, signals_fired):
        """Checks if this specific combination of signals has a history of high-R losses."""
        cluster = tuple(sorted(signals_fired))
        # If this cluster has failed 2 or more times at -0.8R or worse, it's toxic.
        return self.toxic_clusters.get(cluster, 0) >= 2


class GridScorer:
    CATEGORIES = {
        'MEME': ["DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "PUMPUSDT", "PENGUUSDT", "BONKUSDT", "FLOKIUSDT"],
        'LARGE CAP': ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "AVAXUSDT", "XRPUSDT", "DOTUSDT", "MATICUSDT", "ATOMUSDT", "NEARUSDT", "LTCUSDT"],
        'ALT COIN': ["LINKUSDT", "RENDERUSDT", "GRTUSDT", "FETUSDT", "UNIUSDT", "ARUSDT", "IMXUSDT", "INJUSDT", "OPUSDT", "ARBUSDT"],
    }
    GRID_COUNT = 17

    def __init__(self):
        pass

    def score_opportunity(self, symbol, grid, live_price, volume_rank):
        if not grid:
            return None
        score = Decimal('0')
        details = {}
        
        signal_score = 0
        signals_fired = []
        sigs, signals_fired, emc = calculate_signals(grid, {})
        signal_score = sigs
        
        # Use Brain weights for the box score to match the actual trade engine
        weighted = brain.get_weighted_score(signals_fired)
        threshold = brain.get_threshold(symbol)
        
        # Calculate confidence same as trade_executioner
        diff = weighted - threshold
        confidence = int(60 + (diff * Decimal('10.0')))
        confidence = max(0, min(100, confidence))
        
        atr = grid.get('atr', Decimal('0'))
        price = grid.get('price_cur', live_price)
        if atr > Decimal('0') and price > Decimal('0'):
            atr_pct = (atr / price) * 100
        else:
            atr_pct = Decimal('0')
        
        vol_score = Decimal('0')
        if atr_pct >= Decimal('2') and atr_pct <= Decimal('5'):
            vol_score = Decimal('20')
        elif atr_pct >= Decimal('1') and atr_pct <= Decimal('8'):
            vol_score = Decimal('10')
        
        trend_score = Decimal('0')
        if grid.get('bullish', False):
            trend_score = Decimal('10')
        
        range_score = Decimal('0')
        position = Decimal('0.5')
        floor_val = grid.get('floor', price)
        ceil_val = grid.get('ceil', price)
        if ceil_val > floor_val > Decimal('0'):
            position = (price - floor_val) / (ceil_val - floor_val)
            if position >= Decimal('0.3') and position <= Decimal('0.7'):
                range_score = Decimal('5')
        
        volume_score = Decimal(str(15 if volume_rank <= 3 else 10 if volume_rank <= 10 else 5))
        
        signal_weight = Decimal('0.30')
        confidence_weight = Decimal('0.20')
        vol_weight = Decimal('0.20')
        volume_weight = Decimal('0.15')
        trend_weight = Decimal('0.10')
        range_weight = Decimal('0.05')
        
        max_signals = 11
        score += (Decimal(str(signal_score)) / Decimal(str(max_signals))) * signal_weight * Decimal('100')
        score += (Decimal(str(confidence)) / Decimal('100')) * confidence_weight * Decimal('100')
        score += vol_score
        score += volume_score
        score += trend_score
        score += range_score
        
        details = {
            'signal_score': signal_score,
            'signals_fired': signals_fired,
            'confidence': confidence,
            'atr_pct': float(atr_pct),
            'volume_rank': volume_rank,
            'trend': 'BULLISH' if grid.get('bullish', False) else 'BEARISH',
            'range_position': float(position) if ceil_val > floor_val > Decimal('0') else 0.5,
        }
        
        return {'score': float(score), 'details': details}

    def calculate_grid_settings(self, symbol, price, atr):
        if atr <= Decimal('0') or price <= Decimal('0'):
            return None
        
        atr_pct = (atr / price) * 100
        
        upper_mult = Decimal(str(1 + (float(atr_pct) / 100) * 2))
        lower_mult = Decimal(str(1 - (float(atr_pct) / 100) * 2))
        
        upper = price * upper_mult
        lower = price * lower_mult
        
        grid_range = upper - lower
        profit_per_grid = (grid_range / lower / self.GRID_COUNT) * 100
        
        days_low = Decimal('2')
        days_high = Decimal('7')
        if atr_pct >= Decimal('3'):
            days_low = Decimal('1')
            days_high = Decimal('3')
        elif atr_pct < Decimal('1.5'):
            days_low = Decimal('5')
            days_high = Decimal('14')
        
        return {
            'price': float(price),
            'upper': float(upper),
            'lower': float(lower),
            'grids': self.GRID_COUNT,
            'profit_per_grid': float(profit_per_grid.quantize(Decimal('0.01'))),
            'days_low': int(days_low),
            'days_high': int(days_high),
            'atr_pct': float(atr_pct),
        }

    def find_best_in_category(self, category_symbols, grid_data, live_prices, volume_ranks):
        best = None
        best_score = -1
        best_settings = None
        
        best_result = None
        for symbol in category_symbols:
            grid = grid_data.get(symbol)
            live_price = live_prices.get(symbol)
            vol_rank = volume_ranks.get(symbol, 999)
            
            result = self.score_opportunity(symbol, grid, live_price, vol_rank)
            if result and result['score'] > best_score:
                best_score = result['score']
                best = symbol
                best_result = result
                settings = self.calculate_grid_settings(symbol, grid.get('price_cur', live_price) if grid else live_price, grid.get('atr', Decimal('0')) if grid else Decimal('0'))
                best_settings = settings
        
        if best and best_settings and best_result:
            return {
                'symbol': best,
                'score': best_score,
                'details': best_result['details'],
                'settings': best_settings,
                'timestamp': time.time(),
            }
        return None

    def perform_scoring_cycle(self):
        """Unified scoring logic for daily and initial runs."""
        try:
            add_log("GRID SCORER: Running scan...", "yellow")
            volume_ranks = {}
            ticker = requests.get(f"{CONFIG.API_BASE}/api/v3/ticker/24hr", timeout=10).json()
            # Sort by volume to establish ranks
            ticker.sort(key=lambda x: float(x.get('quoteVolume', 0) or 0), reverse=True)
            for i, t in enumerate(ticker):
                sym = t['symbol']
                if sym.endswith("USDT"):
                    volume_ranks[sym] = i + 1
            
            with state.lock:
                grid_snapshot = dict(state.grid_data)
                price_snapshot = dict(state.live_prices)
            
            for category, symbols in self.CATEGORIES.items():
                best = self.find_best_in_category(symbols, grid_snapshot, price_snapshot, volume_ranks)
                if best:
                    state.grid_picks[category] = best
                    state.grid_history.append({
                        'category': category,
                        'symbol': best['symbol'],
                        'score': best['score'],
                        'timestamp': best['timestamp'],
                    })
                    add_log(f"GRID PICK: {category} -> {best['symbol']} ({best['score']:.0f}/100)", "green")
                else:
                    add_log(f"GRID PICK: {category} -> NO SETUP", "red")
            
            if len(state.grid_history) > 21:
                state.grid_history = state.grid_history[-21:]
            add_log("GRID SCORER: Scan complete", "green")
        except Exception as e:
            add_log(f"GRID SCORER error: {e}", "red")

grid_scorer = GridScorer()


class BotState:
    def __init__(self):
        self.balance = CONFIG.initial_balance
        self.positions, self.realized_pnl = {}, Decimal('0.0')
        self.wins, self.losses, self.streak = 0, 0, 0
        self.best_win = Decimal('0.0')
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
        self.ws_ref = None  # type: ignore
        self.peak_equity = CONFIG.initial_balance
        # Pair health tracking
        self.pair_last_valid = {}
        self.blacklist = set()
        # VIP PRO performance tracking
        self.signal_counter = 0
        self.gross_wins   = Decimal('0')
        self.gross_losses = Decimal('0')
        self.total_r_sum  = Decimal('0')
        self.daily_wins   = 0
        self.daily_trades = 0
        self.daily_date   = str(date.today())
        # Grid Bot Daily Picks
        self.grid_picks = {}  # {category: {symbol, score, settings, timestamp}}
        self.grid_history = []  # [{category, symbol, score, timestamp}, ...] last 7 days
        self.market_regime = "STABLE"
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
                            "high":             Decimal(p.get('high', p['entry'])), # RESTORED: Persist high water mark for trailing stops
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
            except Exception as e:
                add_log(f"load_state error: {e}", "red")

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
brain = AdaptiveBrain()
_log_file = open(CONFIG.LOG_FILE, "a", buffering=1, encoding='utf-8')

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
    pf = float(state.gross_wins / state.gross_losses) if state.gross_losses > Decimal('0') else Decimal('99.9')
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
def _wilders_smoothing(series, period):
    if len(series) < period:
        return series[-1] if series else Decimal('0')
    val = sum(series[:period]) / period
    for i in range(period, len(series)):
        val = (val * (period - 1) + series[i]) / period
    return val

def calc_rsi(closes):
    period = CONFIG.rsi_period
    if len(closes) < period + 1:
        return Decimal('50')
    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, Decimal('0')))
        losses.append(max(-diff, Decimal('0')))
    avg_gain = _wilders_smoothing(gains, period)
    avg_loss = _wilders_smoothing(losses, period)
    if avg_loss == Decimal('0'):
        return Decimal('100')
    rs = avg_gain / avg_loss
    return Decimal('100') - (Decimal('100') / (Decimal('1') + rs))
  # type: ignore

def calc_ema_series(closes, period):
    if len(closes) < period:
        return closes[:]
    k = Decimal('2') / (Decimal(period) + 1)
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
    return macd_val, signal_val, hist, macd_val > signal_val and hist > Decimal('0')

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
    upper_band = hl2[period - 1] + mult * atr_series[0]  # type: ignore
    lower_band = hl2[period - 1] - mult * atr_series[0]  # type: ignore
    direction = True
    supertrend = lower_band
    for i in range(period, n):
        atr = atr_series[i - period + 1]
        basic_upper = hl2[i] + mult * atr  # type: ignore
        basic_lower = hl2[i] - mult * atr  # type: ignore
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
    if slow_ma == Decimal('0'):
        return Decimal('0')
    return (fast_ma - slow_ma) / slow_ma * 100

def calc_chandelier(highs, lows, closes, period=14, mult=Decimal('2.75')):
    if len(closes) < period + 1:
        return Decimal('0')
    highest_high = max(highs[-period:])
    atr = calc_atr(highs, lows, closes, period)
    return highest_high - atr * mult  # type: ignore

def calc_atr(highs, lows, closes, period=14):
    if len(closes) < period + 1:
        return Decimal('0')
    tr_list = []
    for i in range(1, len(closes)):
        tr_list.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]),
                           abs(lows[i] - closes[i - 1])))
    return _wilders_smoothing(tr_list, period)

# ------------------------------------------
# MARKET DATA
# ------------------------------------------
def get_market_context(symbol, prev=None):
    """Full 10X indicator suite from 1h + 5m klines."""
    if prev is None:
        prev = {}
    try:
        klines = requests.get(
            f"{CONFIG.API_BASE}/api/v3/klines",
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
        volatility = sum(abs(x - sma) for x in closes) / n  # type: ignore
        vol_pct = volatility / sma if sma > Decimal('0') else Decimal('0.01')  # type: ignore
        
        # Institutional BBW (Standard Deviation based)
        bb_period = Decimal(str(CONFIG.bb_period))
        bb_sma = sum(closes[-CONFIG.bb_period:]) / bb_period
        bb_var = sum((x - bb_sma)**2 for x in closes[-CONFIG.bb_period:]) / bb_period
        bb_std = Decimal(str(float(bb_var) ** 0.5))
        bb_upper = bb_sma + Decimal('2.0') * bb_std
        bb_lower = bb_sma - Decimal('2.0') * bb_std
        bbw = (bb_upper - bb_lower) / bb_sma if bb_sma > Decimal('0') else Decimal('0')
        
        # BBW Average for Squeeze detection
        bbw_series = []
        for i in range(len(closes) - CONFIG.bb_period, len(closes)):
            temp_sma = sum(closes[i-CONFIG.bb_period:i]) / bb_period
            temp_var = sum((x - temp_sma)**2 for x in closes[i-CONFIG.bb_period:i]) / bb_period
            temp_std = Decimal(str(float(temp_var) ** 0.5))
            bbw_series.append(( (temp_sma + Decimal('2.0')*temp_std) - (temp_sma - Decimal('2.0')*temp_std) ) / temp_sma if temp_sma > Decimal('0') else Decimal('0'))
        bbw_avg = sum(bbw_series) / len(bbw_series) if bbw_series else Decimal('0')

        # Golden Pocket (Fib 0.5 - 0.618 of current signal candle)
        c_high, c_low = highs[-1], lows[-1]
        c_range = c_high - c_low
        fib_50 = c_high - (c_range * CONFIG.fib_mid)
        fib_618 = c_high - (c_range * CONFIG.fib_golden)

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
                f"{CONFIG.API_BASE}/api/v3/klines",
                params={"symbol": symbol, "interval": CONFIG.kline_5m_interval, "limit": CONFIG.kline_5m_limit},
                timeout=5
            ).json()
            if isinstance(r5m, list) and len(r5m) >= 3:
                lows_5m = [Decimal(c[3]) for c in r5m]
                higher_lows = lows_5m[-1] > lows_5m[-2] > lows_5m[-3]
        except Exception:
            pass

        # 4h klines for macro trend
        macro_bullish = False  # SAFETY: Default to False (no trade) if data missing
        try:
            r4h = requests.get(
                f"{CONFIG.API_BASE}/api/v3/klines",
                params={"symbol": symbol, "interval": CONFIG.kline_4h_interval, "limit": CONFIG.kline_4h_limit},
                timeout=5
            ).json()
            if isinstance(r4h, list) and len(r4h) >= CONFIG.ema_macro_s:
                c4h = [Decimal(c[4]) for c in r4h]
                ema_m_f = calc_ema_series(c4h, CONFIG.ema_macro_f)
                ema_m_s = calc_ema_series(c4h, CONFIG.ema_macro_s)
                macro_bullish = ema_m_f[-1] > ema_m_s[-1]
        except Exception as e:
            add_log(f"Macro trend fetch failed [{symbol}]: {e}", "dim red")

        # Track valid data time for blacklist
        state.pair_last_valid[symbol] = time.time()
        
        return {
            "floor":              sma - volatility * band_mult,  # type: ignore
            "ceil":               sma + volatility * band_mult,  # type: ignore
            "bullish":            bullish,
            "macro_bullish":      macro_bullish,
            "rsi":                rsi,
            "rsi_3ago":           rsi_3ago,
            "rsi_prev":           rsi_prev,
            "price_3ago":         price_3ago,
            "price_cur":          closes[-1],
            "ema_fast":           ema_f[-1] if ema_f else sma,
            "ema_slow":           ema_s[-1] if ema_s else sma,
            "macd_bullish":       macd_bull,
            "macd_hist":          hist,
            "macd_hist_prev":     prev.get('macd_hist', Decimal('0')),
            "stoch_k":            sk,
            "stoch_d":            sd,
            "stoch_prev_k":       prev.get('stoch_k', Decimal('50')),
            "stoch_prev_d":       prev.get('stoch_d', Decimal('50')),
            "supertrend_dir":     st_dir,
            "supertrend_prev_dir": prev.get('supertrend_dir', False),
            "supertrend_val":     st_val,
            "vo":                 vo,
            "vo_prev":            prev.get('vo', Decimal('0')),
            "chandelier":         chandelier,
            "atr":                atr,
            "fib_618":            fib_618,
            "last_vol":           volumes[-1] if volumes else Decimal('0'),
        }
    except Exception as e:
        add_log(f"Market context error [{symbol}]: {e}", "red")
        return None

def calculate_signals(grid, kd):
    """Consolidated 11-signal confluence engine (EMC)."""
    sc = 0
    signals_fired = []
    
    if not grid:
        return 0, [], 0

    # #a EMC (Entropic Momentum Convergence)
    r_rsi_vel   = grid.get('rsi', Decimal('50')) - grid.get('rsi_prev', Decimal('50'))
    r_hist_vel  = grid.get('macd_hist', Decimal('0'))   - grid.get('macd_hist_prev', Decimal('0'))
    r_stoch_vel = grid.get('stoch_k', Decimal('50'))    - grid.get('stoch_prev_k', Decimal('50'))
    r_vo_vel    = grid.get('vo', Decimal('0'))          - grid.get('vo_prev', Decimal('0'))
    emc_score = sum(1 for v in [r_rsi_vel, r_hist_vel, r_stoch_vel, r_vo_vel] if v > Decimal('0'))
    if emc_score >= 3 and grid.get('rsi', Decimal('50')) < Decimal('55'):
        sc += 1
        signals_fired.append('a')

    # #b EMA(13) > EMA(33)
    if grid.get('ema_fast', Decimal('0')) > grid.get('ema_slow', Decimal('0')):
        sc += 1
        signals_fired.append('b')

    # #c RSI < 42 AND rising vs 3 bars ago
    rsi_v = grid.get('rsi', Decimal('50'))
    rsi_3a = grid.get('rsi_3ago', rsi_v)
    if rsi_v < CONFIG.rsi_oversold and rsi_v > rsi_3a:
        sc += 1
        signals_fired.append('c')

    # #d MACD bullish
    if grid.get('macd_bullish', False):
        sc += 1
        signals_fired.append('d')

    # #e Stoch %K > %D from <35
    g_sk = grid.get('stoch_k', Decimal('50'))
    g_sd = grid.get('stoch_d', Decimal('50'))
    g_spk = grid.get('stoch_prev_k', Decimal('50'))
    g_spd = grid.get('stoch_prev_d', Decimal('50'))
    if g_sk > g_sd and g_spk < g_spd and (g_sk < Decimal('35') or g_spk < Decimal('35')):
        sc += 1
        signals_fired.append('e')

    # #f SuperTrend bullish NO flip
    if grid.get('supertrend_dir', False) and grid.get('supertrend_prev_dir', False):
        sc += 1
        signals_fired.append('f')

    # #g VO > +10%
    vo_val = grid.get('vo', Decimal('0'))
    if vo_val > CONFIG.vo_threshold:
        sc += 1
        signals_fired.append('g')

    # #h 1h close > SMA(20)
    if grid.get('bullish', False):
        sc += 1
        signals_fired.append('h')

    # #i 5m 3-candle higher lows
    if grid.get('higher_lows', False):
        sc += 1
        signals_fired.append('i')

    # #j MACD histogram expanding
    hist_cur  = grid.get('macd_hist', Decimal('0'))
    hist_prev = grid.get('macd_hist_prev', Decimal('0'))
    if hist_cur > hist_prev and hist_cur > Decimal('0'):
        sc += 1
        signals_fired.append('j')

    # #k Volume surge bonus
    if kd:
        r_vol_hist = kd.get('vol_history', [])
        if len(r_vol_hist) >= 5:
            r_avg = sum(r_vol_hist[:-1]) / (len(r_vol_hist) - 1)
            if kd.get('last_vol', Decimal('0')) >= r_avg:
                sc += 1
                signals_fired.append('k')

    return sc, signals_fired, emc_score

def market_pulse():
    while not state.shutdown_flag.is_set():
        try:
            cycle_start = time.time()
            temp_grids = {}
            with state.lock:
                symbols_snapshot = list(state.symbols)
            
            for s in symbols_snapshot:
                try:
                    prev = state.grid_data.get(s, {})
                    ctx = get_market_context(s, prev)
                    if ctx: 
                        temp_grids[s] = ctx
                        # Seed live_prices if missing or zero
                        with state.lock:
                            if state.live_prices.get(s, Decimal('0')) == Decimal('0'):
                                state.live_prices[s] = ctx['price_cur']
                except Exception as sym_err:
                    add_log(f"Pulse error [{s}]: {sym_err}", "dim red")
            
            with state.lock:
                state.grid_data.update(temp_grids)
            
            bulls = sum(1 for g in temp_grids.values() if g.get('bullish'))
            add_log(f"PULSE: {len(temp_grids)} pairs updated ({bulls}B/{len(temp_grids)-bulls}S)", "dim white")
            
            elapsed = time.time() - cycle_start
            time.sleep(max(0.0, 60.0 - elapsed))
        except Exception as e:
            add_log(f"market_pulse error: {e}", "red")
            time.sleep(5)

def perform_market_scan():
    """Consolidated pair discovery and validation logic."""
    try:
        now = time.time()
        # Check for stale pairs to blacklist (no valid data in 1 hour)
        stale_pairs = []
        for sym, last_valid in list(state.pair_last_valid.items()):
            if now - last_valid > 3600 and sym not in state.blacklist:
                state.blacklist.add(sym)
                stale_pairs.append(sym)
        if stale_pairs:
            add_log(f"BLACKLIST: {','.join(stale_pairs)}", "red")

        ticker = requests.get(f"{CONFIG.API_BASE}/api/v3/ticker/24hr", timeout=10).json()
        ticker.sort(key=lambda x: float(x.get('quoteVolume', 0) or 0), reverse=True)
        candidates = list(dict.fromkeys(
            t['symbol'] for t in ticker
            if t['symbol'].endswith("USDT")
            and t['symbol'].replace("USDT", "") not in CONFIG.EXCLUDED_BASES
        ))
        candidates = [s for s in candidates if s not in state.blacklist]
        
        info = requests.get(f"{CONFIG.API_BASE}/api/v3/exchangeInfo", timeout=10).json()
        valid_symbols = {s['symbol'] for s in info['symbols'] if s['status'] == 'TRADING'}
        candidates = [s for s in candidates if s in valid_symbols]
        
        new_symbols = []
        for sym in candidates:
            if len(new_symbols) >= CONFIG.max_pairs:
                break
            if validate_pair(sym):
                new_symbols.append(sym)
                state.pair_last_valid[sym] = time.time()
            time.sleep(0.1)
        
        new_filters = {}
        for s_info in info['symbols']:
            if s_info['symbol'] in new_symbols:
                flts = {filt['filterType']: filt for filt in s_info['filters']}
                min_notional = flts.get('NOTIONAL', {}).get('minNotional') or flts.get('MIN_NOTIONAL', {}).get('minNotional', '10.0')
                new_filters[s_info['symbol']] = {
                    'stepSize':    Decimal(flts['LOT_SIZE']['stepSize']),
                    'minNotional': Decimal(min_notional)
                }
        
        with state.lock:
            added   = [s for s in new_symbols if s not in state.symbols]
            removed = [s for s in state.symbols if s not in new_symbols]
            # Ensure open positions are always in the monitored list
            for pos_sym in state.positions:
                if pos_sym not in new_symbols:
                    new_symbols.append(pos_sym)
            state.symbols = new_symbols
            state.filters.update(new_filters)
            for s in removed:
                state.kline_data.pop(s, None)
                state.grid_data.pop(s, None)
            if added:   add_log(f"SCAN +{','.join(added)}", "dim cyan")
            if removed: add_log(f"SCAN -{','.join(removed)}", "dim yellow")
            if (added or removed) and state.ws_ref:
                try: state.ws_ref.close()
                except Exception: pass
            
            add_log(f"SCAN: Refreshed {len(new_symbols)} total pairs.", "dim cyan")
    except Exception as e:
        add_log(f"Scan error: {e}", "red")

def pair_scanner():
    time.sleep(CONFIG.rescan_interval)
    while not state.shutdown_flag.is_set():
        perform_market_scan()
        time.sleep(CONFIG.rescan_interval)

# ------------------------------------------
# TRADE ENGINE — VIP PRO 10X SYSTEM
# ------------------------------------------
def _get_dynamic_slippage(symbol, price, is_entry=True):
    """Scale slippage based on ATR volatility for realism."""
    grid = state.grid_data.get(symbol, {})
    atr = grid.get('atr', Decimal('0'))
    base_slip = CONFIG.slippage_pct
    if atr > Decimal('0') and price > Decimal('0'):
        atr_pct = (atr / price) * 100
        # Scale: ATR 2% = 1x base, ATR 6%+ = 3x base
        mult = max(Decimal('1'), min(Decimal('3'), atr_pct / Decimal('2')))
        base_slip = base_slip * mult
    
    if is_entry:
        return price * (Decimal('1') + base_slip)
    else:
        return price * (Decimal('1') - base_slip)

def _get_correlation_multiplier(symbol):
    """Reduces size if we already have positions in the same asset group."""
    group_name = None
    for name, symbols in CONFIG.GROUPS.items():
        if symbol in symbols:
            group_name = name
            break
    
    if not group_name:
        return Decimal('1.0') # Not in a tracked group, normal size
    
    count = 0
    for pos_sym in state.positions:
        if pos_sym in CONFIG.GROUPS.get(group_name, set()):
            count += 1
            
    if count >= 2:
        return Decimal('0')   # Block new entries (Max 2 per group)
    elif count == 1:
        return Decimal('0.5') # Half size for the 2nd trade in group
    return Decimal('1.0')     # Full size for the 1st trade in group

def _full_exit(sym, price, reason):
    """Full exit with R-multiple and performance tracking."""
    p = state.positions[sym]
    # Apply Dynamic Slippage to Exit
    slippage_price = _get_dynamic_slippage(sym, price, is_entry=False)
    val = p['qty'] * slippage_price
    remaining_profit = val - (p['qty'] * p['orig_entry'])
    state.balance      += val
    state.realized_pnl += remaining_profit
    total_proceeds = p.get('partial_proceeds', Decimal('0')) + val
    total_cost     = p['orig_qty'] * p['orig_entry']
    total_pnl      = total_proceeds - total_cost
    # R-multiple: 1R = risk at entry
    one_r = p['orig_entry'] * CONFIG.stop_loss_pct * p['orig_qty']
    r_mult = total_pnl / one_r if one_r > Decimal('0') else Decimal('0')
    state.total_r_sum += r_mult
    if r_mult > state.best_win: state.best_win = r_mult
    _daily_check()
    state.daily_trades += 1
    sid = p.get('signal_id', '???')
    if total_pnl > Decimal('0'):
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
    fp = p.get('brain_fingerprint')
    if fp:
        hold_seconds = int(time.time() - fp.get('entry_time', time.time()))
        brain.record_trade(fp, float(r_mult), reason, hold_seconds)
        brain.update()
        brain.save()
    state.positions.pop(sym)
    state.save_state()

def _partial_exit(sym, pct_of_orig, reason, price, stoch_k):
    """Execute partial position exit."""
    p = state.positions[sym]
    f = state.filters.get(sym)
    # Apply Dynamic Slippage to Exit
    slippage_price = _get_dynamic_slippage(sym, price, is_entry=False)
    sell_qty = p['orig_qty'] * pct_of_orig
    if f:
        sell_qty = (sell_qty / f['stepSize']).quantize(Decimal('1'), rounding=ROUND_DOWN) * f['stepSize']
    sell_qty = min(sell_qty, p['qty'])
    if sell_qty <= Decimal('0'):
        return
    proceeds = sell_qty * slippage_price
    cost     = sell_qty * p['orig_entry']
    state.balance        += proceeds
    state.realized_pnl   += proceeds - cost
    p['partial_proceeds'] = p.get('partial_proceeds', Decimal('0')) + proceeds
    p['qty']             -= sell_qty
    # FIX: only arm BE from a partial if price is meaningfully above entry (>=1%)
    # Prevents instant BE loss when first partial fires at a tiny gain and price snaps back
    arm_be = not p.get('breakeven_active', False) and slippage_price >= p['orig_entry'] * Decimal('1.01')
    be_str = "+BE" if arm_be else ""
    if arm_be:
        p['breakeven_active'] = True
    pct_int = int(pct_of_orig * 100)
    sid = p.get('signal_id', '???')
    add_log(f"PART {pct_int}%{be_str} {sym} @{slippage_price:.4f} ST:{int(stoch_k)}", "yellow")

    # --- STEP-UP STOP ESCALATION ---
    # Milestones: [76, 84, 91, 96] (TP1, TP2, TP3, TP4)
    exits_fired = p.get('partial_exits', [])
    num_exits = len(exits_fired)
    
    # Calculate TP Prices (1R = 4.5% distance)
    one_r = p['orig_entry'] * CONFIG.stop_loss_pct
    tp1_px = p['orig_entry'] + one_r
    tp2_px = p['orig_entry'] + (one_r * Decimal('2.0'))
    tp3_px = p['orig_entry'] + (one_r * Decimal('3.5'))
    
    if num_exits == 1: # TP1 HIT
        p['dynamic_stop'] = p['orig_entry']
        add_log(f"STOP STEP-UP: BREAKEVEN #{sid} {sym}", "bold yellow")
    elif num_exits == 2: # TP2 HIT
        p['dynamic_stop'] = tp1_px
        add_log(f"STOP STEP-UP: TP1 LEVEL (+1R) #{sid} {sym}", "bold yellow")
    elif num_exits == 3: # TP3 HIT
        p['dynamic_stop'] = tp2_px
        add_log(f"STOP STEP-UP: TP2 LEVEL (+2R) #{sid} {sym}", "bold yellow")
    elif num_exits == 4: # TP4 HIT -> MOON MODE
        p['dynamic_stop'] = tp3_px
        p['moon_mode'] = True # Only ATR trailing stop remains
        add_log(f"MOON MODE ACTIVE: #{sid} {sym} [NO FIXED TARGETS]", "bold cyan")

    if p['qty'] <= Decimal('0') and not p.get('moon_mode', False):
        total_cost = p['orig_qty'] * p['orig_entry']
        total_pnl  = p['partial_proceeds'] - total_cost
        one_r = p['orig_entry'] * CONFIG.stop_loss_pct * p['orig_qty']
        r_mult = total_pnl / one_r if one_r > Decimal('0') else Decimal('0')
        state.total_r_sum += r_mult
        if r_mult > state.best_win: state.best_win = r_mult
        _daily_check()
        state.daily_trades += 1
        sid = p.get('signal_id', '???')
        state.positions.pop(sym)
        if total_pnl > Decimal('0'):  # FIX: was unconditional WIN; cascades 1+2 can close position at a loss
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

                # Dynamic volatility/direction adjustment
                # Aggregate market sentiment from grid_data
                bullish_count = 0
                total_for_direction = 0
                avg_atr_pct = Decimal('0')
                atr_count = 0
                for sym_check, g in state.grid_data.items():
                    if g:
                        total_for_direction += 1
                        if g.get('bullish', False):
                            bullish_count += 1
                        atr = g.get('atr', Decimal('0'))
                        price_cur = g.get('price_cur', Decimal('1'))
                        if atr > Decimal('0') and price_cur > Decimal('0'):
                            atr_pct = (atr / price_cur) * 100
                            avg_atr_pct += atr_pct
                            atr_count += 1
                # Direction multiplier: bullish market = bigger positions
                if total_for_direction > Decimal('0'):
                    bullish_ratio = Decimal(bullish_count) / Decimal(total_for_direction)
                    direction_mult = Decimal('1.0') + ((bullish_ratio - Decimal('0.5')) * Decimal('0.2'))
                else:
                    direction_mult = Decimal('1.0')
                # Volatility multiplier: higher ATR = smaller position
                market_regime = "STABLE"
                regime_threshold_adj = Decimal('0')
                if atr_count > 0:
                    avg_atr_pct = avg_atr_pct / Decimal(atr_count)
                    # ATR 2% = 1.0x, ATR 5%+ = 0.7x, ATR <1% = 1.2x
                    if avg_atr_pct >= Decimal('4'):
                        vol_mult = Decimal('0.75')
                        market_regime = "VOLATILE"
                        regime_threshold_adj = Decimal('-0.5') # Be more aggressive in volatility
                    elif avg_atr_pct <= Decimal('1.2'):
                        vol_mult = Decimal('1.2')
                        market_regime = "CHOP"
                        regime_threshold_adj = Decimal('1.0') # Be defensive in chop
                    else:
                        vol_mult = Decimal('1.0')
                        market_regime = "STABLE"
                
                state.market_regime = market_regime
                current_bet = current_bet * direction_mult * vol_mult
                current_bet = min(current_bet, CONFIG.alloc_heater)

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

                        # --- RETROACTIVE FIXES FOR EXISTING TRADES ---
                        # 1. Catch-up Stop Escalation (if TP was already hit)
                        exits_fired = p.get('partial_exits', [])
                        one_r = p['orig_entry'] * CONFIG.stop_loss_pct
                        if len(exits_fired) >= 3:
                             if p.get('dynamic_stop', Decimal('0')) < p['orig_entry'] + one_r:
                                 p['dynamic_stop'] = p['orig_entry'] + one_r
                                 add_log(f"RESCUE: Escalate stop to +1R (TP3) for #{p.get('signal_id')}", "dim cyan")
                        elif len(exits_fired) >= 2:
                             if p.get('dynamic_stop', Decimal('0')) < p['orig_entry']:
                                 p['dynamic_stop'] = p['orig_entry']
                                 add_log(f"RESCUE: Escalate stop to BE (TP2) for #{p.get('signal_id')}", "dim cyan")

                        # 2. Retroactive Confidence Calculation
                        fp = p.get('brain_fingerprint', {})
                        if fp and 'confidence' not in fp:
                             ws = fp.get('weighted_score', brain.get_weighted_score(fp.get('signals_fired', [])))
                             et = brain.get_threshold(s)
                             df = Decimal(str(ws)) - et
                             cp = int(60 + (df * Decimal('13.33')))
                             fp['confidence'] = min(100, max(0, cp))
                        # ---------------------------------------------
    
                        # --- STOCHASTIC PARTIALS (when in profit) ---
                        if price > p['orig_entry']:
                            for threshold, pct in CONFIG.partial_levels:
                                if stoch_k > Decimal(threshold) and threshold not in p.get('partial_exits', []):
                                    p.setdefault('partial_exits', []).append(threshold)
                                    if threshold == CONFIG.partial_levels[-1][0]:
                                        # TP4: sweep all remaining qty → clean WIN
                                        _full_exit(s, price, "TP4")
                                    else:
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
                            rem_frac = p['qty'] / p['orig_qty'] if p['orig_qty'] > Decimal('0') else Decimal('0')
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
                            rem_frac = p['qty'] / p['orig_qty'] if p['orig_qty'] > Decimal('0') else Decimal('0')
                            _partial_exit(s, rem_frac * Decimal('0.50'), "VO-CROSS", price, stoch_k)
                            if s not in state.positions:
                                continue
    
                        # CASCADE 3: Chandelier exit (live price vs. slow-moving 1h level)
                        chandelier = grid.get('chandelier', Decimal('0'))
                        if chandelier > Decimal('0') and price < chandelier:
                            _full_exit(s, price, "CHANDELIER")
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
                        # PROFIT BUFFER: only activate trail after price is 0.5% in profit
                        # This prevents the "jump" that causes premature BE exits
                        atr = grid.get('atr', Decimal('0'))
                        if atr > Decimal('0') and p['high'] > p['orig_entry'] * Decimal('1.005'):
                            trail_dist = atr * CONFIG.atr_trail_mult
                            min_dist = p['orig_entry'] * CONFIG.atr_trail_min_pct
                            max_dist = p['orig_entry'] * CONFIG.atr_trail_max_pct
                            trail_dist = max(min_dist, min(max_dist, trail_dist))
                            atr_trail = p['high'] - trail_dist
                            # Never move stop below the original stop or the dynamic milestone stop
                            current_stop = p.get('dynamic_stop', p['orig_entry'] * (Decimal('1') - CONFIG.stop_loss_pct))
                            atr_trail = max(atr_trail, current_stop)
                            
                            if price < atr_trail and price > p['orig_entry']:
                                _full_exit(s, price, "ATR-TRAIL")
                                continue
    
                        # CASCADE 7: Hard stop (dynamic milestone stop)
                        # Default to hard stop (4.5%) if dynamic_stop not yet set
                        stop_lvl = p.get('dynamic_stop', p['orig_entry'] * (Decimal('1') - CONFIG.stop_loss_pct))
                        if price <= stop_lvl:
                            _full_exit(s, price, "STOP-ESC")
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
                        stale    = (loss_pct > brain.stale_loss_pct_30m and age > 1800) or \
                                   (loss_pct > brain.stale_loss_pct_60m and age > CONFIG.max_hold_seconds)
                        if stale:
                            _full_exit(s, price, "STALE")
                            continue

                        # CASCADE 9: Volume-Fade Early Exit (#3 Recommendation)
                        # Detects when "fuel" leaves the market while in profit.
                        cur_vol = grid.get('last_vol', Decimal('0'))
                        entry_vol = p.get('entry_vol', Decimal('0'))
                        # Calculate profit in R-multiples for the trigger
                        one_r_px = p['orig_entry'] * CONFIG.stop_loss_pct
                        current_r = (price - p['orig_entry']) / one_r_px if one_r_px > 0 else Decimal('0')
                        
                        if current_r >= Decimal('0.5') and entry_vol > 0:
                            if cur_vol < entry_vol * Decimal('0.60'): # 40% Fade
                                rem_frac = p['qty'] / p['orig_qty'] if p['orig_qty'] > Decimal('0') else Decimal('0')
                                _partial_exit(s, rem_frac * Decimal('0.25'), "VOL-FADE", price, stoch_k)
                                if s not in state.positions: continue
    
                    # ======= ENTRY LOGIC: EMC CONFLUENCE ENGINE =======
                    else:
                        # --- Pre-flight gates (not signals) ---
                        if time.time() < state.cooldowns.get(s, 0): continue
                        if len(state.positions) >= CONFIG.max_positions: continue
                        if not grid.get('macro_bullish', True): continue  # 4h Trend Filter
                        
                        # Institutional Squeeze Gate: only fire on BBW expansion (explosive move)
                        # Ensure volatility is increasing relative to recent average
                        bbw = grid.get('bbw', Decimal('0'))
                        bbw_avg = grid.get('bbw_avg', Decimal('0'))
                        if bbw <= bbw_avg: continue # Block signals in a "squeeze" or contracting volatility
                        
                        open_val = sum(p2['qty'] * state.live_prices.get(s2, p2['entry'])
                                       for s2, p2 in state.positions.items())
                        if state.balance + open_val < state.peak_equity * CONFIG.drawdown_threshold: continue
    
                        # --- Signal counting (Consolidated) ---
                        sigs, signals_fired, emc_score = calculate_signals(grid, state.kline_data.get(s))
    
                        weighted_score = brain.get_weighted_score(signals_fired)
                        entry_threshold = brain.get_threshold(s) + regime_threshold_adj
    
                        # MANDATORY CORE SIGNALS: 'b' (EMA Trend) and 'g' (Volume Oscillator)
                        # Must be present to ensure high-quality, volume-backed trades.
                        is_core_present = 'b' in signals_fired and 'g' in signals_fired
                        is_toxic = brain.is_cluster_toxic(signals_fired)

                        if weighted_score < entry_threshold or not is_core_present or is_toxic:
                            if is_toxic:
                                add_log(f"BRAIN: Entry Blocked (Toxic Cluster) {s}", "dim red")
                            continue

                        # Confidence calculation (Grade A Signal Attribution)
                        # Scale: 60% base + scaled momentum up to 100%
                        # Range: threshold to threshold + 3.0 (Now requires 9.0+ for ULTRA)
                        diff_v = weighted_score - entry_threshold
                        conf_pct = int(60 + (diff_v * Decimal('10.0'))) 
                        conf_pct = min(100, max(0, conf_pct))
                        
                        conf_label = "STANDARD"
                        if conf_pct >= 90: conf_label = "ULTRA"
                        elif conf_pct >= 75: conf_label = "HIGH"

                        # Apply Dynamic Slippage to Entry
                        slippage_price = _get_dynamic_slippage(s, price, is_entry=True)

                        # FIX: pre-entry Chandelier check — skip if confirmed 1h close
                        # already below Chandelier level (would exit instantly after entry)
                        _chan = grid.get('chandelier', Decimal('0'))
                        if _chan > Decimal('0') and grid.get('price_cur', slippage_price) < _chan:
                            continue

                        # Build fingerprint for brain
                        ema_spread = (grid.get('ema_fast', Decimal('0')) - grid.get('ema_slow', Decimal('0'))) / grid.get('ema_slow', Decimal('1')) if grid.get('ema_slow', Decimal('0')) > Decimal('0') else Decimal('0')
                        rsi_v = grid.get('rsi', Decimal('50'))
                        sk = grid.get('stoch_k', Decimal('50'))
                        vo_v = grid.get('vo', Decimal('0'))
                        
                        fingerprint = {
                            'signals_fired': signals_fired,
                            'signal_score': sigs,
                            'weighted_score': float(weighted_score),
                            'confidence': conf_pct,
                            'rsi_at_entry': float(rsi_v),
                            'stoch_k_at_entry': float(sk),
                            'vo_at_entry': float(vo_v),
                            'ema_spread_at_entry': float(ema_spread),
                            'supertrend_at_entry': bool(grid.get('supertrend_dir', False)),
                            'pair': s,
                            'entry_time': time.time()
                        }
                        
                        corr_mult = _get_correlation_multiplier(s)
                        if corr_mult <= 0:
                            continue # Correlation limit reached for this group
                            
                        target = state.balance * current_bet * Decimal(str(brain.size_multiplier)) * corr_mult
                        if target >= f['minNotional']:
                            qty = (target / slippage_price / f['stepSize']).quantize(Decimal('1'), rounding=ROUND_DOWN) * f['stepSize']
                            if qty > Decimal('0'):
                                if qty * slippage_price > state.balance:
                                    continue
                                state.signal_counter += 1
                                sid = f"{state.signal_counter:03d}"
                                state.balance -= qty * slippage_price
                                state.positions[s] = {
                                    "qty": qty, "entry": slippage_price, "high": slippage_price, "ts": time.time(),
                                    "orig_qty": qty, "orig_entry": slippage_price,
                                    "partial_exits": [], "partial_proceeds": Decimal('0'),
                                    "breakeven_active": False, "vol_confirmed_be": False,
                                    "st_flip_acted": False, "signal_id": sid,
                                    "brain_fingerprint": fingerprint,
                                    "entry_vol": grid.get('last_vol', Decimal('0')),
                                }
                                state.kline_data.pop(s, None)
                                pct = int(current_bet * Decimal(str(brain.size_multiplier)) * Decimal('100'))
                                # Grade A Signal with Golden Pocket re-entry levels
                                gp_50, gp_618 = grid.get('fib_50', price), grid.get('fib_618', price)
                                sig_str = ",".join(signals_fired)
                                add_log(f"LONG #{sid} {s} @{slippage_price:.8g} [SIGS:{sig_str}] [CONF:{conf_pct}%]", "cyan")
                                add_log(f"RE-ENTRY (GP): {gp_50:.8g} - {gp_618:.8g} | Allocation: {pct}%", "dim cyan")
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
            f"{CONFIG.API_BASE}/api/v3/klines",
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
    # ---- Single-instance guard (socket lock — survives force-kill) ----
    import socket as _socket
    import atexit
    _lock_socket = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    try:
        _lock_socket.setsockopt(_socket.SOL_SOCKET, _socket.SO_EXCLUSIVEADDRUSE, 1)
    except (AttributeError, OSError):
        pass  # non-Windows fallback: bind alone provides exclusivity
    try:
        _lock_socket.bind(('127.0.0.1', 21487))
    except OSError:
        print("[ERROR] Gridzilla is already running. Kill it first.")
        return
    
    def cleanup():
        _lock_socket.close()
        try: _log_file.close()
        except: pass
        
    atexit.register(cleanup)
    # -----------------------------------------------

    add_log("Fetching top pairs by volume...", "yellow")
    perform_market_scan()
    
    # Initialize grid_data for symbols loaded by scan
    with state.lock:
        needed = set(state.symbols)
    
    for s in needed:
        ctx = get_market_context(s)
        if ctx:
            with state.lock:
                state.grid_data[s] = ctx
                if state.live_prices.get(s, Decimal('0')) == Decimal('0'):
                    state.live_prices[s] = ctx['price_cur']
    
    state.save_state()
    
    add_log("Starting threads...", "yellow")
    # Descriptive Brain Status
    brain_status = (
        f"Confidence Thresh:{brain.optimal_threshold:.1f} | "
        f"Bet Size Multiplier:{float(brain.size_multiplier):.2f}x | "
        f"Total Performance History:{brain.total_trades_seen} trades | "
        f"Greylisted(Poor Performance):{len(brain.greylisted_pairs)}"
    )
    add_log(f"BRAIN: {brain_status}", "dim white")

    def run_grid_scorer():
        from datetime import timezone
        now = datetime.now(timezone.utc)
        target_hour = 0
        target_minute = 1
        target = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
        if now >= target:
            target = target + timedelta(days=1)
        wait_seconds = (target - now).total_seconds()
        time.sleep(wait_seconds)
        
        while not state.shutdown_flag.is_set():
            grid_scorer.perform_scoring_cycle()
            time.sleep(86400)

    def run_grid_scorer_once():
        try:
            add_log("GRID SCORER: Waiting for market data...", "yellow")
            
            grid_count = 0
            wait_count = 0
            max_wait = 12
            while wait_count < max_wait:
                with state.lock:
                    grid_count = len(state.grid_data)
                if grid_count >= 5:
                    break
                wait_count += 1
                time.sleep(10)
            
            if grid_count < 5:
                add_log(f"GRID SCORER: Warning - only {grid_count} pairs have data", "yellow")
            
            grid_scorer.perform_scoring_cycle()
            add_log("GRID SCORER: Initial scan complete", "green")
        except Exception as e:
            add_log(f"GRID SCORER initial error: {e}", "red")

    threading.Thread(target=run_grid_scorer, daemon=True).start()
    threading.Thread(target=trade_executioner, daemon=True).start()
    threading.Thread(target=market_pulse,      daemon=True).start()
    threading.Thread(target=pair_scanner,      daemon=True).start()
    threading.Thread(target=run_grid_scorer_once, daemon=True).start()

    def heartbeat():
        while not state.shutdown_flag.is_set():
            try:
                state.save_state()
                brain.save()
            except Exception as e:
                add_log(f"heartbeat error: {e}", "red")
            time.sleep(30)
    threading.Thread(target=heartbeat, daemon=True).start()

    def ws_loop():
        while not state.shutdown_flag.is_set():
            try:
                def on_message(ws, m):
                    state.connection_ok = True
                    state.ws_tick += 1
                    msg = json.loads(m)
                    data = msg.get('data', msg)
                    
                    # Handle both array-based and single object miniTickers
                    items = data if isinstance(data, list) else [data]
                    
                    with state.lock:
                        for d in items:
                            if d.get('e') == '24hrMiniTicker':
                                sym = d['s']
                                if sym in state.symbols:
                                    new_p = Decimal(d['c'])
                                    old_p = state.live_prices.get(sym)
                                    state.live_prices[sym] = new_p
                                    if old_p is not None and new_p != old_p:
                                        state.price_dir[sym] = ('up', time.time()) if new_p > old_p else ('dn', time.time())

                def on_error(ws, err):
                    state.connection_ok = False
                    add_log(f"WS error: {err}", "red")

                def on_close(ws, code, msg):
                    state.connection_ok = False
                    add_log("WS closed, reconnecting...", "yellow")
                
                # Use global miniTicker stream from centralized WS_BASE
                url = f"{CONFIG.WS_BASE}/stream?streams=!miniTicker@arr"
                ws = WebSocketApp(
                    url,
                    on_message=on_message,
                    on_error=on_error,
                    on_close=on_close
                )
                state.ws_ref = ws  # type: ignore
                ws.run_forever()
            except Exception as e:
                add_log(f"WS exception: {e}", "red")
                time.sleep(5)
    threading.Thread(target=ws_loop, daemon=True).start()

    MELANIA = [
        [
         '     ♀     ',
         '    \\|/    ',
         '     |     ',
         '    /|\\    ',
         '   /   \\   ',
         "  '     '  ",
         'slay the bears'
        ],
        [
         ' \\   ♀   / ',
         '  \\  |  /  ',
         '   \\ | /   ',
         '    \\|/    ',
         '    / \\    ',
         '   /   \\   ',
         'catch these gains'
        ],
        [
         '     ♀     ',
         '    /|     ',
         '     |     ',
         '    / `--/ ',
         '   /       ',
         '           ',
         'kicked the resistance'
        ],
        [
         '     ♀     ',
         '     |\\    ',
         '     |     ',
         " \\--' \\   ",
         '       \\   ',
         '           ',
         'karate chopped support'
        ],
        [
         '  ♀~~~~~~~~',
         '  |        ',
         '  |        ',
         '  \\        ',
         '   \\       ',
         '   /\\      ',
         'bearish on gravity'
        ],
        [
         '  ~~♀~~    ',
         ' ~ /|\\ ~   ',
         '     |     ',
         '    / \\    ',
         '   /   \\   ',
         '           ',
         'going parabolic'
        ],
        [
         '     ♀     ',
         '    /|\\    ',
         '   / | \\   ',
         '     |     ',
         '   ~~|~~   ',
         '           ',
         'to the moon ser'
        ],
        [
         '           ',
         '     ♀     ',
         '    /|\\    ',
         '  -/ | \\-  ',
         ' /   |   \\ ',
         '           ',
         'loading next setup'
        ],
        [
         '     ♀     ',
         '   --|--   ',
         '     |     ',
         '    / \\    ',
         '   /   \\   ',
         "  '     '  ",
         'entry confirmed'
        ],
        [
         '  \\  ♀  /  ',
         '   \\ | /   ',
         '    \\|/    ',
         '     |     ',
         '    / \\    ',
         '   /   \\   ',
         'green candles only'
        ],
        [
         '       //  ',
         '     ♀/    ',
         '    /|     ',
         '   / \\     ',
         "  '   '    ",
         '           ',
         'waving bye to shorts'
        ],
        [
         '           ',
         '           ',
         '     ♀     ',
         '    /|\\    ',
         '---/ | \\---',
         '           ',
         'support zone located'
        ],
        [
         '    ~♀~    ',
         '   /~|~\\   ',
         '  /~ | ~\\  ',
         '     |     ',
         '    / \\    ',
         "   '   '   ",
         'ATH incoming'
        ],
        [
         ' //  ♀  \\\\ ',
         ' //  |  \\\\ ',
         '     |     ',
         '    /|\\    ',
         '   / | \\   ',
         '           ',
         'profit taking time'
        ],
        [
         '     ♀     ',
         '    /|     ',
         '   / |\\    ',
         '  /  | \\   ',
         ' /   |  \\  ',
         "'         '",
         'thank u bulls'
        ],
        [
         '     ♀     ',
         '     |\\    ',
         '     | \\   ',
         '     |  \\  ',
         '    /    \\ ',
         '   /      \\',
         'riding the trend'
        ],
        [
         '     ♀     ',
         '     |     ',
         '     |     ',
         '/----+----\\',
         '           ',
         '           ',
         'found my entry point'
        ],
        [
         '     ♀     ',
         '    >|     ',
         '     |     ',
         '    / ->   ',
         '   /       ',
         '           ',
         'chasing the breakout'
        ],
        [
         '♀~~~~~~~~~~',
         '\\|         ',
         ' \\         ',
         '  \\        ',
         '   \\       ',
         '           ',
         'proof of twerk'
        ],
        [
         '     ♀     ',
         '    /|     ',
         '     |\\    ',
         '    /  \\   ',
         '   <    >  ',
         '           ',
         'i am the liquidity'
        ],
        [
         '       ♀   ',
         '      /|   ',
         '       |   ',
         ' <----/ \\  ',
         '          \\',
         '           ',
         'moonwalking ur stop'
        ],
        [
         '     ♀     ',
         '     |     ',
         '  ~~~|~~~  ',
         ' /   |   \\ ',
         '|    |    |',
         ' \\---+---/ ',
         'tyvm for the liquidity'
        ],
        [
         '     ♀     ',
         '    ~/     ',
         '   / ~     ',
         '  /        ',
         ' /         ',
         ' ~~        ',
         'dodge the liquidation'
        ],
        [
         '  \\  ♀  /  ',
         '   \\ | /   ',
         '    -+-    ',
         '    | |    ',
         '   /   \\   ',
         '  /     \\  ',
         'rug proof certified'
        ],
    ]

    layout = Layout()
    layout.split_column(Layout(name="h", size=3), Layout(name="m", ratio=1), Layout(name="f", size=13))
    layout["m"].split_row(Layout(name="mkt"), Layout(name="pos"))
    layout["f"].split_row(
        Layout(name="logs", ratio=1), 
        Layout(name="grid1", size=30),
        Layout(name="grid2", size=30),
        Layout(name="grid3", size=30)
    )

    add_log("GRIDZILLA VIP PRO online. Paper trading active.", "bold green")

    PULSE = ['⣾','⣽','⣻','⢿','⡿','⣟','⣯','⣷']

    is_tty = sys.stdout.isatty()
    if not is_tty:
        add_log("Non-TTY environment detected. Running in headless mode.", "dim yellow")
        while not state.shutdown_flag.is_set():
            time.sleep(10)
        return

    with Live(layout, console=console, refresh_per_second=CONFIG.TUI_FPS, screen=console.is_terminal):
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
                snap_regime    = state.market_regime

            open_val   = sum(p['qty'] * snap_prices.get(s, p['entry']) for s, p in snap_positions.items())
            cost_basis = sum(p['qty'] * p['entry'] for p in snap_positions.values())
            unrealized = open_val - cost_basis
            total_val  = snap_balance + open_val
            p_style    = "green" if snap_pnl    >= Decimal('0') else "red"
            u_style    = "green" if unrealized  >= Decimal('0') else "red"
            total_trades = snap_wins + snap_losses
            wr_pct     = int(snap_wins / total_trades * 100) if total_trades else 0
            pf         = float(snap_gw / snap_gl) if snap_gl > Decimal('0') else Decimal('0.0')
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

            regime_color = "yellow" if snap_regime == "CHOP" else "cyan" if snap_regime == "VOLATILE" else "white"
            
            layout["h"].update(Panel(
                Align.center(
                    f"[bold yellow]GRIDZILLA VIP PRO[/]  [dim]|[/]  "
                    f"[bold white]${total_val:,.2f}[/]  "
                    f"PnL:[{p_style}]{snap_pnl:+.2f}[/]  [dim]|[/]  "
                    f"[bold green]{wr_pct}%[/] WR  "
                    f"[green]{snap_wins}[/]W [red]{snap_losses}[/]L  [dim]|[/]  "
                    f"Regime:[{regime_color}]{snap_regime}[/]  [dim]|[/]  "
                    f"[bold white]{snap_streak}x[/] [dim]({cur_alloc}%)[/]  [dim]|[/]  "
                    f"Pos:[bold]{len(snap_positions)}[/]{dd_str}  [dim]|[/]  "
                    f"[dim]{up_str}[/]  "
                    f"[{'green' if snap_conn else 'red'}]{pulse}[/]"
                ),
                border_style="dim blue"
            ))

            # --- RADAR TABLE ---
            m_tab = Table(box=box.HORIZONTALS, expand=True, show_edge=False, border_style="dim blue",
                          header_style="bold cyan")
            m_tab.add_column("TRADING PAIR", no_wrap=True, min_width=12)
            m_tab.add_column("LIVE PRICE",   no_wrap=True, justify="right")
            m_tab.add_column("4H TREND / VOL", no_wrap=True, justify="center")
            m_tab.add_column("LOWER / FLOOR",  style="dim",    no_wrap=True, justify="right")
            m_tab.add_column("UPPER / CEIL",   style="dim",    no_wrap=True, justify="right")
            m_tab.add_column("RSI",   no_wrap=True,   justify="right")
            m_tab.add_column("STOCH", no_wrap=True,   justify="right")
            m_tab.add_column("VOL %", no_wrap=True,   justify="right")
            m_tab.add_column("SIGNAL STATUS", no_wrap=True,   justify="right")
            def _fmt(v):
                f = float(v)
                if f == 0:       return "0.0000"
                if f < 0.0001:   return f"{f:.8f}"
                if f < 0.01:     return f"{f:.6f}"
                return f"{f:.4f}"
            for s in snap_symbols:
                grid  = snap_grid.get(s)
                price = snap_prices.get(s, Decimal('0'))

                # Dynamic Symbol Coloring (Categorized) - Deep Saturated Palette
                MEMES = {"DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "PUMPUSDT", "PENGUUSDT", "BONKUSDT", "FLOKIUSDT"}
                L1S   = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "BNBUSDT", "AVAXUSDT", "LTCUSDT", "XRPUSDT", "DOTUSDT", "MATICUSDT", "ATOMUSDT", "NEARUSDT"}
                AI_INFRA = {"RENDERUSDT", "GRTUSDT", "FETUSDT", "LINKUSDT", "UNIUSDT"}
                
                s_style = "white"
                if s in MEMES: s_style = "blue"
                elif s in L1S: s_style = "yellow"
                elif s in AI_INFRA: s_style = "green"
                s_disp = f"[{s_style}]{s}[/]"

                if grid:
                    rsi_val  = grid.get('rsi', Decimal('50'))
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
                    macro_val = grid.get('macro_bullish', True)
                    macro_disp = "[green]BULL[/]" if macro_val else "[red]BEAR[/]"
                    
                    # Consolidated Signal Calculation (Radar Column)
                    sc, signals_fired, r_emc = calculate_signals(grid, snap_kline.get(s))
                    
                    # Grade A Signal Attribution
                    weighted_score = brain.get_weighted_score(signals_fired)
                    entry_threshold = brain.get_threshold(s)
                    
                    is_core = 'b' in signals_fired and 'g' in signals_fired
                    
                    diff_v = weighted_score - entry_threshold
                    conf_pct = int(60 + (diff_v * Decimal('10.0'))) 
                    conf_pct = min(100, max(0, conf_pct))
                    
                    in_cd = time.time() < snap_cooldowns.get(s, 0)
                    cd_str = " [red]CD[/]" if in_cd else ""
                    emc_str = f"[cyan]E{r_emc}[/]" if r_emc >= 3 else f"[dim]e{r_emc}[/]"
                    
                    # Squeeze detection for Radar
                    bbw = grid.get('bbw', Decimal('0'))
                    bbw_avg = grid.get('bbw_avg', Decimal('0'))
                    sqz_disp = "[bold magenta]SQZ[/]" if bbw <= bbw_avg else "[dim]EXP[/]"

                    if not macro_val:
                        sig = f"[dim red]TREND {sc}/11[/]" # Blocked by 4h filter
                    elif not is_core:
                        core_missing = []
                        if 'b' not in signals_fired: core_missing.append("T")
                        if 'g' not in signals_fired: core_missing.append("V")
                        sig = f"[dim yellow]CORE({'+'.join(core_missing)}) {sc}/11[/]" 
                    elif weighted_score >= entry_threshold:
                        grade = "ULTRA" if conf_pct >= 90 else "HIGH" if conf_pct >= 75 else "STD"
                        sig = f"[bold green]{grade} {sc}/11[/] {emc_str}{cd_str}"
                    elif sc >= 3:
                        sig = f"[yellow]{sc}/11[/] {emc_str}{cd_str}"
                    else:
                        sig = f"[dim]{sc}/11 {emc_str}[/]{cd_str}"
                    floor_str = f"[bold yellow]{_fmt(grid['floor'])}[/]" if price <= grid['floor'] else _fmt(grid['floor'])
                    ceil_str  = _fmt(grid['ceil'])
                    
                    # Add Squeeze to Macro column for space
                    macro_disp = f"{macro_disp} {sqz_disp}"
                else:
                    rsi_disp = macro_disp = floor_str = ceil_str = sig = st_disp = vo_disp = "[dim]...[/]"
                d_info = snap_dirs.get(s)
                now = time.time()
                if d_info and now - d_info[1] < 1.5:
                    arrow = "[bold green]^[/]" if d_info[0] == 'up' else "[bold red]v[/]"
                    p_str = f"{arrow}{_fmt(price)}"
                else:
                    p_str = f" {_fmt(price)}"
                m_tab.add_row(s_disp, p_str, macro_disp, floor_str, ceil_str, rsi_disp, st_disp, vo_disp, sig)
            layout["mkt"].update(Panel(m_tab, title="[dim cyan]Radar[/]", border_style="dim blue"))

            # --- SIGNAL CARDS ---
            card_lines = []
            now = time.time()
            for s, p in snap_positions.items():
                cur     = snap_prices.get(s, p['entry'])
                diff    = (cur - p['entry']) / p['entry']
                usd_pnl = (cur - p['entry']) * p['qty']
                age     = int(now - p['ts'])
                c       = 'green' if diff >= Decimal('0') else 'red'
                sid     = p.get('signal_id', '???')
                rem_pct = int(p['qty'] / p.get('orig_qty', p['qty']) * 100) if p.get('orig_qty', p['qty']) > Decimal('0') else 100
                # direction indicator
                d_info = snap_dirs.get(s)
                arr = ("^" if d_info[0] == 'up' else "v") if (d_info and now - d_info[1] < 1.5) else " "
                # ATR trail stop level
                g = snap_grid.get(s, {})
                atr = g.get('atr', Decimal('0'))
                trail_str = ""
                if atr > Decimal('0') and p['high'] > p['orig_entry']:
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
                
                # Step-Up Stop Display
                stop_val = p.get('dynamic_stop', p['orig_entry'] * (Decimal('1') - CONFIG.stop_loss_pct))
                stop_lvl = _fmt(stop_val)
                
                # Moon Mode Indicator
                moon_str = " [bold cyan][MOON][/]" if p.get('moon_mode') else ""
                
                # Financials
                usd_invested = p['orig_qty'] * p['orig_entry']
                one_r_usd = usd_invested * CONFIG.stop_loss_pct
                total_pnl = (p.get('partial_proceeds', Decimal('0')) + (p['qty'] * cur)) - usd_invested
                r_multiple = total_pnl / one_r_usd if one_r_usd > 0 else Decimal('0')
                
                fp = p.get('brain_fingerprint', {})
                conf_val = fp.get('confidence', 0)
                conf_label = "ULTRA" if conf_val >= 90 else "HIGH" if conf_val >= 75 else "STD"
                conf_str = f"  [bold cyan]CONF:{conf_val}% {conf_label}[/]"
                sigs_at_entry = ",".join(fp.get('signals_fired', []))
                
                card_body = (
                    f"[bold cyan]SIGNAL #{sid}[/]  [bold white]{s}[/]  [dim]{rem_pct}% REM[/]{conf_str}{moon_str}\n"
                    f"[dim]────────────────────────────────────────────────────────────────────────[/]\n"
                    f" [bold white]PROFIT:[/]  [{c}]{diff:+.2%}[/]  [{c}]${float(total_pnl):+.2f}[/]  [bold {c}]({float(r_multiple):+.1f}R)[/]\n"
                    f" [bold white]MARKET:[/]  Entry:[dim]{_fmt(p['entry'])}[/]  Live:[{c}]{_fmt(cur)}[/]{arr}  Age:[dim]{age//60}m{age%60:02d}s[/]\n"
                    f" [bold white]SAFETY:[/]  Stop:[bold yellow]{stop_lvl}[/]{trail_str}{be_str}\n"
                    f"[dim]────────────────────────────────────────────────────────────────────────[/]\n"
                    f" [bold white]TARGETS:[/] {tp_line}\n"
                    f" [bold white]CONTEXT:[/] [dim]StochK:[/] {int(stoch_k)}  [dim]Signals fired at entry:[/] [blue]{sigs_at_entry}[/]"
                )
                card_lines.append(Panel(card_body, border_style="dim green" if diff >= 0 else "dim red", padding=(0, 1)))
            if card_lines:
                layout["pos"].update(Panel(RichGroup(*card_lines), title="[dim green]Open Trades[/]", border_style="dim blue"))
            else:
                layout["pos"].update(Panel("[dim]No open trades[/]", title="[dim green]Open Trades[/]", border_style="dim blue"))
            layout["logs"].update(Panel("\n".join(snap_logs), title="[dim cyan]Logs[/]", border_style="dim cyan"))

            # --- DYNAMIC GRID ADVISOR (3 CATEGORIES - DAILY PICKS) ---
            def _make_grid_panel(category, category_label, title_color, border_color):
                pick = snap_grid_picks.get(category)
                
                if not pick:
                    body = Align.center(
                        f"\n\n[dim]Waiting for daily pick...[/]\n"
                        f"[dim]Scans at 00:01 UTC[/]",
                        vertical="middle"
                    )
                    return Panel(body, title=f"[{title_color}] {category_label} [/]", border_style=border_color, padding=(0,1))
                
                symbol = pick.get('symbol', 'N/A')
                score = pick.get('score', 0)
                details = pick.get('details', {})
                settings = pick.get('settings', {})
                timestamp = pick.get('timestamp', 0)
                
                grade = "A" if score >= 90 else "B" if score >= 75 else "C"
                grade_color = "bold green" if score >= 90 else "green" if score >= 75 else "yellow"
                
                price = settings.get('price', 0)
                upper = settings.get('upper', 0)
                lower = settings.get('lower', 0)
                grids = settings.get('grids', 17)
                profit_per = settings.get('profit_per_grid', 0)
                days_low = settings.get('days_low', 0)
                days_high = settings.get('days_high', 0)
                atr_pct = settings.get('atr_pct', 0)
                
                if price > 0:
                    upper_pct = ((upper - price) / price) * 100
                    lower_pct = ((price - lower) / price) * 100
                else:
                    upper_pct = lower_pct = 0
                
                confidence = details.get('confidence', 0)
                vol_rank = details.get('volume_rank', 999)
                trend = details.get('trend', 'N/A')
                signal_score = details.get('signal_score', 0)
                
                time_since = time.time() - timestamp
                hours_remaining = int(86400 - time_since) // 3600
                mins_remaining = ((86400 - time_since) % 3600) // 60
                
                updated_str = datetime.utcfromtimestamp(timestamp).strftime('%H:%M') if timestamp > 0 else "N/A"
                
                body = (
                    f"[bold white]*{symbol}[/] [bold {grade_color}]{grade}[/] [dim]({score:.0f})[/]\n"
                    f"[dim]Price: [/][white]${price:,.8g}[/]\n"
                    f"[dim]Upper: [/][green]${upper:,.8g}[/] [dim](+{upper_pct:.0f}%)[/]\n"
                    f"[dim]Lower: [/][red]${lower:,.8g}[/] [dim](-{lower_pct:.0f}%)[/]\n"
                    f"[dim]Grid:  [/][white]{grids} ({profit_per:.2f}%)[/]\n"
                    f"[dim]Days:  [/][white]{days_low}-{days_high}[/]\n"
                    f"[dim]Conf:  [/][white]{confidence}%[/] [dim]VolRank:#{vol_rank}[/]\n"
                    f"[dim]Sig:   [/][white]{signal_score}/11[/] [dim]{trend}[/]\n"
                    f"[dim]Upd:   [/][white]{updated_str}[/] [dim]Next:{hours_remaining}h{mins_remaining}m[/]"
                )
                return Panel(body, title=f"[{title_color}] {category_label} [/]", border_style=border_color, padding=(0,1))

            with state.lock:
                snap_grid_picks = dict(state.grid_picks)

            layout["grid1"].update(_make_grid_panel("LARGE CAP", "LARGE CAP", "bold yellow", "bold yellow"))
            layout["grid2"].update(_make_grid_panel("MEME", "MEME", "bold blue", "bold blue"))
            layout["grid3"].update(_make_grid_panel("ALT COIN", "ALT COIN", "bold green", "bold green"))

            time.sleep(0.1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        state.shutdown_flag.set()
        console.print("\n[yellow]Shutting down GRIDZILLA VIP PRO...[/]")
        state.save_state()
        console.print("[green]State saved. Goodbye.[/]")
