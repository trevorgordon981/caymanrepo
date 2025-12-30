#!/usr/bin/env python3
"""
STOCK TICKER APP v7.7.0 — Live Portfolio Tracking
INSTALL: pip install yfinance pandas numpy matplotlib tabulate colorama requests scipy
"""
from __future__ import annotations
import os, re, sys, json, time, math, shlex, logging, warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import requests
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("requests").setLevel(logging.CRITICAL)
import yfinance as yf

try:
    from scipy.stats import norm
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except:
    HAS_MATPLOTLIB = False

try:
    from tabulate import tabulate
except ImportError:
    def tabulate(data, headers=None, tablefmt="simple"):
        if not data: return ""
        lines = []
        if headers:
            lines.append(" | ".join(str(h) for h in headers))
            lines.append("-" * 80)
        for row in data:
            lines.append(" | ".join(str(c) for c in row))
        return "\n".join(lines)

try:
    from colorama import init as _ci, Fore, Style
    _ci(autoreset=True)
except ImportError:
    class Fore:
        GREEN = RED = YELLOW = MAGENTA = CYAN = WHITE = BLUE = RESET = ""
    class Style:
        RESET_ALL = BRIGHT = ""

COMPANY_TO_TICKER = {
    'PURECYCLE TECHNOLOGIES INC COM': 'PCT', 'PURECYCLE TECHNOLOGIES INC': 'PCT',
    'APPLE INC COM': 'AAPL', 'APPLE INC': 'AAPL',
    'MICROSOFT CORP COM': 'MSFT', 'MICROSOFT CORP': 'MSFT',
    'TESLA INC COM': 'TSLA', 'TESLA INC': 'TSLA',
    'NVIDIA CORP COM': 'NVDA', 'NVIDIA CORP': 'NVDA',
    'AMAZON COM INC COM': 'AMZN', 'AMAZON COM INC': 'AMZN',
    'ALPHABET INC CL A': 'GOOGL', 'ALPHABET INC CL C': 'GOOG',
    'META PLATFORMS INC CL A': 'META', 'META PLATFORMS INC': 'META',
    'PALANTIR TECHNOLOGIES INC CL A': 'PLTR', 'PALANTIR TECHNOLOGIES INC': 'PLTR',
    'ADVANCED MICRO DEVICES INC': 'AMD',
    'INTEL CORP COM': 'INTC', 'INTEL CORP': 'INTC',
    'COINBASE GLOBAL INC CL A': 'COIN', 'ROBINHOOD MARKETS INC CL A': 'HOOD',
    'SOFI TECHNOLOGIES INC COM': 'SOFI', 'SOFI TECHNOLOGIES INC': 'SOFI',
    'ROCKET LAB USA INC COM': 'RKLB', 'ROCKET LAB USA INC': 'RKLB',
    'ARCHER AVIATION INC CL A': 'ACHR', 'JOBY AVIATION INC': 'JOBY',
    'AST SPACEMOBILE INC CL A': 'ASTS', 'OKLO INC CL A': 'OKLO',
    'CARVANA CO CL A': 'CVNA', 'UBER TECHNOLOGIES INC COM': 'UBER',
    'REDDIT INC CL A': 'RDDT', 'MICRON TECHNOLOGY INC COM': 'MU',
}

def _now(): return time.time()
def today_str(): return datetime.now().strftime("%Y-%m-%d")

def safe_float(val, default=0.0):
    if val is None: return default
    if isinstance(val, (int, float)):
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)): return default
        return float(val)
    try:
        s = str(val).strip().replace(",", "").replace("$", "").replace('"', "").replace("%", "")
        if s in ("", "--", "N/A", "nan", "None", "-", "NaN"): return default
        if s.startswith("(") and s.endswith(")"): s = "-" + s[1:-1]
        return float(s)
    except: return default

def fmt_money(x, sign=False): return f"${x:+,.2f}" if sign else f"${x:,.2f}"
def fmt_pct(x, sign=False): return f"{x:+.2f}%" if sign else f"{x:.2f}%"
def color_money(x, sign=True): return (Fore.GREEN if x >= 0 else Fore.RED) + fmt_money(x, sign=sign) + Style.RESET_ALL
def color_pct(x, sign=True): return (Fore.GREEN if x >= 0 else Fore.RED) + fmt_pct(x, sign=sign) + Style.RESET_ALL
def color_pnl(pnl, pnl_pct): return f"{color_money(pnl)} ({color_pct(pnl_pct)})"

def color_signal(signal):
    s = str(signal).upper()
    if "BUY" in s or "BULL" in s or "OVERSOLD" in s: return Fore.GREEN + s + Style.RESET_ALL
    if "SELL" in s or "BEAR" in s or "OVERBOUGHT" in s: return Fore.RED + s + Style.RESET_ALL
    return Fore.YELLOW + s + Style.RESET_ALL

def normalize_symbol(s):
    s = str(s).strip().upper()
    s = re.sub(r"[^A-Z0-9\.\-\^\=\:]", "", s)
    return s.replace(".", "-") if re.match(r"^[A-Z]+\.[A-Z]$", s) else s

def parse_date(d):
    d = str(d).strip()
    for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%Y%m%d"]:
        try: return datetime.strptime(d, fmt).strftime("%Y-%m-%d")
        except: pass
    return d

def dte(expiry):
    try:
        exp_dt = datetime.strptime(str(expiry), "%Y-%m-%d")
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        return (exp_dt - today).days
    except: return 0

def warn(msg): print(Fore.YELLOW + "⚠ " + msg + Style.RESET_ALL)
def err(msg): print(Fore.RED + "✗ " + msg + Style.RESET_ALL)
def success(msg): print(Fore.GREEN + "✓ " + msg + Style.RESET_ALL)

class ProgressBar:
    def __init__(self, total, prefix="", width=30):
        self.total = max(total, 1)
        self.prefix = prefix
        self.width = width
        self.current = 0
        self.start = _now()
        self._last = 0
    def update(self, current=None, suffix=""):
        self.current = current if current is not None else self.current + 1
        pct = min(self.current / self.total, 1.0)
        filled = int(self.width * pct)
        bar = "█" * filled + "░" * (self.width - filled)
        elapsed = _now() - self.start
        eta = f"{elapsed/pct*(1-pct):.0f}s" if 0 < pct < 1 else ""
        line = f"\r{self.prefix}|{bar}| {self.current}/{self.total} {eta} {suffix[:15]}"
        print(line + " " * max(0, self._last - len(line)), end="", flush=True)
        self._last = len(line)
    def done(self):
        elapsed = _now() - self.start
        print(f"\r{self.prefix}|{'█'*self.width}| {self.total}/{self.total} done in {elapsed:.1f}s" + " "*20)

@dataclass
class Config:
    data_dir: str = field(default_factory=lambda: os.path.expanduser("~/.stockticker"))
    risk_free_rate: float = 0.043
    theme: str = "dark"
    debug: bool = False
    show_charts: bool = True
    def __post_init__(self): Path(self.data_dir).mkdir(parents=True, exist_ok=True)
    @classmethod
    def load(cls):
        cfg = cls()
        p = Path(os.path.expanduser("~/.stockticker")) / "config.json"
        if p.exists():
            try:
                for k, v in json.loads(p.read_text()).items():
                    if hasattr(cfg, k): setattr(cfg, k, v)
            except: pass
        cfg.__post_init__()
        return cfg
    def save(self):
        p = Path(self.data_dir) / "config.json"
        p.write_text(json.dumps({'theme': self.theme, 'debug': self.debug, 'show_charts': self.show_charts, 'risk_free_rate': self.risk_free_rate}, indent=2))

config = Config.load()

class BlackScholes:
    @staticmethod
    def price(S, K, T, r, sigma, option_type='call'):
        if not HAS_SCIPY or T <= 0 or sigma <= 0 or S <= 0: return 0.0
        try:
            d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            if option_type.lower() == 'call':
                return max(0, S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2))
            return max(0, K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
        except: return 0.0
    @staticmethod
    def delta(S, K, T, r, sigma, option_type='call'):
        if not HAS_SCIPY or T <= 0 or sigma <= 0: return 0.5 if option_type == 'call' else -0.5
        try:
            d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
            return norm.cdf(d1) if option_type.lower() == 'call' else norm.cdf(d1) - 1
        except: return 0.5 if option_type == 'call' else -0.5
    @staticmethod
    def gamma(S, K, T, r, sigma):
        if not HAS_SCIPY or T <= 0 or sigma <= 0 or S <= 0: return 0.0
        try:
            d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
            return norm.pdf(d1) / (S * sigma * math.sqrt(T))
        except: return 0.0
    @staticmethod
    def vega(S, K, T, r, sigma):
        if not HAS_SCIPY or T <= 0 or sigma <= 0: return 0.0
        try:
            d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
            return S * norm.pdf(d1) * math.sqrt(T) / 100
        except: return 0.0
    @staticmethod
    def theta(S, K, T, r, sigma, option_type='call'):
        if not HAS_SCIPY or T <= 0 or sigma <= 0: return 0.0
        try:
            d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            term1 = -S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))
            term2 = -r * K * math.exp(-r * T) * norm.cdf(d2) if option_type.lower() == 'call' else r * K * math.exp(-r * T) * norm.cdf(-d2)
            return (term1 + term2) / 365
        except: return 0.0
    @staticmethod
    def rho(S, K, T, r, sigma, option_type='call'):
        if not HAS_SCIPY or T <= 0 or sigma <= 0: return 0.0
        try:
            d2 = (math.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
            if option_type.lower() == 'call': return K * T * math.exp(-r * T) * norm.cdf(d2) / 100
            return -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100
        except: return 0.0
    @staticmethod
    def implied_volatility(price, S, K, T, r, option_type='call', max_iter=100, tol=0.0001):
        if not HAS_SCIPY or price <= 0 or T <= 0 or S <= 0: return 0.5
        sigma = max(0.1, min(math.sqrt(2 * math.pi / T) * price / S, 2.0))
        for _ in range(max_iter):
            try:
                bs_price = BlackScholes.price(S, K, T, r, sigma, option_type)
                vega = BlackScholes.vega(S, K, T, r, sigma) * 100
                if vega < 1e-10: break
                diff = bs_price - price
                if abs(diff) < tol: break
                sigma = max(0.01, min(sigma - diff / vega, 5.0))
            except: break
        return sigma
    @staticmethod
    def calculate_all_greeks(S, K, T, r, sigma, option_type='call'):
        return {'delta': BlackScholes.delta(S, K, T, r, sigma, option_type), 'gamma': BlackScholes.gamma(S, K, T, r, sigma), 'theta': BlackScholes.theta(S, K, T, r, sigma, option_type), 'vega': BlackScholes.vega(S, K, T, r, sigma), 'rho': BlackScholes.rho(S, K, T, r, sigma, option_type)}

class TechnicalAnalysis:
    @staticmethod
    def sma(data, period): return data.rolling(window=period).mean()
    @staticmethod
    def ema(data, period): return data.ewm(span=period, adjust=False).mean()
    @staticmethod
    def wma(data, period):
        weights = np.arange(1, period + 1)
        return data.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    @staticmethod
    def rsi(data, period=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        return 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line, signal_line, macd_line - signal_line
    @staticmethod
    def bollinger_bands(data, period=20, std_dev=2.0):
        middle = data.rolling(period).mean()
        std = data.rolling(period).std()
        return middle + (std * std_dev), middle, middle - (std * std_dev)
    @staticmethod
    def atr(df, period=14):
        tr = pd.concat([df['High'] - df['Low'], (df['High'] - df['Close'].shift()).abs(), (df['Low'] - df['Close'].shift()).abs()], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    @staticmethod
    def supertrend(df, period=10, multiplier=3):
        atr = TechnicalAnalysis.atr(df, period)
        hl2 = (df['High'] + df['Low']) / 2
        basic_upper = hl2 + (multiplier * atr)
        basic_lower = hl2 - (multiplier * atr)
        close = df['Close']
        final_upper = pd.Series(0.0, index=df.index)
        final_lower = pd.Series(0.0, index=df.index)
        trend = pd.Series(0, index=df.index)
        for i in range(period, len(df)):
            final_upper.iloc[i] = basic_upper.iloc[i] if basic_upper.iloc[i] < final_upper.iloc[i-1] or close.iloc[i-1] > final_upper.iloc[i-1] else final_upper.iloc[i-1]
            final_lower.iloc[i] = basic_lower.iloc[i] if basic_lower.iloc[i] > final_lower.iloc[i-1] or close.iloc[i-1] < final_lower.iloc[i-1] else final_lower.iloc[i-1]
            prev_trend = trend.iloc[i-1] if i > 0 else 1
            trend.iloc[i] = -1 if prev_trend == 1 and close.iloc[i] < final_lower.iloc[i] else 1 if prev_trend == -1 and close.iloc[i] > final_upper.iloc[i] else prev_trend
        return trend, final_upper, final_lower
    @staticmethod
    def stochastic(df, k_period=14, d_period=3):
        low_min = df['Low'].rolling(window=k_period).min()
        high_max = df['High'].rolling(window=k_period).max()
        k = 100 * ((df['Close'] - low_min) / (high_max - low_min + 1e-9))
        return k, k.rolling(window=d_period).mean()
    @staticmethod
    def ichimoku(df):
        tenkan = (df['High'].rolling(9).max() + df['Low'].rolling(9).min()) / 2
        kijun = (df['High'].rolling(26).max() + df['Low'].rolling(26).min()) / 2
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        senkou_b = ((df['High'].rolling(52).max() + df['Low'].rolling(52).min()) / 2).shift(26)
        return {'tenkan': tenkan, 'kijun': kijun, 'senkou_a': senkou_a, 'senkou_b': senkou_b, 'chikou': df['Close'].shift(-26)}
    @staticmethod
    def mfi(df, period=14):
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        mf = tp * df['Volume']
        pos = mf.where(tp > tp.shift(), 0).rolling(period).sum()
        neg = mf.where(tp < tp.shift(), 0).rolling(period).sum()
        return 100 - (100 / (1 + pos / neg.replace(0, np.nan)))
    @staticmethod
    def cci(df, period=20):
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        return (tp - tp.rolling(period).mean()) / (0.015 * tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean()))
    @staticmethod
    def williams_r(df, period=14):
        hh = df['High'].rolling(period).max()
        ll = df['Low'].rolling(period).min()
        return -100 * (hh - df['Close']) / (hh - ll)
    @staticmethod
    def adx(df, period=14):
        plus_dm = df['High'].diff()
        minus_dm = -df['Low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        atr = TechnicalAnalysis.atr(df, period)
        plus_di = 100 * (plus_dm.ewm(span=period).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period).mean() / atr)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        return dx.ewm(span=period).mean(), plus_di, minus_di
    @staticmethod
    def obv(df):
        obv = pd.Series(0.0, index=df.index)
        obv.iloc[0] = df['Volume'].iloc[0]
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]: obv.iloc[i] = obv.iloc[i-1] + df['Volume'].iloc[i]
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]: obv.iloc[i] = obv.iloc[i-1] - df['Volume'].iloc[i]
            else: obv.iloc[i] = obv.iloc[i-1]
        return obv
    @staticmethod
    def vwap(df):
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        return (tp * df['Volume']).cumsum() / df['Volume'].cumsum()
    @staticmethod
    def pivot_points(df):
        h, l, c = df['High'].iloc[-1], df['Low'].iloc[-1], df['Close'].iloc[-1]
        p = (h + l + c) / 3
        return {'P': p, 'R1': 2*p-l, 'S1': 2*p-h, 'R2': p+(h-l), 'S2': p-(h-l), 'R3': h+2*(p-l), 'S3': l-2*(h-p)}
    @staticmethod
    def fibonacci(df, period=120):
        h, l = df['High'].tail(period).max(), df['Low'].tail(period).min()
        d = h - l
        return {'0%': h, '23.6%': h-0.236*d, '38.2%': h-0.382*d, '50%': h-0.5*d, '61.8%': h-0.618*d, '78.6%': h-0.786*d, '100%': l}
    @staticmethod
    def analyze(df, symbol):
        if df is None or len(df) < 30: return {'error': 'Insufficient Data'}
        close = df['Close']
        price = close.iloc[-1]
        rsi = TechnicalAnalysis.rsi(close).iloc[-1]
        _, _, hist = TechnicalAnalysis.macd(close)
        supertrend, _, _ = TechnicalAnalysis.supertrend(df)
        stoch_k, stoch_d = TechnicalAnalysis.stochastic(df)
        mfi = TechnicalAnalysis.mfi(df).iloc[-1]
        williams = TechnicalAnalysis.williams_r(df).iloc[-1]
        cci = TechnicalAnalysis.cci(df).iloc[-1]
        ichimoku = TechnicalAnalysis.ichimoku(df)
        signals = {
            'RSI': "OVERSOLD" if rsi < 30 else "OVERBOUGHT" if rsi > 70 else "Neutral",
            'MACD': "BULLISH" if hist.iloc[-1] > 0 else "BEARISH",
            'Supertrend': "BULLISH" if supertrend.iloc[-1] == 1 else "BEARISH",
            'Stochastic': "OVERSOLD" if stoch_k.iloc[-1] < 20 else "OVERBOUGHT" if stoch_k.iloc[-1] > 80 else "Neutral",
            'Williams %R': "OVERSOLD" if williams < -80 else "OVERBOUGHT" if williams > -20 else "Neutral",
            'CCI': "OVERSOLD" if cci < -100 else "OVERBOUGHT" if cci > 100 else "Neutral",
            'Ichimoku': "BULLISH" if ichimoku['tenkan'].iloc[-1] > ichimoku['kijun'].iloc[-1] and price > ichimoku['senkou_a'].iloc[-1] else "NEUTRAL/BEARISH",
            'MFI': "OVERSOLD" if mfi < 20 else "OVERBOUGHT" if mfi > 80 else "Neutral"
        }
        score = sum(1 if any(x in s for x in ["BULL","OVERSOLD"]) else -1 if any(x in s for x in ["BEAR","OVERBOUGHT"]) else 0 for s in signals.values())
        return {'symbol': symbol, 'price': price, 'change_pct': (price - close.iloc[-2]) / close.iloc[-2] * 100 if len(close) >= 2 else 0, 'indicators': {'RSI': rsi, 'MFI': mfi, 'CCI': cci, 'Williams': williams, 'Stoch_K': stoch_k.iloc[-1], 'Stoch_D': stoch_d.iloc[-1]}, 'signals': signals, 'score': score}

class PriceFetcher:
    def __init__(self):
        self.stock_cache = {}
        self.option_cache = {}
        self.history_cache = {}
        self.cache_ttl = 60
        self.history_ttl = 1800
    def get_stock_price(self, symbol, verbose=False):
        symbol = symbol.upper().strip()
        cached = self.stock_cache.get(symbol)
        if cached and (_now() - cached['time']) < self.cache_ttl: return cached['data']
        result = {'price': 0, 'prev': 0, 'change': 0, 'pct': 0, 'source': 'none'}
        try:
            resp = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}", headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            if resp.status_code == 200:
                meta = resp.json().get('chart', {}).get('result', [{}])[0].get('meta', {})
                price = meta.get('regularMarketPrice', 0)
                prev = meta.get('previousClose', 0) or meta.get('chartPreviousClose', 0)
                if price and float(price) > 0:
                    result = {'price': float(price), 'prev': float(prev) if prev else float(price), 'source': 'yahoo_live', 'change': 0, 'pct': 0}
                    result['change'] = result['price'] - result['prev']
                    result['pct'] = (result['change'] / result['prev'] * 100) if result['prev'] else 0
                    if verbose: print(f"      {symbol}: ${result['price']:.2f} (live)")
        except Exception as e:
            if verbose: print(f"      {symbol}: ERROR - {e}")
        if result['price'] > 0: self.stock_cache[symbol] = {'time': _now(), 'data': result}
        return result
    def get_option_price(self, symbol, expiration, strike, opt_type, underlying_price=0):
        cache_key = f"{symbol}_{expiration}_{strike}_{opt_type}"
        cached = self.option_cache.get(cache_key)
        if cached and (_now() - cached['time']) < self.cache_ttl: return cached['data']
        result = {'price': 0, 'bid': 0, 'ask': 0, 'iv': 0.5, 'source': 'none'}
        is_call = opt_type.lower().startswith('c')
        try:
            ticker = yf.Ticker(symbol)
            if expiration in ticker.options:
                chain = ticker.option_chain(expiration)
                df = chain.calls if is_call else chain.puts
                matches = df[abs(df['strike'] - strike) < 0.01]
                if not matches.empty:
                    row = matches.iloc[0]
                    bid, ask, last = safe_float(row.get('bid')), safe_float(row.get('ask')), safe_float(row.get('lastPrice'))
                    result['bid'], result['ask'] = bid, ask
                    result['iv'] = safe_float(row.get('impliedVolatility')) or 0.5
                    result['price'] = (bid + ask) / 2 if bid > 0 and ask > 0 else last
                    if result['price'] > 0: result['source'] = 'yahoo_chain'
        except: pass
        if result['price'] == 0 and underlying_price > 0 and HAS_SCIPY:
            days = dte(expiration)
            if days > 0:
                bs_price = BlackScholes.price(underlying_price, strike, days/365.0, config.risk_free_rate, result['iv'], 'call' if is_call else 'put')
                if bs_price > 0.01: result['price'], result['source'] = bs_price, 'black_scholes'
        if result['price'] > 0: self.option_cache[cache_key] = {'time': _now(), 'data': result}
        return result
    def get_history(self, symbol, period="6mo"):
        key = f"{symbol}_{period}"
        cached = self.history_cache.get(key)
        if cached and (_now() - cached['time']) < self.history_ttl: return cached['data']
        try:
            df = yf.Ticker(symbol).history(period=period)
            if df is not None and not df.empty:
                self.history_cache[key] = {'time': _now(), 'data': df}
                return df
        except: pass
        return None
    def clear_cache(self):
        self.stock_cache.clear()
        self.option_cache.clear()
        self.history_cache.clear()

class FidelityParser:
    def parse(self, filepath):
        fp = Path(os.path.expanduser(filepath)).resolve()
        if not fp.exists(): raise FileNotFoundError(f"File not found: {fp}")
        df = None
        # CRITICAL FIX: Use utf-8-sig to handle BOM and index_col=False to prevent column shift
        for enc in ['utf-8-sig', 'utf-8', 'latin1', 'cp1252']:
            try: 
                df = pd.read_csv(fp, encoding=enc, index_col=False)
                break
            except: continue
        if df is None: raise ValueError(f"Could not read: {fp}")
        df.columns = [str(c).strip() for c in df.columns]
        print(f"  Found {len(df)} rows, {len(df.columns)} columns")
        col_map = {}
        for col in df.columns:
            cl = col.lower().strip()
            if cl == 'symbol': col_map['symbol'] = col
            elif cl == 'description': col_map['desc'] = col
            elif cl == 'quantity': col_map['qty'] = col
            elif cl == 'last price': col_map['price'] = col
            elif cl == 'current value': col_map['value'] = col
            elif cl == 'cost basis total': col_map['cost'] = col
            elif cl == 'average cost basis': col_map['avg'] = col
            elif cl == 'total gain/loss dollar': col_map['pnl'] = col
            elif cl == "today's gain/loss dollar": col_map['day_pnl'] = col
        print(f"\n  Column mapping:")
        for k, v in col_map.items(): print(f"    {k}: {v}")
        stocks, options, skipped, cash = {}, [], [], 0
        print(f"\n  Parsing positions...")
        for idx, row in df.iterrows():
            raw_sym = str(row.get(col_map.get('symbol', ''), '')).strip()
            desc = str(row.get(col_map.get('desc', ''), '')).strip().upper()
            if not raw_sym or raw_sym.upper() in ('', 'NAN', 'SYMBOL') or 'PENDING' in raw_sym.upper(): continue
            if 'SPAXX' in raw_sym.upper() or 'MONEY MARKET' in desc:
                cash_val = safe_float(row.get(col_map.get('value', ''), 0))
                if cash_val > 0: cash = cash_val; print(f"    [CASH] ${cash:,.2f}")
                continue
            qty = safe_float(row.get(col_map.get('qty', ''), 0))
            if abs(qty) < 0.0001: continue
            last_price = safe_float(row.get(col_map.get('price', ''), 0))
            current_value = safe_float(row.get(col_map.get('value', ''), 0))
            cost_basis = safe_float(row.get(col_map.get('cost', ''), 0))
            avg_cost = safe_float(row.get(col_map.get('avg', ''), 0))
            fidelity_pnl = safe_float(row.get(col_map.get('pnl', ''), 0))
            day_pnl = safe_float(row.get(col_map.get('day_pnl', ''), 0))
            opt = self._parse_option(raw_sym, qty, cost_basis, last_price, current_value, fidelity_pnl, day_pnl)
            if opt:
                options.append(opt)
                print(f"    [OPT] {opt['symbol']:5} {opt['type'][0].upper()} ${opt['strike']:<8.2f} {opt['expiration']} qty:{opt['qty']:>3} cost:{fmt_money(opt['cost']):>10}")
                continue
            ticker = self._get_ticker(raw_sym, desc)
            if ticker:
                if ticker in stocks:
                    stocks[ticker]['qty'] += qty
                    stocks[ticker]['cost'] += abs(cost_basis)
                    stocks[ticker]['fidelity_value'] += abs(current_value)
                    stocks[ticker]['fidelity_pnl'] += fidelity_pnl
                else:
                    stocks[ticker] = {'qty': qty, 'cost': abs(cost_basis), 'avg': avg_cost or (abs(cost_basis)/qty if qty else 0), 'fidelity_price': last_price, 'fidelity_value': abs(current_value), 'fidelity_pnl': fidelity_pnl, 'day_pnl': day_pnl}
                print(f"    [STK] {ticker:5} qty:{qty:>10.4f} cost:{fmt_money(abs(cost_basis)):>10}")
            else: skipped.append(raw_sym[:30])
        print(f"\n  Parsed: {len(stocks)} stocks, {len(options)} options")
        if skipped: print(f"  Skipped: {', '.join(skipped[:5])}")
        return stocks, options, cash
    def _get_ticker(self, raw_sym, desc=""):
        raw = raw_sym.strip().upper()
        if raw.startswith('-'): raw = raw[1:]
        if re.match(r'^[A-Z]+\d{6}[CP][\d\.]+$', raw): return None
        if len(raw) <= 5 and raw.isalpha(): return raw
        for company, ticker in COMPANY_TO_TICKER.items():
            if company in raw or company in desc: return ticker
        return None
    def _parse_option(self, raw_sym, qty, cost_basis, last_price, current_value, fidelity_pnl, day_pnl=0):
        sym = raw_sym.strip().upper()
        is_short = sym.startswith('-')
        if is_short: sym = sym[1:].strip()
        m = re.match(r'^([A-Z]+)(\d{6})([CP])([\d\.]+)$', sym)
        if not m: return None
        root, date_str, opt_type_char, strike_str = m.groups()
        exp = f"20{date_str[0:2]}-{date_str[2:4]}-{date_str[4:6]}"
        # FIX: Fidelity reports contracts as whole numbers - no multiplication needed
        contracts = int(round(abs(qty)))
        if is_short or qty < 0: contracts = -abs(contracts)
        return {'symbol': root, 'type': 'call' if opt_type_char == 'C' else 'put', 'strike': float(strike_str), 'expiration': exp, 'qty': contracts, 'cost': abs(cost_basis), 'fidelity_price': last_price, 'fidelity_value': abs(current_value), 'fidelity_pnl': fidelity_pnl, 'day_pnl': day_pnl}

class Portfolio:
    def __init__(self):
        self.file = Path(config.data_dir) / "portfolio.json"
        self.data = self._load()
    def _load(self):
        if self.file.exists():
            try: return json.loads(self.file.read_text())
            except: pass
        return {'stocks': {}, 'options': [], 'cash': 0}
    def _save(self): self.file.write_text(json.dumps(self.data, indent=2))
    def clear(self): self.data = {'stocks': {}, 'options': [], 'cash': 0}; self._save(); success("Portfolio cleared")
    def import_csv(self, filepath):
        print(Fore.CYAN + f"\n{'═'*60}\n IMPORTING FIDELITY CSV\n{'═'*60}" + Style.RESET_ALL)
        try:
            stocks, options, cash = FidelityParser().parse(filepath)
            self.data = {'stocks': stocks, 'options': options, 'cash': cash, 'imported': datetime.now().isoformat()}
            self._save()
            print(f"\n{Fore.GREEN}✓ Imported {len(stocks)} stocks, {len(options)} options, ${cash:,.2f} cash{Style.RESET_ALL}")
        except Exception as e: err(str(e))
    def display(self, fetcher):
        stocks, options, cash = self.data.get('stocks', {}), self.data.get('options', []), self.data.get('cash', 0)
        if not stocks and not options: warn("No positions. Use 'import FILE' first."); return
        print(Fore.CYAN + f"\n{'═'*80}\n PORTFOLIO — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (LIVE)\n{'═'*80}" + Style.RESET_ALL)
        total_value, total_cost, total_pnl, total_day_pnl = 0, 0, 0, 0
        all_symbols = set(stocks.keys()) | set(o['symbol'] for o in options)
        print(f"\n  Fetching live prices for {len(all_symbols)} symbols...")
        stock_prices = {}
        prog = ProgressBar(len(all_symbols), "  ")
        for i, sym in enumerate(all_symbols): prog.update(i, sym); stock_prices[sym] = fetcher.get_stock_price(sym)
        prog.done()
        if stocks:
            print(f"\n{Style.BRIGHT}STOCKS ({len(stocks)}){Style.RESET_ALL}\n" + "─"*80)
            rows = []
            for sym, pos in sorted(stocks.items()):
                live = stock_prices.get(sym, {})
                price, day_pct = live.get('price', 0), live.get('pct', 0)
                qty, cost, avg = pos['qty'], pos['cost'], pos.get('avg', pos['cost']/pos['qty'] if pos['qty'] else 0)
                if price > 0:
                    value = qty * price
                    pnl = value - cost
                    price_str = fmt_money(price)
                    day_str = color_pct(day_pct)
                else:
                    value = pos.get('fidelity_value', 0)
                    pnl = pos.get('fidelity_pnl', 0)
                    price_str = f"~{fmt_money(pos.get('fidelity_price', 0))}"
                    day_str = "-"
                pnl_pct = (pnl / cost * 100) if cost else 0
                total_value += value; total_cost += cost; total_pnl += pnl
                total_day_pnl += pos.get('day_pnl', 0)
                rows.append([sym, f"{qty:.4f}" if abs(qty) < 10 else f"{qty:.2f}", fmt_money(avg), price_str, day_str, fmt_money(value), color_pnl(pnl, pnl_pct)])
            print(tabulate(rows, headers=["Symbol", "Qty", "Avg", "Price", "Day%", "Value", "P&L"]))
        if options:
            print(f"\n{Style.BRIGHT}OPTIONS ({len(options)}){Style.RESET_ALL}\n" + "─"*80)
            prog = ProgressBar(len(options), "  ")
            rows = []
            for i, o in enumerate(sorted(options, key=lambda x: (x['symbol'], x['expiration'], x['strike']))):
                prog.update(i, o['symbol'])
                underlying_price = stock_prices.get(o['symbol'], {}).get('price', 0)
                opt_data = fetcher.get_option_price(o['symbol'], o['expiration'], o['strike'], o['type'], underlying_price)
                price, source = opt_data['price'], opt_data['source']
                days = dte(o['expiration'])
                qty = o['qty']
                cost = o['cost']
                if price > 0 and source in ('yahoo_chain', 'black_scholes'):
                    value = abs(qty) * price * 100
                    pnl = value - cost if qty > 0 else cost - value
                    price_str = f"${price:.2f}" if source == 'yahoo_chain' else f"~${price:.2f}"
                else:
                    value = abs(o.get('fidelity_value', 0))
                    pnl = o.get('fidelity_pnl', 0)
                    price_str = f"~${o.get('fidelity_price', 0):.2f}"
                pnl_pct = (pnl / cost * 100) if cost else 0
                total_value += value; total_cost += cost; total_pnl += pnl
                total_day_pnl += o.get('day_pnl', 0)
                desc = f"{o['symbol']} {o['expiration'][5:]} ${o['strike']:.0f}{'C' if o['type']=='call' else 'P'}"
                ls = "S" if qty < 0 else "L"
                rows.append([desc, ls, abs(qty), price_str, f"{days}d", fmt_money(value), color_pnl(pnl, pnl_pct)])
            prog.done()
            print(tabulate(rows, headers=["Option", "L/S", "Qty", "Price", "DTE", "Value", "P&L"]))
        print(f"\n{'═'*80}")
        print(f"  {Style.BRIGHT}TOTAL VALUE:{Style.RESET_ALL}  {fmt_money(total_value)}")
        print(f"  {Style.BRIGHT}TOTAL COST:{Style.RESET_ALL}   {fmt_money(total_cost)}")
        print(f"  {Style.BRIGHT}TOTAL P&L:{Style.RESET_ALL}    {color_pnl(total_pnl, (total_pnl/total_cost*100) if total_cost else 0)}")
        if total_day_pnl != 0: print(f"  {Style.BRIGHT}TODAY'S P&L:{Style.RESET_ALL}  {color_money(total_day_pnl)}")
        if cash: print(f"  {Style.BRIGHT}CASH:{Style.RESET_ALL}         {fmt_money(cash)}")
        if cash: print(f"  {Style.BRIGHT}TOTAL ACCT:{Style.RESET_ALL}  {fmt_money(total_value + cash)}")
        print(f"{'═'*80}\n")
    def analyze_symbol(self, symbol, fetcher):
        symbol = symbol.upper()
        print(Fore.CYAN + f"\n{'═'*60}\n TECHNICAL ANALYSIS: {symbol}\n{'═'*60}" + Style.RESET_ALL)
        df = fetcher.get_history(symbol, "6mo")
        if df is None or df.empty: err(f"Could not fetch data for {symbol}"); return
        analysis = TechnicalAnalysis.analyze(df, symbol)
        if 'error' in analysis: err(analysis['error']); return
        live = fetcher.get_stock_price(symbol)
        print(f"\n  {Style.BRIGHT}Price:{Style.RESET_ALL} {fmt_money(live.get('price', analysis['price']))} ({color_pct(live.get('pct', 0))} today)")
        print(f"\n  {Style.BRIGHT}Indicators:{Style.RESET_ALL}")
        for name, val in analysis['indicators'].items(): print(f"    {name:12}: {val:.2f}")
        print(f"\n  {Style.BRIGHT}Signals:{Style.RESET_ALL}")
        for name, signal in analysis['signals'].items(): print(f"    {name:12}: {color_signal(signal)}")
        score = analysis['score']
        print(f"\n  {Style.BRIGHT}Score:{Style.RESET_ALL} {Fore.GREEN if score > 0 else Fore.RED if score < 0 else Fore.YELLOW}{score:+d}{Style.RESET_ALL}")
        for lvl, val in TechnicalAnalysis.pivot_points(df).items(): print(f"    {lvl}: {fmt_money(val)}")
        if HAS_MATPLOTLIB and config.show_charts: self._plot_chart(df, symbol)
    def _plot_chart(self, df, symbol):
        try:
            fig, axes = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1, 1]})
            fig.suptitle(f'{symbol} Technical Analysis', fontsize=14, fontweight='bold')
            close = df['Close']
            upper, middle, lower = TechnicalAnalysis.bollinger_bands(close)
            axes[0].plot(df.index, close, label='Price', color='blue')
            axes[0].plot(df.index, upper, 'r--', alpha=0.5)
            axes[0].plot(df.index, lower, 'r--', alpha=0.5)
            axes[0].fill_between(df.index, lower, upper, alpha=0.1)
            axes[0].legend(); axes[0].grid(True, alpha=0.3)
            rsi = TechnicalAnalysis.rsi(close)
            axes[1].plot(df.index, rsi, color='purple')
            axes[1].axhline(70, color='red', linestyle='--', alpha=0.5)
            axes[1].axhline(30, color='green', linestyle='--', alpha=0.5)
            axes[1].set_ylim(0, 100); axes[1].grid(True, alpha=0.3)
            _, _, hist = TechnicalAnalysis.macd(close)
            axes[2].bar(df.index, hist, color=['g' if h >= 0 else 'r' for h in hist], alpha=0.5)
            axes[2].axhline(0, color='gray'); axes[2].grid(True, alpha=0.3)
            plt.tight_layout(); plt.show()
        except Exception as e: warn(f"Chart error: {e}")
    def option_greeks(self, symbol, fetcher):
        options = [o for o in self.data.get('options', []) if o['symbol'].upper() == symbol.upper()]
        if not options: warn(f"No options for {symbol}"); return
        print(Fore.CYAN + f"\n{'═'*60}\n GREEKS: {symbol.upper()}\n{'═'*60}" + Style.RESET_ALL)
        underlying = fetcher.get_stock_price(symbol).get('price', 0)
        if underlying <= 0: err("Could not get underlying price"); return
        print(f"\n  Underlying: {fmt_money(underlying)}")
        rows = []
        for o in sorted(options, key=lambda x: (x['expiration'], x['strike'])):
            T = max(dte(o['expiration']) / 365.0, 0.001)
            iv = fetcher.get_option_price(o['symbol'], o['expiration'], o['strike'], o['type'], underlying).get('iv', 0.5)
            g = BlackScholes.calculate_all_greeks(underlying, o['strike'], T, config.risk_free_rate, iv, o['type'])
            rows.append([f"{o['expiration'][5:]} ${o['strike']:.0f}{'C' if o['type']=='call' else 'P'}", o['qty'], f"{dte(o['expiration'])}d", f"{iv*100:.1f}%", f"{g['delta']:.3f}", f"{g['gamma']:.4f}", f"{g['theta']:.3f}", f"{g['vega']:.3f}"])
        print(tabulate(rows, headers=["Option", "Qty", "DTE", "IV", "Delta", "Gamma", "Theta", "Vega"]))
        td = sum(float(r[4])*int(r[1])*100 for r in rows)
        print(f"\n  Portfolio Delta: {td:+.2f} (${td*underlying:,.0f} exposure)")
    def summary(self):
        stocks, options, cash = self.data.get('stocks', {}), self.data.get('options', []), self.data.get('cash', 0)
        if not stocks and not options: warn("No positions. Use 'import FILE' first."); return
        print(Fore.CYAN + f"\n{'═'*60}\n PORTFOLIO SUMMARY (from import)\n{'═'*60}" + Style.RESET_ALL)
        total_value = sum(s.get('fidelity_value', 0) for s in stocks.values()) + sum(abs(o.get('fidelity_value', 0)) for o in options)
        total_cost = sum(s.get('cost', 0) for s in stocks.values()) + sum(o.get('cost', 0) for o in options)
        total_pnl = sum(s.get('fidelity_pnl', 0) for s in stocks.values()) + sum(o.get('fidelity_pnl', 0) for o in options)
        print(f"\n  Stocks: {len(stocks)}\n  Options: {len(options)}\n  Cash: {fmt_money(cash)}")
        print(f"\n  Total Value: {fmt_money(total_value)}\n  Total Cost: {fmt_money(total_cost)}")
        print(f"  Total P&L: {color_pnl(total_pnl, (total_pnl/total_cost*100) if total_cost else 0)}")
        if cash: print(f"  Total Account: {fmt_money(total_value + cash)}")
        print()
    def expiring_soon(self, days=7):
        options = self.data.get('options', [])
        if not options: warn("No options in portfolio"); return
        expiring = [o for o in options if 0 <= dte(o['expiration']) <= days]
        if not expiring: print(f"\n  No options expiring within {days} days"); return
        print(Fore.CYAN + f"\n{'═'*60}\n OPTIONS EXPIRING WITHIN {days} DAYS\n{'═'*60}" + Style.RESET_ALL)
        rows = []
        for o in sorted(expiring, key=lambda x: (x['expiration'], x['symbol'])):
            d = dte(o['expiration'])
            desc = f"{o['symbol']} {o['expiration']} ${o['strike']:.0f}{'C' if o['type']=='call' else 'P'}"
            ls = "S" if o['qty'] < 0 else "L"
            rows.append([desc, ls, abs(o['qty']), f"{d}d", fmt_money(abs(o.get('fidelity_value', 0))), color_money(o.get('fidelity_pnl', 0))])
        print(tabulate(rows, headers=["Option", "L/S", "Qty", "DTE", "Value", "P&L"]))

class Watchlist:
    def __init__(self):
        self.file = Path(config.data_dir) / "watchlist.json"
        self.symbols = json.loads(self.file.read_text()) if self.file.exists() else []
    def _save(self): self.file.write_text(json.dumps(self.symbols))
    def add(self, sym): sym = sym.upper(); self.symbols.append(sym) if sym not in self.symbols else None; self._save(); success(f"Added {sym}")
    def remove(self, sym): sym = sym.upper(); self.symbols.remove(sym) if sym in self.symbols else None; self._save(); success(f"Removed {sym}")
    def display(self, fetcher):
        if not self.symbols: warn("Watchlist empty"); return
        print(Fore.CYAN + f"\n{'═'*60}\n WATCHLIST\n{'═'*60}" + Style.RESET_ALL)
        rows = [[s, fmt_money(d['price']), color_pct(d['pct'])] if (d := fetcher.get_stock_price(s))['price'] > 0 else [s, "N/A", "-"] for s in self.symbols]
        print(tabulate(rows, headers=["Symbol", "Price", "Change"]))

def print_help():
    print(Fore.CYAN + """
╔═══════════════════════════════════════════════════════════════╗
║  STOCK TICKER v7.7.0 COMMANDS                                 ║
╠═══════════════════════════════════════════════════════════════╣
║  pf          Show portfolio    │  import FILE  Import CSV     ║
║  summary     Quick summary     │  expiring [N] Options <N days║
║  q SYMBOL    Quote             │  ta SYMBOL    Analysis       ║
║  greeks SYM  Option Greeks     │  watch        Watchlist      ║
║  watch add/rm SYMBOL           │  clear        Clear portfolio║
║  debug/charts on|off           │  refresh      Clear cache    ║
║  help        This help         │  quit         Exit           ║
╚═══════════════════════════════════════════════════════════════╝
""" + Style.RESET_ALL)

def main():
    print(Fore.CYAN + Style.BRIGHT + "\n    STOCK TICKER v7.7.0 — Live Portfolio Tracking\n" + Style.RESET_ALL)
    fetcher, portfolio, watchlist = PriceFetcher(), Portfolio(), Watchlist()
    print(f"  Data: {config.data_dir}\n  Type 'help' for commands\n")
    while True:
        try:
            raw = input(Fore.GREEN + "▶ " + Style.RESET_ALL).strip()
            if not raw: continue
            parts = shlex.split(raw)
            cmd, args = parts[0].lower(), parts[1:] if len(parts) > 1 else []
            if cmd in ('quit', 'exit') or (cmd == 'q' and not args): print("Goodbye!"); break
            elif cmd in ('help', 'h', '?'): print_help()
            elif cmd in ('pf', 'portfolio'): portfolio.display(fetcher)
            elif cmd == 'summary': portfolio.summary()
            elif cmd == 'expiring': portfolio.expiring_soon(int(args[0]) if args else 7)
            elif cmd == 'import': portfolio.import_csv(args[0]) if args else err("Usage: import FILE")
            elif cmd == 'clear': portfolio.clear()
            elif cmd in ('q', 'quote') and args:
                d = fetcher.get_stock_price(args[0].upper())
                print(f"  {args[0].upper()}: {fmt_money(d['price'])} ({color_pct(d['pct'])})") if d['price'] > 0 else err(f"Not found: {args[0]}")
            elif cmd == 'ta' and args: portfolio.analyze_symbol(args[0], fetcher)
            elif cmd == 'greeks' and args: portfolio.option_greeks(args[0], fetcher)
            elif cmd == 'watch':
                if not args: watchlist.display(fetcher)
                elif args[0] == 'add' and len(args) > 1: watchlist.add(args[1])
                elif args[0] in ('rm', 'del') and len(args) > 1: watchlist.remove(args[1])
            elif cmd == 'debug': config.debug = not config.debug if not args else args[0] in ('on', '1'); config.save(); print(f"  Debug: {'ON' if config.debug else 'OFF'}")
            elif cmd == 'charts': config.show_charts = not config.show_charts if not args else args[0] in ('on', '1'); config.save(); print(f"  Charts: {'ON' if config.show_charts else 'OFF'}")
            elif cmd == 'refresh': fetcher.clear_cache(); success("Cache cleared")
            elif len(cmd) <= 5 and cmd.isalpha():
                d = fetcher.get_stock_price(cmd.upper())
                print(f"  {cmd.upper()}: {fmt_money(d['price'])} ({color_pct(d['pct'])})") if d['price'] > 0 else err(f"Unknown: {cmd}")
            else: err(f"Unknown: {cmd}")
        except KeyboardInterrupt: print("\n  Use 'quit' to exit")
        except EOFError: break
        except Exception as e: err(str(e))

if __name__ == "__main__":
    main()
