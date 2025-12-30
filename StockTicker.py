#!/usr/bin/env python3
"""
STOCK TICKER APP v10.1.0 — Universal Portfolio Tracker & Analyzer
INSTALL: pip install yfinance pandas numpy matplotlib tabulate colorama requests scipy

NEW IN v10.1.0:
- Smart Context: Commands remember last symbol used
- News Command: Get latest headlines for any stock
- Dashboard: Quick overview with price, trend, and headlines
- Extended Technical Analysis (ta2): Stochastic, ADX, Williams %R, CCI, MFI, VWAP, Ichimoku, OBV
- Support/Resistance Levels (levels): Fibonacci, pivot points, key price levels
- Trend Analysis (trend): Comprehensive trend strength scoring
- Detailed Quote (quote/info): Full fundamentals and analyst ratings
"""
from __future__ import annotations
import os, re, sys, json, time, math, shlex, logging, warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import requests

# Set Logging Levels
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("requests").setLevel(logging.CRITICAL)
import yfinance as yf

# --- OPTIONAL IMPORTS ---
try:
    from scipy.stats import norm, linregress
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

# --- CONSTANTS ---
COMPANY_TO_TICKER = {
    'PURECYCLE TECHNOLOGIES INC': 'PCT', 'APPLE INC': 'AAPL', 'MICROSOFT CORP': 'MSFT',
    'TESLA INC': 'TSLA', 'NVIDIA CORP': 'NVDA', 'AMAZON COM INC': 'AMZN',
    'ALPHABET INC': 'GOOGL', 'META PLATFORMS INC': 'META', 'PALANTIR TECHNOLOGIES': 'PLTR',
    'ADVANCED MICRO DEVICES': 'AMD', 'INTEL CORP': 'INTC', 'COINBASE GLOBAL': 'COIN',
    'ROBINHOOD MARKETS': 'HOOD', 'SOFI TECHNOLOGIES': 'SOFI', 'ROCKET LAB USA': 'RKLB',
    'ARCHER AVIATION': 'ACHR', 'JOBY AVIATION': 'JOBY', 'AST SPACEMOBILE': 'ASTS',
    'OKLO INC': 'OKLO', 'CARVANA CO': 'CVNA', 'UBER TECHNOLOGIES': 'UBER',
    'REDDIT INC': 'RDDT', 'MICRON TECHNOLOGY': 'MU'
}

# --- UTILS ---
def _now(): return time.time()
def today_str(): return datetime.now().strftime("%Y-%m-%d")

def safe_float(val, default=0.0):
    if val is None: return default
    if isinstance(val, (int, float)):
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)): return default
        return float(val)
    try:
        s = str(val).strip().replace(",", "").replace("$", "").replace('"', "").replace("%", "")
        if s in ("", "--", "N/A", "nan", "None", "-", "NaN", "null"): return default
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
    benchmark: str = "SPY"
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
        p.write_text(json.dumps({'theme': self.theme, 'debug': self.debug, 'show_charts': self.show_charts, 'risk_free_rate': self.risk_free_rate, 'benchmark': self.benchmark}, indent=2))

config = Config.load()

# --- MATH ---
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
    def calculate_all_greeks(S, K, T, r, sigma, option_type='call'):
        if not HAS_SCIPY or T <= 0 or sigma <= 0 or S <= 0: return {'delta':0,'gamma':0,'theta':0,'vega':0,'rho':0}
        try:
            d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            delta = norm.cdf(d1) if option_type=='call' else norm.cdf(d1)-1
            gamma = norm.pdf(d1)/(S*sigma*math.sqrt(T))
            vega = S*norm.pdf(d1)*math.sqrt(T)/100
            theta = (-S*norm.pdf(d1)*sigma/(2*math.sqrt(T)) - r*K*math.exp(-r*T)*norm.cdf(d2 if option_type=='call' else -d2))/365
            rho = K * T * math.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2) / 100
            if option_type == 'put': rho = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100
            return {'delta':delta, 'gamma':gamma, 'theta':theta, 'vega':vega, 'rho':rho}
        except: return {'delta':0,'gamma':0,'theta':0,'vega':0,'rho':0}
    @staticmethod
    def implied_volatility(market_price, S, K, T, r, option_type='call', max_iterations=100, precision=1.0e-5):
        """Calculate implied volatility using Newton-Raphson method"""
        if not HAS_SCIPY or T <= 0 or market_price <= 0: return 0.5
        sigma = 0.5
        for i in range(max_iterations):
            price = BlackScholes.price(S, K, T, r, sigma, option_type)
            vega = BlackScholes.calculate_all_greeks(S, K, T, r, sigma, option_type)['vega'] * 100
            if abs(vega) < 1e-10: break
            diff = market_price - price
            if abs(diff) < precision: return sigma
            sigma += diff / vega
            sigma = max(0.01, min(sigma, 5.0))
        return sigma
    @staticmethod
    def probability_itm(S, K, T, r, sigma, option_type='call'):
        """Calculate probability of option expiring in the money"""
        if not HAS_SCIPY or T <= 0 or sigma <= 0: return 0.0
        try:
            d2 = (math.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
            return norm.cdf(d2) if option_type == 'call' else norm.cdf(-d2)
        except: return 0.0
    @staticmethod
    def probability_profit(S, K, premium, T, r, sigma, option_type='call', is_long=True):
        """Calculate probability of profit for an option position"""
        if not HAS_SCIPY or T <= 0 or sigma <= 0: return 0.0
        try:
            breakeven = K + premium if option_type == 'call' else K - premium
            d2 = (math.log(S / breakeven) + (r - 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
            if option_type == 'call':
                return norm.cdf(d2) if is_long else norm.cdf(-d2)
            return norm.cdf(-d2) if is_long else norm.cdf(d2)
        except: return 0.0

class TechnicalAnalysis:
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
    def pivot_points(df):
        h, l, c = df['High'].iloc[-1], df['Low'].iloc[-1], df['Close'].iloc[-1]
        p = (h + l + c) / 3
        return {'P': p, 'R1': 2*p-l, 'S1': 2*p-h, 'R2': p+(h-l), 'S2': p-(h-l), 'R3': h+2*(p-l), 'S3': l-2*(h-p)}

    @staticmethod
    def stochastic(df, k_period=14, d_period=3, smooth_k=3):
        """Stochastic Oscillator (%K and %D)"""
        low_min = df['Low'].rolling(k_period).min()
        high_max = df['High'].rolling(k_period).max()
        stoch_k = 100 * (df['Close'] - low_min) / (high_max - low_min)
        stoch_k = stoch_k.rolling(smooth_k).mean()
        stoch_d = stoch_k.rolling(d_period).mean()
        return stoch_k, stoch_d

    @staticmethod
    def adx(df, period=14):
        """Average Directional Index (ADX) with +DI and -DI"""
        high, low, close = df['High'], df['Low'], df['Close']
        plus_dm = high.diff()
        minus_dm = low.diff().abs() * -1
        plus_dm = plus_dm.where((plus_dm > minus_dm.abs()) & (plus_dm > 0), 0)
        minus_dm = minus_dm.abs().where((minus_dm.abs() > plus_dm) & (minus_dm.abs() > 0), 0)
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx_val = dx.rolling(period).mean()
        return adx_val, plus_di, minus_di

    @staticmethod
    def obv(df):
        """On-Balance Volume"""
        obv = pd.Series(0.0, index=df.index)
        obv.iloc[0] = df['Volume'].iloc[0]
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['Volume'].iloc[i]
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['Volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        return obv

    @staticmethod
    def vwap(df):
        """Volume Weighted Average Price"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        return (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()

    @staticmethod
    def williams_r(df, period=14):
        """Williams %R"""
        high_max = df['High'].rolling(period).max()
        low_min = df['Low'].rolling(period).min()
        return -100 * (high_max - df['Close']) / (high_max - low_min)

    @staticmethod
    def cci(df, period=20):
        """Commodity Channel Index"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        sma = typical_price.rolling(period).mean()
        mad = typical_price.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
        return (typical_price - sma) / (0.015 * mad)

    @staticmethod
    def mfi(df, period=14):
        """Money Flow Index"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        raw_money_flow = typical_price * df['Volume']
        positive_flow = pd.Series(0.0, index=df.index)
        negative_flow = pd.Series(0.0, index=df.index)
        for i in range(1, len(df)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_flow.iloc[i] = raw_money_flow.iloc[i]
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                negative_flow.iloc[i] = raw_money_flow.iloc[i]
        positive_mf = positive_flow.rolling(period).sum()
        negative_mf = negative_flow.rolling(period).sum()
        return 100 - (100 / (1 + positive_mf / negative_mf.replace(0, np.nan)))

    @staticmethod
    def fibonacci_retracement(df, lookback=50):
        """Calculate Fibonacci retracement levels"""
        high = df['High'].tail(lookback).max()
        low = df['Low'].tail(lookback).min()
        diff = high - low
        return {
            '0.0% (High)': high, '23.6%': high - diff * 0.236, '38.2%': high - diff * 0.382,
            '50.0%': high - diff * 0.5, '61.8%': high - diff * 0.618, '78.6%': high - diff * 0.786,
            '100.0% (Low)': low
        }

    @staticmethod
    def ichimoku(df, tenkan=9, kijun=26, senkou_b=52):
        """Ichimoku Cloud"""
        tenkan_sen = (df['High'].rolling(tenkan).max() + df['Low'].rolling(tenkan).min()) / 2
        kijun_sen = (df['High'].rolling(kijun).max() + df['Low'].rolling(kijun).min()) / 2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
        senkou_span_b = ((df['High'].rolling(senkou_b).max() + df['Low'].rolling(senkou_b).min()) / 2).shift(kijun)
        return {'tenkan_sen': tenkan_sen, 'kijun_sen': kijun_sen, 'senkou_span_a': senkou_span_a, 'senkou_span_b': senkou_span_b}

    @staticmethod
    def support_resistance(df, window=20, num_levels=3):
        """Find key support and resistance levels"""
        highs = df['High'].rolling(window, center=True).max()
        lows = df['Low'].rolling(window, center=True).min()
        resistance_levels, support_levels = [], []
        for i in range(window, len(df) - window):
            if df['High'].iloc[i] == highs.iloc[i]: resistance_levels.append(df['High'].iloc[i])
            if df['Low'].iloc[i] == lows.iloc[i]: support_levels.append(df['Low'].iloc[i])
        current_price = df['Close'].iloc[-1]
        resistance = sorted(set([r for r in resistance_levels if r > current_price]))[:num_levels]
        support = sorted(set([s for s in support_levels if s < current_price]), reverse=True)[:num_levels]
        return {'support': support, 'resistance': resistance}

    @staticmethod
    def trend_strength(df):
        """Calculate trend strength using multiple indicators"""
        close = df['Close']
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1]
        sma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else sma_50
        current = close.iloc[-1]
        ma_score = sum([1 if current > sma_20 else -1, 1 if current > sma_50 else -1, 1 if current > sma_200 else -1,
                       1 if sma_20 > sma_50 else -1, 1 if sma_50 > sma_200 else -1])
        adx_val, plus_di, minus_di = TechnicalAnalysis.adx(df)
        adx = adx_val.iloc[-1] if not pd.isna(adx_val.iloc[-1]) else 0
        trend_direction = 1 if plus_di.iloc[-1] > minus_di.iloc[-1] else -1
        rsi_score = 1 if TechnicalAnalysis.rsi(close).iloc[-1] > 50 else -1
        _, _, hist = TechnicalAnalysis.macd(close)
        macd_score = 1 if hist.iloc[-1] > 0 else -1
        total_score = ma_score + (2 * trend_direction if adx > 25 else trend_direction) + rsi_score + macd_score
        if total_score >= 6: strength = "STRONG UPTREND"
        elif total_score >= 3: strength = "UPTREND"
        elif total_score >= 1: strength = "WEAK UPTREND"
        elif total_score >= -1: strength = "NEUTRAL"
        elif total_score >= -3: strength = "WEAK DOWNTREND"
        elif total_score >= -6: strength = "DOWNTREND"
        else: strength = "STRONG DOWNTREND"
        return {'score': total_score, 'strength': strength, 'adx': adx}

    @staticmethod
    def analyze(df, symbol):
        if df is None or len(df) < 30: return {'error': 'Insufficient Data'}
        close = df['Close']
        price = close.iloc[-1]
        rsi = TechnicalAnalysis.rsi(close).iloc[-1]
        _, _, hist = TechnicalAnalysis.macd(close)
        supertrend, _, _ = TechnicalAnalysis.supertrend(df)
        signals = {
            'RSI': "OVERSOLD" if rsi < 30 else "OVERBOUGHT" if rsi > 70 else "Neutral",
            'MACD': "BULLISH" if hist.iloc[-1] > 0 else "BEARISH",
            'Supertrend': "BULLISH" if supertrend.iloc[-1] == 1 else "BEARISH"
        }
        score = sum(1 if "BULL" in s or "OVERSOLD" in s else -1 if "BEAR" in s or "OVERBOUGHT" in s else 0 for s in signals.values())
        return {'symbol': symbol, 'price': price, 'indicators': {'RSI': rsi}, 'signals': signals, 'score': score}

# --- DATA FETCHING ---
class PriceFetcher:
    def __init__(self):
        self.stock_cache = {}
        self.option_cache = {}
        self.history_cache = {}
        self.meta_cache = {}
        self.cache_ttl = 60
        self.history_ttl = 1800
        self.meta_ttl = 86400
        
    def get_stock_price(self, symbol, verbose=False):
        symbol = symbol.upper().strip()
        cached = self.stock_cache.get(symbol)
        if cached and (_now() - cached['time']) < self.cache_ttl: return cached['data']
        result = {'price': 0, 'prev': 0, 'change': 0, 'pct': 0, 'source': 'none'}
        
        # Method 1: Direct Yahoo API (most reliable, try first)
        try:
            resp = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}", 
                              headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                chart_result = data.get('chart', {}).get('result')
                if chart_result and len(chart_result) > 0:
                    meta = chart_result[0].get('meta', {})
                    price = meta.get('regularMarketPrice', 0)
                    prev = meta.get('chartPreviousClose') or meta.get('previousClose', 0)
                    if price:
                        result = {'price': float(price), 'prev': float(prev) if prev else float(price), 'source': 'yahoo_api', 'change': 0, 'pct': 0}
                        result['change'] = result['price'] - result['prev']
                        result['pct'] = (result['change']/result['prev']*100) if result['prev'] else 0
        except Exception as e:
            if config.debug: print(f"  Debug [{symbol}] yahoo_api error: {e}")
        
        # Method 2: yfinance history (reliable fallback)
        if result['price'] == 0:
            try:
                t = yf.Ticker(symbol)
                hist = t.history(period='5d')
                if hist is not None and not hist.empty and 'Close' in hist.columns:
                    price = float(hist['Close'].iloc[-1])
                    prev = float(hist['Close'].iloc[-2]) if len(hist) > 1 else price
                    result = {'price': price, 'prev': prev, 'source': 'yfinance_history', 'change': price - prev, 'pct': (price - prev) / prev * 100 if prev else 0}
            except Exception as e:
                if config.debug: print(f"  Debug [{symbol}] history error: {e}")
        
        # Method 3: yfinance info dict
        if result['price'] == 0:
            try:
                t = yf.Ticker(symbol)
                info = t.info
                if info:
                    price = info.get('regularMarketPrice') or info.get('currentPrice') or info.get('navPrice') or info.get('ask') or 0
                    prev = info.get('regularMarketPreviousClose') or info.get('previousClose') or price
                    if price and price > 0:
                        result = {'price': float(price), 'prev': float(prev) if prev else float(price), 'source': 'yfinance_info', 'change': 0, 'pct': 0}
                        result['change'] = result['price'] - result['prev']
                        result['pct'] = (result['change']/result['prev']*100) if result['prev'] else 0
            except Exception as e:
                if config.debug: print(f"  Debug [{symbol}] info error: {e}")
        
        # Method 4: fast_info (can be buggy with some tickers)
        if result['price'] == 0:
            try:
                t = yf.Ticker(symbol)
                if t.fast_info and hasattr(t.fast_info, 'last_price') and t.fast_info.last_price:
                    price, prev = t.fast_info.last_price, t.fast_info.previous_close or t.fast_info.last_price
                    result = {'price': float(price), 'prev': float(prev), 'source': 'yfinance_fast', 'change': price - prev, 'pct': (price - prev) / prev * 100 if prev else 0}
            except Exception as e:
                if config.debug: print(f"  Debug [{symbol}] fast_info error: {e}")
        
        if config.debug and result['price'] == 0:
            print(f"  Debug [{symbol}] ALL METHODS FAILED - no price found")
        
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

    def get_history(self, symbol, period="6mo", interval="1d"):
        symbol = symbol.upper().strip()
        key = f"{symbol}_{period}_{interval}"
        cached = self.history_cache.get(key)
        if cached and (_now() - cached['time']) < self.history_ttl: return cached['data']
        
        df = None
        
        # Method 1: Direct Yahoo API (more reliable)
        try:
            # Convert period to timestamps
            period_days = {'1d': 1, '5d': 5, '1mo': 30, '3mo': 90, '6mo': 180, '1y': 365, '2y': 730, '5y': 1825}
            days = period_days.get(period, 180)
            end_ts = int(time.time())
            start_ts = end_ts - (days * 86400)
            
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={start_ts}&period2={end_ts}&interval={interval}"
            resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                result = data.get('chart', {}).get('result', [])
                if result and len(result) > 0:
                    timestamps = result[0].get('timestamp', [])
                    quote = result[0].get('indicators', {}).get('quote', [{}])[0]
                    
                    if timestamps and quote:
                        df = pd.DataFrame({
                            'Open': quote.get('open', []),
                            'High': quote.get('high', []),
                            'Low': quote.get('low', []),
                            'Close': quote.get('close', []),
                            'Volume': quote.get('volume', [])
                        }, index=pd.to_datetime(timestamps, unit='s'))
                        df = df.dropna()
                        if not df.empty:
                            if config.debug: print(f"  Debug [{symbol}] history from yahoo_api: {len(df)} rows")
        except Exception as e:
            if config.debug: print(f"  Debug [{symbol}] yahoo_api history error: {e}")
        
        # Method 2: yfinance (fallback)
        if df is None or df.empty:
            try:
                df = yf.Ticker(symbol).history(period=period, interval=interval)
                if df is not None and not df.empty:
                    if config.debug: print(f"  Debug [{symbol}] history from yfinance: {len(df)} rows")
            except Exception as e:
                if config.debug: print(f"  Debug [{symbol}] yfinance history error: {e}")
        
        if df is not None and not df.empty:
            self.history_cache[key] = {'time': _now(), 'data': df}
            return df
        
        return None

    def get_meta(self, symbol):
        symbol = symbol.upper()
        if symbol in self.meta_cache and (_now() - self.meta_cache[symbol]['time']) < self.meta_ttl: return self.meta_cache[symbol]['data']
        try:
            info = yf.Ticker(symbol).info
            data = {'sector': info.get('sector','Unknown'), 'beta': info.get('beta', 1.0), 'name': info.get('shortName', symbol)}
            self.meta_cache[symbol] = {'time': _now(), 'data': data}
            return data
        except: return {'sector':'Unknown', 'beta':1.0, 'name':symbol}

    def get_news(self, symbol):
        try: return yf.Ticker(symbol).news
        except: return []

    def get_calendar(self, symbol):
        try: 
            t = yf.Ticker(symbol)
            return {'earnings': t.earnings_dates, 'dividends': t.dividends}
        except: return None
    
    def clear_cache(self):
        self.stock_cache.clear(); self.option_cache.clear(); self.history_cache.clear()

# --- UNIVERSAL PARSER ---
# Supports: Fidelity, Schwab, E*Trade, Robinhood, TD Ameritrade, Vanguard, Interactive Brokers
class UniversalParser:
    ALIASES = {
        'symbol': ['Symbol', 'Ticker', 'Stock Symbol', 'Security', 'Instrument', 'Security ID'],
        'desc': ['Description', 'Security Description', 'Security Name', 'Instrument Description'],  # Removed 'Name' - too generic
        'qty': ['Quantity', 'Qty', 'Shares', 'Position', 'Share Quantity', 'Open Quantity', 'Current Qty'],
        'price': ['Last Price', 'Price', 'Close', 'Current Price', 'Market Price', 'Mkt Price', 'Last', 'Mark'],
        'value': ['Current Value', 'Market Value', 'Value', 'Amount', 'Total Value', 'Mkt Value', 'Position Value'],
        'cost': ['Cost Basis', 'Cost', 'Total Cost', 'Basis', 'Cost Basis Total', 'Purchase Price', 'Avg Cost', 'Average Cost'],
        'pnl': ['Total Gain/Loss', 'Gain/Loss', 'P&L', 'Unrealized', 'Total Gain/Loss Dollar', 'Unrealized Gain', 'Unrealized P/L'],
        'day_pnl': ["Today's Gain/Loss", "Day Gain", "Day's Gain/Loss Dollar", "Change $", "Day Change", "Daily P/L"],
        'type': ['Type', 'Security Type', 'Asset Type', 'Asset Class', 'Position Type']
    }
    
    # Cash/Money Market identifiers across brokers
    CASH_IDENTIFIERS = [
        'SPAXX', 'FDRXX', 'FCASH', 'CASH', 'MMDA', 'VMFXX', 'SWVXX', 'SPRXX',  # Fidelity, Schwab, Vanguard
        'MONEY MARKET', 'FDIC', 'CORE', 'SWEEP', 'GOVERNMENT MONEY', 'CASH & CASH INVESTMENTS',
        'FREE BALANCE', 'SETTLED CASH', 'CASH BALANCE', 'AVAILABLE CASH'
    ]
    
    # Option symbol formats by broker:
    # Fidelity/OCC Standard: -AAPL260117C00150000 or AAPL260117C00150000
    # Schwab: AAPL 01/17/2026 150.00 C
    # E*Trade: AAPL Jan 17 2026 150.0 Call
    # Robinhood: AAPL $150 Call 1/17/26
    OPTION_PATTERNS = [
        # OCC Standard: SYMBOL + YYMMDD + C/P + STRIKE (with optional leading -, spaces)
        (r'^[- ]*([A-Z]{1,5})(\d{6})([CP])(\d+\.?\d*)$', 'occ'),
        # Schwab style: SYMBOL MM/DD/YYYY STRIKE C/P
        (r'^([A-Z]{1,5})\s+(\d{1,2}/\d{1,2}/\d{4})\s+(\d+\.?\d*)\s*([CP])$', 'schwab'),
        # Verbose style: SYMBOL Mon DD YYYY STRIKE Call/Put
        (r'^([A-Z]{1,5})\s+([A-Za-z]{3})\s+(\d{1,2})\s+(\d{4})\s+\$?(\d+\.?\d*)\s+(Call|Put|C|P)$', 'verbose'),
        # Robinhood style: SYMBOL $STRIKE Call/Put MM/DD/YY
        (r'^([A-Z]{1,5})\s+\$(\d+\.?\d*)\s+(Call|Put)\s+(\d{1,2}/\d{1,2}/\d{2,4})$', 'robinhood'),
    ]

    def parse(self, filepath):
        fp = Path(os.path.expanduser(filepath)).resolve()
        if not fp.exists(): raise FileNotFoundError(f"File not found: {fp}")
        
        df, header_map, broker = self._load_and_map_columns(fp)
        stocks, options, skipped, cash = {}, [], [], 0
        print(f"\n  Detected broker format: {broker}")
        print(f"  Parsing {len(df)} rows using detected columns...")
        if config.debug:
            print(f"  Column mapping: {header_map}")
        
        for idx, row in df.iterrows():
            raw_sym = str(row.get(header_map.get('symbol', ''), '')).strip()
            desc = str(row.get(header_map.get('desc', ''), raw_sym)).strip().upper()
            row_type = str(row.get(header_map.get('type', ''), '')).strip().upper()
            
            # Skip empty/header rows
            if not raw_sym or raw_sym.upper() in ('', 'NAN', 'SYMBOL', 'TOTAL', 'ACCOUNT TOTAL', 'PENDING ACTIVITY'): 
                continue
            
            # Detect and accumulate cash positions
            if self._is_cash_position(raw_sym, desc, row_type):
                cash_val = safe_float(row.get(header_map.get('value', ''), 0))
                if cash_val > 0: cash += cash_val
                continue
            
            # Parse numeric fields with sign preservation
            qty = safe_float(row.get(header_map.get('qty', ''), 0))
            if abs(qty) < 0.0001: continue
            
            last_price = abs(safe_float(row.get(header_map.get('price', ''), 0)))
            current_value = safe_float(row.get(header_map.get('value', ''), 0))
            cost_basis = safe_float(row.get(header_map.get('cost', ''), 0))
            broker_pnl = safe_float(row.get(header_map.get('pnl', ''), 0))
            day_pnl = safe_float(row.get(header_map.get('day_pnl', ''), 0))
            
            # Try parsing as option first (check symbol AND description)
            opt = self._parse_option(raw_sym, desc, qty, cost_basis, last_price, current_value, broker_pnl, day_pnl)
            
            if opt:
                options.append(opt)
                print(f"    [OPT] {opt['symbol']:5} {opt['type'][0].upper()} ${opt['strike']:<8.2f} {opt['expiration']} qty:{opt['qty']:>3} {'(short)' if opt['qty'] < 0 else '(long)'}")
                continue
                
            # Otherwise parse as stock
            ticker = self._clean_ticker(raw_sym, desc)
            if ticker:
                if ticker in stocks:
                    stocks[ticker]['qty'] += qty
                    stocks[ticker]['cost'] += abs(cost_basis)
                    stocks[ticker]['broker_value'] += abs(current_value)
                    stocks[ticker]['broker_pnl'] += broker_pnl
                else:
                    stocks[ticker] = {
                        'qty': qty, 
                        'cost': abs(cost_basis), 
                        'avg': (abs(cost_basis)/abs(qty)) if qty else 0, 
                        'broker_price': last_price, 
                        'broker_value': abs(current_value), 
                        'broker_pnl': broker_pnl, 
                        'day_pnl': day_pnl,
                        # Legacy field names for compatibility
                        'fidelity_price': last_price,
                        'fidelity_value': abs(current_value),
                        'fidelity_pnl': broker_pnl
                    }
                print(f"    [STK] {ticker:5} qty:{qty:>10.4f} val:{fmt_money(current_value)}")
            else:
                skipped.append(raw_sym[:20])
                if config.debug:
                    print(f"    [SKIP] {raw_sym[:30]}")

        if skipped and config.debug:
            print(f"\n  Skipped {len(skipped)} unrecognized rows")
        print(f"\n  Parsed: {len(stocks)} stocks, {len(options)} options")
        if cash > 0: print(f"  Cash Detected: {fmt_money(cash)}")
        return stocks, options, cash

    def _is_cash_position(self, symbol, desc, row_type):
        """Detect cash/money market positions across all broker formats.
        
        Note: We check Symbol and Description, but NOT the Type column because
        brokers like Fidelity use 'Cash' as an account type for settled positions
        (vs 'Margin'), which is different from actual cash/money market holdings.
        """
        # Only check symbol and description - NOT row_type
        # row_type of 'Cash' just means the position is in a cash account, not that it IS cash
        combined = f"{symbol} {desc}".upper()
        
        # Check for money market fund symbols (exact matches on symbol)
        money_market_symbols = ['SPAXX', 'FDRXX', 'FCASH', 'VMFXX', 'SWVXX', 'SPRXX', 'FZFXX', 'FTEXX']
        symbol_clean = symbol.upper().replace('*', '').strip()
        if symbol_clean in money_market_symbols:
            return True
        
        # Check for cash-related descriptions
        cash_descriptions = [
            'MONEY MARKET', 'FDIC', 'CORE POSITION', 'SWEEP', 'GOVERNMENT MONEY',
            'CASH & CASH INVESTMENTS', 'FREE BALANCE', 'SETTLED CASH', 
            'CASH BALANCE', 'AVAILABLE CASH', 'HELD IN MONEY MARKET'
        ]
        return any(cash_desc in combined for cash_desc in cash_descriptions)

    def _load_and_map_columns(self, fp):
        """Load CSV and auto-detect column mappings and broker format"""
        with open(fp, 'r', encoding='utf-8-sig', errors='replace') as f:
            lines = f.readlines()
        
        # Find header row (skip broker disclaimers/metadata at top)
        header_row_idx = 0
        for i, line in enumerate(lines[:30]):
            line_lower = line.lower()
            has_symbol = any(k in line_lower for k in ['symbol', 'ticker', 'security', 'instrument'])
            has_qty = any(k in line_lower for k in ['quantity', 'qty', 'shares', 'position'])
            if has_symbol and has_qty:
                header_row_idx = i
                break
        
        df = pd.read_csv(fp, skiprows=header_row_idx, index_col=False)
        df.columns = [str(c).strip() for c in df.columns]
        
        # Auto-detect broker from column patterns
        broker = self._detect_broker(df.columns, lines[:10])
        
        # Build column mapping
        header_map = {}
        for col in df.columns:
            cl = col.lower().strip()
            for key, aliases in self.ALIASES.items():
                if key not in header_map:
                    for alias in aliases:
                        if alias.lower() == cl or alias.lower() in cl:
                            header_map[key] = col
                            break
        
        if 'symbol' not in header_map:
            header_map['symbol'] = df.columns[0]
            warn(f"Could not find 'Symbol' column, assuming '{df.columns[0]}'")
            
        return df, header_map, broker
    
    def _detect_broker(self, columns, first_lines):
        """Detect which broker the CSV came from"""
        cols_lower = ' '.join(c.lower() for c in columns)
        lines_lower = ' '.join(first_lines).lower()
        
        if 'fidelity' in lines_lower or 'account number' in cols_lower:
            return 'Fidelity'
        elif 'schwab' in lines_lower or 'charles schwab' in lines_lower:
            return 'Schwab'
        elif 'e*trade' in lines_lower or 'etrade' in lines_lower:
            return 'E*Trade'
        elif 'robinhood' in lines_lower:
            return 'Robinhood'
        elif 'td ameritrade' in lines_lower or 'thinkorswim' in lines_lower:
            return 'TD Ameritrade'
        elif 'vanguard' in lines_lower:
            return 'Vanguard'
        elif 'interactive brokers' in lines_lower or 'ibkr' in lines_lower:
            return 'Interactive Brokers'
        return 'Unknown (Generic)'

    def _clean_ticker(self, raw, desc):
        """Extract clean stock ticker from raw symbol string"""
        raw = raw.strip().upper()
        # Remove common broker prefixes (-, +, *, etc.)
        raw = re.sub(r'^[-+*\s]+', '', raw)
        # Remove trailing special chars
        raw = re.sub(r'[*]+$', '', raw)
        # Extract just letters/numbers
        clean = re.sub(r'[^A-Z0-9]', '', raw)
        
        # Valid stock ticker: 1-5 letters
        if 1 <= len(clean) <= 5 and clean.isalpha(): 
            return clean
        
        # Try to match from company name in description
        for company, ticker in COMPANY_TO_TICKER.items():
            if company in desc.upper(): 
                return ticker
        return None

    def _parse_option(self, raw_sym, desc, qty, cost, price, value, pnl, day_pnl):
        """
        Parse option from symbol or description string.
        
        CRITICAL: Position direction (long/short) is determined ONLY by:
        1. The SIGN of the quantity field from the CSV
        2. The SIGN of the current value field (negative value = liability = short)
        
        Symbol prefixes like '-' are IGNORED as they are inconsistent across brokers.
        """
        # Try each pattern against both symbol and description
        for source in [raw_sym, desc]:
            opt_data = self._try_parse_option_string(source)
            if opt_data:
                root, exp_date, strike, opt_type = opt_data
                
                # POSITION DIRECTION LOGIC (broker-agnostic):
                # Primary signal: quantity sign from CSV
                # Secondary signal: value sign (short positions often show negative value)
                contracts = int(round(qty))  # Preserve original sign from CSV
                
                # Cross-validate with value sign if available
                # Short options typically have negative current value (liability)
                if value < 0 and contracts > 0:
                    # Value is negative but qty is positive - this indicates short
                    contracts = -abs(contracts)
                elif value > 0 and contracts < 0:
                    # Value is positive but qty is negative - trust the qty sign
                    pass  # Keep contracts negative
                
                return {
                    'symbol': root, 
                    'type': opt_type, 
                    'strike': strike, 
                    'expiration': exp_date, 
                    'qty': contracts, 
                    'cost': abs(cost), 
                    'broker_price': price, 
                    'broker_value': abs(value), 
                    'broker_pnl': pnl, 
                    'day_pnl': day_pnl,
                    # Legacy field names for compatibility
                    'fidelity_price': price,
                    'fidelity_value': abs(value),
                    'fidelity_pnl': pnl
                }
        return None
    
    def _try_parse_option_string(self, s):
        """Try to parse an option from a string, returns (root, expiration, strike, type) or None"""
        if not s: return None
        clean = s.upper().replace(' ', '').replace('-', '')
        
        # Pattern 1: OCC Standard (most common) - AAPL260117C00150000
        m = re.match(r'^([A-Z]{1,5})(\d{6})([CP])(\d+\.?\d*)$', clean)
        if m:
            root, dstr, type_char, strike_str = m.groups()
            exp = f"20{dstr[0:2]}-{dstr[2:4]}-{dstr[4:6]}"
            strike = float(strike_str)
            # Handle strike encoding (150000 = $150.00)
            if strike > 10000 and '.' not in strike_str: 
                strike = strike / 1000.0
            opt_type = 'call' if type_char == 'C' else 'put'
            return (root, exp, strike, opt_type)
        
        # Pattern 2: Schwab - AAPL 01/17/2026 150.00 C
        m = re.match(r'^([A-Z]{1,5})\s+(\d{1,2})/(\d{1,2})/(\d{4})\s+(\d+\.?\d*)\s*([CP])$', s.upper().strip())
        if m:
            root, month, day, year, strike_str, type_char = m.groups()
            exp = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            strike = float(strike_str)
            opt_type = 'call' if type_char == 'C' else 'put'
            return (root, exp, strike, opt_type)
        
        # Pattern 3: Verbose - AAPL Jan 17 2026 $150 Call
        months = {'JAN':'01','FEB':'02','MAR':'03','APR':'04','MAY':'05','JUN':'06',
                  'JUL':'07','AUG':'08','SEP':'09','OCT':'10','NOV':'11','DEC':'12'}
        m = re.match(r'^([A-Z]{1,5})\s+([A-Z]{3})\s+(\d{1,2})[\s,]+(\d{4})\s+\$?(\d+\.?\d*)\s+(CALL|PUT|C|P)$', s.upper().strip())
        if m:
            root, mon, day, year, strike_str, type_str = m.groups()
            month = months.get(mon, '01')
            exp = f"{year}-{month}-{day.zfill(2)}"
            strike = float(strike_str)
            opt_type = 'call' if type_str in ('CALL', 'C') else 'put'
            return (root, exp, strike, opt_type)
        
        # Pattern 4: Robinhood - AAPL $150 Call 1/17/26
        m = re.match(r'^([A-Z]{1,5})\s+\$(\d+\.?\d*)\s+(CALL|PUT)\s+(\d{1,2})/(\d{1,2})/(\d{2,4})$', s.upper().strip())
        if m:
            root, strike_str, type_str, month, day, year = m.groups()
            if len(year) == 2: year = '20' + year
            exp = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            strike = float(strike_str)
            opt_type = 'call' if type_str == 'CALL' else 'put'
            return (root, exp, strike, opt_type)
        
        return None

# --- PORTFOLIO MANAGER ---
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
        print(Fore.CYAN + f"\n{'═'*60}\n IMPORTING CSV (UNIVERSAL PARSER)\n{'═'*60}" + Style.RESET_ALL)
        try:
            stocks, options, cash = UniversalParser().parse(filepath)
            self.data = {'stocks': stocks, 'options': options, 'cash': cash, 'imported': datetime.now().isoformat()}
            self._save()
            success(f"Imported {len(stocks)} stocks, {len(options)} options, {fmt_money(cash)} cash")
        except Exception as e: err(str(e))

    def display(self, fetcher):
        stocks, options, cash = self.data.get('stocks', {}), self.data.get('options', []), self.data.get('cash', 0)
        if not stocks and not options: warn("No positions. Use 'import FILE' first."); return
        
        print(Fore.CYAN + f"\n{'═'*95}\n PORTFOLIO — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (LIVE)\n{'═'*95}" + Style.RESET_ALL)
        
        # 1. Stocks Fetch
        all_symbols = set(stocks.keys()) | set(o['symbol'] for o in options)
        stock_prices = {}
        if all_symbols:
            prog = ProgressBar(len(all_symbols), "  Fetching Stocks")
            for i, sym in enumerate(all_symbols): 
                prog.update(i, sym)
                stock_prices[sym] = fetcher.get_stock_price(sym)
            prog.done()

        total_val, total_cost, total_pnl, total_day = 0, 0, 0, 0
        stock_holdings = {k: v['qty'] for k,v in stocks.items()}
        
        # --- STOCKS ---
        if stocks:
            print(f"\n{Style.BRIGHT}STOCKS{Style.RESET_ALL}\n" + "─"*95)
            rows = []
            for sym, pos in sorted(stocks.items()):
                live = stock_prices.get(sym, {})
                price, day_pct = live.get('price', 0), live.get('pct', 0)
                qty, cost = pos['qty'], pos['cost']
                
                if price > 0:
                    val = qty * price
                    pnl = val - cost
                    day = val * (day_pct/100)
                    d_str = color_pct(day_pct)
                else:
                    val = pos.get('fidelity_value', 0)
                    pnl = pos.get('fidelity_pnl', 0)
                    day = pos.get('day_pnl', 0)
                    d_str = "-"
                
                total_val += val; total_cost += cost; total_pnl += pnl; total_day += day
                rows.append([sym, f"{qty:.2f}", fmt_money(pos.get('avg',0)), fmt_money(cost), d_str, fmt_money(val), color_pnl(pnl, (pnl/cost*100) if cost else 0)])
            print(tabulate(rows, headers=["Symbol", "Qty", "Avg", "Cost", "Day%", "Value", "P&L"]))

        # --- OPTIONS ---
        # Map Long Calls for PMCC (Qty > 0)
        long_calls = {o['symbol']: [] for o in options if o['qty'] > 0 and o['type'] == 'call'}
        for o in options:
            if o['qty'] > 0 and o['type'] == 'call':
                long_calls[o['symbol']].append(o['expiration'])

        if options:
            print(f"\n{Style.BRIGHT}OPTIONS{Style.RESET_ALL}\n" + "─"*95)
            rows = []
            prog = ProgressBar(len(options), "  Updating")
            
            sorted_opts = sorted(options, key=lambda x: (x['expiration'], x['symbol']))
            
            for i, o in enumerate(sorted_opts):
                prog.update(i, o['symbol'])
                
                und_price = stock_prices.get(o['symbol'], {}).get('price', 0)
                opt_live = fetcher.get_option_price(o['symbol'], o['expiration'], o['strike'], o['type'], und_price)
                
                qty, cost = o['qty'], o['cost']
                price = opt_live['price'] if opt_live['price'] > 0 else o.get('fidelity_price', 0)
                
                # Logic
                is_short = qty < 0
                strat = "Long"
                
                if is_short:
                    if o['type'] == 'put': 
                        strat = "CSP"
                    else: # Short Call
                        shares = stock_holdings.get(o['symbol'], 0)
                        if shares >= abs(qty)*100: strat = "CC"
                        elif any(dte(x) > dte(o['expiration']) for x in long_calls.get(o['symbol'], [])): 
                            strat = "PMCC"
                        elif shares > 0: strat = "PtCC"
                        else: strat = "Naked"
                else:
                    strat = "Long"
                
                # Values & P&L
                intr = max(0, und_price - o['strike']) if o['type']=='call' else max(0, o['strike'] - und_price)
                extr = max(0, price - intr)
                days = dte(o['expiration'])
                
                # P&L Calculation: (Current Price - Avg Price) * Qty * 100
                avg_price = (cost / abs(qty) / 100) if abs(qty) > 0 else 0
                pnl = (price - avg_price) * qty * 100
                
                val = abs(qty) * price * 100
                if is_short: total_val -= val # Liability
                else: total_val += val # Asset
                
                total_cost += cost
                total_pnl += pnl
                
                if opt_live['price']>0: total_day += (val * 0.0) # Placeholder
                else: total_day += o.get('day_pnl', 0)
                
                desc = f"{o['symbol']} {o['expiration'][5:]} ${o['strike']:.0f}{o['type'][0].upper()}"
                
                rows.append([desc, strat, f"{qty:.0f}", fmt_money(cost), f"{days}d", f"I:{intr:.1f}/E:{extr:.1f}", fmt_money(val), color_pnl(pnl, (pnl/cost*100) if cost else 0)])
            
            prog.done()
            print(tabulate(rows, headers=["Option", "Strat", "Qty", "Cost", "DTE", "Intr/Extr", "Value", "P&L"]))

        print(f"\n{'═'*95}")
        print(f"  {Style.BRIGHT}NET LIQ:{Style.RESET_ALL}      {fmt_money(total_val + cash)}")
        print(f"  {Style.BRIGHT}TOTAL P&L:{Style.RESET_ALL}    {color_pnl(total_pnl, (total_pnl/total_cost*100) if total_cost else 0)}")
        if cash: print(f"  {Style.BRIGHT}CASH:{Style.RESET_ALL}         {fmt_money(cash)}")
        print(f"{'═'*95}\n")

    def analyze_risk(self, fetcher):
        print(Fore.CYAN + f"\n{'═'*60}\n RISK & BETA ANALYSIS\n{'═'*60}" + Style.RESET_ALL)
        stocks = self.data.get('stocks', {})
        if not stocks: warn("No stocks."); return
        bench = fetcher.get_history(config.benchmark, "1y")
        if bench is None: err("Benchmark error"); return
        bench_ret = bench['Close'].pct_change().dropna()
        
        rows, w_beta, w_tot = [], 0, 0
        for sym, pos in stocks.items():
            val = pos['fidelity_value']
            if val < 100: continue
            h = fetcher.get_history(sym, "1y")
            beta = 1.0
            if h is not None:
                comb = pd.concat([h['Close'].pct_change(), bench_ret], axis=1).dropna()
                if len(comb) > 30: beta = linregress(comb.iloc[:,1], comb.iloc[:,0])[0]
            w_beta += beta * val; w_tot += val
            rows.append([sym, fmt_money(val), f"{beta:.2f}"])
        
        rows.sort(key=lambda x: float(x[2]), reverse=True)
        print(tabulate(rows[:8], headers=["Top Holdings", "Value", "Beta"]))
        pf_beta = w_beta / w_tot if w_tot else 1.0
        print(f"\n  Portfolio Beta: {pf_beta:.2f} ({'High Volatility' if pf_beta>1.2 else 'Defensive' if pf_beta<0.8 else 'Market'})")

    def calendar(self, fetcher):
        print(Fore.CYAN + f"\n{'═'*60}\n EARNINGS SCANNER\n{'═'*60}" + Style.RESET_ALL)
        events = []
        for sym in self.data.get('stocks', {}):
            cal = fetcher.get_calendar(sym)
            if cal and cal['earnings'] is not None:
                now = pd.Timestamp.now().tz_localize(None)
                fut = cal['earnings'].index[cal['earnings'].index > now]
                if not fut.empty:
                    dt = fut[-1].tz_localize(None) if fut[-1].tzinfo else fut[-1]
                    days = (dt - now).days
                    if days < 45: events.append([sym, dt.strftime("%Y-%m-%d"), f"{days}d"])
        print(tabulate(sorted(events, key=lambda x: x[2]), headers=["Symbol", "Date", "In"]))

# --- WATCHLIST ---
class Watchlist:
    def __init__(self):
        self.file = Path(config.data_dir) / "watchlist.json"
        self.symbols = json.loads(self.file.read_text()) if self.file.exists() else []
    def save(self): self.file.write_text(json.dumps(self.symbols))
    def add(self, s): self.symbols.append(s.upper()) if s.upper() not in self.symbols else None; self.save()
    def remove(self, s): self.symbols.remove(s.upper()) if s.upper() in self.symbols else None; self.save()
    def show(self, fetcher):
        rows = [[s, fmt_money(d['price']), color_pct(d['pct'])] if (d:=fetcher.get_stock_price(s))['price']>0 else [s,"-","-"] for s in self.symbols]
        print(tabulate(rows, headers=["Symbol", "Price", "Change"]))

def print_help():
    print(Fore.CYAN + """
╔══════════════════════════════════════════════════════════════════════════╗
║  STOCK TICKER v10.1.0 — HELP & MANUAL                                    ║
╠══════════════════════════════════════════════════════════════════════════╣
║  SMART CONTEXT: Commands remember last symbol. Just type 'ta' again!     ║
║                                                                          ║
║  PORTFOLIO COMMANDS:                                                     ║
║  pf              View full portfolio with P&L, Strategy, and Greeks      ║
║  import FILE     Load CSV (Fidelity, Schwab, E*Trade, Robinhood)         ║
║  risk            Calculate Portfolio Beta (volatility vs S&P 500)        ║
║  cal             Scan portfolio for upcoming earnings                    ║
║  clear           Delete all portfolio data (reset)                       ║
║                                                                          ║
║  QUOTE & LOOKUP:                                                         ║
║  AAPL            Just type ticker for quick quote (sets context)         ║
║  q SYMBOL        Quick quote (e.g., 'q AAPL')                            ║
║  quote SYMBOL    Detailed stock quote with fundamentals                  ║
║  info SYMBOL     Full company info and metrics                           ║
║  dash SYMBOL     Dashboard: price, trend, and headlines                  ║
║  news SYMBOL     Latest news headlines for a stock                       ║
║                                                                          ║
║  TECHNICAL ANALYSIS:                                                     ║
║  ta SYMBOL       Basic TA (RSI, MACD, Bollinger, Supertrend)             ║
║  ta2 SYMBOL      Extended TA (Stochastic, ADX, Ichimoku, Williams %R)    ║
║  trend SYMBOL    Trend strength analysis with scoring                    ║
║  levels SYMBOL   Support/Resistance and Fibonacci levels                 ║
║  compare A B     Compare two stocks side-by-side                         ║
║                                                                          ║
║  WATCHLIST:                                                              ║
║  watch           View watchlist                                          ║
║  watch add SYM   Add symbol to watchlist                                 ║
║  watch rm SYM    Remove symbol from watchlist                            ║
║                                                                          ║
║  SETTINGS:                                                               ║
║  refresh         Clear cached data for fresh updates                     ║
║  debug           Toggle debug mode                                       ║
║                                                                          ║
║  STRATEGY LEGEND:                                                        ║
║  Long  = Bought option     CSP  = Cash Secured Put (short put)           ║
║  CC    = Covered Call      PMCC = Poor Man's Covered Call                ║
║  Naked = Uncovered short call                                            ║
╚══════════════════════════════════════════════════════════════════════════╝
""" + Style.RESET_ALL)

# --- MAIN ---
def main():
    print(Fore.CYAN + Style.BRIGHT + "\n    STOCK TICKER v10.1.0 — Universal + Smart Context\n" + Style.RESET_ALL)
    fetcher, pf, wl = PriceFetcher(), Portfolio(), Watchlist()
    last_symbol = None  # Context memory
    
    while True:
        try:
            # Smart prompt showing active context if available
            prompt_sym = f"[{last_symbol}] " if last_symbol else ""
            raw = input(Fore.GREEN + f"▶ {prompt_sym}" + Style.RESET_ALL).strip()
            if not raw: continue
            
            parts = shlex.split(raw)
            cmd = parts[0].lower()
            args = parts[1:]
            
            # --- CONTEXT HANDLING ---
            # If command requires a symbol but none provided, use last_symbol
            symbol_commands = ('quote', 'q', 'ta', 'ta2', 'trend', 'levels', 'info', 'news', 'dash')
            
            if cmd in symbol_commands:
                if args:
                    last_symbol = args[0].upper()  # Update context
                elif last_symbol:
                    args = [last_symbol]  # Inject context
                    print(Fore.YELLOW + f"  Using active ticker: {last_symbol}" + Style.RESET_ALL)
                else:
                    err(f"Usage: {cmd} <symbol>"); continue
            
            # --- COMMANDS ---
            if cmd in ('quit', 'exit'): break
            elif cmd == 'import': pf.import_csv(args[0]) if args else err("Usage: import <file>")
            elif cmd in ('pf', 'portfolio'): pf.display(fetcher)
            elif cmd == 'risk': pf.analyze_risk(fetcher)
            elif cmd == 'cal': pf.calendar(fetcher)
            elif cmd == 'watch': 
                if not args: wl.show(fetcher)
                elif args[0]=='add': wl.add(args[1])
                elif args[0] in ('rm','del'): wl.remove(args[1])
            elif cmd == 'clear': pf.clear()
            elif cmd == 'debug': config.debug = not config.debug; print(f"Debug: {config.debug}")
            elif cmd == 'refresh': fetcher.clear_cache(); success("Cache cleared")
            elif cmd in ('help', 'h', '?'): print_help()
            
            # --- NEWS COMMAND ---
            elif cmd == 'news' and args:
                symbol = args[0].upper()
                print(Fore.CYAN + f"\n{'═'*70}\n NEWS — {symbol}\n{'═'*70}" + Style.RESET_ALL)
                news_items = fetcher.get_news(symbol)
                if not news_items: warn("No recent news found.")
                else:
                    for i, item in enumerate(news_items[:5]):
                        pub = datetime.fromtimestamp(item.get('providerPublishTime', 0)).strftime('%Y-%m-%d %H:%M')
                        print(f"\n  {Style.BRIGHT}{item.get('title')}{Style.RESET_ALL}")
                        print(f"  {Fore.CYAN}{item.get('publisher')}{Style.RESET_ALL} • {pub}")
                        if item.get('link'): print(f"  {item.get('link')}")
                print(f"{'═'*70}\n")

            # --- DASHBOARD COMMAND ---
            elif cmd == 'dash' and args:
                symbol = args[0].upper()
                print(Fore.CYAN + f"\n{'═'*70}\n DASHBOARD — {symbol}\n{'═'*70}" + Style.RESET_ALL)
                
                # Parallel fetch for dashboard components
                with ThreadPoolExecutor() as executor:
                    f_price = executor.submit(fetcher.get_stock_price, symbol)
                    f_hist = executor.submit(fetcher.get_history, symbol, "6mo")
                    f_meta = executor.submit(fetcher.get_meta, symbol)
                    f_news = executor.submit(fetcher.get_news, symbol)
                    
                    d = f_price.result()
                    df = f_hist.result()
                    meta = f_meta.result()
                    news = f_news.result()

                # 1. Price Header
                print(f"\n{Style.BRIGHT}{meta.get('name', symbol)}{Style.RESET_ALL} ({meta.get('sector', 'Unknown')})")
                print(f"  {Style.BRIGHT}PRICE:{Style.RESET_ALL}   {fmt_money(d['price'])}  {color_money(d['change'])} ({color_pct(d['pct'])})")
                
                # 2. Trend Snapshot
                if df is not None and len(df) > 30:
                    trend = TechnicalAnalysis.trend_strength(df)
                    rsi = TechnicalAnalysis.rsi(df['Close']).iloc[-1]
                    print(f"  {Style.BRIGHT}TREND:{Style.RESET_ALL}   {color_signal(trend['strength'])} (Score: {trend['score']})")
                    print(f"  {Style.BRIGHT}RSI:{Style.RESET_ALL}     {rsi:.1f} ({color_signal('OVERSOLD' if rsi<30 else 'OVERBOUGHT' if rsi>70 else 'Neutral')})")
                    
                    # Mini Sparkline (Text based)
                    closes = df['Close'].tail(10).tolist()
                    min_c, max_c = min(closes), max(closes)
                    spark = ""
                    if max_c > min_c:
                        chars = "  ▂▃▄▅▆▇█"
                        spark = "".join([chars[int((c - min_c) / (max_c - min_c) * 8)] for c in closes])
                    print(f"  {Style.BRIGHT}10D:{Style.RESET_ALL}     {spark}  {fmt_money(closes[0])} -> {fmt_money(closes[-1])}")

                # 3. Recent Headlines
                if news:
                    print(f"\n{Style.BRIGHT}LATEST HEADLINES{Style.RESET_ALL}")
                    for item in news[:3]:
                        title = item.get('title', '')[:60]
                        print(f"  • {title}...")
                print(f"{'═'*70}\n")

            elif cmd == 'ta' and args:
                # Technical Analysis command: ta SYMBOL
                symbol = args[0].upper()
                print(Fore.CYAN + f"\n{'═'*70}\n TECHNICAL ANALYSIS — {symbol}\n{'═'*70}" + Style.RESET_ALL)
                df = fetcher.get_history(symbol, "6mo")
                if df is None or len(df) < 30:
                    err(f"Insufficient data for {symbol}")
                else:
                    close = df['Close']
                    high, low = df['High'], df['Low']
                    price = close.iloc[-1]
                    
                    # Price Info
                    d = fetcher.get_stock_price(symbol)
                    print(f"\n{Style.BRIGHT}PRICE{Style.RESET_ALL}")
                    print(f"  Current:      {fmt_money(price)}")
                    print(f"  Day Change:   {color_pct(d['pct'])}")
                    print(f"  52W High:     {fmt_money(high.max())}")
                    print(f"  52W Low:      {fmt_money(low.min())}")
                    pct_from_high = ((price - high.max()) / high.max()) * 100
                    print(f"  From 52W High: {color_pct(pct_from_high)}")
                    
                    # RSI
                    rsi = TechnicalAnalysis.rsi(close)
                    rsi_val = rsi.iloc[-1]
                    rsi_signal = "OVERSOLD" if rsi_val < 30 else "OVERBOUGHT" if rsi_val > 70 else "Neutral"
                    print(f"\n{Style.BRIGHT}RSI (14){Style.RESET_ALL}")
                    print(f"  Value:  {rsi_val:.2f}")
                    print(f"  Signal: {color_signal(rsi_signal)}")
                    
                    # MACD
                    macd_line, signal_line, histogram = TechnicalAnalysis.macd(close)
                    macd_signal = "BULLISH" if histogram.iloc[-1] > 0 else "BEARISH"
                    print(f"\n{Style.BRIGHT}MACD (12/26/9){Style.RESET_ALL}")
                    print(f"  MACD Line:    {macd_line.iloc[-1]:.4f}")
                    print(f"  Signal Line:  {signal_line.iloc[-1]:.4f}")
                    print(f"  Histogram:    {histogram.iloc[-1]:.4f}")
                    print(f"  Signal:       {color_signal(macd_signal)}")
                    
                    # Bollinger Bands
                    upper, middle, lower = TechnicalAnalysis.bollinger_bands(close)
                    bb_pct = (price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1]) * 100 if (upper.iloc[-1] - lower.iloc[-1]) > 0 else 50
                    bb_signal = "OVERSOLD" if bb_pct < 20 else "OVERBOUGHT" if bb_pct > 80 else "Neutral"
                    print(f"\n{Style.BRIGHT}BOLLINGER BANDS (20/2){Style.RESET_ALL}")
                    print(f"  Upper:    {fmt_money(upper.iloc[-1])}")
                    print(f"  Middle:   {fmt_money(middle.iloc[-1])}")
                    print(f"  Lower:    {fmt_money(lower.iloc[-1])}")
                    print(f"  %B:       {bb_pct:.1f}%")
                    print(f"  Signal:   {color_signal(bb_signal)}")
                    
                    # Supertrend
                    trend, st_upper, st_lower = TechnicalAnalysis.supertrend(df)
                    st_signal = "BULLISH" if trend.iloc[-1] == 1 else "BEARISH"
                    print(f"\n{Style.BRIGHT}SUPERTREND (10/3){Style.RESET_ALL}")
                    print(f"  Signal: {color_signal(st_signal)}")
                    
                    # ATR
                    atr = TechnicalAnalysis.atr(df)
                    atr_pct = (atr.iloc[-1] / price) * 100
                    print(f"\n{Style.BRIGHT}ATR (14){Style.RESET_ALL}")
                    print(f"  Value:    {fmt_money(atr.iloc[-1])}")
                    print(f"  % of Price: {atr_pct:.2f}%")
                    
                    # Pivot Points
                    pivots = TechnicalAnalysis.pivot_points(df)
                    print(f"\n{Style.BRIGHT}PIVOT POINTS{Style.RESET_ALL}")
                    print(f"  R3: {fmt_money(pivots['R3'])}  R2: {fmt_money(pivots['R2'])}  R1: {fmt_money(pivots['R1'])}")
                    print(f"  Pivot: {fmt_money(pivots['P'])}")
                    print(f"  S1: {fmt_money(pivots['S1'])}  S2: {fmt_money(pivots['S2'])}  S3: {fmt_money(pivots['S3'])}")
                    
                    # Moving Averages
                    sma_20 = close.rolling(20).mean().iloc[-1]
                    sma_50 = close.rolling(50).mean().iloc[-1]
                    sma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None
                    print(f"\n{Style.BRIGHT}MOVING AVERAGES{Style.RESET_ALL}")
                    print(f"  SMA 20:  {fmt_money(sma_20)} {'▲' if price > sma_20 else '▼'}")
                    print(f"  SMA 50:  {fmt_money(sma_50)} {'▲' if price > sma_50 else '▼'}")
                    if sma_200:
                        print(f"  SMA 200: {fmt_money(sma_200)} {'▲' if price > sma_200 else '▼'}")
                    
                    # Overall Score
                    signals = [rsi_signal, macd_signal, bb_signal, st_signal]
                    bull_count = sum(1 for s in signals if "BULL" in s or "OVERSOLD" in s)
                    bear_count = sum(1 for s in signals if "BEAR" in s or "OVERBOUGHT" in s)
                    overall = "BULLISH" if bull_count > bear_count else "BEARISH" if bear_count > bull_count else "NEUTRAL"
                    print(f"\n{Style.BRIGHT}OVERALL{Style.RESET_ALL}")
                    print(f"  Score: {bull_count} Bull / {bear_count} Bear")
                    print(f"  Bias:  {color_signal(overall)}")
                    print(f"{'═'*70}\n")
            
            elif cmd == 'compare' and len(args) >= 2:
                # Compare two stocks: compare SYMBOL1 SYMBOL2
                sym1, sym2 = args[0].upper(), args[1].upper()
                print(Fore.CYAN + f"\n{'═'*80}\n COMPARISON — {sym1} vs {sym2}\n{'═'*80}" + Style.RESET_ALL)
                
                df1, df2 = fetcher.get_history(sym1, "6mo"), fetcher.get_history(sym2, "6mo")
                d1, d2 = fetcher.get_stock_price(sym1), fetcher.get_stock_price(sym2)
                
                if df1 is None or df2 is None or len(df1) < 30 or len(df2) < 30:
                    err("Insufficient data for comparison")
                else:
                    # Helper to get all metrics
                    def get_metrics(df, d):
                        close = df['Close']
                        price = close.iloc[-1]
                        rsi = TechnicalAnalysis.rsi(close).iloc[-1]
                        _, _, hist = TechnicalAnalysis.macd(close)
                        upper, middle, lower = TechnicalAnalysis.bollinger_bands(close)
                        bb_pct = (price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1]) * 100 if (upper.iloc[-1] - lower.iloc[-1]) > 0 else 50
                        trend, _, _ = TechnicalAnalysis.supertrend(df)
                        atr = TechnicalAnalysis.atr(df).iloc[-1]
                        
                        # Performance
                        pct_1w = ((price - close.iloc[-5]) / close.iloc[-5] * 100) if len(close) >= 5 else 0
                        pct_1m = ((price - close.iloc[-21]) / close.iloc[-21] * 100) if len(close) >= 21 else 0
                        pct_3m = ((price - close.iloc[-63]) / close.iloc[-63] * 100) if len(close) >= 63 else 0
                        
                        return {
                            'price': price, 'day_pct': d['pct'],
                            'pct_1w': pct_1w, 'pct_1m': pct_1m, 'pct_3m': pct_3m,
                            'rsi': rsi, 'macd_hist': hist.iloc[-1], 'bb_pct': bb_pct,
                            'trend': trend.iloc[-1], 'atr': atr, 'atr_pct': (atr/price)*100,
                            'volatility': close.pct_change().std() * np.sqrt(252) * 100
                        }
                    
                    m1, m2 = get_metrics(df1, d1), get_metrics(df2, d2)
                    
                    # Build comparison table
                    def winner(v1, v2, higher_better=True):
                        if higher_better:
                            return (Fore.GREEN + "◀" + Style.RESET_ALL, "") if v1 > v2 else ("", Fore.GREEN + "▶" + Style.RESET_ALL) if v2 > v1 else ("", "")
                        else:
                            return (Fore.GREEN + "◀" + Style.RESET_ALL, "") if v1 < v2 else ("", Fore.GREEN + "▶" + Style.RESET_ALL) if v2 < v1 else ("", "")
                    
                    rows = []
                    
                    # Price
                    rows.append(["Price", fmt_money(m1['price']), "", fmt_money(m2['price'])])
                    
                    # Performance
                    w = winner(m1['day_pct'], m2['day_pct'])
                    rows.append(["Day %", color_pct(m1['day_pct']), w[0], color_pct(m2['day_pct']) + " " + w[1]])
                    w = winner(m1['pct_1w'], m2['pct_1w'])
                    rows.append(["1 Week %", color_pct(m1['pct_1w']), w[0], color_pct(m2['pct_1w']) + " " + w[1]])
                    w = winner(m1['pct_1m'], m2['pct_1m'])
                    rows.append(["1 Month %", color_pct(m1['pct_1m']), w[0], color_pct(m2['pct_1m']) + " " + w[1]])
                    w = winner(m1['pct_3m'], m2['pct_3m'])
                    rows.append(["3 Month %", color_pct(m1['pct_3m']), w[0], color_pct(m2['pct_3m']) + " " + w[1]])
                    
                    rows.append(["─"*12, "─"*15, "─", "─"*15])
                    
                    # Technicals
                    rsi1_sig = "OVERSOLD" if m1['rsi'] < 30 else "OVERBOUGHT" if m1['rsi'] > 70 else "Neutral"
                    rsi2_sig = "OVERSOLD" if m2['rsi'] < 30 else "OVERBOUGHT" if m2['rsi'] > 70 else "Neutral"
                    rows.append(["RSI", f"{m1['rsi']:.1f} ({color_signal(rsi1_sig)})", "", f"{m2['rsi']:.1f} ({color_signal(rsi2_sig)})"])
                    
                    macd1_sig = "BULLISH" if m1['macd_hist'] > 0 else "BEARISH"
                    macd2_sig = "BULLISH" if m2['macd_hist'] > 0 else "BEARISH"
                    rows.append(["MACD", color_signal(macd1_sig), "", color_signal(macd2_sig)])
                    
                    bb1_sig = "OVERSOLD" if m1['bb_pct'] < 20 else "OVERBOUGHT" if m1['bb_pct'] > 80 else "Neutral"
                    bb2_sig = "OVERSOLD" if m2['bb_pct'] < 20 else "OVERBOUGHT" if m2['bb_pct'] > 80 else "Neutral"
                    rows.append(["Bollinger %B", f"{m1['bb_pct']:.0f}% ({color_signal(bb1_sig)})", "", f"{m2['bb_pct']:.0f}% ({color_signal(bb2_sig)})"])
                    
                    st1_sig = "BULLISH" if m1['trend'] == 1 else "BEARISH"
                    st2_sig = "BULLISH" if m2['trend'] == 1 else "BEARISH"
                    rows.append(["Supertrend", color_signal(st1_sig), "", color_signal(st2_sig)])
                    
                    rows.append(["─"*12, "─"*15, "─", "─"*15])
                    
                    # Risk
                    w = winner(m1['volatility'], m2['volatility'], higher_better=False)
                    rows.append(["Volatility", f"{m1['volatility']:.1f}%", w[0], f"{m2['volatility']:.1f}% " + w[1]])
                    w = winner(m1['atr_pct'], m2['atr_pct'], higher_better=False)
                    rows.append(["ATR %", f"{m1['atr_pct']:.2f}%", w[0], f"{m2['atr_pct']:.2f}% " + w[1]])
                    
                    print(tabulate(rows, headers=["Metric", sym1, "", sym2], tablefmt="simple"))
                    print(f"{'═'*80}\n")
            
            # Extended Technical Analysis
            elif cmd == 'ta2' and args:
                symbol = args[0].upper()
                df = fetcher.get_history(symbol, "1y")
                if df is None or len(df) < 50: err(f"Insufficient data for {symbol}")
                else:
                    close, price = df['Close'], df['Close'].iloc[-1]
                    d = fetcher.get_stock_price(symbol)
                    print(Fore.CYAN + f"\n{'═'*70}\n EXTENDED TA — {symbol}\n{'═'*70}" + Style.RESET_ALL)
                    print(f"\n{Style.BRIGHT}PRICE{Style.RESET_ALL}: {fmt_money(price)} ({color_pct(d['pct'])})")
                    stoch_k, stoch_d = TechnicalAnalysis.stochastic(df)
                    k_val = stoch_k.iloc[-1]
                    print(f"\n{Style.BRIGHT}STOCHASTIC{Style.RESET_ALL}: %K={k_val:.1f} %D={stoch_d.iloc[-1]:.1f} -> {color_signal('OVERSOLD' if k_val < 20 else 'OVERBOUGHT' if k_val > 80 else 'Neutral')}")
                    adx, plus_di, minus_di = TechnicalAnalysis.adx(df)
                    adx_val = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0
                    print(f"{Style.BRIGHT}ADX{Style.RESET_ALL}: {adx_val:.1f} +DI={plus_di.iloc[-1]:.1f} -DI={minus_di.iloc[-1]:.1f} -> {color_signal('BULLISH' if plus_di.iloc[-1] > minus_di.iloc[-1] else 'BEARISH')}")
                    wr = TechnicalAnalysis.williams_r(df).iloc[-1]
                    print(f"{Style.BRIGHT}WILLIAMS %R{Style.RESET_ALL}: {wr:.1f} -> {color_signal('OVERSOLD' if wr < -80 else 'OVERBOUGHT' if wr > -20 else 'Neutral')}")
                    cci = TechnicalAnalysis.cci(df).iloc[-1]
                    print(f"{Style.BRIGHT}CCI{Style.RESET_ALL}: {cci:.1f} -> {color_signal('OVERSOLD' if cci < -100 else 'OVERBOUGHT' if cci > 100 else 'Neutral')}")
                    mfi = TechnicalAnalysis.mfi(df).iloc[-1]
                    print(f"{Style.BRIGHT}MFI{Style.RESET_ALL}: {mfi:.1f} -> {color_signal('OVERSOLD' if mfi < 20 else 'OVERBOUGHT' if mfi > 80 else 'Neutral')}")
                    vwap = TechnicalAnalysis.vwap(df).iloc[-1]
                    print(f"{Style.BRIGHT}VWAP{Style.RESET_ALL}: {fmt_money(vwap)} -> {color_signal('BULLISH' if price > vwap else 'BEARISH')}")
                    ich = TechnicalAnalysis.ichimoku(df)
                    print(f"{Style.BRIGHT}ICHIMOKU{Style.RESET_ALL}: Tenkan={fmt_money(ich['tenkan_sen'].iloc[-1])} Kijun={fmt_money(ich['kijun_sen'].iloc[-1])} -> {color_signal('BULLISH' if ich['tenkan_sen'].iloc[-1] > ich['kijun_sen'].iloc[-1] else 'BEARISH')}")
                    obv = TechnicalAnalysis.obv(df)
                    print(f"{Style.BRIGHT}OBV{Style.RESET_ALL}: {color_signal('BULLISH' if obv.iloc[-1] > obv.rolling(20).mean().iloc[-1] else 'BEARISH')}")
                    print(f"{'═'*70}\n")
            
            # Trend Analysis
            elif cmd == 'trend' and args:
                symbol = args[0].upper()
                df = fetcher.get_history(symbol, "1y")
                if df is None or len(df) < 50: err(f"Insufficient data for {symbol}")
                else:
                    close, price = df['Close'], df['Close'].iloc[-1]
                    d = fetcher.get_stock_price(symbol)
                    print(Fore.CYAN + f"\n{'═'*70}\n TREND — {symbol}\n{'═'*70}" + Style.RESET_ALL)
                    print(f"\n{Style.BRIGHT}CURRENT{Style.RESET_ALL}: {fmt_money(price)} ({color_pct(d['pct'])})")
                    trend = TechnicalAnalysis.trend_strength(df)
                    print(f"\n{Style.BRIGHT}TREND STRENGTH{Style.RESET_ALL}")
                    print(f"  Score: {trend['score']}/9  Strength: {color_signal(trend['strength'])}  ADX: {trend['adx']:.1f}")
                    sma_20, sma_50 = close.rolling(20).mean().iloc[-1], close.rolling(50).mean().iloc[-1]
                    print(f"\n{Style.BRIGHT}PERFORMANCE{Style.RESET_ALL}")
                    for label, days in [('1 Week', 5), ('1 Month', 21), ('3 Months', 63)]:
                        if len(close) > days: print(f"  {label}: {color_pct((price - close.iloc[-days]) / close.iloc[-days] * 100)}")
                    print(f"{'═'*70}\n")
            
            # Support/Resistance Levels
            elif cmd == 'levels' and args:
                symbol = args[0].upper()
                df = fetcher.get_history(symbol, "6mo")
                if df is None or len(df) < 50: err(f"Insufficient data for {symbol}")
                else:
                    price = df['Close'].iloc[-1]
                    d = fetcher.get_stock_price(symbol)
                    print(Fore.CYAN + f"\n{'═'*70}\n LEVELS — {symbol}\n{'═'*70}" + Style.RESET_ALL)
                    print(f"\n{Style.BRIGHT}CURRENT{Style.RESET_ALL}: {fmt_money(price)} ({color_pct(d['pct'])})")
                    fib = TechnicalAnalysis.fibonacci_retracement(df)
                    print(f"\n{Style.BRIGHT}FIBONACCI{Style.RESET_ALL}")
                    for lvl, val in fib.items():
                        ind = " <--" if abs(price - val) / price < 0.01 else ""
                        print(f"  {lvl:<15} {fmt_money(val)}{ind}")
                    pivots = TechnicalAnalysis.pivot_points(df)
                    print(f"\n{Style.BRIGHT}PIVOTS{Style.RESET_ALL}")
                    print(f"  R3: {fmt_money(pivots['R3'])}  R2: {fmt_money(pivots['R2'])}  R1: {fmt_money(pivots['R1'])}")
                    print(f"  Pivot: {fmt_money(pivots['P'])}")
                    print(f"  S1: {fmt_money(pivots['S1'])}  S2: {fmt_money(pivots['S2'])}  S3: {fmt_money(pivots['S3'])}")
                    sr = TechnicalAnalysis.support_resistance(df)
                    print(f"\n{Style.BRIGHT}SUPPORT/RESISTANCE{Style.RESET_ALL}")
                    for i, r in enumerate(sr['resistance'][:3], 1): print(f"  R{i}: {fmt_money(r)}")
                    for i, s in enumerate(sr['support'][:3], 1): print(f"  S{i}: {fmt_money(s)}")
                    print(f"{'═'*70}\n")
            
            # Detailed Quote
            elif cmd in ('quote', 'info') and args:
                symbol = args[0].upper()
                d = fetcher.get_stock_price(symbol)
                meta = fetcher.get_meta(symbol)
                print(Fore.CYAN + f"\n{'═'*70}\n QUOTE — {symbol}\n{'═'*70}" + Style.RESET_ALL)
                print(f"\n{Style.BRIGHT}{meta.get('name', symbol)}{Style.RESET_ALL}")
                print(f"  Sector: {meta.get('sector', 'N/A')}  Industry: {meta.get('industry', 'N/A')}")
                print(f"\n{Style.BRIGHT}PRICE{Style.RESET_ALL}")
                print(f"  Current: {fmt_money(d['price'])}  Change: {color_money(d['change'])} ({color_pct(d['pct'])})")
                h52, l52 = meta.get('52w_high', 0), meta.get('52w_low', 0)
                if h52 and l52:
                    print(f"\n{Style.BRIGHT}52 WEEK{Style.RESET_ALL}")
                    print(f"  High: {fmt_money(h52)}  Low: {fmt_money(l52)}  Position: {(d['price'] - l52) / (h52 - l52) * 100:.0f}%")
                print(f"\n{Style.BRIGHT}FUNDAMENTALS{Style.RESET_ALL}")
                mc = meta.get('market_cap', 0)
                if mc: print(f"  Market Cap: ${mc/1e9:.2f}B" if mc >= 1e9 else f"  Market Cap: ${mc/1e6:.2f}M")
                if meta.get('pe_ratio'): print(f"  P/E: {meta['pe_ratio']:.2f}")
                if meta.get('eps'): print(f"  EPS: {fmt_money(meta['eps'])}")
                if meta.get('dividend_yield'): print(f"  Div Yield: {meta['dividend_yield']*100:.2f}%")
                if meta.get('beta'): print(f"  Beta: {meta['beta']:.2f}")
                if meta.get('target_price'):
                    print(f"\n{Style.BRIGHT}ANALYST{Style.RESET_ALL}")
                    print(f"  Target: {fmt_money(meta['target_price'])}  Upside: {color_pct((meta['target_price'] - d['price']) / d['price'] * 100)}")
                print(f"{'═'*70}\n")
            
            elif cmd == 'q' and args:
                # Quote command: q SYMBOL
                symbol = args[0].upper()
                d = fetcher.get_stock_price(symbol)
                print(f"  {symbol}: {fmt_money(d['price'])} ({color_pct(d['pct'])})")
                last_symbol = symbol  # Update context
            elif len(cmd) <= 5 and cmd.isalpha(): 
                # Direct ticker lookup (e.g., typing "AAPL" directly)
                d = fetcher.get_stock_price(cmd)
                print(f"  {cmd.upper()}: {fmt_money(d['price'])} ({color_pct(d['pct'])})")
                last_symbol = cmd.upper()  # Update context
        except Exception as e: err(str(e))

if __name__ == "__main__":
    main()