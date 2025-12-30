#!/usr/bin/env python3
"""
STOCK TICKER APP v11.0.0 — Universal Portfolio Tracker & Analyzer
INSTALL: pip install yfinance pandas numpy matplotlib tabulate colorama requests scipy

NEW IN v11.0.0:
- 🚨 Price Alerts (alert): Set price targets with notifications
- 📊 Sector Heatmap (sectors): Visual sector performance breakdown
- 🎯 AI Trade Signals (signals): Aggregated buy/sell scoring system
- 💰 Dividend Tracker (divs): Track dividend income & upcoming payments
- 🏆 Gainers/Losers (movers): Top market movers today
- 📈 Backtest (backtest): Simple strategy backtesting
- 🔔 Options Screener (optscreen): Find options by criteria
- ⚡ Quick Portfolio Stats (stats): At-a-glance portfolio health
- 📰 Sentiment Analysis (sentiment): News sentiment scoring
- 🎲 Random Stock (random): Discover new stocks to research

PREVIOUS (v10.2.0):
- Market Overview (market): Indices, Crypto, and VIX snapshot
- ASCII Charts (chart): View price history directly in the terminal
- Watchlist Scanner (scan): Find oversold/strong trend stocks in your list
- Multi-threaded Fetching: Significantly faster portfolio loading
- Smart Context: Commands remember the last used symbol
"""
from __future__ import annotations
import os, re, sys, json, time, math, shlex, logging, warnings, random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# Popular stocks for random discovery feature
DISCOVERY_STOCKS = [
    # Tech Giants
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
    # Growth Tech
    'PLTR', 'SNOW', 'CRWD', 'DDOG', 'NET', 'SHOP', 'SQ', 'PYPL',
    # Semiconductors  
    'AMD', 'INTC', 'MU', 'QCOM', 'AVGO', 'TSM', 'ASML',
    # EV & Clean Energy
    'RIVN', 'LCID', 'NIO', 'ENPH', 'SEDG', 'FSLR',
    # Finance
    'JPM', 'BAC', 'GS', 'MS', 'V', 'MA', 'AXP', 'COIN', 'HOOD', 'SOFI',
    # Healthcare
    'UNH', 'JNJ', 'PFE', 'MRNA', 'LLY', 'ABBV',
    # Consumer
    'NKE', 'SBUX', 'MCD', 'DIS', 'NFLX', 'COST', 'WMT', 'TGT',
    # Industrials
    'CAT', 'DE', 'BA', 'GE', 'HON', 'UPS', 'FDX',
    # Space & Defense
    'RKLB', 'ASTS', 'LMT', 'RTX', 'NOC',
    # REITs
    'O', 'SPG', 'AMT', 'CCI',
    # Energy
    'XOM', 'CVX', 'COP', 'OXY',
    # Dividend Kings
    'KO', 'PG', 'MMM', 'JNJ', 'PEP',
]

# Sector ETFs for heatmap
SECTOR_ETFS = {
    'Technology': 'XLK',
    'Healthcare': 'XLV',
    'Financials': 'XLF',
    'Consumer Disc': 'XLY',
    'Communication': 'XLC',
    'Industrials': 'XLI',
    'Consumer Stpl': 'XLP',
    'Energy': 'XLE',
    'Utilities': 'XLU',
    'Real Estate': 'XLRE',
    'Materials': 'XLB',
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

# --- VISUALIZATION ---
class AsciiChart:
    @staticmethod
    def plot(df, height=12, title=""):
        if df is None or len(df) < 2: return
        # Downsample if too long for terminal width
        if len(df) > 80: df = df.iloc[::len(df)//80]
        
        prices = df['Close'].tolist()
        min_p, max_p = min(prices), max(prices)
        range_p = max_p - min_p
        if range_p == 0: range_p = 1
        
        print(Fore.CYAN + f"\n  {title} ({len(prices)} periods)" + Style.RESET_ALL)
        
        # Draw Chart
        for r in range(height, -1, -1):
            line = ""
            label = min_p + (range_p * (r / height))
            prefix = f"{label:>8.2f} | "
            
            for i in range(len(prices)):
                normalized = (prices[i] - min_p) / range_p * height
                if int(normalized) == r:
                    # Determine character based on slope
                    if i > 0:
                        if prices[i] > prices[i-1]: line += Fore.GREEN + "/" + Style.RESET_ALL
                        elif prices[i] < prices[i-1]: line += Fore.RED + "\\" + Style.RESET_ALL
                        else: line += "-"
                    else: line += "-"
                elif int(normalized) > r:
                    # Fill area below (optional, looks cleaner empty for line charts)
                    line += " " 
                else:
                    line += " "
            print(f"{Fore.WHITE}{prefix}{line}")
            
        # X-Axis
        print(f"{' '*11}{'-' * len(prices)}")
        dates = [df.index[0].strftime('%Y-%m-%d'), df.index[-1].strftime('%Y-%m-%d')]
        print(f"{' '*11}{dates[0]}{' ' * (len(prices) - len(dates[0]) - len(dates[1]))}{dates[1]}\n")

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

    def _save(self): 
        self.file.write_text(json.dumps(self.data, indent=2))

    def clear(self): 
        self.data = {'stocks': {}, 'options': [], 'cash': 0}
        self._save()
        success("Portfolio cleared")
    
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
        
        # 1. Stocks Fetch (Parallel)
        all_symbols = set(stocks.keys()) | set(o['symbol'] for o in options)
        stock_prices = {}
        if all_symbols:
            prog = ProgressBar(len(all_symbols), "  Fetching Stocks")
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(fetcher.get_stock_price, sym): sym for sym in all_symbols}
                for i, f in enumerate(as_completed(futures)):
                    sym = futures[f]
                    stock_prices[sym] = f.result()
                    prog.update(i, sym)
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
            sorted_opts = sorted(options, key=lambda x: (x['expiration'], x['symbol']))
            
            # PARALLEL OPTION FETCH
            prog = ProgressBar(len(options), "  Updating Opts")
            opt_live_data = {}
            
            def fetch_single_opt(o):
                und_p = stock_prices.get(o['symbol'], {}).get('price', 0)
                return fetcher.get_option_price(o['symbol'], o['expiration'], o['strike'], o['type'], und_p)

            with ThreadPoolExecutor(max_workers=10) as executor:
                # Create a map of future -> index to maintain sort order later if needed
                future_to_idx = {executor.submit(fetch_single_opt, o): i for i, o in enumerate(sorted_opts)}
                for f in as_completed(future_to_idx):
                    idx = future_to_idx[f]
                    try: opt_live_data[idx] = f.result()
                    except: opt_live_data[idx] = {'price':0, 'iv':0}
                    prog.update()
            prog.done()
            
            for i, o in enumerate(sorted_opts):
                opt_live = opt_live_data.get(i, {'price': 0})
                und_price = stock_prices.get(o['symbol'], {}).get('price', 0)
                
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
                
                # P&L Calculation
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
    
    def dividends(self, fetcher):
        """Show dividend information for portfolio holdings"""
        print(Fore.CYAN + f"\n{'═'*70}\n DIVIDEND TRACKER\n{'═'*70}" + Style.RESET_ALL)
        stocks = self.data.get('stocks', {})
        if not stocks:
            warn("No stocks in portfolio")
            return
        
        rows = []
        total_annual = 0
        
        prog = ProgressBar(len(stocks), "  Fetching Dividends")
        for i, (sym, pos) in enumerate(stocks.items()):
            prog.update(i, sym)
            try:
                ticker = yf.Ticker(sym)
                info = ticker.info
                div_yield = info.get('dividendYield', 0) or 0
                div_rate = info.get('dividendRate', 0) or 0
                ex_date = info.get('exDividendDate')
                
                if div_rate > 0:
                    qty = pos['qty']
                    annual_income = qty * div_rate
                    total_annual += annual_income
                    
                    # Format ex-dividend date
                    if ex_date:
                        ex_dt = datetime.fromtimestamp(ex_date)
                        ex_str = ex_dt.strftime('%Y-%m-%d')
                        days_to_ex = (ex_dt - datetime.now()).days
                        if 0 <= days_to_ex <= 30:
                            ex_str = Fore.YELLOW + ex_str + f" ({days_to_ex}d)" + Style.RESET_ALL
                    else:
                        ex_str = "-"
                    
                    rows.append([
                        sym,
                        f"{qty:.0f}",
                        f"{div_yield*100:.2f}%",
                        fmt_money(div_rate),
                        fmt_money(annual_income),
                        ex_str
                    ])
            except:
                pass
        prog.done()
        
        if rows:
            rows.sort(key=lambda x: float(x[4].replace('$', '').replace(',', '')), reverse=True)
            print(tabulate(rows, headers=["Symbol", "Shares", "Yield", "Rate", "Annual $", "Ex-Div Date"]))
            print(f"\n  {Style.BRIGHT}Total Annual Dividend Income:{Style.RESET_ALL} {Fore.GREEN}{fmt_money(total_annual)}{Style.RESET_ALL}")
            print(f"  {Style.BRIGHT}Monthly Average:{Style.RESET_ALL} {fmt_money(total_annual/12)}")
        else:
            warn("No dividend-paying stocks found in portfolio")
        print(f"{'═'*70}\n")
    
    def quick_stats(self, fetcher):
        """Quick portfolio health stats at a glance - options friendly with delta exposure"""
        stocks = self.data.get('stocks', {})
        options = self.data.get('options', [])
        cash = self.data.get('cash', 0)
        
        if not stocks and not options:
            warn("No positions in portfolio")
            return
        
        print(Fore.CYAN + f"\n{'═'*70}\n ⚡ QUICK STATS\n{'═'*70}" + Style.RESET_ALL)
        
        # Gather all underlying symbols for price fetching
        all_symbols = set(stocks.keys()) | set(o['symbol'] for o in options)
        stock_prices = {}
        
        # Phase 1: Fetch underlying prices
        if all_symbols:
            prog = ProgressBar(len(all_symbols), "  Stocks ")
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(fetcher.get_stock_price, sym): sym for sym in all_symbols}
                for i, f in enumerate(as_completed(futures)):
                    sym = futures[f]
                    stock_prices[sym] = f.result()
                    prog.update(i + 1, sym)
            prog.done()
        
        # Track all positions (stocks + options)
        all_positions = []  # List of (name, value, cost, pnl_pct, type)
        total_stock_value = 0
        total_option_value = 0
        total_cost = 0
        delta_exposure = {}  # Track DELTA-ADJUSTED exposure by underlying (in equivalent shares)
        total_portfolio_delta = 0  # Net delta in dollar terms
        
        # Process stocks - stocks have delta of 1.0 per share
        for sym, pos in stocks.items():
            price = stock_prices.get(sym, {}).get('price', 0)
            if price > 0:
                val = pos['qty'] * price
                cost = pos['cost']
                pnl_pct = ((val - cost) / cost * 100) if cost > 0 else 0
                
                total_stock_value += val
                total_cost += cost
                
                # Stock delta = qty shares (delta of 1.0 each)
                delta_exposure[sym] = delta_exposure.get(sym, 0) + pos['qty']
                total_portfolio_delta += val  # Dollar delta
                
                all_positions.append((sym, val, cost, pnl_pct, 'stock'))
        
        # Phase 2: Process options with delta calculation
        option_stats = {'long_calls': 0, 'long_puts': 0, 'short_calls': 0, 'short_puts': 0}
        options_by_expiry = {}
        option_data_cache = {}
        
        # Fetch option prices in parallel
        if options:
            prog = ProgressBar(len(options), "  Options")
            
            def fetch_opt(o):
                und_price = stock_prices.get(o['symbol'], {}).get('price', 0)
                return fetcher.get_option_price(o['symbol'], o['expiration'], o['strike'], o['type'], und_price)
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(fetch_opt, o): i for i, o in enumerate(options)}
                for f in as_completed(futures):
                    idx = futures[f]
                    option_data_cache[idx] = f.result()
                    prog.update(len(option_data_cache), options[idx]['symbol'])
            prog.done()
        
        # Now process all options with cached data
        for i, o in enumerate(options):
            und_price = stock_prices.get(o['symbol'], {}).get('price', 0)
            opt_data = option_data_cache.get(i, {'price': 0, 'iv': 0.5})
            
            qty = o['qty']
            cost = o['cost']
            price = opt_data['price'] if opt_data['price'] > 0 else o.get('broker_price', 0)
            iv = opt_data.get('iv', 0.5)  # Implied volatility, default 50%
            
            # Calculate current value
            val = abs(qty) * price * 100
            
            # Calculate delta using Black-Scholes
            days = dte(o['expiration'])
            T = max(days / 365.0, 0.001)  # Time to expiry in years, minimum to avoid div by zero
            
            if und_price > 0 and HAS_SCIPY:
                # Get Greeks from Black-Scholes
                greeks = BlackScholes.calculate_all_greeks(
                    und_price, o['strike'], T, config.risk_free_rate, iv, o['type']
                )
                delta = greeks['delta']
            else:
                # Fallback: estimate delta based on moneyness
                if o['type'] == 'call':
                    if und_price > o['strike'] * 1.1:  # Deep ITM
                        delta = 0.9
                    elif und_price > o['strike']:  # ITM
                        delta = 0.6 + 0.3 * ((und_price - o['strike']) / o['strike'])
                    elif und_price > o['strike'] * 0.95:  # ATM
                        delta = 0.5
                    elif und_price > o['strike'] * 0.9:  # OTM
                        delta = 0.3
                    else:  # Deep OTM
                        delta = 0.1
                else:  # Put
                    if und_price < o['strike'] * 0.9:  # Deep ITM put
                        delta = -0.9
                    elif und_price < o['strike']:  # ITM put
                        delta = -0.6 - 0.3 * ((o['strike'] - und_price) / o['strike'])
                    elif und_price < o['strike'] * 1.05:  # ATM put
                        delta = -0.5
                    elif und_price < o['strike'] * 1.1:  # OTM put
                        delta = -0.3
                    else:  # Deep OTM put
                        delta = -0.1
            
            # For short options, value is negative (liability)
            is_short = qty < 0
            if is_short:
                avg_price = (cost / abs(qty) / 100) if abs(qty) > 0 else 0
                pnl = (avg_price - price) * abs(qty) * 100
                pnl_pct = (pnl / cost * 100) if cost > 0 else 0
                total_option_value -= val  # Liability
            else:
                pnl = val - cost
                pnl_pct = ((val - cost) / cost * 100) if cost > 0 else 0
                total_option_value += val  # Asset
            
            total_cost += cost
            
            # DELTA-ADJUSTED EXPOSURE
            equivalent_shares = delta * qty * 100
            delta_exposure[o['symbol']] = delta_exposure.get(o['symbol'], 0) + equivalent_shares
            
            # Dollar delta for portfolio
            if und_price > 0:
                total_portfolio_delta += equivalent_shares * und_price
            
            # Option type stats
            if is_short:
                if o['type'] == 'call':
                    option_stats['short_calls'] += abs(qty)
                else:
                    option_stats['short_puts'] += abs(qty)
            else:
                if o['type'] == 'call':
                    option_stats['long_calls'] += abs(qty)
                else:
                    option_stats['long_puts'] += abs(qty)
            
            # Track by expiry
            exp = o['expiration']
            if exp not in options_by_expiry:
                options_by_expiry[exp] = {'count': 0, 'value': 0}
            options_by_expiry[exp]['count'] += abs(qty)
            options_by_expiry[exp]['value'] += val if not is_short else -val
            
            # Create position name
            pos_name = f"{o['symbol']} {o['expiration'][5:]} ${o['strike']:.0f}{o['type'][0].upper()}"
            if is_short:
                pos_name = f"-{pos_name}"
            
            all_positions.append((pos_name, val if not is_short else -val, cost, pnl_pct, 'option'))
        
        # Calculate totals
        net_liq = cash + total_stock_value + total_option_value
        total_pnl = net_liq - total_cost - cash
        total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
        
        # Count winners/losers
        gainers = sum(1 for p in all_positions if p[3] > 0)
        losers = sum(1 for p in all_positions if p[3] <= 0)
        
        # Find best/worst performers
        if all_positions:
            best = max(all_positions, key=lambda x: x[3])
            worst = min(all_positions, key=lambda x: x[3])
        else:
            best = worst = None
        
        # Display results
        print(f"\n  {Style.BRIGHT}NET LIQUIDATION:{Style.RESET_ALL}   {fmt_money(net_liq)}")
        print(f"  {Style.BRIGHT}Total P&L:{Style.RESET_ALL}          {color_pnl(total_pnl, total_pnl_pct)}")
        
        print(f"\n  {Style.BRIGHT}BREAKDOWN:{Style.RESET_ALL}")
        print(f"    Cash:              {fmt_money(cash)}")
        print(f"    Stock Value:       {fmt_money(total_stock_value)}")
        print(f"    Options Value:     {fmt_money(total_option_value)} {'(net liability)' if total_option_value < 0 else ''}")
        
        print(f"\n  {Style.BRIGHT}POSITIONS:{Style.RESET_ALL}         {len(stocks)} stocks, {len(options)} options")
        print(f"  {Style.BRIGHT}Win/Loss:{Style.RESET_ALL}          {Fore.GREEN}{gainers} winners{Style.RESET_ALL} / {Fore.RED}{losers} losers{Style.RESET_ALL}")
        
        # Options breakdown
        if options:
            print(f"\n  {Style.BRIGHT}OPTIONS BREAKDOWN:{Style.RESET_ALL}")
            print(f"    Long Calls:  {option_stats['long_calls']:>3} contracts")
            print(f"    Long Puts:   {option_stats['long_puts']:>3} contracts")
            print(f"    Short Calls: {option_stats['short_calls']:>3} contracts")
            print(f"    Short Puts:  {option_stats['short_puts']:>3} contracts")
        
        # Expiration timeline
        if options_by_expiry:
            print(f"\n  {Style.BRIGHT}EXPIRATION TIMELINE:{Style.RESET_ALL}")
            sorted_expiries = sorted(options_by_expiry.items(), key=lambda x: x[0])[:5]
            for exp, data in sorted_expiries:
                days = dte(exp)
                urgency = Fore.RED if days <= 7 else Fore.YELLOW if days <= 30 else Fore.RESET
                print(f"    {urgency}{exp}{Style.RESET_ALL}: {data['count']} contracts ({days}d)")
        
        # Best/Worst performers
        if best:
            print(f"\n  {Style.BRIGHT}Best Performer:{Style.RESET_ALL}   {best[0]} ({color_pct(best[3])})")
        if worst:
            print(f"  {Style.BRIGHT}Worst Performer:{Style.RESET_ALL}  {worst[0]} ({color_pct(worst[3])})")
        
        # DELTA-ADJUSTED EXPOSURE (equivalent shares)
        if delta_exposure:
            print(f"\n  {Style.BRIGHT}DELTA EXPOSURE (equivalent shares):{Style.RESET_ALL}")
            sorted_exp = sorted(delta_exposure.items(), key=lambda x: abs(x[1]), reverse=True)[:6]
            for sym, eq_shares in sorted_exp:
                if abs(eq_shares) < 1:  # Skip negligible positions
                    continue
                direction = Fore.GREEN + "LONG" if eq_shares > 0 else Fore.RED + "SHORT"
                price = stock_prices.get(sym, {}).get('price', 0)
                dollar_exp = abs(eq_shares) * price if price > 0 else 0
                print(f"    {sym:<6} {direction}{Style.RESET_ALL} {abs(eq_shares):>7.0f} shares (≈{fmt_money(dollar_exp)})")
            
            # Net portfolio delta
            net_direction = "BULLISH" if total_portfolio_delta > 0 else "BEARISH"
            net_color = Fore.GREEN if total_portfolio_delta > 0 else Fore.RED
            print(f"\n  {Style.BRIGHT}NET PORTFOLIO DELTA:{Style.RESET_ALL} {net_color}{net_direction}{Style.RESET_ALL} {fmt_money(abs(total_portfolio_delta))}")
            print(f"  {Fore.CYAN}(If market moves $1, portfolio moves ~${abs(total_portfolio_delta)/1000:.0f} per $1000 exposure){Style.RESET_ALL}")
        
        print(f"\n{'═'*70}\n")

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

# --- PRICE ALERTS ---
class PriceAlerts:
    def __init__(self):
        self.file = Path(config.data_dir) / "alerts.json"
        self.alerts = json.loads(self.file.read_text()) if self.file.exists() else []
    
    def save(self): 
        self.file.write_text(json.dumps(self.alerts, indent=2))
    
    def add(self, symbol, target, direction='above'):
        """Add a price alert"""
        alert = {
            'symbol': symbol.upper(),
            'target': float(target),
            'direction': direction.lower(),  # 'above' or 'below'
            'created': datetime.now().isoformat(),
            'triggered': False
        }
        self.alerts.append(alert)
        self.save()
        success(f"Alert set: {symbol.upper()} {direction} ${target}")
    
    def remove(self, idx):
        """Remove alert by index"""
        if 0 <= idx < len(self.alerts):
            removed = self.alerts.pop(idx)
            self.save()
            success(f"Removed alert for {removed['symbol']}")
        else:
            err("Invalid alert index")
    
    def check(self, fetcher):
        """Check all alerts and return triggered ones"""
        triggered = []
        for i, alert in enumerate(self.alerts):
            if alert['triggered']:
                continue
            d = fetcher.get_stock_price(alert['symbol'])
            price = d['price']
            if price <= 0:
                continue
            
            if alert['direction'] == 'above' and price >= alert['target']:
                alert['triggered'] = True
                alert['trigger_price'] = price
                triggered.append(alert)
            elif alert['direction'] == 'below' and price <= alert['target']:
                alert['triggered'] = True
                alert['trigger_price'] = price
                triggered.append(alert)
        
        if triggered:
            self.save()
        return triggered
    
    def show(self, fetcher):
        """Display all alerts with current prices"""
        if not self.alerts:
            warn("No price alerts set. Use 'alert add SYMBOL PRICE [above/below]'")
            return
        
        print(Fore.CYAN + f"\n{'═'*70}\n PRICE ALERTS\n{'═'*70}" + Style.RESET_ALL)
        rows = []
        for i, alert in enumerate(self.alerts):
            d = fetcher.get_stock_price(alert['symbol'])
            current = d['price']
            target = alert['target']
            direction = "▲" if alert['direction'] == 'above' else "▼"
            
            # Calculate distance to target
            if current > 0:
                dist_pct = ((target - current) / current) * 100
                dist_str = color_pct(dist_pct)
            else:
                dist_str = "-"
            
            status = Fore.GREEN + "✓ TRIGGERED" + Style.RESET_ALL if alert['triggered'] else Fore.YELLOW + "Active" + Style.RESET_ALL
            
            rows.append([
                i,
                alert['symbol'],
                fmt_money(current),
                f"{direction} {fmt_money(target)}",
                dist_str,
                status
            ])
        
        print(tabulate(rows, headers=["#", "Symbol", "Current", "Target", "Distance", "Status"]))
        print(f"\n  Use 'alert rm <#>' to remove an alert")
        print(f"{'═'*70}\n")

# --- SENTIMENT ANALYZER ---
class SentimentAnalyzer:
    """Simple keyword-based sentiment analysis for news headlines"""
    
    POSITIVE_WORDS = {
        'surge', 'soar', 'jump', 'rally', 'gain', 'rise', 'up', 'high', 'record',
        'beat', 'exceed', 'strong', 'growth', 'profit', 'bullish', 'buy', 'upgrade',
        'positive', 'boost', 'success', 'win', 'breakthrough', 'innovation', 'expand',
        'outperform', 'optimistic', 'momentum', 'accelerate', 'recover', 'boom'
    }
    
    NEGATIVE_WORDS = {
        'drop', 'fall', 'plunge', 'crash', 'decline', 'loss', 'down', 'low', 'miss',
        'weak', 'bearish', 'sell', 'downgrade', 'negative', 'concern', 'risk', 'fear',
        'warning', 'cut', 'layoff', 'lawsuit', 'investigation', 'recall', 'delay',
        'underperform', 'pessimistic', 'slowdown', 'recession', 'bankruptcy', 'fraud'
    }
    
    @classmethod
    def analyze_headline(cls, headline):
        """Returns sentiment score from -1 (bearish) to +1 (bullish)"""
        words = headline.lower().split()
        pos_count = sum(1 for w in words if any(pw in w for pw in cls.POSITIVE_WORDS))
        neg_count = sum(1 for w in words if any(nw in w for nw in cls.NEGATIVE_WORDS))
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        return (pos_count - neg_count) / total
    
    @classmethod
    def analyze_news(cls, news_items):
        """Analyze a list of news items and return overall sentiment"""
        if not news_items:
            return {'score': 0, 'label': 'Neutral', 'positive': 0, 'negative': 0, 'neutral': 0}
        
        scores = []
        pos, neg, neu = 0, 0, 0
        
        for item in news_items:
            title = item.get('title', '')
            score = cls.analyze_headline(title)
            scores.append(score)
            if score > 0.1:
                pos += 1
            elif score < -0.1:
                neg += 1
            else:
                neu += 1
        
        avg_score = sum(scores) / len(scores) if scores else 0
        
        if avg_score > 0.15:
            label = "BULLISH"
        elif avg_score > 0.05:
            label = "Slightly Bullish"
        elif avg_score < -0.15:
            label = "BEARISH"
        elif avg_score < -0.05:
            label = "Slightly Bearish"
        else:
            label = "Neutral"
        
        return {
            'score': avg_score,
            'label': label,
            'positive': pos,
            'negative': neg,
            'neutral': neu
        }

# --- TRADE SIGNALS AGGREGATOR ---
class TradeSignals:
    """Aggregates multiple indicators into a single trade signal"""
    
    @staticmethod
    def calculate(df, price_data, meta=None):
        """Calculate comprehensive trade signal from -100 (strong sell) to +100 (strong buy)"""
        if df is None or len(df) < 50:
            return None
        
        signals = {}
        close = df['Close']
        price = close.iloc[-1]
        
        # RSI Signal (-20 to +20)
        rsi = TechnicalAnalysis.rsi(close).iloc[-1]
        if rsi < 30:
            signals['rsi'] = 20  # Oversold = bullish
        elif rsi < 40:
            signals['rsi'] = 10
        elif rsi > 70:
            signals['rsi'] = -20  # Overbought = bearish
        elif rsi > 60:
            signals['rsi'] = -10
        else:
            signals['rsi'] = 0
        
        # MACD Signal (-20 to +20)
        _, _, hist = TechnicalAnalysis.macd(close)
        macd_val = hist.iloc[-1]
        macd_prev = hist.iloc[-2] if len(hist) > 1 else 0
        if macd_val > 0 and macd_val > macd_prev:
            signals['macd'] = 20  # Bullish and strengthening
        elif macd_val > 0:
            signals['macd'] = 10
        elif macd_val < 0 and macd_val < macd_prev:
            signals['macd'] = -20  # Bearish and weakening
        elif macd_val < 0:
            signals['macd'] = -10
        else:
            signals['macd'] = 0
        
        # Moving Averages Signal (-20 to +20)
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1]
        sma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else sma_50
        
        ma_score = 0
        if price > sma_20: ma_score += 5
        if price > sma_50: ma_score += 5
        if price > sma_200: ma_score += 5
        if sma_20 > sma_50: ma_score += 5  # Golden cross forming
        signals['ma'] = ma_score - 10  # Center around 0
        
        # Supertrend Signal (-15 to +15)
        trend, _, _ = TechnicalAnalysis.supertrend(df)
        signals['supertrend'] = 15 if trend.iloc[-1] == 1 else -15
        
        # Volume Signal (-10 to +10)
        avg_vol = df['Volume'].rolling(20).mean().iloc[-1]
        curr_vol = df['Volume'].iloc[-1]
        vol_ratio = curr_vol / avg_vol if avg_vol > 0 else 1
        day_change = price_data.get('pct', 0)
        if vol_ratio > 1.5 and day_change > 0:
            signals['volume'] = 10  # High volume on up day
        elif vol_ratio > 1.5 and day_change < 0:
            signals['volume'] = -10  # High volume on down day
        else:
            signals['volume'] = 0
        
        # Bollinger Bands Signal (-15 to +15)
        upper, _, lower = TechnicalAnalysis.bollinger_bands(close)
        bb_range = upper.iloc[-1] - lower.iloc[-1]
        if bb_range > 0:
            bb_pos = (price - lower.iloc[-1]) / bb_range
            if bb_pos < 0.2:
                signals['bollinger'] = 15  # Near lower band = potential bounce
            elif bb_pos > 0.8:
                signals['bollinger'] = -15  # Near upper band = potential pullback
            else:
                signals['bollinger'] = 0
        else:
            signals['bollinger'] = 0
        
        # Calculate total score
        total_score = sum(signals.values())
        max_possible = 100
        normalized = (total_score / max_possible) * 100
        
        # Determine signal label
        if normalized >= 50:
            label = "STRONG BUY"
        elif normalized >= 25:
            label = "BUY"
        elif normalized >= 10:
            label = "Lean Bullish"
        elif normalized <= -50:
            label = "STRONG SELL"
        elif normalized <= -25:
            label = "SELL"
        elif normalized <= -10:
            label = "Lean Bearish"
        else:
            label = "HOLD"
        
        return {
            'score': normalized,
            'label': label,
            'breakdown': signals,
            'rsi': rsi
        }

def print_help():
    print(Fore.CYAN + """
╔══════════════════════════════════════════════════════════════════════════════╗
║  STOCK TICKER v11.0.0 — HELP & MANUAL                                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  SMART CONTEXT: Commands remember last symbol. Just type 'ta' again!         ║
║                                                                              ║
║  PORTFOLIO COMMANDS:                                                         ║
║  pf              View full portfolio with P&L, Strategy, and Greeks          ║
║  import FILE     Load CSV (Fidelity, Schwab, E*Trade, Robinhood)             ║
║  risk            Calculate Portfolio Beta (volatility vs S&P 500)            ║
║  cal             Scan portfolio for upcoming earnings                        ║
║  divs            Show dividend income & upcoming ex-dates                    ║
║  stats           Quick portfolio health stats at a glance                    ║
║  clear           Delete all portfolio data (reset)                           ║
║                                                                              ║
║  QUOTE & LOOKUP:                                                             ║
║  AAPL            Just type ticker for quick quote (sets context)             ║
║  q SYMBOL        Quick quote (e.g., 'q AAPL')                                ║
║  quote SYMBOL    Detailed stock quote with fundamentals                      ║
║  info SYMBOL     Full company info and metrics                               ║
║  dash SYMBOL     Dashboard: price, trend, and headlines                      ║
║  news SYMBOL     Latest news headlines for a stock                           ║
║  chart SYMBOL    Show ASCII chart in terminal                                ║
║                                                                              ║
║  MARKET OVERVIEW:                                                            ║
║  market          Show Market Indices, VIX, and Bitcoin                       ║
║  sectors         Sector Heatmap - visual performance breakdown               ║
║  movers          Top gainers and losers today                                ║
║  random          Discover a random stock to research                         ║
║                                                                              ║
║  TECHNICAL ANALYSIS:                                                         ║
║  ta SYMBOL       Basic TA (RSI, MACD, Bollinger, Supertrend)                 ║
║  ta2 SYMBOL      Extended TA (Stochastic, ADX, Ichimoku, Williams %R)        ║
║  trend SYMBOL    Trend strength analysis with scoring                        ║
║  levels SYMBOL   Support/Resistance and Fibonacci levels                     ║
║  signals SYMBOL  AI Trade Signal aggregator (buy/sell scoring)               ║
║  sentiment SYM   News sentiment analysis for a stock                         ║
║  compare A B     Compare two stocks side-by-side                             ║
║  backtest SYM    Simple moving average crossover backtest                    ║
║                                                                              ║
║  WATCHLIST & ALERTS:                                                         ║
║  watch           View watchlist                                              ║
║  watch add SYM   Add symbol to watchlist                                     ║
║  scan            Scan watchlist for Oversold/Trending stocks                 ║
║  alert           View all price alerts                                       ║
║  alert add SYM PRICE [above/below]   Set a price alert                       ║
║  alert rm #      Remove alert by number                                      ║
║                                                                              ║
║  SETTINGS:                                                                   ║
║  refresh         Clear cached data for fresh updates                         ║
║  debug           Toggle debug mode                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
""" + Style.RESET_ALL)

# --- MAIN ---
def main():
    print(Fore.CYAN + Style.BRIGHT + "\n    STOCK TICKER v11.0.0 — Universal + Charts + Signals + Alerts\n" + Style.RESET_ALL)
    fetcher, pf, wl, alerts = PriceFetcher(), Portfolio(), Watchlist(), PriceAlerts()
    last_symbol = None
    
    # Check alerts on startup
    triggered = alerts.check(fetcher)
    if triggered:
        print(Fore.YELLOW + Style.BRIGHT + "\n  🔔 TRIGGERED ALERTS:" + Style.RESET_ALL)
        for a in triggered:
            print(f"     {a['symbol']} hit ${a['target']:.2f} (now ${a.get('trigger_price', 0):.2f})")
        print()
    
    while True:
        try:
            prompt_sym = f"[{last_symbol}] " if last_symbol else ""
            raw = input(Fore.GREEN + f"▶ {prompt_sym}" + Style.RESET_ALL).strip()
            if not raw: continue
            
            parts = shlex.split(raw)
            cmd = parts[0].lower()
            args = parts[1:]
            
            # --- CONTEXT HANDLING ---
            symbol_commands = ('quote', 'q', 'ta', 'ta2', 'trend', 'levels', 'info', 'news', 'dash', 'chart', 'signals', 'sentiment', 'backtest')
            if cmd in symbol_commands:
                if args: last_symbol = args[0].upper()
                elif last_symbol:
                    args = [last_symbol]
                    print(Fore.YELLOW + f"  Using active ticker: {last_symbol}" + Style.RESET_ALL)
                else: err(f"Usage: {cmd} <symbol>"); continue
            
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

            # --- NEW: MARKET OVERVIEW ---
            elif cmd == 'market':
                indices = {'S&P 500': '^GSPC', 'Nasdaq': '^IXIC', 'Dow Jones': '^DJI', 'VIX': '^VIX', 'Bitcoin': 'BTC-USD'}
                print(Fore.CYAN + f"\n{'═'*60}\n MARKET OVERVIEW\n{'═'*60}" + Style.RESET_ALL)
                rows = []
                with ThreadPoolExecutor() as executor:
                    futures = {executor.submit(fetcher.get_stock_price, sym): name for name, sym in indices.items()}
                    for f in futures:
                        name = futures[f]
                        try:
                            d = f.result()
                            rows.append([name, fmt_money(d['price']), color_money(d['change']), color_pct(d['pct'])])
                        except: pass
                print(tabulate(rows, headers=["Index", "Price", "Change", "% Change"]))
                print(f"{'═'*60}\n")

            # --- NEW v11: SECTOR HEATMAP ---
            elif cmd == 'sectors':
                print(Fore.CYAN + f"\n{'═'*70}\n SECTOR HEATMAP\n{'═'*70}" + Style.RESET_ALL)
                rows = []
                with ThreadPoolExecutor() as executor:
                    futures = {executor.submit(fetcher.get_stock_price, sym): name for name, sym in SECTOR_ETFS.items()}
                    for f in as_completed(futures):
                        name = futures[f]
                        try:
                            d = f.result()
                            pct = d['pct']
                            # Visual bar
                            bar_len = min(int(abs(pct) * 2), 20)
                            if pct >= 0:
                                bar = Fore.GREEN + "█" * bar_len + Style.RESET_ALL
                            else:
                                bar = Fore.RED + "█" * bar_len + Style.RESET_ALL
                            rows.append([name, color_pct(pct), bar, fmt_money(d['price'])])
                        except: pass
                # Sort by performance
                rows.sort(key=lambda x: float(x[1].replace(Fore.GREEN, '').replace(Fore.RED, '').replace(Style.RESET_ALL, '').replace('%', '').replace('+', '')), reverse=True)
                print(tabulate(rows, headers=["Sector", "Change", "Performance", "ETF Price"]))
                print(f"\n  {Fore.CYAN}ETFs shown: XLK, XLV, XLF, XLY, XLC, XLI, XLP, XLE, XLU, XLRE, XLB{Style.RESET_ALL}")
                print(f"{'═'*70}\n")

            # --- NEW v11: MARKET MOVERS ---
            elif cmd == 'movers':
                print(Fore.CYAN + f"\n{'═'*70}\n TOP MOVERS TODAY\n{'═'*70}" + Style.RESET_ALL)
                # Check a subset of popular stocks
                check_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC',
                               'JPM', 'BAC', 'GS', 'NFLX', 'DIS', 'NKE', 'COIN', 'HOOD', 'PLTR', 'SOFI',
                               'RIVN', 'LCID', 'NIO', 'F', 'GM', 'UBER', 'SNAP', 'PINS', 'SQ', 'PYPL']
                results = []
                prog = ProgressBar(len(check_symbols), "  Scanning")
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = {executor.submit(fetcher.get_stock_price, s): s for s in check_symbols}
                    for i, f in enumerate(as_completed(futures)):
                        sym = futures[f]
                        prog.update(i, sym)
                        try:
                            d = f.result()
                            if d['price'] > 0:
                                results.append((sym, d['price'], d['pct']))
                        except: pass
                prog.done()
                
                # Sort and display
                results.sort(key=lambda x: x[2], reverse=True)
                
                print(f"\n  {Style.BRIGHT}{Fore.GREEN}TOP GAINERS{Style.RESET_ALL}")
                for sym, price, pct in results[:5]:
                    print(f"    {sym:<6} {fmt_money(price):>10}  {color_pct(pct)}")
                
                print(f"\n  {Style.BRIGHT}{Fore.RED}TOP LOSERS{Style.RESET_ALL}")
                for sym, price, pct in results[-5:]:
                    print(f"    {sym:<6} {fmt_money(price):>10}  {color_pct(pct)}")
                print(f"\n{'═'*70}\n")

            # --- NEW v11: RANDOM STOCK DISCOVERY ---
            elif cmd == 'random':
                symbol = random.choice(DISCOVERY_STOCKS)
                print(Fore.MAGENTA + f"\n  🎲 Random Pick: {symbol}" + Style.RESET_ALL)
                d = fetcher.get_stock_price(symbol)
                meta = fetcher.get_meta(symbol)
                df = fetcher.get_history(symbol, "3mo")
                
                print(f"\n  {Style.BRIGHT}{meta.get('name', symbol)}{Style.RESET_ALL}")
                print(f"  Sector: {meta.get('sector', 'Unknown')}")
                print(f"  Price: {fmt_money(d['price'])} ({color_pct(d['pct'])})")
                
                if df is not None and len(df) > 30:
                    rsi = TechnicalAnalysis.rsi(df['Close']).iloc[-1]
                    trend = TechnicalAnalysis.trend_strength(df)
                    print(f"  RSI: {rsi:.1f} ({color_signal('OVERSOLD' if rsi < 30 else 'OVERBOUGHT' if rsi > 70 else 'Neutral')})")
                    print(f"  Trend: {color_signal(trend['strength'])}")
                
                print(f"\n  {Fore.CYAN}Type 'ta {symbol}' for full analysis{Style.RESET_ALL}\n")
                last_symbol = symbol

            # --- NEW v11: PRICE ALERTS ---
            elif cmd == 'alert':
                if not args:
                    alerts.show(fetcher)
                elif args[0] == 'add' and len(args) >= 3:
                    sym = args[1].upper()
                    target = float(args[2])
                    direction = args[3] if len(args) > 3 else 'above'
                    alerts.add(sym, target, direction)
                elif args[0] in ('rm', 'del', 'remove') and len(args) >= 2:
                    alerts.remove(int(args[1]))
                elif args[0] == 'check':
                    triggered = alerts.check(fetcher)
                    if triggered:
                        for a in triggered:
                            print(Fore.GREEN + f"  🔔 {a['symbol']} hit ${a['target']:.2f}!" + Style.RESET_ALL)
                    else:
                        print("  No alerts triggered")
                else:
                    err("Usage: alert | alert add SYM PRICE [above/below] | alert rm #")

            # --- NEW v11: AI TRADE SIGNALS ---
            elif cmd == 'signals' and args:
                symbol = args[0].upper()
                print(Fore.CYAN + f"\n{'═'*70}\n 🎯 TRADE SIGNALS — {symbol}\n{'═'*70}" + Style.RESET_ALL)
                
                df = fetcher.get_history(symbol, "1y")
                d = fetcher.get_stock_price(symbol)
                
                if df is None or len(df) < 50:
                    err(f"Insufficient data for {symbol}")
                else:
                    signals = TradeSignals.calculate(df, d)
                    if signals:
                        # Main signal display
                        score = signals['score']
                        label = signals['label']
                        
                        # Color based on signal
                        if 'BUY' in label or 'Bullish' in label:
                            label_colored = Fore.GREEN + Style.BRIGHT + label + Style.RESET_ALL
                        elif 'SELL' in label or 'Bearish' in label:
                            label_colored = Fore.RED + Style.BRIGHT + label + Style.RESET_ALL
                        else:
                            label_colored = Fore.YELLOW + Style.BRIGHT + label + Style.RESET_ALL
                        
                        print(f"\n  {Style.BRIGHT}SIGNAL:{Style.RESET_ALL}  {label_colored}")
                        print(f"  {Style.BRIGHT}SCORE:{Style.RESET_ALL}   {score:+.1f} / 100")
                        
                        # Visual score bar
                        bar_pos = int((score + 100) / 200 * 40)  # -100 to +100 mapped to 0-40
                        bar = "─" * 40
                        bar = bar[:bar_pos] + "│" + bar[bar_pos+1:]
                        print(f"\n  SELL {Fore.RED}{'─'*20}{Style.RESET_ALL}│{Fore.GREEN}{'─'*20}{Style.RESET_ALL} BUY")
                        print(f"       {bar[:bar_pos]}{Fore.YELLOW}●{Style.RESET_ALL}{bar[bar_pos+1:]}")
                        
                        # Breakdown
                        print(f"\n  {Style.BRIGHT}BREAKDOWN:{Style.RESET_ALL}")
                        for indicator, value in signals['breakdown'].items():
                            indicator_color = Fore.GREEN if value > 0 else Fore.RED if value < 0 else Fore.RESET
                            print(f"    {indicator.upper():<12} {indicator_color}{value:+.0f}{Style.RESET_ALL}")
                        
                        print(f"\n  {Style.BRIGHT}RSI:{Style.RESET_ALL} {signals['rsi']:.1f}")
                print(f"{'═'*70}\n")
                last_symbol = symbol

            # --- NEW v11: SENTIMENT ANALYSIS ---
            elif cmd == 'sentiment' and args:
                symbol = args[0].upper()
                print(Fore.CYAN + f"\n{'═'*70}\n 📰 NEWS SENTIMENT — {symbol}\n{'═'*70}" + Style.RESET_ALL)
                
                news_items = fetcher.get_news(symbol)
                if not news_items:
                    warn("No recent news found")
                else:
                    result = SentimentAnalyzer.analyze_news(news_items)
                    
                    # Sentiment label with color
                    label = result['label']
                    if 'BULLISH' in label.upper():
                        label_colored = Fore.GREEN + Style.BRIGHT + label + Style.RESET_ALL
                    elif 'BEARISH' in label.upper():
                        label_colored = Fore.RED + Style.BRIGHT + label + Style.RESET_ALL
                    else:
                        label_colored = Fore.YELLOW + label + Style.RESET_ALL
                    
                    print(f"\n  {Style.BRIGHT}SENTIMENT:{Style.RESET_ALL}  {label_colored}")
                    print(f"  {Style.BRIGHT}SCORE:{Style.RESET_ALL}      {result['score']:+.2f} (-1 to +1)")
                    print(f"\n  Headlines: {Fore.GREEN}{result['positive']} positive{Style.RESET_ALL}, "
                          f"{Fore.RED}{result['negative']} negative{Style.RESET_ALL}, "
                          f"{result['neutral']} neutral")
                    
                    # Show headlines with individual sentiment
                    print(f"\n  {Style.BRIGHT}RECENT HEADLINES:{Style.RESET_ALL}")
                    for item in news_items[:5]:
                        title = item.get('title', '')[:60]
                        score = SentimentAnalyzer.analyze_headline(title)
                        if score > 0.1:
                            indicator = Fore.GREEN + "▲" + Style.RESET_ALL
                        elif score < -0.1:
                            indicator = Fore.RED + "▼" + Style.RESET_ALL
                        else:
                            indicator = Fore.YELLOW + "─" + Style.RESET_ALL
                        print(f"    {indicator} {title}...")
                print(f"{'═'*70}\n")
                last_symbol = symbol

            # --- NEW v11: SIMPLE BACKTEST ---
            elif cmd == 'backtest' and args:
                symbol = args[0].upper()
                print(Fore.CYAN + f"\n{'═'*70}\n 📈 BACKTEST — {symbol} (SMA Crossover)\n{'═'*70}" + Style.RESET_ALL)
                
                df = fetcher.get_history(symbol, "2y")
                if df is None or len(df) < 200:
                    err(f"Need at least 200 days of data for backtest")
                else:
                    close = df['Close']
                    sma_20 = close.rolling(20).mean()
                    sma_50 = close.rolling(50).mean()
                    
                    # Simple crossover strategy
                    position = 0  # 0 = no position, 1 = long
                    trades = []
                    entry_price = 0
                    
                    for i in range(51, len(df)):
                        if position == 0 and sma_20.iloc[i] > sma_50.iloc[i] and sma_20.iloc[i-1] <= sma_50.iloc[i-1]:
                            # Golden cross - buy
                            position = 1
                            entry_price = close.iloc[i]
                            trades.append({'type': 'BUY', 'price': entry_price, 'date': df.index[i]})
                        elif position == 1 and sma_20.iloc[i] < sma_50.iloc[i] and sma_20.iloc[i-1] >= sma_50.iloc[i-1]:
                            # Death cross - sell
                            position = 0
                            exit_price = close.iloc[i]
                            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                            trades.append({'type': 'SELL', 'price': exit_price, 'date': df.index[i], 'pnl': pnl_pct})
                    
                    # Results
                    buy_hold_return = ((close.iloc[-1] - close.iloc[50]) / close.iloc[50]) * 100
                    
                    strategy_return = 0
                    wins, losses = 0, 0
                    for t in trades:
                        if t['type'] == 'SELL':
                            strategy_return += t['pnl']
                            if t['pnl'] > 0:
                                wins += 1
                            else:
                                losses += 1
                    
                    print(f"\n  {Style.BRIGHT}Strategy:{Style.RESET_ALL} SMA 20/50 Crossover")
                    print(f"  {Style.BRIGHT}Period:{Style.RESET_ALL}   ~2 years ({len(df)} trading days)")
                    print(f"\n  {Style.BRIGHT}RESULTS:{Style.RESET_ALL}")
                    print(f"    Buy & Hold Return:  {color_pct(buy_hold_return)}")
                    print(f"    Strategy Return:    {color_pct(strategy_return)}")
                    print(f"    Total Trades:       {len([t for t in trades if t['type'] == 'SELL'])}")
                    print(f"    Win Rate:           {(wins/(wins+losses)*100) if (wins+losses) > 0 else 0:.1f}%")
                    
                    # Recent trades
                    if trades:
                        print(f"\n  {Style.BRIGHT}RECENT SIGNALS:{Style.RESET_ALL}")
                        for t in trades[-6:]:
                            date_str = t['date'].strftime('%Y-%m-%d')
                            if t['type'] == 'BUY':
                                print(f"    {Fore.GREEN}BUY {Style.RESET_ALL} {date_str}  @ {fmt_money(t['price'])}")
                            else:
                                print(f"    {Fore.RED}SELL{Style.RESET_ALL} {date_str}  @ {fmt_money(t['price'])}  P&L: {color_pct(t['pnl'])}")
                    
                    print(f"\n  {Fore.YELLOW}⚠ Past performance doesn't guarantee future results{Style.RESET_ALL}")
                print(f"{'═'*70}\n")
                last_symbol = symbol

            # --- NEW v11: DIVIDEND TRACKER ---
            elif cmd == 'divs':
                pf.dividends(fetcher)

            # --- NEW v11: QUICK STATS ---
            elif cmd == 'stats':
                pf.quick_stats(fetcher)

            # --- NEW: ASCII CHART ---
            elif cmd == 'chart' and args:
                symbol = args[0].upper()
                period = args[1] if len(args) > 1 else "3mo"
                df = fetcher.get_history(symbol, period)
                if df is None or df.empty: err("No data found")
                else: AsciiChart.plot(df, title=f"{symbol} - {period}")

            # --- NEW: WATCHLIST SCANNER ---
            elif cmd == 'scan':
                if not wl.symbols: err("Watchlist empty. Add symbols first."); continue
                print(Fore.CYAN + f"\n{'═'*60}\n SCANNING WATCHLIST...\n{'═'*60}" + Style.RESET_ALL)
                rows = []
                prog = ProgressBar(len(wl.symbols), "  Scanning")
                
                def scan_ticker(s):
                    df = fetcher.get_history(s, "6mo")
                    if df is None or len(df) < 30: return None
                    rsi = TechnicalAnalysis.rsi(df['Close']).iloc[-1]
                    trend = TechnicalAnalysis.trend_strength(df)
                    return [s, f"{rsi:.1f}", trend['strength'], trend['score']]

                with ThreadPoolExecutor() as executor:
                    futures = {executor.submit(scan_ticker, s): s for s in wl.symbols}
                    for i, f in enumerate(futures):
                        prog.update(i)
                        res = f.result()
                        if res: rows.append(res)
                prog.done()
                
                # Sort by Trend Score (Strongest to Weakest)
                rows.sort(key=lambda x: x[3], reverse=True)
                
                # Colorize Output
                formatted_rows = []
                for r in rows:
                    rsi_val = float(r[1])
                    rsi_str = (Fore.GREEN if rsi_val < 30 else Fore.RED if rsi_val > 70 else Fore.RESET) + r[1] + Style.RESET_ALL
                    trend_str = color_signal(r[2])
                    formatted_rows.append([r[0], rsi_str, trend_str, r[3]])
                    
                print(tabulate(formatted_rows, headers=["Symbol", "RSI", "Trend", "Score"]))
                print(f"{'═'*60}\n")

            # --- EXISTING COMMANDS (UNCHANGED) ---
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

            elif cmd == 'dash' and args:
                symbol = args[0].upper()
                print(Fore.CYAN + f"\n{'═'*70}\n DASHBOARD — {symbol}\n{'═'*70}" + Style.RESET_ALL)
                with ThreadPoolExecutor() as executor:
                    f_price = executor.submit(fetcher.get_stock_price, symbol)
                    f_hist = executor.submit(fetcher.get_history, symbol, "6mo")
                    f_meta = executor.submit(fetcher.get_meta, symbol)
                    f_news = executor.submit(fetcher.get_news, symbol)
                    d, df, meta, news = f_price.result(), f_hist.result(), f_meta.result(), f_news.result()

                print(f"\n{Style.BRIGHT}{meta.get('name', symbol)}{Style.RESET_ALL} ({meta.get('sector', 'Unknown')})")
                print(f"  {Style.BRIGHT}PRICE:{Style.RESET_ALL}   {fmt_money(d['price'])}  {color_money(d['change'])} ({color_pct(d['pct'])})")
                
                if df is not None and len(df) > 30:
                    trend = TechnicalAnalysis.trend_strength(df)
                    rsi = TechnicalAnalysis.rsi(df['Close']).iloc[-1]
                    print(f"  {Style.BRIGHT}TREND:{Style.RESET_ALL}   {color_signal(trend['strength'])} (Score: {trend['score']})")
                    print(f"  {Style.BRIGHT}RSI:{Style.RESET_ALL}     {rsi:.1f} ({color_signal('OVERSOLD' if rsi<30 else 'OVERBOUGHT' if rsi>70 else 'Neutral')})")
                    closes = df['Close'].tail(10).tolist()
                    min_c, max_c = min(closes), max(closes)
                    spark = ""
                    if max_c > min_c:
                        chars = "  ▂▃▄▅▆▇█"
                        spark = "".join([chars[int((c - min_c) / (max_c - min_c) * 8)] for c in closes])
                    print(f"  {Style.BRIGHT}10D:{Style.RESET_ALL}     {spark}  {fmt_money(closes[0])} -> {fmt_money(closes[-1])}")

                if news:
                    print(f"\n{Style.BRIGHT}LATEST HEADLINES{Style.RESET_ALL}")
                    for item in news[:3]:
                        title = item.get('title', '')[:60]
                        print(f"  • {title}...")
                print(f"{'═'*70}\n")

            elif cmd == 'ta' and args:
                symbol = args[0].upper()
                print(Fore.CYAN + f"\n{'═'*70}\n TECHNICAL ANALYSIS — {symbol}\n{'═'*70}" + Style.RESET_ALL)
                df = fetcher.get_history(symbol, "6mo")
                if df is None or len(df) < 30: err(f"Insufficient data for {symbol}")
                else:
                    close = df['Close']
                    high, low = df['High'], df['Low']
                    price = close.iloc[-1]
                    d = fetcher.get_stock_price(symbol)
                    print(f"\n{Style.BRIGHT}PRICE{Style.RESET_ALL}")
                    print(f"  Current:      {fmt_money(price)}")
                    print(f"  Day Change:   {color_pct(d['pct'])}")
                    print(f"  52W High:     {fmt_money(high.max())}")
                    print(f"  52W Low:      {fmt_money(low.min())}")
                    pct_from_high = ((price - high.max()) / high.max()) * 100
                    print(f"  From 52W High: {color_pct(pct_from_high)}")
                    
                    rsi = TechnicalAnalysis.rsi(close)
                    rsi_val = rsi.iloc[-1]
                    rsi_signal = "OVERSOLD" if rsi_val < 30 else "OVERBOUGHT" if rsi_val > 70 else "Neutral"
                    print(f"\n{Style.BRIGHT}RSI (14){Style.RESET_ALL}")
                    print(f"  Value:  {rsi_val:.2f}")
                    print(f"  Signal: {color_signal(rsi_signal)}")
                    
                    macd_line, signal_line, histogram = TechnicalAnalysis.macd(close)
                    macd_signal = "BULLISH" if histogram.iloc[-1] > 0 else "BEARISH"
                    print(f"\n{Style.BRIGHT}MACD (12/26/9){Style.RESET_ALL}")
                    print(f"  MACD Line:    {macd_line.iloc[-1]:.4f}")
                    print(f"  Signal Line:  {signal_line.iloc[-1]:.4f}")
                    print(f"  Histogram:    {histogram.iloc[-1]:.4f}")
                    print(f"  Signal:       {color_signal(macd_signal)}")
                    
                    upper, middle, lower = TechnicalAnalysis.bollinger_bands(close)
                    bb_pct = (price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1]) * 100 if (upper.iloc[-1] - lower.iloc[-1]) > 0 else 50
                    bb_signal = "OVERSOLD" if bb_pct < 20 else "OVERBOUGHT" if bb_pct > 80 else "Neutral"
                    print(f"\n{Style.BRIGHT}BOLLINGER BANDS (20/2){Style.RESET_ALL}")
                    print(f"  Upper:    {fmt_money(upper.iloc[-1])}")
                    print(f"  Middle:   {fmt_money(middle.iloc[-1])}")
                    print(f"  Lower:    {fmt_money(lower.iloc[-1])}")
                    print(f"  %B:       {bb_pct:.1f}%")
                    print(f"  Signal:   {color_signal(bb_signal)}")
                    
                    trend, st_upper, st_lower = TechnicalAnalysis.supertrend(df)
                    st_signal = "BULLISH" if trend.iloc[-1] == 1 else "BEARISH"
                    print(f"\n{Style.BRIGHT}SUPERTREND (10/3){Style.RESET_ALL}")
                    print(f"  Signal: {color_signal(st_signal)}")
                    
                    atr = TechnicalAnalysis.atr(df)
                    atr_pct = (atr.iloc[-1] / price) * 100
                    print(f"\n{Style.BRIGHT}ATR (14){Style.RESET_ALL}")
                    print(f"  Value:    {fmt_money(atr.iloc[-1])}")
                    print(f"  % of Price: {atr_pct:.2f}%")
                    
                    pivots = TechnicalAnalysis.pivot_points(df)
                    print(f"\n{Style.BRIGHT}PIVOT POINTS{Style.RESET_ALL}")
                    print(f"  R3: {fmt_money(pivots['R3'])}  R2: {fmt_money(pivots['R2'])}  R1: {fmt_money(pivots['R1'])}")
                    print(f"  Pivot: {fmt_money(pivots['P'])}")
                    print(f"  S1: {fmt_money(pivots['S1'])}  S2: {fmt_money(pivots['S2'])}  S3: {fmt_money(pivots['S3'])}")
                    
                    sma_20 = close.rolling(20).mean().iloc[-1]
                    sma_50 = close.rolling(50).mean().iloc[-1]
                    sma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None
                    print(f"\n{Style.BRIGHT}MOVING AVERAGES{Style.RESET_ALL}")
                    print(f"  SMA 20:  {fmt_money(sma_20)} {'▲' if price > sma_20 else '▼'}")
                    print(f"  SMA 50:  {fmt_money(sma_50)} {'▲' if price > sma_50 else '▼'}")
                    if sma_200:
                        print(f"  SMA 200: {fmt_money(sma_200)} {'▲' if price > sma_200 else '▼'}")
                    
                    signals = [rsi_signal, macd_signal, bb_signal, st_signal]
                    bull_count = sum(1 for s in signals if "BULL" in s or "OVERSOLD" in s)
                    bear_count = sum(1 for s in signals if "BEAR" in s or "OVERBOUGHT" in s)
                    overall = "BULLISH" if bull_count > bear_count else "BEARISH" if bear_count > bull_count else "NEUTRAL"
                    print(f"\n{Style.BRIGHT}OVERALL{Style.RESET_ALL}")
                    print(f"  Score: {bull_count} Bull / {bear_count} Bear")
                    print(f"  Bias:  {color_signal(overall)}")
                    print(f"{'═'*70}\n")
            
            elif cmd == 'compare' and len(args) >= 2:
                sym1, sym2 = args[0].upper(), args[1].upper()
                print(Fore.CYAN + f"\n{'═'*80}\n COMPARISON — {sym1} vs {sym2}\n{'═'*80}" + Style.RESET_ALL)
                df1, df2 = fetcher.get_history(sym1, "6mo"), fetcher.get_history(sym2, "6mo")
                d1, d2 = fetcher.get_stock_price(sym1), fetcher.get_stock_price(sym2)
                
                if df1 is None or df2 is None or len(df1) < 30 or len(df2) < 30:
                    err("Insufficient data for comparison")
                else:
                    def get_metrics(df, d):
                        close = df['Close']
                        price = close.iloc[-1]
                        rsi = TechnicalAnalysis.rsi(close).iloc[-1]
                        _, _, hist = TechnicalAnalysis.macd(close)
                        upper, middle, lower = TechnicalAnalysis.bollinger_bands(close)
                        bb_pct = (price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1]) * 100 if (upper.iloc[-1] - lower.iloc[-1]) > 0 else 50
                        trend, _, _ = TechnicalAnalysis.supertrend(df)
                        atr = TechnicalAnalysis.atr(df).iloc[-1]
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
                    
                    def winner(v1, v2, higher_better=True):
                        if higher_better:
                            return (Fore.GREEN + "◀" + Style.RESET_ALL, "") if v1 > v2 else ("", Fore.GREEN + "▶" + Style.RESET_ALL) if v2 > v1 else ("", "")
                        else:
                            return (Fore.GREEN + "◀" + Style.RESET_ALL, "") if v1 < v2 else ("", Fore.GREEN + "▶" + Style.RESET_ALL) if v2 < v1 else ("", "")
                    rows = []
                    rows.append(["Price", fmt_money(m1['price']), "", fmt_money(m2['price'])])
                    w = winner(m1['day_pct'], m2['day_pct'])
                    rows.append(["Day %", color_pct(m1['day_pct']), w[0], color_pct(m2['day_pct']) + " " + w[1]])
                    w = winner(m1['pct_1w'], m2['pct_1w'])
                    rows.append(["1 Week %", color_pct(m1['pct_1w']), w[0], color_pct(m2['pct_1w']) + " " + w[1]])
                    w = winner(m1['pct_1m'], m2['pct_1m'])
                    rows.append(["1 Month %", color_pct(m1['pct_1m']), w[0], color_pct(m2['pct_1m']) + " " + w[1]])
                    w = winner(m1['pct_3m'], m2['pct_3m'])
                    rows.append(["3 Month %", color_pct(m1['pct_3m']), w[0], color_pct(m2['pct_3m']) + " " + w[1]])
                    rows.append(["─"*12, "─"*15, "─", "─"*15])
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
                    w = winner(m1['volatility'], m2['volatility'], higher_better=False)
                    rows.append(["Volatility", f"{m1['volatility']:.1f}%", w[0], f"{m2['volatility']:.1f}% " + w[1]])
                    w = winner(m1['atr_pct'], m2['atr_pct'], higher_better=False)
                    rows.append(["ATR %", f"{m1['atr_pct']:.2f}%", w[0], f"{m2['atr_pct']:.2f}% " + w[1]])
                    print(tabulate(rows, headers=["Metric", sym1, "", sym2], tablefmt="simple"))
                    print(f"{'═'*80}\n")
            
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
                symbol = args[0].upper()
                d = fetcher.get_stock_price(symbol)
                print(f"  {symbol}: {fmt_money(d['price'])} ({color_pct(d['pct'])})")
                last_symbol = symbol
            
            elif len(cmd) <= 5 and cmd.isalpha(): 
                d = fetcher.get_stock_price(cmd)
                print(f"  {cmd.upper()}: {fmt_money(d['price'])} ({color_pct(d['pct'])})")
                last_symbol = cmd.upper()
                
        except Exception as e: err(str(e))

if __name__ == "__main__":
    main()