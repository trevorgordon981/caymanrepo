#!/usr/bin/env python3
"""
STOCK TICKER APP v12.2.0 — Universal Portfolio Tracker & Analyzer
INSTALL: pip install yfinance pandas numpy matplotlib tabulate colorama requests scipy

NEW IN v12.2.0 (LEAPS FOCUS):
- 📊 LEAPS Dashboard (leaps): Dedicated view for long-term options
- 🔄 Roll Optimizer (leapsroll): Analyze roll strategies with cost/theta analysis
- 🔍 LEAPS Chain Finder (leapschain): Find stock replacement opportunities
- 📐 Portfolio Greeks (greeks): Full delta/gamma/theta/vega exposure analysis
- 🏥 LEAPS Health Scoring: Automatic assessment of position health
- ⚡ Roll Recommendations: Alerts when positions need attention
- 📈 Breakeven Tracking: See your profit thresholds at a glance

NEW IN v12.1.0 (UX IMPROVEMENTS):
- 🎨 Cleaner visual design with improved spacing and headers
- 💡 Smart command suggestions for typos ("Did you mean...?")
- 📋 Quick-start guide shown on first launch
- ⌨️  More command shortcuts (p=portfolio, m=market, t=ta, d=dash)
- 🔔 Contextual tips to guide you to related commands
- 📊 Portfolio value shown in prompt after import
- 🎯 Improved error messages with helpful hints
- ⏳ Better progress bars with time estimates
- 📖 Searchable help (help portfolio, help analysis)
- 🎪 Cleaner tables and section headers

PREVIOUS (v12.0.0):
- 📄 Export Reports (export): Generate PDF/CSV portfolio reports
- 📐 Position Sizing (size): Kelly criterion & risk-based position calculator
- 🔗 Correlation Matrix (corr): Portfolio diversification analysis
- 🎯 Stop-Loss Tracker (stops): Manage stop-loss and take-profit levels
- 📈 Performance History (perf): Track portfolio performance over time
- ⏱️ Multi-Timeframe TA (mtf): Analyze across multiple timeframes
- 🔄 Rebalance Suggestions (rebalance): Get portfolio allocation advice
- 🛡️ Improved reliability: Retry logic, rate limiting, validation
- 💾 Data persistence: Historical snapshots saved automatically

PREVIOUS (v11.0.0):
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
import os, re, sys, json, time, math, shlex, logging, warnings, random, hashlib, atexit
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from collections import defaultdict
import threading

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

# --- RATE LIMITER ---
class RateLimiter:
    """Thread-safe rate limiter for API calls"""
    def __init__(self, calls_per_second: float = 2.0, burst: int = 5):
        self.calls_per_second = calls_per_second
        self.burst = burst
        self.tokens = burst
        self.last_update = _now()
        self.lock = threading.Lock()
    
    def acquire(self, timeout: float = 30.0) -> bool:
        """Acquire a token, blocking if necessary. Returns True if acquired."""
        deadline = _now() + timeout
        while _now() < deadline:
            with self.lock:
                now = _now()
                # Add tokens based on time elapsed
                elapsed = now - self.last_update
                self.tokens = min(self.burst, self.tokens + elapsed * self.calls_per_second)
                self.last_update = now
                
                if self.tokens >= 1:
                    self.tokens -= 1
import difflib

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

# ═══════════════════════════════════════════════════════════════════════════════
# NEW v12.1: ENHANCED UX COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

# All available commands with descriptions for smart suggestions
ALL_COMMANDS = {
    'pf': 'View portfolio', 'portfolio': 'View portfolio', 'import': 'Import CSV',
    'risk': 'Risk analysis', 'cal': 'Earnings calendar', 'divs': 'Dividends',
    'stats': 'Quick stats', 'clear': 'Clear portfolio', 'export': 'Export report',
    'stops': 'Stop-loss tracker', 'size': 'Position sizing', 'perf': 'Performance',
    'corr': 'Correlation', 'rebalance': 'Rebalancing', 'mtf': 'Multi-timeframe',
    'q': 'Quick quote', 'quote': 'Detailed quote', 'info': 'Company info',
    'dash': 'Dashboard', 'news': 'News', 'chart': 'ASCII chart',
    'market': 'Market overview', 'sectors': 'Sector heatmap', 'movers': 'Top movers',
    'random': 'Random stock', 'ta': 'Technical analysis', 'ta2': 'Extended TA',
    'trend': 'Trend analysis', 'levels': 'Support/resistance', 'signals': 'Trade signals',
    'sentiment': 'News sentiment', 'compare': 'Compare stocks', 'backtest': 'Backtest',
    'watch': 'Watchlist', 'scan': 'Scan watchlist', 'alert': 'Price alerts',
    'refresh': 'Clear cache', 'debug': 'Toggle debug', 'help': 'Show help',
    'h': 'Show help', '?': 'Show help', 'quit': 'Exit', 'exit': 'Exit',
    # LEAPS Commands (NEW)
    'leaps': 'LEAPS portfolio dashboard', 'leapsroll': 'LEAPS roll optimizer',
    'leapschain': 'Find LEAPS opportunities', 'greeks': 'Portfolio Greeks exposure',
}

# Command shortcuts for faster typing
COMMAND_ALIASES = {
    'p': 'pf', 'port': 'pf', 'i': 'import', 'load': 'import',
    's': 'stats', 'summary': 'stats', 'm': 'market', 'mkt': 'market',
    'sec': 'sectors', 'heat': 'sectors', 'w': 'watch', 'wl': 'watch',
    'a': 'alert', 'alerts': 'alert', 'n': 'news', 'd': 'dash',
    'dashboard': 'dash', 'c': 'chart', 'l': 'levels', 't': 'ta',
    'tech': 'ta', 'sig': 'signals', 'sent': 'sentiment',
    'comp': 'compare', 'vs': 'compare', 'bt': 'backtest',
    'r': 'random', 'discover': 'random', 'x': 'export',
    'st': 'stops', 'stop': 'stops', 'sz': 'size', 'rb': 'rebalance',
    'bye': 'quit', 'q!': 'quit',
    # LEAPS aliases (NEW)
    'leap': 'leaps', 'lp': 'leaps', 'roll': 'leapsroll', 'lr': 'leapsroll',
    'chain': 'leapschain', 'lc': 'leapschain', 'gr': 'greeks', 'g': 'greeks',
}

def suggest_command(user_input: str):
    """Find the closest matching command for typo correction."""
    user_cmd = user_input.lower().split()[0] if user_input else ""
    if not user_cmd or user_cmd in ALL_COMMANDS or user_cmd in COMMAND_ALIASES:
        return None
    all_cmds = list(ALL_COMMANDS.keys()) + list(COMMAND_ALIASES.keys())
    matches = difflib.get_close_matches(user_cmd, all_cmds, n=1, cutoff=0.6)
    return matches[0] if matches else None

def print_welcome():
    """Print a welcoming startup screen."""
    print(Fore.CYAN + Style.BRIGHT + """
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                                                                       ║
    ║   ███████╗████████╗ ██████╗  ██████╗██╗  ██╗                          ║
    ║   ██╔════╝╚══██╔══╝██╔═══██╗██╔════╝██║ ██╔╝                          ║
    ║   ███████╗   ██║   ██║   ██║██║     █████╔╝                           ║
    ║   ╚════██║   ██║   ██║   ██║██║     ██╔═██╗                           ║
    ║   ███████║   ██║   ╚██████╔╝╚██████╗██║  ██╗                          ║
    ║   ╚══════╝   ╚═╝    ╚═════╝  ╚═════╝╚═╝  ╚═╝  TICKER v12.1           ║
    ║                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """ + Style.RESET_ALL)

def print_quick_start():
    """Print quick-start tips for new users."""
    print(Fore.WHITE + """
    ┌─────────────────────────────────────────────────────────────────────┐
    │  """ + Style.BRIGHT + """⚡ QUICK START""" + Style.RESET_ALL + Fore.WHITE + """                                                      │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │  """ + Fore.GREEN + """AAPL""" + Fore.WHITE + """          → Quick quote (just type any ticker)             │
    │  """ + Fore.GREEN + """market""" + Fore.WHITE + """        → See indices, VIX & Bitcoin                     │
    │  """ + Fore.GREEN + """import file""" + Fore.WHITE + """   → Load your portfolio CSV                        │
    │  """ + Fore.GREEN + """ta AAPL""" + Fore.WHITE + """       → Technical analysis                             │
    │  """ + Fore.GREEN + """help""" + Fore.WHITE + """          → Full command list                              │
    │                                                                     │
    │  """ + Fore.YELLOW + """💡 Tip: Commands remember last symbol. Just type 'ta' again!""" + Fore.WHITE + """     │
    │  """ + Fore.YELLOW + """💡 Tip: Use shortcuts: p=portfolio, m=market, t=ta, d=dash""" + Fore.WHITE + """      │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
    """ + Style.RESET_ALL)

def print_tip(tip_type):
    """Print contextual tips based on what the user just did."""
    tips = {
        'import': f"  {Fore.YELLOW}💡 Next: 'pf' for portfolio, 'stats' for summary, 'risk' for beta analysis{Style.RESET_ALL}",
        'pf': f"  {Fore.YELLOW}💡 Try: 'stats' for summary, 'divs' for dividends, 'corr' for diversification{Style.RESET_ALL}",
        'ta': f"  {Fore.YELLOW}💡 Try: 'ta2' for more indicators, 'signals' for buy/sell score, 'levels' for S/R{Style.RESET_ALL}",
        'quote': f"  {Fore.YELLOW}💡 Try: 'dash' for overview, 'news' for headlines, 'chart' for price history{Style.RESET_ALL}",
        'market': f"  {Fore.YELLOW}💡 Try: 'sectors' for heatmap, 'movers' for top gainers/losers{Style.RESET_ALL}",
        'watch': f"  {Fore.YELLOW}💡 Try: 'scan' to find oversold/trending stocks in your watchlist{Style.RESET_ALL}",
        'first_stock': f"  {Fore.YELLOW}💡 Now try: 'ta' for technicals, 'news' for headlines, 'dash' for overview{Style.RESET_ALL}",
    }
    if tip_type in tips and config.show_tips:
        print(tips[tip_type])

def format_prompt(pf_value=0, last_symbol=None):
    """Create a rich prompt showing portfolio value and context."""
    parts = []
    if pf_value > 0:
        if pf_value >= 1_000_000:
            val_str = f"${pf_value/1_000_000:.2f}M"
        elif pf_value >= 1_000:
            val_str = f"${pf_value/1_000:.1f}K"
        else:
            val_str = f"${pf_value:.0f}"
        parts.append(Fore.GREEN + val_str + Style.RESET_ALL)
    if last_symbol:
        parts.append(Fore.CYAN + last_symbol + Style.RESET_ALL)
    if parts:
        return f"[{' | '.join(parts)}] " + Fore.GREEN + "▶ " + Style.RESET_ALL
    return Fore.GREEN + "▶ " + Style.RESET_ALL

def print_header(title, width=70, icon=""):
    """Print a clean section header."""
    if icon:
        title = f"{icon} {title}"
    print(Fore.CYAN + f"\n{'─'*width}\n {title}\n{'─'*width}" + Style.RESET_ALL)

def print_footer(width=70):
    """Print a clean section footer."""
    print(Fore.CYAN + f"{'─'*width}\n" + Style.RESET_ALL)

def print_kv(key, value, indent=2):
    """Print a formatted key-value pair."""
    print(f"{' '*indent}{Style.BRIGHT}{key}:{Style.RESET_ALL} {value}")

def print_error(msg, hint=None):
    """Print an error message with an optional helpful hint."""
    print(Fore.RED + f"  ✗ {msg}" + Style.RESET_ALL)
    if hint:
        print(Fore.YELLOW + f"    💡 {hint}" + Style.RESET_ALL)

def print_searchable_help(keyword=None):
    """Print help filtered by keyword."""
    help_sections = {
        'portfolio': """
  """ + Style.BRIGHT + """PORTFOLIO COMMANDS""" + Style.RESET_ALL + """
    pf, p             View full portfolio with P&L and strategy labels
    import FILE       Load CSV (Fidelity, Schwab, E*Trade, Robinhood, etc.)
    stats, s          Quick portfolio health stats at a glance
    risk              Calculate portfolio beta vs S&P 500
    cal               Scan portfolio for upcoming earnings
    divs              Show dividend income & upcoming ex-dates
    export [csv]      Export portfolio report
    clear             Delete all portfolio data
""",
        'leaps': """
  """ + Style.BRIGHT + """LEAPS OPTIONS (v12.2)""" + Style.RESET_ALL + """
    leaps, lp         LEAPS portfolio dashboard with health scores
    leapsroll SYM     Roll optimizer for LEAPS positions
    leapschain SYM    Find new LEAPS opportunities [--budget AMT]
    greeks, g         Portfolio-wide Greeks exposure analysis
    
  """ + Style.BRIGHT + """LEAPS TIPS""" + Style.RESET_ALL + """
    • LEAPS = options with >270 days to expiration
    • Health score considers DTE, moneyness, theta decay
    • Roll when DTE drops below 180 or if deep OTM
    • High delta LEAPS (0.70+) work as stock replacement
""",
        'risk': """
  """ + Style.BRIGHT + """RISK MANAGEMENT (v12)""" + Style.RESET_ALL + """
    stops             View/manage stop-loss and take-profit levels
    stops add SYM --stop PRICE [--tp PRICE] [--trail PCT]
    size SYM          Position sizing calculator
    perf              View portfolio performance history
    corr              Portfolio correlation matrix
    rebalance         Get rebalancing suggestions
    mtf SYMBOL        Multi-timeframe analysis
""",
        'quote': """
  """ + Style.BRIGHT + """QUOTE & LOOKUP""" + Style.RESET_ALL + """
    AAPL              Just type ticker for quick quote
    q SYMBOL          Quick quote
    quote SYMBOL      Detailed quote with fundamentals
    dash, d SYMBOL    Dashboard: price, trend, headlines
    news, n SYMBOL    Latest news headlines
    chart, c SYMBOL   ASCII price chart
""",
        'market': """
  """ + Style.BRIGHT + """MARKET OVERVIEW""" + Style.RESET_ALL + """
    market, m         Show indices, VIX, and Bitcoin
    sectors           Sector heatmap - visual performance
    movers            Top gainers and losers today
    random, r         Discover a random stock to research
""",
        'analysis': """
  """ + Style.BRIGHT + """TECHNICAL ANALYSIS""" + Style.RESET_ALL + """
    ta, t SYMBOL      Basic TA (RSI, MACD, Bollinger, Supertrend)
    ta2 SYMBOL        Extended TA (Stochastic, ADX, Ichimoku)
    trend SYMBOL      Trend strength analysis
    levels, l SYMBOL  Support/Resistance and Fibonacci
    signals SYMBOL    AI trade signal aggregator (buy/sell score)
    sentiment SYMBOL  News sentiment analysis
    compare A B       Compare two stocks side-by-side
    backtest SYM      Simple MA crossover backtest
""",
        'watchlist': """
  """ + Style.BRIGHT + """WATCHLIST & ALERTS""" + Style.RESET_ALL + """
    watch, w          View watchlist
    watch add SYM     Add symbol to watchlist
    scan              Scan watchlist for oversold/trending
    alert, a          View all price alerts
    alert add SYM PRICE [above/below]  Set a price alert
    alert rm #        Remove alert by number
""",
        'settings': """
  """ + Style.BRIGHT + """SETTINGS""" + Style.RESET_ALL + """
    refresh           Clear cached data for fresh updates
    debug             Toggle debug mode
    tips              Toggle contextual tips on/off
""",
    }
    
    if keyword:
        keyword = keyword.lower()
        found = False
        for section, content in help_sections.items():
            if keyword in section or keyword in content.lower():
                print(Fore.CYAN + content + Style.RESET_ALL)
                found = True
        if not found:
            print(f"  No help found for '{keyword}'. Try: portfolio, leaps, risk, quote, market, analysis, watchlist")
    else:
        # Print full help
        print(Fore.CYAN + """
╔══════════════════════════════════════════════════════════════════════════════╗
║  STOCK TICKER v12.2.0 — HELP                                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  💡 Smart Context: Commands remember last symbol. Just type 'ta' again!      ║
║  💡 Shortcuts: p=portfolio, m=market, t=ta, lp=leaps, g=greeks              ║
║  💡 Search Help: 'help portfolio', 'help leaps', 'help analysis'            ║
╚══════════════════════════════════════════════════════════════════════════════╝""" + Style.RESET_ALL)
        for content in help_sections.values():
            print(Fore.CYAN + content + Style.RESET_ALL)

# ═══════════════════════════════════════════════════════════════════════════════
# END NEW UX COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

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

# --- RATE LIMITER ---
class RateLimiter:
    """Thread-safe rate limiter for API calls"""
    def __init__(self, calls_per_second: float = 2.0, burst: int = 5):
        self.calls_per_second = calls_per_second
        self.burst = burst
        self.tokens = burst
        self.last_update = _now()
        self.lock = threading.Lock()
    
    def acquire(self, timeout: float = 30.0) -> bool:
        """Acquire a token, blocking if necessary. Returns True if acquired."""
        deadline = _now() + timeout
        while _now() < deadline:
            with self.lock:
                now = _now()
                # Add tokens based on time elapsed
                elapsed = now - self.last_update
                self.tokens = min(self.burst, self.tokens + elapsed * self.calls_per_second)
                self.last_update = now
                
                if self.tokens >= 1:
                    self.tokens -= 1
                    return True
            time.sleep(0.1)
        return False

# Global rate limiter for API calls
_api_limiter = RateLimiter(calls_per_second=3.0, burst=10)

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator for retrying failed operations with exponential backoff"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            for attempt in range(max_retries):
                try:
                    _api_limiter.acquire()
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        time.sleep(current_delay)
                        current_delay *= backoff
            if last_exception:
                raise last_exception
            return None
        return wrapper
    return decorator

def validate_symbol(symbol: str) -> Optional[str]:
    """Validate and clean a stock symbol"""
    if not symbol:
        return None
    cleaned = re.sub(r'[^A-Za-z0-9.-]', '', symbol.upper().strip())
    if len(cleaned) < 1 or len(cleaned) > 10:
        return None
    return cleaned

def validate_number(value: Any, min_val: float = None, max_val: float = None, default: float = 0.0) -> float:
    """Validate and sanitize numeric input"""
    try:
        num = float(value)
        if math.isnan(num) or math.isinf(num):
            return default
        if min_val is not None and num < min_val:
            return min_val
        if max_val is not None and num > max_val:
            return max_val
        return num
    except (TypeError, ValueError):
        return default

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
    show_tips: bool = True  # NEW v12.1: Toggle contextual tips
    first_run: bool = True  # NEW v12.1: Track first run for welcome
    
    def __post_init__(self): Path(self.data_dir).mkdir(parents=True, exist_ok=True)
    @classmethod
    def load(cls):
        cfg = cls()
        p = Path(os.path.expanduser("~/.stockticker")) / "config.json"
        if p.exists():
            try:
                data = json.loads(p.read_text())
                for k, v in data.items():
                    if hasattr(cfg, k): setattr(cfg, k, v)
                cfg.first_run = False  # Not first run if config exists
            except: pass
        cfg.__post_init__()
        return cfg
    def save(self):
        p = Path(self.data_dir) / "config.json"
        p.write_text(json.dumps({
            'theme': self.theme, 'debug': self.debug, 'show_charts': self.show_charts, 
            'risk_free_rate': self.risk_free_rate, 'benchmark': self.benchmark,
            'show_tips': self.show_tips, 'first_run': False
        }, indent=2))

config = Config.load()

# --- NEW v12: STOP-LOSS TRACKER ---
class StopLossTracker:
    """Track stop-loss and take-profit levels for positions"""
    def __init__(self):
        self.fp = Path(config.data_dir) / "stops.json"
        self.stops = self._load()
    
    def _load(self) -> Dict:
        if self.fp.exists():
            try:
                return json.loads(self.fp.read_text())
            except:
                pass
        return {}
    
    def _save(self):
        self.fp.write_text(json.dumps(self.stops, indent=2))
    
    def add(self, symbol: str, stop_loss: float = None, take_profit: float = None, trailing_pct: float = None):
        """Add stop-loss and/or take-profit for a symbol"""
        symbol = validate_symbol(symbol)
        if not symbol:
            err("Invalid symbol")
            return
        
        entry = self.stops.get(symbol, {})
        if stop_loss:
            entry['stop_loss'] = validate_number(stop_loss, min_val=0.01)
        if take_profit:
            entry['take_profit'] = validate_number(take_profit, min_val=0.01)
        if trailing_pct:
            entry['trailing_pct'] = validate_number(trailing_pct, min_val=0.1, max_val=50.0)
            entry['trailing_high'] = entry.get('trailing_high', 0)
        
        entry['created'] = datetime.now().isoformat()
        self.stops[symbol] = entry
        self._save()
        success(f"Stop levels set for {symbol}")
    
    def remove(self, symbol: str):
        symbol = validate_symbol(symbol)
        if symbol in self.stops:
            del self.stops[symbol]
            self._save()
            success(f"Removed stops for {symbol}")
        else:
            err(f"No stops found for {symbol}")
    
    def check(self, fetcher) -> List[Dict]:
        """Check all stops against current prices and return triggered ones"""
        triggered = []
        updated = False
        
        for symbol, levels in list(self.stops.items()):
            try:
                price_data = fetcher.get_stock_price(symbol)
                price = price_data.get('price', 0)
                if price <= 0:
                    continue
                
                # Update trailing stop high watermark
                if 'trailing_pct' in levels:
                    if price > levels.get('trailing_high', 0):
                        levels['trailing_high'] = price
                        updated = True
                    
                    # Calculate trailing stop level
                    trailing_stop = levels['trailing_high'] * (1 - levels['trailing_pct'] / 100)
                    if price <= trailing_stop:
                        triggered.append({
                            'symbol': symbol,
                            'type': 'trailing_stop',
                            'level': trailing_stop,
                            'price': price,
                            'high': levels['trailing_high']
                        })
                
                # Check fixed stop loss
                if 'stop_loss' in levels and price <= levels['stop_loss']:
                    triggered.append({
                        'symbol': symbol,
                        'type': 'stop_loss',
                        'level': levels['stop_loss'],
                        'price': price
                    })
                
                # Check take profit
                if 'take_profit' in levels and price >= levels['take_profit']:
                    triggered.append({
                        'symbol': symbol,
                        'type': 'take_profit',
                        'level': levels['take_profit'],
                        'price': price
                    })
            except:
                pass
        
        if updated:
            self._save()
        
        return triggered
    
    def show(self, fetcher):
        """Display all stop levels with current prices"""
        if not self.stops:
            warn("No stop levels set. Use 'stops add SYMBOL --stop PRICE' to add.")
            return
        
        print(Fore.CYAN + f"\n{'═'*80}\n 🎯 STOP-LOSS / TAKE-PROFIT TRACKER\n{'═'*80}" + Style.RESET_ALL)
        
        rows = []
        for symbol, levels in sorted(self.stops.items()):
            try:
                price_data = fetcher.get_stock_price(symbol)
                price = price_data.get('price', 0)
                
                stop = levels.get('stop_loss', '-')
                tp = levels.get('take_profit', '-')
                trail = levels.get('trailing_pct', '-')
                
                # Calculate distances
                stop_dist = ""
                tp_dist = ""
                
                if price > 0:
                    if isinstance(stop, (int, float)):
                        pct = (price - stop) / price * 100
                        stop_dist = f" ({pct:+.1f}%)"
                        stop = f"${stop:.2f}"
                    if isinstance(tp, (int, float)):
                        pct = (tp - price) / price * 100
                        tp_dist = f" ({pct:+.1f}%)"
                        tp = f"${tp:.2f}"
                    if isinstance(trail, (int, float)):
                        trail_level = levels.get('trailing_high', price) * (1 - trail / 100)
                        trail = f"{trail:.1f}% (${trail_level:.2f})"
                
                rows.append([
                    symbol,
                    fmt_money(price) if price > 0 else '-',
                    f"{stop}{stop_dist}",
                    f"{tp}{tp_dist}",
                    trail
                ])
            except:
                rows.append([symbol, '-', '-', '-', '-'])
        
        print(tabulate(rows, headers=["Symbol", "Price", "Stop Loss", "Take Profit", "Trailing"]))
        print(f"{'═'*80}\n")

# --- NEW v12: POSITION SIZER ---
class PositionSizer:
    """Calculate optimal position sizes using various methods"""
    
    @staticmethod
    def fixed_risk(account_size: float, risk_pct: float, entry: float, stop: float) -> Dict:
        """Calculate position size based on fixed % risk per trade"""
        if entry <= 0 or stop <= 0 or entry == stop:
            return {'error': 'Invalid entry or stop price'}
        
        risk_amount = account_size * (risk_pct / 100)
        risk_per_share = abs(entry - stop)
        shares = int(risk_amount / risk_per_share)
        position_value = shares * entry
        
        return {
            'shares': shares,
            'position_value': position_value,
            'risk_amount': risk_amount,
            'risk_per_share': risk_per_share,
            'account_pct': (position_value / account_size) * 100
        }
    
    @staticmethod
    def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate Kelly Criterion percentage for position sizing"""
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
        
        # Kelly % = W - [(1-W) / R] where W = win rate, R = win/loss ratio
        r = abs(avg_win / avg_loss)
        kelly = win_rate - ((1 - win_rate) / r)
        
        # Cap at 25% (half-Kelly is often recommended)
        return max(0, min(kelly, 0.25))
    
    @staticmethod
    def volatility_adjusted(account_size: float, target_risk: float, price: float, atr: float) -> Dict:
        """Calculate position size adjusted for volatility (ATR-based)"""
        if atr <= 0 or price <= 0:
            return {'error': 'Invalid ATR or price'}
        
        # Risk amount based on 2x ATR as stop distance
        risk_amount = account_size * (target_risk / 100)
        stop_distance = atr * 2
        shares = int(risk_amount / stop_distance)
        position_value = shares * price
        
        return {
            'shares': shares,
            'position_value': position_value,
            'atr': atr,
            'stop_distance': stop_distance,
            'suggested_stop': price - stop_distance,
            'account_pct': (position_value / account_size) * 100
        }

# --- NEW v12: PERFORMANCE TRACKER ---
class PerformanceTracker:
    """Track portfolio performance over time"""
    def __init__(self):
        self.fp = Path(config.data_dir) / "performance.json"
        self.history = self._load()
    
    def _load(self) -> List[Dict]:
        if self.fp.exists():
            try:
                return json.loads(self.fp.read_text())
            except:
                pass
        return []
    
    def _save(self):
        # Keep last 365 days of snapshots
        cutoff = (datetime.now() - timedelta(days=365)).isoformat()
        self.history = [h for h in self.history if h.get('date', '') > cutoff]
        self.fp.write_text(json.dumps(self.history, indent=2))
    
    def record_snapshot(self, total_value: float, total_cost: float, cash: float, positions: int):
        """Record a portfolio snapshot"""
        today = today_str()
        
        # Update or add today's snapshot
        for snap in self.history:
            if snap.get('date') == today:
                snap['value'] = total_value
                snap['cost'] = total_cost
                snap['cash'] = cash
                snap['positions'] = positions
                snap['pnl'] = total_value - total_cost
                snap['pnl_pct'] = ((total_value - total_cost) / total_cost * 100) if total_cost > 0 else 0
                self._save()
                return
        
        # Add new snapshot
        self.history.append({
            'date': today,
            'value': total_value,
            'cost': total_cost,
            'cash': cash,
            'positions': positions,
            'pnl': total_value - total_cost,
            'pnl_pct': ((total_value - total_cost) / total_cost * 100) if total_cost > 0 else 0
        })
        self._save()
    
    def get_stats(self) -> Dict:
        """Calculate performance statistics"""
        if len(self.history) < 2:
            return {'error': 'Need at least 2 snapshots for statistics'}
        
        sorted_hist = sorted(self.history, key=lambda x: x['date'])
        values = [h['value'] for h in sorted_hist]
        
        # Calculate returns
        returns = []
        for i in range(1, len(values)):
            if values[i-1] > 0:
                returns.append((values[i] - values[i-1]) / values[i-1])
        
        if not returns:
            return {'error': 'Insufficient data'}
        
        # Calculate metrics
        total_return = (values[-1] - values[0]) / values[0] * 100 if values[0] > 0 else 0
        
        avg_return = sum(returns) / len(returns) * 100
        std_dev = (sum((r - avg_return/100)**2 for r in returns) / len(returns)) ** 0.5 * 100
        
        # Sharpe ratio (annualized, assuming daily data)
        rf_daily = config.risk_free_rate / 252
        excess_returns = [r - rf_daily for r in returns]
        sharpe = 0
        if std_dev > 0:
            sharpe = (sum(excess_returns) / len(excess_returns)) / (std_dev/100) * (252 ** 0.5)
        
        # Max drawdown
        peak = values[0]
        max_dd = 0
        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        # Win days
        up_days = sum(1 for r in returns if r > 0)
        down_days = sum(1 for r in returns if r < 0)
        
        return {
            'total_return': total_return,
            'avg_daily_return': avg_return,
            'volatility': std_dev,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd * 100,
            'up_days': up_days,
            'down_days': down_days,
            'win_rate': up_days / (up_days + down_days) * 100 if (up_days + down_days) > 0 else 0,
            'start_value': values[0],
            'end_value': values[-1],
            'days_tracked': len(sorted_hist)
        }
    
    def show(self):
        """Display performance statistics"""
        print(Fore.CYAN + f"\n{'═'*70}\n 📈 PERFORMANCE HISTORY\n{'═'*70}" + Style.RESET_ALL)
        
        stats = self.get_stats()
        if 'error' in stats:
            warn(stats['error'])
            return
        
        print(f"\n  {Style.BRIGHT}OVERALL PERFORMANCE{Style.RESET_ALL}")
        print(f"    Total Return:      {color_pct(stats['total_return'])}")
        print(f"    Start Value:       {fmt_money(stats['start_value'])}")
        print(f"    Current Value:     {fmt_money(stats['end_value'])}")
        
        print(f"\n  {Style.BRIGHT}RISK METRICS{Style.RESET_ALL}")
        print(f"    Volatility:        {stats['volatility']:.2f}% daily")
        print(f"    Sharpe Ratio:      {stats['sharpe_ratio']:.2f}")
        print(f"    Max Drawdown:      {Fore.RED}{stats['max_drawdown']:.1f}%{Style.RESET_ALL}")
        
        print(f"\n  {Style.BRIGHT}TRADING STATS{Style.RESET_ALL}")
        print(f"    Days Tracked:      {stats['days_tracked']}")
        print(f"    Up Days:           {Fore.GREEN}{stats['up_days']}{Style.RESET_ALL}")
        print(f"    Down Days:         {Fore.RED}{stats['down_days']}{Style.RESET_ALL}")
        print(f"    Win Rate:          {stats['win_rate']:.1f}%")
        
        # Show recent history
        if self.history:
            print(f"\n  {Style.BRIGHT}RECENT SNAPSHOTS{Style.RESET_ALL}")
            recent = sorted(self.history, key=lambda x: x['date'], reverse=True)[:7]
            for snap in recent:
                pnl_color = Fore.GREEN if snap.get('pnl', 0) >= 0 else Fore.RED
                print(f"    {snap['date']}  Value: {fmt_money(snap['value'])}  P&L: {pnl_color}{fmt_money(snap.get('pnl', 0))}{Style.RESET_ALL}")
        
        print(f"{'═'*70}\n")

# --- NEW v12: CORRELATION ANALYZER ---
class CorrelationAnalyzer:
    """Analyze correlation between portfolio holdings"""
    
    @staticmethod
    def calculate_correlation_matrix(symbols: List[str], fetcher, period: str = "6mo") -> Optional[pd.DataFrame]:
        """Calculate correlation matrix for given symbols"""
        if len(symbols) < 2:
            return None
        
        # Fetch historical data
        price_data = {}
        for sym in symbols[:15]:  # Limit to 15 symbols for performance
            df = fetcher.get_history(sym, period)
            if df is not None and not df.empty:
                price_data[sym] = df['Close'].pct_change().dropna()
        
        if len(price_data) < 2:
            return None
        
        # Align data
        combined = pd.DataFrame(price_data)
        combined = combined.dropna()
        
        if len(combined) < 20:
            return None
        
        return combined.corr()
    
    @staticmethod
    def find_highly_correlated(corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """Find pairs of highly correlated assets"""
        pairs = []
        symbols = corr_matrix.columns.tolist()
        
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols):
                if i < j:  # Only upper triangle
                    corr = corr_matrix.loc[sym1, sym2]
                    if abs(corr) >= threshold:
                        pairs.append((sym1, sym2, corr))
        
        return sorted(pairs, key=lambda x: abs(x[2]), reverse=True)
    
    @staticmethod
    def portfolio_diversification_score(corr_matrix: pd.DataFrame) -> float:
        """Calculate diversification score (0-100, higher is better)"""
        if corr_matrix is None or corr_matrix.empty:
            return 0.0
        
        # Average absolute correlation (excluding diagonal)
        n = len(corr_matrix)
        if n < 2:
            return 0.0
        
        total_corr = 0
        count = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    total_corr += abs(corr_matrix.iloc[i, j])
                    count += 1
        
        avg_corr = total_corr / count if count > 0 else 0
        # Score: 100 means perfectly uncorrelated, 0 means perfectly correlated
        return max(0, (1 - avg_corr) * 100)

# --- NEW v12: REBALANCER ---
class PortfolioRebalancer:
    """Suggest portfolio rebalancing actions"""
    
    # Default target allocations by sector
    DEFAULT_TARGETS = {
        'Technology': 25,
        'Healthcare': 15,
        'Financials': 15,
        'Consumer Cyclical': 12,
        'Industrials': 10,
        'Communication Services': 8,
        'Consumer Defensive': 7,
        'Energy': 5,
        'Utilities': 3,
    }
    
    @staticmethod
    def analyze(stocks: Dict, fetcher, targets: Dict = None) -> Dict:
        """Analyze current allocation vs targets and suggest rebalancing"""
        if not stocks:
            return {'error': 'No stocks in portfolio'}
        
        targets = targets or PortfolioRebalancer.DEFAULT_TARGETS
        
        # Get sector for each holding
        sector_values = defaultdict(float)
        total_value = 0
        
        for sym, pos in stocks.items():
            try:
                meta = fetcher.get_meta(sym)
                sector = meta.get('sector', 'Unknown')
                price_data = fetcher.get_stock_price(sym)
                value = pos['qty'] * price_data.get('price', pos.get('broker_price', 0))
                sector_values[sector] += value
                total_value += value
            except:
                pass
        
        if total_value <= 0:
            return {'error': 'Could not calculate portfolio value'}
        
        # Calculate current allocations
        current_alloc = {sector: (value / total_value * 100) for sector, value in sector_values.items()}
        
        # Generate suggestions
        suggestions = []
        for sector, target_pct in targets.items():
            current_pct = current_alloc.get(sector, 0)
            diff = current_pct - target_pct
            
            if abs(diff) > 3:  # Only suggest if off by more than 3%
                action = "REDUCE" if diff > 0 else "INCREASE"
                suggestions.append({
                    'sector': sector,
                    'current': current_pct,
                    'target': target_pct,
                    'diff': diff,
                    'action': action,
                    'amount': abs(diff / 100 * total_value)
                })
        
        # Sort by magnitude of imbalance
        suggestions.sort(key=lambda x: abs(x['diff']), reverse=True)
        
        return {
            'total_value': total_value,
            'current_allocation': current_alloc,
            'target_allocation': targets,
            'suggestions': suggestions,
            'num_sectors': len(sector_values)
        }

# --- NEW v12: MULTI-TIMEFRAME ANALYZER ---
class MultiTimeframeAnalyzer:
    """Analyze a symbol across multiple timeframes"""
    
    TIMEFRAMES = [
        ('1d', '15m', 'Intraday'),
        ('5d', '1h', 'Short-term'),
        ('1mo', '1d', 'Medium-term'),
        ('3mo', '1d', 'Swing'),
        ('1y', '1d', 'Long-term'),
    ]
    
    @staticmethod
    def analyze(symbol: str, fetcher) -> Dict:
        """Analyze symbol across multiple timeframes"""
        results = {}
        
        for period, interval, label in MultiTimeframeAnalyzer.TIMEFRAMES:
            try:
                df = fetcher.get_history(symbol, period)
                if df is None or len(df) < 14:
                    results[label] = {'error': 'Insufficient data'}
                    continue
                
                close = df['Close']
                
                # RSI
                rsi = TechnicalAnalysis.rsi(close).iloc[-1]
                
                # MACD
                _, _, hist = TechnicalAnalysis.macd(close)
                macd_signal = "BULLISH" if hist.iloc[-1] > 0 else "BEARISH"
                
                # Trend (price vs SMA)
                sma_20 = close.rolling(min(20, len(close)-1)).mean().iloc[-1]
                trend = "UP" if close.iloc[-1] > sma_20 else "DOWN"
                
                # Momentum
                momentum = ((close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100) if close.iloc[0] > 0 else 0
                
                results[label] = {
                    'rsi': rsi,
                    'rsi_signal': 'OVERSOLD' if rsi < 30 else 'OVERBOUGHT' if rsi > 70 else 'Neutral',
                    'macd': macd_signal,
                    'trend': trend,
                    'momentum': momentum,
                    'price': close.iloc[-1]
                }
            except Exception as e:
                results[label] = {'error': str(e)}
        
        # Calculate overall confluence
        bullish_count = sum(1 for tf in results.values() if isinstance(tf, dict) and 
                          ('BULLISH' in str(tf.get('macd', '')) or 'OVERSOLD' in str(tf.get('rsi_signal', ''))))
        bearish_count = sum(1 for tf in results.values() if isinstance(tf, dict) and 
                          ('BEARISH' in str(tf.get('macd', '')) or 'OVERBOUGHT' in str(tf.get('rsi_signal', ''))))
        
        results['confluence'] = {
            'bullish': bullish_count,
            'bearish': bearish_count,
            'signal': 'BULLISH' if bullish_count > bearish_count else 'BEARISH' if bearish_count > bullish_count else 'MIXED'
        }
        
        return results

# ═══════════════════════════════════════════════════════════════════════════════
# NEW v12.2: LEAPS ANALYZER & MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

class LEAPSAnalyzer:
    """
    Comprehensive LEAPS (Long-Term Equity Anticipation Securities) analyzer.
    LEAPS are options with expiration dates typically 1-3 years out.
    """
    
    # LEAPS are typically defined as options with > 270 days (9 months) to expiration
    LEAPS_MIN_DTE = 270
    
    # Risk thresholds
    THETA_DANGER_PCT = 0.15  # If theta > 0.15% of value per day, warn
    DTE_DANGER = 90  # Warn when LEAPS drops below 90 DTE
    DELTA_DEEP_ITM = 0.80
    DELTA_ATM = 0.50
    DELTA_OTM = 0.30
    
    @staticmethod
    def is_leaps(expiration: str) -> bool:
        """Check if an option qualifies as a LEAP"""
        days = dte(expiration)
        return days >= LEAPSAnalyzer.LEAPS_MIN_DTE
    
    @staticmethod
    def classify_leaps(option: Dict, und_price: float) -> Dict:
        """
        Classify a LEAPS option and return detailed metrics.
        
        Returns:
            Dict with classification, health score, and recommendations
        """
        days = dte(option['expiration'])
        strike = option['strike']
        opt_type = option['type']
        qty = option['qty']
        
        # Moneyness calculation
        if opt_type == 'call':
            moneyness = (und_price - strike) / strike * 100
            itm = und_price > strike
        else:
            moneyness = (strike - und_price) / strike * 100
            itm = und_price < strike
        
        # Classification
        if abs(moneyness) < 5:
            classification = "ATM"
        elif itm and abs(moneyness) >= 20:
            classification = "Deep ITM"
        elif itm:
            classification = "ITM"
        elif not itm and abs(moneyness) >= 20:
            classification = "Deep OTM"
        else:
            classification = "OTM"
        
        # Time category
        if days >= 730:  # 2+ years
            time_category = "Long LEAPS"
        elif days >= 365:  # 1-2 years
            time_category = "LEAPS"
        elif days >= LEAPSAnalyzer.LEAPS_MIN_DTE:
            time_category = "Short LEAPS"
        elif days >= 90:
            time_category = "Medium-term"
        else:
            time_category = "Short-term"
        
        # Health scoring (0-100)
        health_score = 100
        warnings = []
        
        # Penalize for time decay acceleration zone
        if days < 90:
            health_score -= 40
            warnings.append("⚠️ In theta decay acceleration zone (<90 DTE)")
        elif days < 180:
            health_score -= 20
            warnings.append("⏰ Approaching theta acceleration zone")
        elif days < 270:
            health_score -= 10
            warnings.append("📅 No longer qualifies as LEAPS")
        
        # Penalize deep OTM
        if classification == "Deep OTM":
            health_score -= 25
            warnings.append("🎯 Deep OTM - low probability of profit")
        elif classification == "OTM":
            health_score -= 10
        
        # Bonus for deep ITM (stock replacement)
        if classification == "Deep ITM":
            health_score += 10
        
        # Check if roll should be considered
        should_roll = days < 180 or (days < 270 and classification in ["OTM", "Deep OTM"])
        
        return {
            'classification': classification,
            'time_category': time_category,
            'days': days,
            'moneyness': moneyness,
            'itm': itm,
            'health_score': max(0, min(100, health_score)),
            'warnings': warnings,
            'should_roll': should_roll,
            'is_leaps': days >= LEAPSAnalyzer.LEAPS_MIN_DTE
        }
    
    @staticmethod
    def calculate_roll_analysis(option: Dict, und_price: float, iv: float, fetcher) -> Dict:
        """
        Analyze potential rolls for a LEAPS position.
        
        Returns recommendations for rolling out, up, down, or out-and-up/down.
        """
        current_days = dte(option['expiration'])
        strike = option['strike']
        opt_type = option['type']
        
        T = max(current_days / 365.0, 0.001)
        
        # Current position value
        current_price = BlackScholes.price(und_price, strike, T, config.risk_free_rate, iv, opt_type)
        current_greeks = BlackScholes.calculate_all_greeks(und_price, strike, T, config.risk_free_rate, iv, opt_type)
        
        roll_options = []
        
        # Generate roll candidates (out 6 months, 12 months)
        for months_out in [6, 12]:
            target_days = current_days + (months_out * 30)
            target_T = target_days / 365.0
            
            # Same strike (roll out)
            new_price = BlackScholes.price(und_price, strike, target_T, config.risk_free_rate, iv, opt_type)
            new_greeks = BlackScholes.calculate_all_greeks(und_price, strike, target_T, config.risk_free_rate, iv, opt_type)
            
            roll_cost = new_price - current_price
            theta_improvement = new_greeks['theta'] - current_greeks['theta']  # Less negative is better
            
            roll_options.append({
                'type': f"Roll Out {months_out}mo",
                'new_strike': strike,
                'new_dte': target_days,
                'roll_cost': roll_cost,
                'new_delta': new_greeks['delta'],
                'new_theta': new_greeks['theta'],
                'theta_improvement': theta_improvement,
                'new_price': new_price
            })
            
            # Roll out and up/down (adjust strike by ~5%)
            if opt_type == 'call':
                new_strike = round(strike * 1.05, 0)  # Roll up
            else:
                new_strike = round(strike * 0.95, 0)  # Roll down
            
            new_price_adjusted = BlackScholes.price(und_price, new_strike, target_T, config.risk_free_rate, iv, opt_type)
            new_greeks_adjusted = BlackScholes.calculate_all_greeks(und_price, new_strike, target_T, config.risk_free_rate, iv, opt_type)
            
            roll_cost_adjusted = new_price_adjusted - current_price
            
            direction = "Up" if opt_type == 'call' else "Down"
            roll_options.append({
                'type': f"Roll Out+{direction} {months_out}mo",
                'new_strike': new_strike,
                'new_dte': target_days,
                'roll_cost': roll_cost_adjusted,
                'new_delta': new_greeks_adjusted['delta'],
                'new_theta': new_greeks_adjusted['theta'],
                'theta_improvement': new_greeks_adjusted['theta'] - current_greeks['theta'],
                'new_price': new_price_adjusted
            })
        
        # Find best roll (minimize cost while maximizing theta improvement)
        for r in roll_options:
            # Score: higher is better
            # Prioritize theta improvement (less decay) and lower cost
            r['score'] = (r['theta_improvement'] * 365 * 100) - (r['roll_cost'] / current_price * 10)
        
        roll_options.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'current_price': current_price,
            'current_delta': current_greeks['delta'],
            'current_theta': current_greeks['theta'],
            'current_gamma': current_greeks['gamma'],
            'current_vega': current_greeks['vega'],
            'roll_options': roll_options,
            'best_roll': roll_options[0] if roll_options else None
        }
    
    @staticmethod
    def find_leaps_opportunities(symbol: str, fetcher, budget: float = 10000) -> List[Dict]:
        """
        Find LEAPS buying opportunities for a given symbol.
        
        Looks for options with:
        - High delta (0.60-0.80) for stock replacement
        - Reasonable cost relative to stock
        - Good time value
        """
        opportunities = []
        
        try:
            ticker = yf.Ticker(symbol)
            und_price_data = fetcher.get_stock_price(symbol)
            und_price = und_price_data.get('price', 0)
            
            if und_price <= 0:
                return []
            
            # Get available expirations
            expirations = ticker.options
            leaps_expirations = [exp for exp in expirations if dte(exp) >= LEAPSAnalyzer.LEAPS_MIN_DTE]
            
            for exp in leaps_expirations[:3]:  # Check first 3 LEAPS expirations
                try:
                    chain = ticker.option_chain(exp)
                    calls = chain.calls
                    
                    # Filter for reasonable strikes (60-100% of current price for calls)
                    target_strikes = calls[(calls['strike'] >= und_price * 0.60) & 
                                          (calls['strike'] <= und_price * 1.0)]
                    
                    for _, row in target_strikes.iterrows():
                        strike = row['strike']
                        bid = safe_float(row.get('bid', 0))
                        ask = safe_float(row.get('ask', 0))
                        mid_price = (bid + ask) / 2 if bid > 0 and ask > 0 else safe_float(row.get('lastPrice', 0))
                        iv = safe_float(row.get('impliedVolatility', 0.3))
                        
                        if mid_price <= 0:
                            continue
                        
                        days = dte(exp)
                        T = days / 365.0
                        
                        # Calculate Greeks
                        greeks = BlackScholes.calculate_all_greeks(und_price, strike, T, config.risk_free_rate, iv, 'call')
                        delta = greeks['delta']
                        
                        # Filter for target delta range (stock replacement candidates)
                        if delta < 0.55 or delta > 0.90:
                            continue
                        
                        # Cost analysis
                        contract_cost = mid_price * 100
                        max_contracts = int(budget / contract_cost)
                        
                        if max_contracts < 1:
                            continue
                        
                        # Equivalent stock exposure
                        shares_controlled = delta * 100
                        stock_cost_equivalent = shares_controlled * und_price
                        leverage = stock_cost_equivalent / contract_cost
                        
                        # Breakeven
                        breakeven = strike + mid_price
                        breakeven_pct = (breakeven - und_price) / und_price * 100
                        
                        # Intrinsic and extrinsic
                        intrinsic = max(0, und_price - strike)
                        extrinsic = mid_price - intrinsic
                        extrinsic_pct = extrinsic / mid_price * 100 if mid_price > 0 else 0
                        
                        opportunities.append({
                            'symbol': symbol,
                            'expiration': exp,
                            'dte': days,
                            'strike': strike,
                            'price': mid_price,
                            'contract_cost': contract_cost,
                            'delta': delta,
                            'theta': greeks['theta'],
                            'gamma': greeks['gamma'],
                            'vega': greeks['vega'],
                            'iv': iv,
                            'intrinsic': intrinsic,
                            'extrinsic': extrinsic,
                            'extrinsic_pct': extrinsic_pct,
                            'breakeven': breakeven,
                            'breakeven_pct': breakeven_pct,
                            'leverage': leverage,
                            'max_contracts': max_contracts,
                            'shares_controlled': shares_controlled,
                            'und_price': und_price
                        })
                
                except Exception as e:
                    if config.debug:
                        print(f"  Debug: Error processing {exp}: {e}")
                    continue
            
            # Sort by leverage (best stock replacement value)
            opportunities.sort(key=lambda x: x['leverage'], reverse=True)
            
        except Exception as e:
            if config.debug:
                print(f"  Debug: LEAPS chain error: {e}")
        
        return opportunities[:10]  # Return top 10


class PortfolioGreeksAnalyzer:
    """Analyze total portfolio Greeks exposure across all options positions"""
    
    @staticmethod
    def calculate_portfolio_greeks(options: List[Dict], fetcher, stock_prices: Dict) -> Dict:
        """
        Calculate aggregate Greeks for entire options portfolio.
        
        Returns:
            Dict with total delta, gamma, theta, vega, and breakdown by underlying
        """
        totals = {
            'delta': 0,
            'gamma': 0,
            'theta': 0,
            'vega': 0,
            'delta_dollars': 0,
            'theta_daily': 0,
            'vega_dollars': 0
        }
        
        by_underlying = defaultdict(lambda: {
            'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0,
            'contracts': 0, 'value': 0, 'delta_dollars': 0
        })
        
        by_expiry = defaultdict(lambda: {
            'delta': 0, 'theta': 0, 'contracts': 0, 'value': 0
        })
        
        leaps_positions = []
        non_leaps_positions = []
        
        for o in options:
            sym = o['symbol']
            und_price = stock_prices.get(sym, {}).get('price', 0)
            
            if und_price <= 0:
                continue
            
            days = dte(o['expiration'])
            T = max(days / 365.0, 0.001)
            qty = o['qty']
            
            # Get option price and IV
            opt_data = fetcher.get_option_price(sym, o['expiration'], o['strike'], o['type'], und_price)
            iv = opt_data.get('iv', 0.3)
            opt_price = opt_data.get('price', 0)
            
            # Calculate Greeks
            greeks = BlackScholes.calculate_all_greeks(
                und_price, o['strike'], T, config.risk_free_rate, iv, o['type']
            )
            
            # Scale by position size (qty contracts * 100 shares)
            position_delta = greeks['delta'] * qty * 100
            position_gamma = greeks['gamma'] * qty * 100
            position_theta = greeks['theta'] * qty * 100
            position_vega = greeks['vega'] * qty * 100
            
            # Dollar exposures
            delta_dollars = position_delta * und_price
            
            # Aggregate totals
            totals['delta'] += position_delta
            totals['gamma'] += position_gamma
            totals['theta'] += position_theta
            totals['vega'] += position_vega
            totals['delta_dollars'] += delta_dollars
            totals['theta_daily'] += position_theta
            
            # By underlying
            by_underlying[sym]['delta'] += position_delta
            by_underlying[sym]['gamma'] += position_gamma
            by_underlying[sym]['theta'] += position_theta
            by_underlying[sym]['vega'] += position_vega
            by_underlying[sym]['contracts'] += abs(qty)
            by_underlying[sym]['value'] += abs(qty) * opt_price * 100
            by_underlying[sym]['delta_dollars'] += delta_dollars
            by_underlying[sym]['und_price'] = und_price
            
            # By expiry
            exp = o['expiration']
            by_expiry[exp]['delta'] += position_delta
            by_expiry[exp]['theta'] += position_theta
            by_expiry[exp]['contracts'] += abs(qty)
            by_expiry[exp]['value'] += abs(qty) * opt_price * 100
            
            # Track LEAPS vs non-LEAPS
            position_info = {
                'symbol': sym,
                'expiration': o['expiration'],
                'strike': o['strike'],
                'type': o['type'],
                'qty': qty,
                'dte': days,
                'delta': greeks['delta'],
                'theta': greeks['theta'],
                'position_delta': position_delta,
                'position_theta': position_theta,
                'value': abs(qty) * opt_price * 100
            }
            
            if LEAPSAnalyzer.is_leaps(o['expiration']):
                leaps_positions.append(position_info)
            else:
                non_leaps_positions.append(position_info)
        
        return {
            'totals': totals,
            'by_underlying': dict(by_underlying),
            'by_expiry': dict(by_expiry),
            'leaps_positions': leaps_positions,
            'non_leaps_positions': non_leaps_positions,
            'leaps_count': len(leaps_positions),
            'non_leaps_count': len(non_leaps_positions)
        }

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
        
        # Auto-record performance snapshot
        try:
            perf = PerformanceTracker()
            perf.record_snapshot(total_val + cash, total_cost, cash, len(stocks) + len(options))
        except:
            pass  # Silently fail if performance tracking has issues

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
║  STOCK TICKER v12.0.0 — HELP & MANUAL                                        ║
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
║  NEW v12 — RISK MANAGEMENT:                                                  ║
║  stops           View all stop-loss and take-profit levels                   ║
║  stops add SYM --stop PRICE [--tp PRICE] [--trail PCT]                       ║
║  stops rm SYM    Remove stops for symbol                                     ║
║  size SYM        Position sizing calculator (risk-based)                     ║
║  perf            View portfolio performance history & metrics                ║
║  corr            Portfolio correlation matrix & diversification              ║
║  rebalance       Get portfolio rebalancing suggestions                       ║
║  mtf SYMBOL      Multi-timeframe analysis                                    ║
║  export [csv]    Export portfolio report (PDF default, or CSV)               ║
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
    # NEW v12.1: Enhanced welcome experience
    print_welcome()
    if config.first_run:
        print_quick_start()
        config.first_run = False
        config.save()
    else:
        print(f"  {Fore.CYAN}Type 'help' for commands or just enter a ticker symbol{Style.RESET_ALL}\n")
    
    fetcher, pf, wl, alerts = PriceFetcher(), Portfolio(), Watchlist(), PriceAlerts()
    stops = StopLossTracker()
    perf_tracker = PerformanceTracker()
    last_symbol = None
    portfolio_value = 0  # NEW v12.1: Track for prompt display
    
    # Check alerts and stops on startup
    triggered = alerts.check(fetcher)
    if triggered:
        print(Fore.YELLOW + Style.BRIGHT + "\n  🔔 TRIGGERED PRICE ALERTS:" + Style.RESET_ALL)
        for a in triggered:
            print(f"     {a['symbol']} hit ${a['target']:.2f} (now ${a.get('trigger_price', 0):.2f})")
        print()
    
    triggered_stops = stops.check(fetcher)
    if triggered_stops:
        print(Fore.RED + Style.BRIGHT + "\n  🚨 TRIGGERED STOPS:" + Style.RESET_ALL)
        for s in triggered_stops:
            if s['type'] == 'stop_loss':
                print(f"     {s['symbol']} hit STOP LOSS at ${s['level']:.2f} (now ${s['price']:.2f})")
            elif s['type'] == 'take_profit':
                print(Fore.GREEN + f"     {s['symbol']} hit TAKE PROFIT at ${s['level']:.2f} (now ${s['price']:.2f})" + Style.RESET_ALL)
            elif s['type'] == 'trailing_stop':
                print(f"     {s['symbol']} hit TRAILING STOP at ${s['level']:.2f} (high: ${s['high']:.2f}, now: ${s['price']:.2f})")
        print()
    
    while True:
        try:
            # NEW v12.1: Rich prompt showing portfolio value and context
            prompt = format_prompt(portfolio_value, last_symbol)
            raw = input(prompt).strip()
            if not raw: continue
            
            parts = shlex.split(raw)
            cmd = parts[0].lower()
            args = parts[1:]
            
            # NEW v12.1: Apply command aliases
            if cmd in COMMAND_ALIASES:
                cmd = COMMAND_ALIASES[cmd]
            
            # NEW v12.1: Smart command suggestions for typos
            if cmd not in ALL_COMMANDS and cmd not in COMMAND_ALIASES and len(cmd) > 2:
                suggestion = suggest_command(cmd)
                if suggestion:
                    print(f"  {Fore.YELLOW}Did you mean '{suggestion}'?{Style.RESET_ALL}")
                    continue
            
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
            elif cmd == 'import': 
                if args:
                    pf.import_csv(args[0])
                    print_tip('import')
                else: 
                    print_error("Usage: import <file>", "Example: import ~/Downloads/portfolio.csv")
            elif cmd in ('pf', 'portfolio'): 
                pf.display(fetcher)
                # Update portfolio value for prompt
                stocks = pf.data.get('stocks', {})
                if stocks:
                    total = sum(fetcher.get_stock_price(s).get('price', 0) * p['qty'] for s, p in stocks.items())
                    portfolio_value = total + pf.data.get('cash', 0)
                print_tip('pf')
            elif cmd == 'risk': pf.analyze_risk(fetcher)
            elif cmd == 'cal': pf.calendar(fetcher)
            elif cmd == 'watch': 
                if not args: wl.show(fetcher)
                elif args[0]=='add': wl.add(args[1])
                elif args[0] in ('rm','del'): wl.remove(args[1])
            elif cmd == 'clear': pf.clear()
            elif cmd == 'debug': config.debug = not config.debug; print(f"  Debug: {config.debug}"); config.save()
            elif cmd == 'tips': 
                config.show_tips = not config.show_tips
                print(f"  Contextual tips: {'ON' if config.show_tips else 'OFF'}")
                config.save()
            elif cmd == 'refresh': fetcher.clear_cache(); success("Cache cleared")
            elif cmd in ('help', 'h', '?'): 
                # NEW v12.1: Searchable help
                if args:
                    print_searchable_help(args[0])
                else:
                    print_searchable_help()

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
                print_tip('market')

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
            
            # --- NEW v12: STOP-LOSS TRACKER ---
            elif cmd == 'stops':
                if not args:
                    stops.show(fetcher)
                elif args[0] == 'add' and len(args) >= 2:
                    symbol = args[1].upper()
                    stop_loss = None
                    take_profit = None
                    trailing = None
                    
                    # Parse optional arguments
                    i = 2
                    while i < len(args):
                        if args[i] in ('--stop', '-s') and i + 1 < len(args):
                            stop_loss = float(args[i + 1])
                            i += 2
                        elif args[i] in ('--tp', '-t') and i + 1 < len(args):
                            take_profit = float(args[i + 1])
                            i += 2
                        elif args[i] in ('--trail', '-tr') and i + 1 < len(args):
                            trailing = float(args[i + 1])
                            i += 2
                        else:
                            # Assume it's a stop loss price
                            try:
                                stop_loss = float(args[i])
                            except:
                                pass
                            i += 1
                    
                    if stop_loss or take_profit or trailing:
                        stops.add(symbol, stop_loss, take_profit, trailing)
                    else:
                        err("Usage: stops add SYMBOL [--stop PRICE] [--tp PRICE] [--trail PCT]")
                elif args[0] in ('rm', 'remove', 'del') and len(args) >= 2:
                    stops.remove(args[1].upper())
                elif args[0] == 'check':
                    triggered = stops.check(fetcher)
                    if triggered:
                        for s in triggered:
                            print(f"  🚨 {s['symbol']} triggered {s['type']} at ${s['level']:.2f}")
                    else:
                        print("  No stops triggered")
                else:
                    err("Usage: stops | stops add SYM --stop PRICE | stops rm SYM")
            
            # --- NEW v12: POSITION SIZING ---
            elif cmd == 'size':
                if not args:
                    err("Usage: size SYMBOL [--risk PCT] [--stop PRICE] [--account SIZE]")
                    continue
                
                symbol = args[0].upper()
                risk_pct = 2.0  # Default 2% risk
                stop_price = None
                account_size = 100000  # Default $100k
                
                # Parse arguments
                i = 1
                while i < len(args):
                    if args[i] in ('--risk', '-r') and i + 1 < len(args):
                        risk_pct = float(args[i + 1])
                        i += 2
                    elif args[i] in ('--stop', '-s') and i + 1 < len(args):
                        stop_price = float(args[i + 1])
                        i += 2
                    elif args[i] in ('--account', '-a') and i + 1 < len(args):
                        account_size = float(args[i + 1])
                        i += 2
                    else:
                        i += 1
                
                print(Fore.CYAN + f"\n{'═'*70}\n 📐 POSITION SIZING — {symbol}\n{'═'*70}" + Style.RESET_ALL)
                
                d = fetcher.get_stock_price(symbol)
                price = d.get('price', 0)
                
                if price <= 0:
                    err(f"Could not get price for {symbol}")
                else:
                    df = fetcher.get_history(symbol, "3mo")
                    
                    print(f"\n  {Style.BRIGHT}INPUTS{Style.RESET_ALL}")
                    print(f"    Current Price:    {fmt_money(price)}")
                    print(f"    Account Size:     {fmt_money(account_size)}")
                    print(f"    Risk Tolerance:   {risk_pct}%")
                    
                    # Method 1: Fixed Risk (if stop provided)
                    if stop_price:
                        result = PositionSizer.fixed_risk(account_size, risk_pct, price, stop_price)
                        print(f"\n  {Style.BRIGHT}FIXED RISK METHOD{Style.RESET_ALL}")
                        print(f"    Stop Loss:        {fmt_money(stop_price)}")
                        print(f"    Risk per Share:   {fmt_money(result['risk_per_share'])}")
                        print(f"    Position Size:    {Fore.GREEN}{result['shares']} shares{Style.RESET_ALL}")
                        print(f"    Position Value:   {fmt_money(result['position_value'])}")
                        print(f"    Account %:        {result['account_pct']:.1f}%")
                    
                    # Method 2: Volatility-adjusted (ATR-based)
                    if df is not None and len(df) > 14:
                        atr = TechnicalAnalysis.atr(df).iloc[-1]
                        result = PositionSizer.volatility_adjusted(account_size, risk_pct, price, atr)
                        
                        print(f"\n  {Style.BRIGHT}VOLATILITY-ADJUSTED (ATR){Style.RESET_ALL}")
                        print(f"    ATR (14-day):     {fmt_money(result['atr'])}")
                        print(f"    Stop Distance:    {fmt_money(result['stop_distance'])} (2x ATR)")
                        print(f"    Suggested Stop:   {fmt_money(result['suggested_stop'])}")
                        print(f"    Position Size:    {Fore.GREEN}{result['shares']} shares{Style.RESET_ALL}")
                        print(f"    Position Value:   {fmt_money(result['position_value'])}")
                        print(f"    Account %:        {result['account_pct']:.1f}%")
                    
                    # Kelly Criterion hint
                    print(f"\n  {Style.BRIGHT}KELLY CRITERION{Style.RESET_ALL}")
                    print(f"    {Fore.YELLOW}Tip: Use 'backtest {symbol}' to get win rate for Kelly calculation{Style.RESET_ALL}")
                    
                print(f"{'═'*70}\n")
                last_symbol = symbol
            
            # --- NEW v12: PERFORMANCE TRACKING ---
            elif cmd == 'perf':
                perf_tracker.show()
            
            # --- NEW v12: CORRELATION ANALYSIS ---
            elif cmd == 'corr':
                stocks = pf.data.get('stocks', {})
                if not stocks:
                    print_error("No stocks in portfolio", "Use 'import <file.csv>' to load your portfolio")
                    continue
                
                symbols = list(stocks.keys())
                if len(symbols) < 2:
                    print_error("Need at least 2 stocks for correlation analysis", "Add more positions to analyze diversification")
                    continue
                
                print(Fore.CYAN + f"\n{'═'*80}\n 🔗 CORRELATION ANALYSIS\n{'═'*80}" + Style.RESET_ALL)
                
                prog = ProgressBar(len(symbols), "  Analyzing")
                corr_matrix = CorrelationAnalyzer.calculate_correlation_matrix(symbols, fetcher)
                prog.done()
                
                if corr_matrix is None:
                    err("Could not calculate correlation matrix")
                else:
                    # Diversification score
                    div_score = CorrelationAnalyzer.portfolio_diversification_score(corr_matrix)
                    div_color = Fore.GREEN if div_score >= 60 else Fore.YELLOW if div_score >= 40 else Fore.RED
                    
                    print(f"\n  {Style.BRIGHT}DIVERSIFICATION SCORE:{Style.RESET_ALL} {div_color}{div_score:.1f}/100{Style.RESET_ALL}")
                    if div_score >= 60:
                        print(f"    {Fore.GREEN}Well diversified portfolio{Style.RESET_ALL}")
                    elif div_score >= 40:
                        print(f"    {Fore.YELLOW}Moderately diversified - consider adding uncorrelated assets{Style.RESET_ALL}")
                    else:
                        print(f"    {Fore.RED}Highly correlated - significant concentration risk{Style.RESET_ALL}")
                    
                    # Highly correlated pairs
                    pairs = CorrelationAnalyzer.find_highly_correlated(corr_matrix, threshold=0.7)
                    if pairs:
                        print(f"\n  {Style.BRIGHT}HIGHLY CORRELATED PAIRS (>70%):{Style.RESET_ALL}")
                        for sym1, sym2, corr in pairs[:5]:
                            color = Fore.RED if corr > 0.85 else Fore.YELLOW
                            print(f"    {sym1} ↔ {sym2}: {color}{corr:.2f}{Style.RESET_ALL}")
                    
                    # Show matrix for small portfolios
                    if len(symbols) <= 8:
                        print(f"\n  {Style.BRIGHT}CORRELATION MATRIX:{Style.RESET_ALL}")
                        # Format matrix for display
                        header = [""] + list(corr_matrix.columns)
                        rows = []
                        for idx in corr_matrix.index:
                            row = [idx]
                            for col in corr_matrix.columns:
                                val = corr_matrix.loc[idx, col]
                                if idx == col:
                                    row.append("1.00")
                                else:
                                    color = Fore.RED if abs(val) > 0.7 else Fore.YELLOW if abs(val) > 0.4 else Fore.RESET
                                    row.append(f"{color}{val:.2f}{Style.RESET_ALL}")
                            rows.append(row)
                        print(tabulate(rows, headers=header, tablefmt="simple"))
                
                print(f"{'═'*80}\n")
            
            # --- NEW v12: REBALANCING ---
            elif cmd == 'rebalance':
                stocks = pf.data.get('stocks', {})
                if not stocks:
                    print_error("No stocks in portfolio", "Use 'import <file.csv>' to load your portfolio")
                    continue
                
                print_header("PORTFOLIO REBALANCING", icon="🔄")
                
                result = PortfolioRebalancer.analyze(stocks, fetcher)
                
                if 'error' in result:
                    err(result['error'])
                else:
                    print(f"\n  {Style.BRIGHT}PORTFOLIO VALUE:{Style.RESET_ALL} {fmt_money(result['total_value'])}")
                    print(f"  {Style.BRIGHT}SECTORS HELD:{Style.RESET_ALL} {result['num_sectors']}")
                    
                    # Current allocation
                    print(f"\n  {Style.BRIGHT}CURRENT ALLOCATION:{Style.RESET_ALL}")
                    for sector, pct in sorted(result['current_allocation'].items(), key=lambda x: x[1], reverse=True):
                        target = result['target_allocation'].get(sector, 0)
                        diff = pct - target
                        diff_color = Fore.RED if abs(diff) > 5 else Fore.YELLOW if abs(diff) > 2 else Fore.GREEN
                        bar_len = int(pct / 2)
                        bar = "█" * bar_len
                        print(f"    {sector:<20} {bar:<15} {pct:>5.1f}% (target: {target}%) {diff_color}{diff:+.1f}%{Style.RESET_ALL}")
                    
                    # Suggestions
                    if result['suggestions']:
                        print(f"\n  {Style.BRIGHT}REBALANCING SUGGESTIONS:{Style.RESET_ALL}")
                        for sug in result['suggestions'][:5]:
                            action_color = Fore.RED if sug['action'] == 'REDUCE' else Fore.GREEN
                            print(f"    {action_color}{sug['action']}{Style.RESET_ALL} {sug['sector']}: "
                                  f"{sug['current']:.1f}% → {sug['target']:.1f}% (${abs(sug['amount']):,.0f})")
                    else:
                        print(f"\n  {Fore.GREEN}Portfolio is well balanced!{Style.RESET_ALL}")
                
                print(f"{'═'*80}\n")
            
            # --- NEW v12: MULTI-TIMEFRAME ANALYSIS ---
            elif cmd == 'mtf' and args:
                symbol = args[0].upper()
                print(Fore.CYAN + f"\n{'═'*80}\n ⏱️ MULTI-TIMEFRAME ANALYSIS — {symbol}\n{'═'*80}" + Style.RESET_ALL)
                
                results = MultiTimeframeAnalyzer.analyze(symbol, fetcher)
                
                d = fetcher.get_stock_price(symbol)
                print(f"\n  {Style.BRIGHT}CURRENT PRICE:{Style.RESET_ALL} {fmt_money(d.get('price', 0))} ({color_pct(d.get('pct', 0))})")
                
                print(f"\n  {Style.BRIGHT}TIMEFRAME BREAKDOWN:{Style.RESET_ALL}")
                rows = []
                for label in ['Intraday', 'Short-term', 'Medium-term', 'Swing', 'Long-term']:
                    data = results.get(label, {})
                    if 'error' in data:
                        rows.append([label, '-', '-', '-', '-'])
                    else:
                        rsi = data.get('rsi', 0)
                        rsi_str = f"{rsi:.1f}" if rsi else '-'
                        rsi_signal = data.get('rsi_signal', '-')
                        macd = data.get('macd', '-')
                        trend = data.get('trend', '-')
                        momentum = data.get('momentum', 0)
                        
                        rows.append([
                            label,
                            rsi_str,
                            color_signal(rsi_signal),
                            color_signal(macd),
                            f"{Fore.GREEN if momentum > 0 else Fore.RED}{momentum:+.1f}%{Style.RESET_ALL}"
                        ])
                
                print(tabulate(rows, headers=["Timeframe", "RSI", "RSI Signal", "MACD", "Momentum"]))
                
                # Confluence
                conf = results.get('confluence', {})
                print(f"\n  {Style.BRIGHT}CONFLUENCE:{Style.RESET_ALL}")
                print(f"    Bullish Signals: {Fore.GREEN}{conf.get('bullish', 0)}{Style.RESET_ALL}")
                print(f"    Bearish Signals: {Fore.RED}{conf.get('bearish', 0)}{Style.RESET_ALL}")
                overall = conf.get('signal', 'MIXED')
                overall_color = Fore.GREEN if overall == 'BULLISH' else Fore.RED if overall == 'BEARISH' else Fore.YELLOW
                print(f"    Overall: {overall_color}{Style.BRIGHT}{overall}{Style.RESET_ALL}")
                
                print(f"{'═'*80}\n")
                last_symbol = symbol
            
            # --- NEW v12: EXPORT REPORT ---
            elif cmd == 'export':
                stocks = pf.data.get('stocks', {})
                options = pf.data.get('options', [])
                
                if not stocks and not options:
                    warn("No positions to export. Import data first.")
                    continue
                
                fmt = 'csv' if args and args[0].lower() == 'csv' else 'txt'
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = Path(config.data_dir) / f"portfolio_report_{timestamp}.{fmt}"
                
                try:
                    with open(filename, 'w') as f:
                        if fmt == 'csv':
                            # CSV export
                            f.write("Type,Symbol,Quantity,Cost,Value,P&L,P&L %\n")
                            
                            for sym, pos in stocks.items():
                                price_data = fetcher.get_stock_price(sym)
                                price = price_data.get('price', pos.get('broker_price', 0))
                                val = pos['qty'] * price
                                pnl = val - pos['cost']
                                pnl_pct = (pnl / pos['cost'] * 100) if pos['cost'] > 0 else 0
                                f.write(f"Stock,{sym},{pos['qty']},{pos['cost']:.2f},{val:.2f},{pnl:.2f},{pnl_pct:.2f}\n")
                            
                            for o in options:
                                f.write(f"Option,{o['symbol']} {o['expiration']} ${o['strike']} {o['type']},{o['qty']},{o['cost']:.2f},-,-\n")
                        else:
                            # Text report
                            f.write("=" * 80 + "\n")
                            f.write(f"PORTFOLIO REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                            f.write("=" * 80 + "\n\n")
                            
                            f.write("STOCKS\n")
                            f.write("-" * 80 + "\n")
                            total_val = 0
                            total_pnl = 0
                            for sym, pos in sorted(stocks.items()):
                                price_data = fetcher.get_stock_price(sym)
                                price = price_data.get('price', pos.get('broker_price', 0))
                                val = pos['qty'] * price
                                pnl = val - pos['cost']
                                total_val += val
                                total_pnl += pnl
                                f.write(f"{sym:<8} {pos['qty']:>10.2f} shares @ ${pos.get('avg', 0):>8.2f} = ${val:>12,.2f}  P&L: ${pnl:>10,.2f}\n")
                            
                            f.write("\n" + "=" * 80 + "\n")
                            f.write(f"TOTAL VALUE: ${total_val:,.2f}\n")
                            f.write(f"TOTAL P&L:   ${total_pnl:,.2f}\n")
                            f.write("=" * 80 + "\n")
                    
                    success(f"Report exported to: {filename}")
                except Exception as e:
                    err(f"Export failed: {e}")
            
            # ═══════════════════════════════════════════════════════════════════════════════
            # NEW v12.2: LEAPS COMMANDS
            # ═══════════════════════════════════════════════════════════════════════════════
            
            # --- LEAPS DASHBOARD ---
            elif cmd == 'leaps':
                options = pf.data.get('options', [])
                if not options:
                    print_error("No options in portfolio", "Use 'import <file.csv>' to load your portfolio")
                    continue
                
                print_header("LEAPS PORTFOLIO DASHBOARD", icon="📊")
                
                # Fetch underlying prices first
                all_symbols = set(o['symbol'] for o in options)
                stock_prices = {}
                prog = ProgressBar(len(all_symbols), "  Fetching prices")
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = {executor.submit(fetcher.get_stock_price, sym): sym for sym in all_symbols}
                    for i, f in enumerate(as_completed(futures)):
                        sym = futures[f]
                        stock_prices[sym] = f.result()
                        prog.update(i, sym)
                prog.done()
                
                # Separate LEAPS from other options
                leaps_options = []
                other_options = []
                
                for o in options:
                    days = dte(o['expiration'])
                    if days >= LEAPSAnalyzer.LEAPS_MIN_DTE:
                        leaps_options.append(o)
                    else:
                        other_options.append(o)
                
                if not leaps_options:
                    warn(f"No LEAPS found (options with >{LEAPSAnalyzer.LEAPS_MIN_DTE} DTE)")
                    if other_options:
                        print(f"\n  You have {len(other_options)} non-LEAPS options. Consider rolling them out.")
                    continue
                
                # Process each LEAPS
                leaps_data = []
                total_leaps_value = 0
                total_leaps_cost = 0
                total_theta = 0
                total_delta = 0
                
                prog = ProgressBar(len(leaps_options), "  Analyzing LEAPS")
                for i, o in enumerate(leaps_options):
                    sym = o['symbol']
                    und_price = stock_prices.get(sym, {}).get('price', 0)
                    
                    # Get option price
                    opt_data = fetcher.get_option_price(sym, o['expiration'], o['strike'], o['type'], und_price)
                    opt_price = opt_data.get('price', 0) or o.get('broker_price', 0)
                    iv = opt_data.get('iv', 0.3)
                    
                    # Classify LEAPS
                    classification = LEAPSAnalyzer.classify_leaps(o, und_price)
                    
                    # Calculate Greeks
                    days = dte(o['expiration'])
                    T = max(days / 365.0, 0.001)
                    greeks = BlackScholes.calculate_all_greeks(und_price, o['strike'], T, config.risk_free_rate, iv, o['type'])
                    
                    # Values
                    qty = o['qty']
                    cost = o['cost']
                    val = abs(qty) * opt_price * 100
                    pnl = val - cost if qty > 0 else cost - val
                    pnl_pct = (pnl / cost * 100) if cost > 0 else 0
                    
                    # Position Greeks
                    pos_delta = greeks['delta'] * qty * 100
                    pos_theta = greeks['theta'] * qty * 100
                    
                    total_leaps_value += val if qty > 0 else -val
                    total_leaps_cost += cost
                    total_theta += pos_theta
                    total_delta += pos_delta
                    
                    # Intrinsic/Extrinsic
                    intrinsic = max(0, und_price - o['strike']) if o['type'] == 'call' else max(0, o['strike'] - und_price)
                    extrinsic = max(0, opt_price - intrinsic)
                    extr_pct = (extrinsic / opt_price * 100) if opt_price > 0 else 0
                    
                    # Breakeven
                    if o['type'] == 'call':
                        breakeven = o['strike'] + (cost / abs(qty) / 100) if qty > 0 else o['strike'] + opt_price
                    else:
                        breakeven = o['strike'] - (cost / abs(qty) / 100) if qty > 0 else o['strike'] - opt_price
                    
                    leaps_data.append({
                        'option': o,
                        'und_price': und_price,
                        'opt_price': opt_price,
                        'classification': classification,
                        'greeks': greeks,
                        'pos_delta': pos_delta,
                        'pos_theta': pos_theta,
                        'iv': iv,
                        'val': val,
                        'cost': cost,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'intrinsic': intrinsic,
                        'extrinsic': extrinsic,
                        'extr_pct': extr_pct,
                        'breakeven': breakeven
                    })
                    prog.update(i, sym)
                prog.done()
                
                # Sort by DTE
                leaps_data.sort(key=lambda x: x['classification']['days'])
                
                # Summary
                print(f"\n  {Style.BRIGHT}LEAPS SUMMARY{Style.RESET_ALL}")
                print(f"    Total LEAPS:       {len(leaps_options)} positions")
                print(f"    Total Value:       {fmt_money(total_leaps_value)}")
                print(f"    Total Cost Basis:  {fmt_money(total_leaps_cost)}")
                total_pnl = total_leaps_value - total_leaps_cost
                print(f"    Total P&L:         {color_pnl(total_pnl, (total_pnl/total_leaps_cost*100) if total_leaps_cost else 0)}")
                print(f"    Net Delta:         {total_delta:+.1f} shares")
                print(f"    Daily Theta:       {Fore.RED if total_theta < 0 else Fore.GREEN}${total_theta:.2f}/day{Style.RESET_ALL}")
                
                # Table of LEAPS
                print(f"\n  {Style.BRIGHT}LEAPS POSITIONS{Style.RESET_ALL}\n" + "─"*100)
                rows = []
                for ld in leaps_data:
                    o = ld['option']
                    c = ld['classification']
                    g = ld['greeks']
                    
                    # Format description
                    desc = f"{o['symbol']} {o['expiration'][5:]} ${o['strike']:.0f}{o['type'][0].upper()}"
                    if o['qty'] < 0:
                        desc = f"-{desc}"
                    
                    # Health color
                    health = c['health_score']
                    if health >= 80:
                        health_str = Fore.GREEN + f"{health}" + Style.RESET_ALL
                    elif health >= 60:
                        health_str = Fore.YELLOW + f"{health}" + Style.RESET_ALL
                    else:
                        health_str = Fore.RED + f"{health}" + Style.RESET_ALL
                    
                    # DTE color
                    days = c['days']
                    if days >= 365:
                        dte_str = Fore.GREEN + f"{days}d" + Style.RESET_ALL
                    elif days >= 180:
                        dte_str = Fore.YELLOW + f"{days}d" + Style.RESET_ALL
                    else:
                        dte_str = Fore.RED + f"{days}d" + Style.RESET_ALL
                    
                    rows.append([
                        desc,
                        c['classification'],
                        dte_str,
                        f"Δ{g['delta']:.2f}",
                        f"θ${ld['pos_theta']:.2f}",
                        f"{ld['iv']*100:.0f}%",
                        f"I:{ld['intrinsic']:.1f}/E:{ld['extr_pct']:.0f}%",
                        fmt_money(ld['val']),
                        color_pnl(ld['pnl'], ld['pnl_pct']),
                        health_str
                    ])
                
                print(tabulate(rows, headers=["Position", "Type", "DTE", "Delta", "Theta/d", "IV", "Int/Ext", "Value", "P&L", "Health"]))
                
                # Warnings
                warnings = []
                for ld in leaps_data:
                    for w in ld['classification']['warnings']:
                        warnings.append(f"{ld['option']['symbol']}: {w}")
                
                if warnings:
                    print(f"\n  {Style.BRIGHT}⚠️  ALERTS{Style.RESET_ALL}")
                    for w in warnings[:5]:
                        print(f"    {w}")
                
                # Roll recommendations
                roll_candidates = [ld for ld in leaps_data if ld['classification']['should_roll']]
                if roll_candidates:
                    print(f"\n  {Style.BRIGHT}🔄 ROLL RECOMMENDATIONS{Style.RESET_ALL}")
                    for ld in roll_candidates[:3]:
                        o = ld['option']
                        print(f"    {o['symbol']} {o['expiration']} ${o['strike']}{o['type'][0].upper()} — "
                              f"{ld['classification']['days']} DTE, consider rolling out")
                    print(f"\n  {Fore.CYAN}Use 'leapsroll {roll_candidates[0]['option']['symbol']}' for detailed roll analysis{Style.RESET_ALL}")
                
                # Non-LEAPS summary
                if other_options:
                    print(f"\n  {Style.BRIGHT}NON-LEAPS OPTIONS{Style.RESET_ALL}")
                    print(f"    You have {len(other_options)} options under {LEAPSAnalyzer.LEAPS_MIN_DTE} DTE")
                    print(f"    Consider converting to LEAPS for reduced time decay")
                
                print_footer(100)
            
            # --- LEAPS ROLL OPTIMIZER ---
            elif cmd == 'leapsroll':
                if not args:
                    err("Usage: leapsroll SYMBOL [--current EXP] [--budget AMOUNT]")
                    continue
                
                symbol = args[0].upper()
                options = pf.data.get('options', [])
                
                # Find LEAPS for this symbol
                symbol_leaps = [o for o in options if o['symbol'] == symbol and LEAPSAnalyzer.is_leaps(o['expiration'])]
                
                if not symbol_leaps:
                    # Check for any options on this symbol
                    symbol_opts = [o for o in options if o['symbol'] == symbol]
                    if symbol_opts:
                        warn(f"No LEAPS found for {symbol}, but you have {len(symbol_opts)} options. Consider rolling to LEAPS.")
                    else:
                        warn(f"No options found for {symbol} in portfolio")
                    continue
                
                print_header(f"LEAPS ROLL OPTIMIZER — {symbol}", icon="🔄")
                
                # Get underlying price
                d = fetcher.get_stock_price(symbol)
                und_price = d.get('price', 0)
                
                if und_price <= 0:
                    err(f"Could not get price for {symbol}")
                    continue
                
                print(f"\n  {Style.BRIGHT}CURRENT PRICE:{Style.RESET_ALL} {fmt_money(und_price)} ({color_pct(d.get('pct', 0))})")
                
                # Analyze each LEAPS position
                for o in symbol_leaps:
                    opt_data = fetcher.get_option_price(symbol, o['expiration'], o['strike'], o['type'], und_price)
                    iv = opt_data.get('iv', 0.3)
                    
                    print(f"\n  {Style.BRIGHT}ANALYZING:{Style.RESET_ALL} {o['expiration']} ${o['strike']} {o['type'].upper()}")
                    print(f"    Quantity:    {o['qty']} contracts")
                    print(f"    Cost Basis:  {fmt_money(o['cost'])}")
                    print(f"    Days Left:   {dte(o['expiration'])} DTE")
                    
                    # Get roll analysis
                    analysis = LEAPSAnalyzer.calculate_roll_analysis(o, und_price, iv, fetcher)
                    
                    print(f"\n    {Style.BRIGHT}CURRENT POSITION GREEKS:{Style.RESET_ALL}")
                    print(f"      Delta: {analysis['current_delta']:.3f}")
                    print(f"      Theta: ${analysis['current_theta']:.4f}/day")
                    print(f"      Gamma: {analysis['current_gamma']:.4f}")
                    print(f"      Vega:  ${analysis['current_vega']:.2f}")
                    print(f"      Price: ${analysis['current_price']:.2f}")
                    
                    print(f"\n    {Style.BRIGHT}ROLL OPTIONS:{Style.RESET_ALL}")
                    print("    " + "─"*80)
                    
                    roll_rows = []
                    for r in analysis['roll_options']:
                        cost_color = Fore.RED if r['roll_cost'] > 0 else Fore.GREEN
                        theta_color = Fore.GREEN if r['theta_improvement'] > 0 else Fore.RED
                        
                        roll_rows.append([
                            r['type'],
                            f"${r['new_strike']:.0f}",
                            f"{r['new_dte']}d",
                            f"Δ{r['new_delta']:.2f}",
                            f"θ${r['new_theta']:.4f}",
                            f"{cost_color}${r['roll_cost']:.2f}{Style.RESET_ALL}",
                            f"{theta_color}{r['theta_improvement']*365:.2f}/yr{Style.RESET_ALL}"
                        ])
                    
                    print("    " + tabulate(roll_rows, headers=["Roll Type", "Strike", "New DTE", "Delta", "Theta/d", "Net Cost", "θ Improv"]).replace("\n", "\n    "))
                    
                    # Best recommendation
                    best = analysis['best_roll']
                    if best:
                        print(f"\n    {Style.BRIGHT}💡 RECOMMENDATION:{Style.RESET_ALL}")
                        print(f"      {Fore.GREEN}{best['type']}{Style.RESET_ALL} to ${best['new_strike']} strike")
                        print(f"      Net debit/credit: ${best['roll_cost']:.2f} per contract")
                        print(f"      Total for {abs(o['qty'])} contracts: ${best['roll_cost'] * abs(o['qty']) * 100:.2f}")
                
                print_footer(90)
            
            # --- LEAPS CHAIN FINDER ---
            elif cmd == 'leapschain':
                if not args:
                    err("Usage: leapschain SYMBOL [--budget AMOUNT]")
                    continue
                
                symbol = args[0].upper()
                budget = 10000  # Default
                
                # Parse budget
                for i, arg in enumerate(args):
                    if arg in ('--budget', '-b') and i + 1 < len(args):
                        budget = float(args[i + 1])
                
                print_header(f"LEAPS OPPORTUNITIES — {symbol}", icon="🔍")
                
                d = fetcher.get_stock_price(symbol)
                und_price = d.get('price', 0)
                
                if und_price <= 0:
                    err(f"Could not get price for {symbol}")
                    continue
                
                meta = fetcher.get_meta(symbol)
                print(f"\n  {Style.BRIGHT}{meta.get('name', symbol)}{Style.RESET_ALL}")
                print(f"  Current Price: {fmt_money(und_price)} ({color_pct(d.get('pct', 0))})")
                print(f"  Budget: {fmt_money(budget)}")
                
                print(f"\n  {Fore.CYAN}Scanning LEAPS chains...{Style.RESET_ALL}")
                opportunities = LEAPSAnalyzer.find_leaps_opportunities(symbol, fetcher, budget)
                
                if not opportunities:
                    warn(f"No suitable LEAPS found for {symbol}")
                    continue
                
                # Group by expiration
                by_exp = defaultdict(list)
                for opp in opportunities:
                    by_exp[opp['expiration']].append(opp)
                
                for exp in sorted(by_exp.keys()):
                    opps = by_exp[exp]
                    print(f"\n  {Style.BRIGHT}EXPIRATION: {exp} ({opps[0]['dte']} DTE){Style.RESET_ALL}")
                    print("  " + "─"*85)
                    
                    rows = []
                    for opp in opps[:5]:  # Top 5 per expiration
                        rows.append([
                            f"${opp['strike']:.0f}C",
                            f"${opp['price']:.2f}",
                            f"${opp['contract_cost']:.0f}",
                            f"Δ{opp['delta']:.2f}",
                            f"{opp['leverage']:.1f}x",
                            f"{opp['iv']*100:.0f}%",
                            f"${opp['breakeven']:.2f} ({opp['breakeven_pct']:+.1f}%)",
                            f"{opp['max_contracts']}"
                        ])
                    
                    print("  " + tabulate(rows, headers=["Strike", "Price", "Cost", "Delta", "Leverage", "IV", "Breakeven", "Max Qty"]).replace("\n", "\n  "))
                
                # Best overall recommendation
                if opportunities:
                    best = opportunities[0]
                    print(f"\n  {Style.BRIGHT}💡 TOP PICK (Best Leverage){Style.RESET_ALL}")
                    print(f"    {symbol} {best['expiration']} ${best['strike']:.0f} Call")
                    print(f"    Price: ${best['price']:.2f} (${best['contract_cost']:.0f}/contract)")
                    print(f"    Delta: {best['delta']:.2f} = Controls {best['shares_controlled']:.0f} equivalent shares")
                    print(f"    Leverage: {best['leverage']:.1f}x vs stock")
                    print(f"    Breakeven: ${best['breakeven']:.2f} ({best['breakeven_pct']:+.1f}% from current)")
                    print(f"    Max purchase: {best['max_contracts']} contracts with ${budget:,.0f}")
                    print(f"    Extrinsic value: ${best['extrinsic']:.2f} ({best['extrinsic_pct']:.0f}% of premium)")
                
                print_footer(90)
            
            # --- PORTFOLIO GREEKS ---
            elif cmd == 'greeks':
                options = pf.data.get('options', [])
                stocks = pf.data.get('stocks', {})
                
                if not options:
                    print_error("No options in portfolio", "Use 'import <file.csv>' to load your portfolio")
                    continue
                
                print_header("PORTFOLIO GREEKS EXPOSURE", icon="📐")
                
                # Fetch all prices
                all_symbols = set(o['symbol'] for o in options) | set(stocks.keys())
                stock_prices = {}
                prog = ProgressBar(len(all_symbols), "  Fetching prices")
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = {executor.submit(fetcher.get_stock_price, sym): sym for sym in all_symbols}
                    for i, f in enumerate(as_completed(futures)):
                        sym = futures[f]
                        stock_prices[sym] = f.result()
                        prog.update(i, sym)
                prog.done()
                
                # Calculate portfolio Greeks
                print(f"  {Fore.CYAN}Calculating Greeks...{Style.RESET_ALL}")
                greeks_analysis = PortfolioGreeksAnalyzer.calculate_portfolio_greeks(options, fetcher, stock_prices)
                
                totals = greeks_analysis['totals']
                
                # Overall summary
                print(f"\n  {Style.BRIGHT}AGGREGATE GREEKS{Style.RESET_ALL}")
                print(f"    Net Delta:     {totals['delta']:+.1f} shares")
                print(f"    Delta $:       {fmt_money(totals['delta_dollars'])} exposure")
                print(f"    Net Gamma:     {totals['gamma']:+.4f}")
                print(f"    Net Theta:     {Fore.RED if totals['theta'] < 0 else Fore.GREEN}${totals['theta']:.2f}/day{Style.RESET_ALL}")
                print(f"    Net Vega:      ${totals['vega']:.2f}")
                
                # Add stock delta
                stock_delta = sum(pos['qty'] for pos in stocks.values())
                stock_delta_dollars = sum(pos['qty'] * stock_prices.get(sym, {}).get('price', 0) for sym, pos in stocks.items())
                
                print(f"\n  {Style.BRIGHT}INCLUDING STOCKS{Style.RESET_ALL}")
                print(f"    Stock Delta:   {stock_delta:+.1f} shares ({fmt_money(stock_delta_dollars)})")
                print(f"    Total Delta:   {totals['delta'] + stock_delta:+.1f} shares")
                print(f"    Total $ Exp:   {fmt_money(totals['delta_dollars'] + stock_delta_dollars)}")
                
                # By underlying
                print(f"\n  {Style.BRIGHT}BY UNDERLYING{Style.RESET_ALL}")
                by_und = greeks_analysis['by_underlying']
                und_rows = []
                for sym, data in sorted(by_und.items(), key=lambda x: abs(x[1]['delta']), reverse=True):
                    stock_qty = stocks.get(sym, {}).get('qty', 0)
                    total_delta = data['delta'] + stock_qty
                    und_rows.append([
                        sym,
                        fmt_money(data.get('und_price', 0)),
                        f"{data['delta']:+.0f}",
                        f"{stock_qty:+.0f}",
                        f"{total_delta:+.0f}",
                        f"${data['theta']:.2f}",
                        f"{data['contracts']}"
                    ])
                
                print(tabulate(und_rows[:10], headers=["Symbol", "Price", "Opt Δ", "Stock", "Total Δ", "Theta/d", "Contracts"]))
                
                # By expiry
                print(f"\n  {Style.BRIGHT}BY EXPIRATION{Style.RESET_ALL}")
                by_exp = greeks_analysis['by_expiry']
                exp_rows = []
                for exp in sorted(by_exp.keys()):
                    data = by_exp[exp]
                    days = dte(exp)
                    is_leaps = days >= LEAPSAnalyzer.LEAPS_MIN_DTE
                    exp_str = f"{exp} ({days}d)"
                    if is_leaps:
                        exp_str = Fore.GREEN + exp_str + " LEAPS" + Style.RESET_ALL
                    elif days < 90:
                        exp_str = Fore.RED + exp_str + Style.RESET_ALL
                    
                    exp_rows.append([
                        exp_str,
                        f"{data['delta']:+.0f}",
                        f"${data['theta']:.2f}",
                        f"{data['contracts']}",
                        fmt_money(data['value'])
                    ])
                
                print(tabulate(exp_rows, headers=["Expiration", "Delta", "Theta/d", "Contracts", "Value"]))
                
                # LEAPS vs non-LEAPS breakdown
                print(f"\n  {Style.BRIGHT}LEAPS vs NON-LEAPS{Style.RESET_ALL}")
                leaps_count = greeks_analysis['leaps_count']
                non_leaps_count = greeks_analysis['non_leaps_count']
                print(f"    LEAPS positions:     {leaps_count}")
                print(f"    Non-LEAPS positions: {non_leaps_count}")
                
                if non_leaps_count > 0 and leaps_count > 0:
                    ratio = leaps_count / (leaps_count + non_leaps_count) * 100
                    print(f"    LEAPS allocation:    {ratio:.0f}%")
                
                # Risk assessment
                print(f"\n  {Style.BRIGHT}RISK ASSESSMENT{Style.RESET_ALL}")
                
                # Delta exposure
                if abs(totals['delta_dollars']) > 100000:
                    print(f"    ⚠️  High delta exposure: {fmt_money(totals['delta_dollars'])}")
                else:
                    print(f"    ✓ Delta exposure manageable")
                
                # Theta decay
                if totals['theta'] < -50:
                    print(f"    ⚠️  High theta decay: ${abs(totals['theta']):.0f}/day = ${abs(totals['theta'])*30:.0f}/month")
                elif totals['theta'] < 0:
                    print(f"    ⚡ Theta decay: ${abs(totals['theta']):.0f}/day")
                else:
                    print(f"    ✓ Net theta positive (short options)")
                
                print_footer(90)
            
            elif len(cmd) <= 5 and cmd.isalpha(): 
                d = fetcher.get_stock_price(cmd)
                print(f"  {cmd.upper()}: {fmt_money(d['price'])} ({color_pct(d['pct'])})")
                # Show first-time tip if this is their first lookup
                if last_symbol is None:
                    print_tip('first_stock')
                last_symbol = cmd.upper()
                
        except Exception as e: err(str(e))

if __name__ == "__main__":
    main()