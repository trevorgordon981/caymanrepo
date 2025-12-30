"""
Enhanced Stock Ticker Analysis App with Multi-Brokerage Import Support
Supports: Charles Schwab, Fidelity, E*TRADE, Robinhood, TD Ameritrade/thinkorswim

pip install requests pandas numpy matplotlib yfinance tabulate colorama
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
import time
import json
import warnings
import os
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

# Optional imports with fallbacks
try:
    from tabulate import tabulate
except ImportError:
    def tabulate(data, headers=None, tablefmt="simple"):
        lines = []
        if headers:
            lines.append(" | ".join(str(h) for h in headers))
            lines.append("-" * 60)
        for row in data:
            lines.append(" | ".join(str(c) for c in row))
        return "\n".join(lines)

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
except ImportError:
    class Fore:
        GREEN = RED = YELLOW = MAGENTA = CYAN = WHITE = BLUE = RESET = ''
    class Style:
        RESET_ALL = BRIGHT = ''


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    show_charts: bool = True
    save_charts: bool = False
    theme: str = 'dark'
    cache_duration: int = 300
    max_concurrent_requests: int = 5
    data_dir: str = field(default_factory=lambda: os.path.expanduser("~/.stockticker"))
    
    def __post_init__(self):
        os.makedirs(self.data_dir, exist_ok=True)
    
    def get_path(self, filename: str) -> str:
        return os.path.join(self.data_dir, filename)


config = Config()


# =============================================================================
# PRICE CACHE
# =============================================================================

class PriceCache:
    def __init__(self, cache_duration: int = 300):
        self.cache: Dict[str, Tuple[float, float]] = {}
        self.cache_duration = cache_duration
        self.last_api_call = 0
        self.min_delay = 0.5  # Increased from 0.3 to avoid rate limiting

    def get(self, symbol: str) -> Optional[float]:
        symbol = symbol.upper()
        if symbol in self.cache:
            timestamp, price = self.cache[symbol]
            if time.time() - timestamp < self.cache_duration:
                return price
        return None

    def set(self, symbol: str, price: float):
        if price is not None:
            self.cache[symbol.upper()] = (time.time(), price)

    def wait_for_rate_limit(self):
        elapsed = time.time() - self.last_api_call
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)
        self.last_api_call = time.time()

    def clear(self):
        self.cache.clear()


price_cache = PriceCache()


# =============================================================================
# BROKERAGE CSV PARSERS
# =============================================================================

class BrokerageParser(ABC):
    """Abstract base class for brokerage CSV parsers"""
    name: str = "Unknown"
    
    @abstractmethod
    def can_parse(self, headers: List[str], content: str) -> bool:
        pass
    
    @abstractmethod
    def parse(self, filepath: str) -> Tuple[Dict, Dict, float]:
        pass
    
    @staticmethod
    def clean_number(val: Any) -> float:
        if val is None or val == '' or val == '--':
            return 0.0
        clean = str(val).replace(',', '').replace('$', '').replace('"', '').replace('(', '-').replace(')', '').strip()
        try:
            return float(clean) if clean and clean != '-' else 0.0
        except ValueError:
            return 0.0
    
    @staticmethod
    def parse_csv_line(line: str) -> List[str]:
        fields, current, in_quotes = [], '', False
        for char in line:
            if char == '"':
                in_quotes = not in_quotes
            elif char == ',' and not in_quotes:
                fields.append(current.strip())
                current = ''
            else:
                current += char
        fields.append(current.strip())
        return fields


class FidelityParser(BrokerageParser):
    name = "Fidelity"
    
    def can_parse(self, headers: List[str], content: str) -> bool:
        hl = [h.lower() for h in headers]
        return 'symbol' in hl and 'quantity' in hl and any('cost basis' in h for h in hl)
    
    def parse(self, filepath: str) -> Tuple[Dict, Dict, float]:
        stocks, options, cash = {}, {}, 0.0
        
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
        
        header_idx = next((i for i, l in enumerate(lines) if 'Symbol' in l and 'Quantity' in l), 0)
        headers = [h.strip().lower() for h in lines[header_idx].split(',')]
        
        col_map = {}
        for i, h in enumerate(headers):
            if h == 'symbol': col_map['symbol'] = i
            elif h == 'description': col_map['desc'] = i
            elif h == 'quantity': col_map['qty'] = i
            elif 'cost basis total' in h: col_map['cost'] = i
            elif 'average cost basis' in h: col_map['avg_cost'] = i
            elif h == 'current value': col_map['value'] = i
            elif 'total gain/loss dollar' in h: col_map['gl'] = i
        
        skip_terms = ['MONEY MARKET', 'SPAXX', 'FCASH', 'FDRXX', 'PENDING', 'CORE', 'CASH']
        
        for line in lines[header_idx + 1:]:
            fields = self.parse_csv_line(line)
            if len(fields) < 5:
                continue
            
            symbol = fields[col_map.get('symbol', 0)].strip().upper()
            desc = fields[col_map.get('desc', 1)].strip().upper() if 'desc' in col_map else ''
            
            if not symbol or any(s in f"{symbol} {desc}" for s in skip_terms):
                continue
            
            qty = self.clean_number(fields[col_map.get('qty', 2)])
            if qty == 0:
                continue
            
            cost_total = abs(self.clean_number(fields[col_map.get('cost', -1)])) if 'cost' in col_map else 0
            avg_cost = abs(self.clean_number(fields[col_map.get('avg_cost', -1)])) if 'avg_cost' in col_map else 0
            value = self.clean_number(fields[col_map.get('value', -1)]) if 'value' in col_map else 0
            gl = self.clean_number(fields[col_map.get('gl', -1)]) if 'gl' in col_map else 0
            
            if 'CALL' in desc or 'PUT' in desc:
                opt = self._parse_option(desc, qty, cost_total, avg_cost, value, gl)
                if opt:
                    options[opt['id']] = opt['data']
            else:
                if avg_cost == 0 and cost_total > 0 and qty > 0:
                    avg_cost = cost_total / qty
                stocks[symbol] = {'shares': qty, 'cost_basis': avg_cost, 'current_value': value, 'total_gl': gl}
        
        return stocks, options, cash
    
    def _parse_option(self, desc: str, qty: float, cost: float, avg: float, value: float, gl: float) -> Optional[Dict]:
        months = {'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12}
        words = desc.split()
        if len(words) < 5:
            return None
        
        underlying = ''.join(c for c in words[0] if c.isalpha())[:5]
        if not underlying:
            return None
        
        otype = 'call' if 'CALL' in desc else 'put'
        expiry = None
        for i, w in enumerate(words):
            if w in months and i + 2 < len(words):
                try:
                    mon, day = months[w], int(words[i+1].replace(',', ''))
                    yr = int(words[i+2].replace(',', ''))
                    yr = yr + 2000 if yr < 100 else yr
                    expiry = f"{yr}-{mon:02d}-{day:02d}"
                    break
                except ValueError:
                    continue
        
        if not expiry:
            return None
        
        match = re.search(r'\$(\d+(?:\.\d+)?)', desc)
        strike = float(match.group(1)) if match else 0
        if strike == 0:
            return None
        
        contracts = int(abs(qty))
        if contracts == 0:
            return None
        
        premium = avg if avg > 0 else (cost / (contracts * 100) if cost > 0 else 0.01)
        
        return {
            'id': f"{underlying}_{expiry}_{otype[0].upper()}{int(strike)}",
            'data': {
                'symbol': underlying, 'type': otype, 'strike': strike, 'expiration': expiry,
                'contracts': contracts, 'premium': premium, 'position_type': 'long' if qty > 0 else 'short',
                'total_cost': cost, 'current_value': value, 'total_gl': gl
            }
        }


class SchwabParser(BrokerageParser):
    name = "Charles Schwab"
    
    def can_parse(self, headers: List[str], content: str) -> bool:
        hl = ','.join(h.lower() for h in headers)
        return 'action' in hl and 'symbol' in hl and ('fees & comm' in hl or 'amount' in hl)
    
    def parse(self, filepath: str) -> Tuple[Dict, Dict, float]:
        stocks, options, positions = {}, {}, {}
        
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
        
        header_idx = next((i for i, l in enumerate(lines) if 'Date' in l and 'Action' in l and 'Symbol' in l), 0)
        headers = [h.strip() for h in lines[header_idx].split(',')]
        hmap = {h.lower(): i for i, h in enumerate(headers)}
        
        for line in lines[header_idx + 1:]:
            fields = self.parse_csv_line(line)
            if len(fields) < 4:
                continue
            
            action = fields[hmap.get('action', 1)].upper() if 'action' in hmap else ''
            symbol = fields[hmap.get('symbol', 2)].strip().upper() if 'symbol' in hmap else ''
            
            if not symbol or not action or any(s in action for s in ['DIVIDEND', 'INTEREST', 'TRANSFER', 'JOURNAL', 'ADR']):
                continue
            
            qty = abs(self.clean_number(fields[hmap.get('quantity', 3)])) if 'quantity' in hmap else 0
            price = abs(self.clean_number(fields[hmap.get('price', 4)])) if 'price' in hmap else 0
            
            if qty == 0:
                continue
            
            if symbol not in positions:
                positions[symbol] = {'shares': 0, 'cost_total': 0}
            
            if any(b in action for b in ['BUY', 'REINVEST']):
                positions[symbol]['shares'] += qty
                positions[symbol]['cost_total'] += qty * price
            elif 'SELL' in action:
                positions[symbol]['shares'] -= qty
        
        for sym, pos in positions.items():
            if pos['shares'] > 0.001:
                stocks[sym] = {'shares': pos['shares'], 'cost_basis': pos['cost_total'] / pos['shares'] if pos['shares'] > 0 else 0, 'current_value': 0, 'total_gl': 0}
        
        return stocks, options, 0.0


class ETRADEParser(BrokerageParser):
    name = "E*TRADE"
    
    def can_parse(self, headers: List[str], content: str) -> bool:
        hl = ','.join(h.lower() for h in headers)
        return ('transactiontype' in hl or 'transaction type' in hl) and 'symbol' in hl
    
    def parse(self, filepath: str) -> Tuple[Dict, Dict, float]:
        stocks, positions = {}, {}
        
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
        
        header_idx = next((i for i, l in enumerate(lines) if 'Symbol' in l), 0)
        headers = [h.strip() for h in lines[header_idx].split(',')]
        hmap = {h.lower().replace(' ', ''): i for i, h in enumerate(headers)}
        
        action_col = hmap.get('transactiontype', hmap.get('type', hmap.get('action', 1)))
        symbol_col = hmap.get('symbol', 2)
        qty_col = hmap.get('quantity', hmap.get('qty', 3))
        price_col = hmap.get('price', 4)
        
        for line in lines[header_idx + 1:]:
            fields = self.parse_csv_line(line)
            if len(fields) < 4:
                continue
            
            action = fields[action_col].upper() if action_col < len(fields) else ''
            symbol = fields[symbol_col].strip().upper() if symbol_col < len(fields) else ''
            qty = abs(self.clean_number(fields[qty_col])) if qty_col < len(fields) else 0
            price = abs(self.clean_number(fields[price_col])) if price_col < len(fields) else 0
            
            if not symbol or qty == 0:
                continue
            
            if symbol not in positions:
                positions[symbol] = {'shares': 0, 'cost_total': 0}
            
            if 'BOUGHT' in action or 'BUY' in action:
                positions[symbol]['shares'] += qty
                positions[symbol]['cost_total'] += qty * price
            elif 'SOLD' in action or 'SELL' in action:
                positions[symbol]['shares'] -= qty
        
        for sym, pos in positions.items():
            if pos['shares'] > 0.001:
                stocks[sym] = {'shares': pos['shares'], 'cost_basis': pos['cost_total'] / pos['shares'] if pos['shares'] > 0 else 0, 'current_value': 0, 'total_gl': 0}
        
        return stocks, {}, 0.0


class RobinhoodParser(BrokerageParser):
    name = "Robinhood"
    
    def can_parse(self, headers: List[str], content: str) -> bool:
        hl = ','.join(h.lower() for h in headers)
        return ('activity date' in hl or 'trans code' in hl) and ('instrument' in hl or 'symbol' in hl)
    
    def parse(self, filepath: str) -> Tuple[Dict, Dict, float]:
        stocks, positions = {}, {}
        
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
        
        header_idx = next((i for i, l in enumerate(lines) if any(h in l.lower() for h in ['activity date', 'trans code', 'instrument'])), 0)
        headers = [h.strip().lower() for h in lines[header_idx].split(',')]
        
        def find_col(*names):
            for n in names:
                if n in headers:
                    return headers.index(n)
            return -1
        
        action_col = find_col('trans code', 'type', 'description')
        symbol_col = find_col('instrument', 'symbol')
        qty_col = find_col('quantity', 'qty')
        price_col = find_col('price', 'amount')
        
        for line in lines[header_idx + 1:]:
            fields = self.parse_csv_line(line)
            if len(fields) <= max(symbol_col, qty_col, 0):
                continue
            
            action = fields[action_col].upper() if 0 <= action_col < len(fields) else ''
            symbol = fields[symbol_col].strip().upper() if 0 <= symbol_col < len(fields) else ''
            qty = abs(self.clean_number(fields[qty_col])) if 0 <= qty_col < len(fields) else 0
            price = abs(self.clean_number(fields[price_col])) if 0 <= price_col < len(fields) else 0
            
            if not symbol or qty == 0:
                continue
            
            if symbol not in positions:
                positions[symbol] = {'shares': 0, 'cost_total': 0}
            
            if 'BUY' in action:
                positions[symbol]['shares'] += qty
                positions[symbol]['cost_total'] += qty * price
            elif 'SELL' in action:
                positions[symbol]['shares'] -= qty
        
        for sym, pos in positions.items():
            if pos['shares'] > 0.001:
                stocks[sym] = {'shares': pos['shares'], 'cost_basis': pos['cost_total'] / pos['shares'] if pos['shares'] > 0 else 0, 'current_value': 0, 'total_gl': 0}
        
        return stocks, {}, 0.0


class ThinkorswimParser(BrokerageParser):
    name = "thinkorswim"
    
    def can_parse(self, headers: List[str], content: str) -> bool:
        hl = ','.join(h.lower() for h in headers)
        return ('exec time' in hl or 'side' in hl) and ('symbol' in hl or 'underlying' in hl)
    
    def parse(self, filepath: str) -> Tuple[Dict, Dict, float]:
        stocks, positions = {}, {}
        
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            lines = f.read().split('\n')
        
        header_idx = next((i for i, l in enumerate(lines) if ('symbol' in l.lower() and ('side' in l.lower() or 'type' in l.lower())) or 'exec time' in l.lower()), 0)
        headers = [h.strip().lower() for h in lines[header_idx].split(',')]
        
        def find_col(*names):
            for n in names:
                for i, h in enumerate(headers):
                    if n in h:
                        return i
            return -1
        
        symbol_col = find_col('symbol', 'underlying')
        action_col = find_col('side', 'type', 'action')
        qty_col = find_col('qty', 'quantity')
        price_col = find_col('price', 'avg price')
        
        for line in lines[header_idx + 1:]:
            if not line.strip():
                continue
            fields = self.parse_csv_line(line)
            if len(fields) <= max(symbol_col, qty_col, 0):
                continue
            
            symbol = fields[symbol_col].strip().upper() if symbol_col >= 0 else ''
            action = fields[action_col].upper() if 0 <= action_col < len(fields) else ''
            qty = abs(self.clean_number(fields[qty_col])) if qty_col >= 0 and qty_col < len(fields) else 0
            price = abs(self.clean_number(fields[price_col])) if price_col >= 0 and price_col < len(fields) else 0
            
            if ' ' in symbol:
                symbol = symbol.split()[0]
            symbol = ''.join(c for c in symbol if c.isalpha())[:5]
            
            if not symbol or qty == 0:
                continue
            
            if symbol not in positions:
                positions[symbol] = {'shares': 0, 'cost_total': 0}
            
            if any(b in action for b in ['BUY', 'BTO', 'BOT']):
                positions[symbol]['shares'] += qty
                positions[symbol]['cost_total'] += qty * price
            elif any(s in action for s in ['SELL', 'STC', 'SLD']):
                positions[symbol]['shares'] -= qty
        
        for sym, pos in positions.items():
            if pos['shares'] > 0.001:
                stocks[sym] = {'shares': pos['shares'], 'cost_basis': pos['cost_total'] / pos['shares'] if pos['shares'] > 0 else 0, 'current_value': 0, 'total_gl': 0}
        
        return stocks, {}, 0.0


class GenericParser(BrokerageParser):
    name = "Generic CSV"
    
    def can_parse(self, headers: List[str], content: str) -> bool:
        return True
    
    def parse(self, filepath: str) -> Tuple[Dict, Dict, float]:
        stocks = {}
        
        try:
            df = pd.read_csv(filepath, encoding='utf-8-sig')
        except:
            df = pd.read_csv(filepath, encoding='latin-1')
        
        df.columns = [c.strip().lower() for c in df.columns]
        
        def find_col(candidates):
            for c in candidates:
                for col in df.columns:
                    if c in col:
                        return col
            return None
        
        sym_col = find_col(['symbol', 'ticker', 'security', 'stock'])
        qty_col = find_col(['quantity', 'qty', 'shares', 'units'])
        price_col = find_col(['price', 'cost', 'avg cost', 'average cost', 'cost basis'])
        action_col = find_col(['action', 'type', 'transaction', 'side'])
        
        if not sym_col or not qty_col:
            return {}, {}, 0.0
        
        positions = {}
        for _, row in df.iterrows():
            symbol = str(row.get(sym_col, '')).strip().upper()
            if not symbol or len(symbol) > 6:
                continue
            
            qty = self.clean_number(row.get(qty_col, 0))
            price = self.clean_number(row.get(price_col, 0)) if price_col else 0
            action = str(row.get(action_col, 'BUY')).upper() if action_col else 'BUY'
            
            if qty == 0:
                continue
            
            if symbol not in positions:
                positions[symbol] = {'shares': 0, 'cost_total': 0}
            
            if 'SELL' in action or 'SLD' in action:
                positions[symbol]['shares'] -= abs(qty)
            else:
                positions[symbol]['shares'] += abs(qty)
                positions[symbol]['cost_total'] += abs(qty) * price
        
        for sym, pos in positions.items():
            if pos['shares'] > 0.001:
                stocks[sym] = {'shares': pos['shares'], 'cost_basis': pos['cost_total'] / pos['shares'] if pos['shares'] > 0 else 0, 'current_value': 0, 'total_gl': 0}
        
        return stocks, {}, 0.0


# =============================================================================
# CSV IMPORT MANAGER
# =============================================================================

class CSVImportManager:
    def __init__(self):
        self.parsers = [FidelityParser(), SchwabParser(), ETRADEParser(), RobinhoodParser(), ThinkorswimParser(), GenericParser()]
    
    def detect_and_parse(self, filepath: str) -> Tuple[str, Dict, Dict, float]:
        filepath = os.path.expanduser(filepath)
        filepath = os.path.abspath(filepath)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            content = f.read()
        
        headers = []
        for line in content.split('\n')[:10]:
            if ',' in line and any(c.isalpha() for c in line):
                headers = [h.strip() for h in line.split(',')]
                break
        
        for parser in self.parsers:
            if parser.can_parse(headers, content):
                print(Fore.CYAN + f"Detected format: {parser.name}")
                stocks, options, cash = parser.parse(filepath)
                return parser.name, stocks, options, cash
        
        raise ValueError("Could not detect CSV format")
    
    def parse_with_format(self, filepath: str, format_name: str) -> Tuple[Dict, Dict, float]:
        filepath = os.path.expanduser(filepath)
        parser_map = {p.name.lower(): p for p in self.parsers}
        parser = parser_map.get(format_name.lower())
        
        if not parser:
            raise ValueError(f"Unknown format: {format_name}")
        
        return parser.parse(filepath)


# =============================================================================
# DATA FETCHER
# =============================================================================

class DataFetcher:
    def __init__(self):
        self.cache: Dict[str, Tuple[float, Any, pd.DataFrame]] = {}
        self.source = None
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'})

    def get_price(self, symbol: str) -> Optional[float]:
        symbol = symbol.upper().strip()
        cached = price_cache.get(symbol)
        if cached is not None:
            return cached

        price_cache.wait_for_rate_limit()

        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            r = self.session.get(url, params={"interval": "1d", "range": "1d"}, timeout=10)
            if r.status_code == 200:
                data = r.json()
                result = data.get('chart', {}).get('result', [])
                if result:
                    price = result[0].get('meta', {}).get('regularMarketPrice')
                    if price:
                        price_cache.set(symbol, price)
                        return price
        except:
            pass

        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d')
            if not data.empty:
                price = float(data['Close'].iloc[-1])
                price_cache.set(symbol, price)
                return price
        except:
            pass

        return None

    def get_prices_batch(self, symbols: List[str]) -> Dict[str, float]:
        results = {}
        symbols = [s.upper().strip() for s in symbols if s]
        to_fetch = []
        
        for sym in symbols:
            cached = price_cache.get(sym)
            if cached is not None:
                results[sym] = cached
            else:
                to_fetch.append(sym)
        
        if not to_fetch:
            return results
        
        with ThreadPoolExecutor(max_workers=config.max_concurrent_requests) as executor:
            futures = {executor.submit(self.get_price, sym): sym for sym in to_fetch}
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    price = future.result()
                    if price:
                        results[sym] = price
                except:
                    pass
        
        return results

    def fetch(self, symbol: str, quick: bool = False) -> Tuple[Optional[Dict], Optional[pd.DataFrame]]:
        symbol = symbol.upper().strip()
        
        if quick and symbol in self.cache:
            t, info, hist = self.cache[symbol]
            if time.time() - t < 300:
                return info, hist

        price_cache.wait_for_rate_limit()

        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1y")
            if not hist.empty:
                self.source = "yfinance"
                self.cache[symbol] = (time.time(), info, hist)
                if 'regularMarketPrice' in info:
                    price_cache.set(symbol, info['regularMarketPrice'])
                return info, hist
        except:
            pass

        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            r = self.session.get(url, params={"interval": "1d", "range": "1y"}, timeout=15)
            if r.status_code == 200:
                data = r.json()
                result = data.get('chart', {}).get('result', [])
                if result:
                    result = result[0]
                    meta = result.get('meta', {})
                    ts = result.get('timestamp', [])
                    q = result.get('indicators', {}).get('quote', [{}])[0]
                    if ts and q:
                        df = pd.DataFrame({'Open': q.get('open'), 'High': q.get('high'), 'Low': q.get('low'), 'Close': q.get('close'), 'Volume': q.get('volume')}, index=pd.to_datetime(ts, unit='s')).dropna()
                        info = {'regularMarketPrice': meta.get('regularMarketPrice'), 'previousClose': meta.get('previousClose'), 'longName': meta.get('longName', symbol)}
                        self.source = "Direct API"
                        self.cache[symbol] = (time.time(), info, df)
                        return info, df
        except:
            pass

        return None, None


# =============================================================================
# PORTFOLIO MANAGER
# =============================================================================

class Portfolio:
    def __init__(self):
        self.stocks = self._load('portfolio.json')
        self.options = self._load('options_portfolio.json')
        self.cash = self._load_value('cash.json', 'amount', 0.0)
        self.csv_manager = CSVImportManager()

    def _load(self, filename: str) -> Dict:
        path = config.get_path(filename)
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return json.load(f)
        except:
            pass
        return {}

    def _save(self, filename: str, data: Dict):
        with open(config.get_path(filename), 'w') as f:
            json.dump(data, f, indent=2)

    def _load_value(self, filename: str, key: str, default: Any) -> Any:
        return self._load(filename).get(key, default)

    def _save_stocks(self):
        self._save('portfolio.json', self.stocks)

    def _save_options(self):
        self._save('options_portfolio.json', self.options)

    def _save_cash(self):
        self._save('cash.json', {'amount': self.cash})

    def set_cash(self, amount: float):
        self.cash = float(amount)
        self._save_cash()
        print(Fore.GREEN + f"Cash balance set to ${self.cash:,.2f}")

    def add_stock(self, symbol: str, shares: float, cost: float):
        symbol = symbol.upper()
        self.stocks[symbol] = {'shares': shares, 'cost_basis': cost, 'current_value': 0, 'total_gl': 0}
        self._save_stocks()
        print(Fore.GREEN + f"Added {symbol}: {shares} shares @ ${cost:.2f}")

    def remove_stock(self, symbol: str):
        symbol = symbol.upper()
        if symbol in self.stocks:
            del self.stocks[symbol]
            self._save_stocks()
            print(Fore.YELLOW + f"Removed {symbol}")
        else:
            print(Fore.RED + f"Stock not found: {symbol}")

    def add_option(self, symbol: str, otype: str, strike: float, expiry: str, contracts: int, premium: float, pos_type: str = 'long'):
        symbol = symbol.upper()
        oid = f"{symbol}_{expiry}_{otype[0].upper()}{int(strike)}"
        self.options[oid] = {'symbol': symbol, 'type': otype.lower(), 'strike': strike, 'expiration': expiry, 'contracts': contracts, 'premium': premium, 'position_type': pos_type, 'total_cost': contracts * 100 * premium, 'current_value': 0, 'total_gl': 0}
        self._save_options()
        print(Fore.GREEN + f"{'Bought' if pos_type == 'long' else 'Sold'} {contracts} {symbol} ${strike} {otype}s expiring {expiry}")

    def remove_option(self, oid: str):
        if oid in self.options:
            del self.options[oid]
            self._save_options()
            print(Fore.YELLOW + f"Removed option: {oid}")
        else:
            print(Fore.RED + f"Option not found: {oid}")
            if self.options:
                print("Available: " + ", ".join(list(self.options.keys())[:5]))

    def clear(self, stocks: bool = True, options: bool = True):
        if stocks:
            self.stocks = {}
            self._save_stocks()
            print(Fore.YELLOW + "Stock positions cleared")
        if options:
            self.options = {}
            self._save_options()
            print(Fore.YELLOW + "Options positions cleared")

    def import_csv(self, filepath: str, format_name: str = None):
        filepath = os.path.expanduser(filepath.strip().strip('"').strip("'"))
        print(Fore.CYAN + f"\nImporting from: {filepath}")
        
        if not os.path.exists(filepath):
            print(Fore.RED + f"File not found: {filepath}")
            downloads = os.path.expanduser("~/Downloads")
            if os.path.exists(downloads):
                csvs = [f for f in os.listdir(downloads) if f.endswith('.csv')]
                if csvs:
                    print(Fore.YELLOW + "CSV files in Downloads: " + ", ".join(csvs[:5]))
            return 0, 0
        
        try:
            if format_name:
                brokerage = format_name
                stocks, options, cash = self.csv_manager.parse_with_format(filepath, format_name)
            else:
                brokerage, stocks, options, cash = self.csv_manager.detect_and_parse(filepath)
            
            for sym, data in stocks.items():
                self.stocks[sym] = data
                print(Fore.GREEN + f"  Stock: {sym} | {data['shares']:.2f} @ ${data['cost_basis']:.2f}")
            
            for oid, data in options.items():
                self.options[oid] = data
                print(Fore.GREEN + f"  Option: {oid}")
            
            if cash > 0:
                self.cash = cash
                self._save_cash()
            
            self._save_stocks()
            self._save_options()
            
            print(Fore.GREEN + f"\n✓ Import complete from {brokerage}! Stocks: {len(stocks)}, Options: {len(options)}")
            return len(stocks), len(options)
        except Exception as e:
            print(Fore.RED + f"Import error: {e}")
            return 0, 0

    def export_csv(self, filepath: str):
        filepath = os.path.expanduser(filepath.strip().strip('"').strip("'"))
        rows = []
        for sym, pos in self.stocks.items():
            rows.append({'Type': 'Stock', 'Symbol': sym, 'Quantity': pos['shares'], 'Cost Basis': pos['cost_basis'], 'Value': pos.get('current_value', 0), 'P&L': pos.get('total_gl', 0)})
        for oid, opt in self.options.items():
            rows.append({'Type': 'Option', 'Symbol': f"{opt['symbol']} ${opt['strike']}{opt['type'][0].upper()}", 'Quantity': opt['contracts'], 'Cost Basis': opt['premium'], 'Value': opt.get('current_value', 0), 'P&L': opt.get('total_gl', 0), 'Expiration': opt['expiration']})
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        print(Fore.GREEN + f"Exported portfolio to {filepath}")

    def display(self, fetcher: DataFetcher):
        if not self.stocks and not self.options:
            print(Fore.YELLOW + "\nPortfolio empty. Use: buy SYMBOL SHARES PRICE  or  import /path/to/file.csv")
            return

        print(Fore.CYAN + "\n" + "=" * 70 + "\n  PORTFOLIO SUMMARY\n" + "=" * 70 + Style.RESET_ALL)

        all_symbols = set(self.stocks.keys()) | {o['symbol'] for o in self.options.values()}
        print(f"\nFetching prices for {len(all_symbols)} symbols...")
        prices = fetcher.get_prices_batch(list(all_symbols))
        for sym, price in sorted(prices.items()):
            print(f"  {sym}: ${price:.2f}")

        total_stock_cost = total_stock_value = total_stock_pnl = 0
        total_opts_cost = total_opts_value = total_opts_pnl = 0

        if self.stocks:
            print(Fore.CYAN + f"\nSTOCKS ({len(self.stocks)})" + Style.RESET_ALL + "\n" + "-" * 70)
            data = []
            for sym, pos in sorted(self.stocks.items()):
                shares, cost_basis = pos['shares'], pos['cost_basis']
                cost_total = shares * cost_basis
                current_price = prices.get(sym, 0)
                current_value = shares * current_price if current_price else pos.get('current_value', 0)
                pnl = current_value - cost_total if current_value else pos.get('total_gl', 0)
                total_stock_cost += cost_total
                total_stock_value += current_value
                total_stock_pnl += pnl
                pnl_color = Fore.GREEN if pnl >= 0 else Fore.RED
                data.append([sym[:12], f"{shares:.2f}", f"${cost_total:,.2f}", f"${current_value:,.2f}", f"{pnl_color}${pnl:+,.2f}{Style.RESET_ALL}"])
            print(tabulate(data, headers=["Symbol", "Qty", "Cost", "Value", "P&L"]))
            color = Fore.GREEN if total_stock_pnl >= 0 else Fore.RED
            print(f"\n  Stock Cost: ${total_stock_cost:,.2f} | Value: ${total_stock_value:,.2f} | P&L: {color}${total_stock_pnl:+,.2f}{Style.RESET_ALL}")

        if self.options:
            long_c = sum(1 for o in self.options.values() if o['position_type'] == 'long')
            print(Fore.CYAN + f"\nOPTIONS ({len(self.options)}: {long_c} long, {len(self.options)-long_c} short)" + Style.RESET_ALL + "\n" + "-" * 100)
            data = []
            for oid, opt in sorted(self.options.items(), key=lambda x: x[1]['expiration']):
                stock_price = prices.get(opt['symbol'], 0)
                cost = opt.get('total_cost', 0)
                value = opt.get('current_value', 0)
                pnl = opt.get('total_gl', 0) if opt.get('total_gl', 0) != 0 else (value - cost)
                try:
                    dte = (datetime.strptime(opt['expiration'], '%Y-%m-%d') - datetime.now()).days
                except:
                    dte = 0
                dte_str = Fore.RED + "EXP" + Style.RESET_ALL if dte < 0 else (Fore.YELLOW + f"{dte}d" + Style.RESET_ALL if dte <= 7 else f"{dte}d")
                itm_str = "—"
                if stock_price > 0:
                    is_itm = (stock_price > opt['strike']) if opt['type'] == 'call' else (stock_price < opt['strike'])
                    itm_str = Fore.GREEN + "ITM" + Style.RESET_ALL if is_itm else "OTM"
                total_opts_cost += cost
                total_opts_value += abs(value)
                total_opts_pnl += pnl
                pos_str = Fore.GREEN + "LONG" + Style.RESET_ALL if opt['position_type'] == 'long' else Fore.MAGENTA + "SHORT" + Style.RESET_ALL
                pnl_color = Fore.GREEN if pnl >= 0 else Fore.RED
                opt_type = opt.get('type', 'call')
                data.append([f"{opt['symbol']} ${opt['strike']:.0f}{opt_type[0].upper()}", opt['expiration'], dte_str, opt['contracts'], pos_str, f"${cost:,.2f}", f"${abs(value):,.2f}", itm_str, f"{pnl_color}${pnl:+,.2f}{Style.RESET_ALL}"])
            print(tabulate(data, headers=["Option", "Expiry", "DTE", "Qty", "Type", "Cost", "Value", "ITM", "P&L"]))
            color = Fore.GREEN if total_opts_pnl >= 0 else Fore.RED
            print(f"\n  Options Cost: ${total_opts_cost:,.2f} | Value: ${total_opts_value:,.2f} | P&L: {color}${total_opts_pnl:+,.2f}{Style.RESET_ALL}")

        print(Fore.CYAN + "\n" + "=" * 70 + "\nPORTFOLIO TOTAL" + Style.RESET_ALL)
        grand_cost = total_stock_cost + total_opts_cost
        grand_value = total_stock_value + total_opts_value + self.cash
        grand_pnl = total_stock_pnl + total_opts_pnl
        color = Fore.GREEN if grand_pnl >= 0 else Fore.RED
        print(f"  Total Cost: ${grand_cost:,.2f} | Total Value: ${grand_value:,.2f}" + (f" | Cash: ${self.cash:,.2f}" if self.cash > 0 else "") + f" | P&L: {color}${grand_pnl:+,.2f}{Style.RESET_ALL}")


# =============================================================================
# WATCHLIST MANAGER
# =============================================================================

class WatchlistManager:
    def __init__(self):
        self.filename = config.get_path('watchlist.json')
        self.data = self._load()

    def _load(self) -> Dict:
        try:
            if os.path.exists(self.filename):
                with open(self.filename, 'r') as f:
                    return json.load(f)
        except:
            pass
        return {}

    def _save(self):
        with open(self.filename, 'w') as f:
            json.dump(self.data, f, indent=2)

    def add(self, symbol: str, high: float = None, low: float = None):
        self.data[symbol.upper()] = {'high': high, 'low': low}
        self._save()
        print(Fore.GREEN + f"Added {symbol.upper()} to watchlist")

    def remove(self, symbol: str):
        symbol = symbol.upper()
        if symbol in self.data:
            del self.data[symbol]
            self._save()
            print(Fore.YELLOW + f"Removed {symbol}")
        else:
            print(Fore.RED + f"{symbol} not in watchlist")

    def display(self, fetcher: DataFetcher):
        if not self.data:
            print(Fore.YELLOW + "Watchlist empty")
            return
        print(Fore.CYAN + "\nWATCHLIST\n" + "-" * 50)
        prices = fetcher.get_prices_batch(list(self.data.keys()))
        for sym, alerts in sorted(self.data.items()):
            price = prices.get(sym)
            if price:
                status = ""
                if alerts.get('high') and price >= alerts['high']:
                    status = Fore.RED + " ▲ HIGH ALERT" + Style.RESET_ALL
                elif alerts.get('low') and price <= alerts['low']:
                    status = Fore.GREEN + " ▼ LOW ALERT" + Style.RESET_ALL
                alert_info = f" (H:{alerts.get('high', '-')} L:{alerts.get('low', '-')})" if alerts.get('high') or alerts.get('low') else ""
                print(f"  {sym}: ${price:.2f}{alert_info}{status}")
            else:
                print(f"  {sym}: N/A")


# =============================================================================
# TECHNICAL ANALYSIS
# =============================================================================

def calc_rsi(data: pd.DataFrame, periods: int = 14) -> pd.Series:
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(periods).mean()
    return 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

def calc_macd(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    exp12, exp26 = data['Close'].ewm(span=12).mean(), data['Close'].ewm(span=26).mean()
    macd = exp12 - exp26
    return macd, macd.ewm(span=9).mean(), macd - macd.ewm(span=9).mean()

def calc_bb(data: pd.DataFrame, window: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
    sma, std = data['Close'].rolling(window).mean(), data['Close'].rolling(window).std()
    return sma + 2 * std, sma, sma - 2 * std

def analyze(symbol: str, fetcher: DataFetcher):
    symbol = symbol.upper().strip()
    print(Fore.CYAN + f"\nAnalyzing {symbol}..." + Style.RESET_ALL)
    info, hist = fetcher.fetch(symbol)
    if not info or hist is None or hist.empty:
        print(Fore.RED + "Could not fetch data. Try again in a few seconds (rate limited).")
        return None, None

    print(Fore.GREEN + f"Source: {fetcher.source}" + Style.RESET_ALL)
    
    # Safely get price with fallbacks
    price = info.get('regularMarketPrice')
    if price is None and hist is not None and not hist.empty:
        price = float(hist['Close'].iloc[-1])
    if price is None:
        print(Fore.RED + "Could not get current price")
        return None, None
    
    # Safely get previous close with fallback
    prev = info.get('previousClose')
    if prev is None and hist is not None and len(hist) > 1:
        prev = float(hist['Close'].iloc[-2])
    if prev is None:
        prev = price  # Use current price if no previous available
    
    chg = price - prev
    pct = (chg / prev * 100) if prev and prev != 0 else 0
    print(f"\n  Price: ${price:.2f}")
    print(f"  Change: {Fore.GREEN if chg >= 0 else Fore.RED}${chg:+.2f} ({pct:+.2f}%){Style.RESET_ALL}")

    if len(hist) >= 20:
        hist['RSI'] = calc_rsi(hist)
        hist['MACD'], hist['Signal'], hist['Hist'] = calc_macd(hist)
        hist['BB_U'], hist['BB_M'], hist['BB_L'] = calc_bb(hist)
        hist['SMA20'], hist['SMA50'] = hist['Close'].rolling(20).mean(), hist['Close'].rolling(50).mean()
        hist['EMA200'] = hist['Close'].ewm(span=200).mean()
        
        rsi = hist['RSI'].iloc[-1]
        if pd.notna(rsi):
            sig = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
            color = Fore.RED if rsi > 70 else Fore.GREEN if rsi < 30 else ""
            print(f"  RSI: {color}{rsi:.1f} ({sig}){Style.RESET_ALL}")
        if pd.notna(hist['SMA50'].iloc[-1]):
            print(f"  SMA50: ${hist['SMA50'].iloc[-1]:.2f} (price {'above' if price > hist['SMA50'].iloc[-1] else 'below'})")
        if pd.notna(hist['EMA200'].iloc[-1]):
            print(f"  EMA200: ${hist['EMA200'].iloc[-1]:.2f} (price {'above' if price > hist['EMA200'].iloc[-1] else 'below'})")

    if config.show_charts and len(hist) > 5:
        create_chart(hist, symbol, info, price)
    return hist, info

def create_chart(hist: pd.DataFrame, symbol: str, info: Dict, price: float):
    try:
        plt.style.use('dark_background' if config.theme == 'dark' else 'default')
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1, 1]})
        fig.suptitle(f"{info.get('longName', symbol)} ({symbol}) - ${price:.2f}", fontsize=14, fontweight='bold')

        ax1 = axes[0]
        ax1.plot(hist.index, hist['Close'], label='Price', color='white', lw=1.5)
        for col, lbl, clr in [('SMA20', 'SMA20', '#00ff88'), ('SMA50', 'SMA50', '#ff8800'), ('EMA200', 'EMA200', '#ff00ff')]:
            if col in hist.columns:
                ax1.plot(hist.index, hist[col], label=lbl, alpha=0.7 if col != 'EMA200' else 1, color=clr, lw=1.5 if col != 'EMA200' else 2)
        if 'BB_U' in hist.columns:
            ax1.fill_between(hist.index, hist['BB_U'], hist['BB_L'], alpha=0.1, color='cyan')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylabel('Price ($)')

        ax2 = axes[1]
        colors = ['#00ff00' if hist['Close'].iloc[i] >= hist['Open'].iloc[i] else '#ff0000' for i in range(len(hist))]
        ax2.bar(hist.index, hist['Volume'], color=colors, alpha=0.6)
        ax2.set_ylabel('Volume')
        ax2.grid(True, alpha=0.3)

        ax3 = axes[2]
        if 'RSI' in hist.columns:
            ax3.plot(hist.index, hist['RSI'], color='#aa88ff', lw=1.5)
            ax3.axhline(70, color='#ff4444', ls='--', alpha=0.7)
            ax3.axhline(30, color='#44ff44', ls='--', alpha=0.7)
            ax3.axhline(50, color='gray', ls=':', alpha=0.5)
            ax3.fill_between(hist.index, 70, 100, alpha=0.1, color='red')
            ax3.fill_between(hist.index, 0, 30, alpha=0.1, color='green')
            ax3.set_ylim(0, 100)
        ax3.set_ylabel('RSI')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        if config.save_charts:
            fn = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(fn, dpi=150, bbox_inches='tight')
            print(Fore.GREEN + f"Saved: {fn}")
    except Exception as e:
        print(Fore.RED + f"Chart error: {e}")

def compare(symbols: List[str], fetcher: DataFetcher):
    symbols = [s.upper() for s in symbols[:8]]
    print(Fore.CYAN + f"\nComparing: {', '.join(symbols)}" + Style.RESET_ALL)
    data = []
    for sym in symbols:
        info, hist = fetcher.fetch(sym, quick=True)
        if info:
            price = info.get('regularMarketPrice', 0)
            prev = info.get('previousClose', price)
            pct = ((price - prev) / prev * 100) if prev else 0
            rsi = calc_rsi(hist).iloc[-1] if hist is not None and len(hist) >= 14 else float('nan')
            color = Fore.GREEN if pct >= 0 else Fore.RED
            rsi_color = Fore.RED if pd.notna(rsi) and rsi > 70 else (Fore.GREEN if pd.notna(rsi) and rsi < 30 else "")
            data.append([sym, f"${price:.2f}", f"{color}{pct:+.2f}%{Style.RESET_ALL}", f"{rsi_color}{rsi:.1f}{Style.RESET_ALL}" if pd.notna(rsi) else "N/A"])
    print(tabulate(data, headers=["Symbol", "Price", "Change", "RSI"]))

def scan(fetcher: DataFetcher, symbols: List[str] = None):
    if not symbols:
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'AMD', 'NFLX', 'SPY']
    print(Fore.CYAN + f"\nScanning {len(symbols)} stocks..." + Style.RESET_ALL)
    oversold, overbought, golden_cross = [], [], []
    
    for sym in symbols:
        info, hist = fetcher.fetch(sym, quick=True)
        if hist is not None and len(hist) >= 50:
            rsi = calc_rsi(hist).iloc[-1]
            price = info.get('regularMarketPrice', 0) if info else 0
            sma50, ema200 = hist['Close'].rolling(50).mean().iloc[-1], hist['Close'].ewm(span=200).mean().iloc[-1]
            if pd.notna(rsi):
                if rsi < 30:
                    oversold.append((sym, rsi, price))
                elif rsi > 70:
                    overbought.append((sym, rsi, price))
            if pd.notna(sma50) and pd.notna(ema200) and sma50 > ema200:
                prev_sma50, prev_ema200 = hist['Close'].rolling(50).mean().iloc[-5], hist['Close'].ewm(span=200).mean().iloc[-5]
                if pd.notna(prev_sma50) and pd.notna(prev_ema200) and prev_sma50 <= prev_ema200:
                    golden_cross.append((sym, price))

    print("\n" + "=" * 50 + "\nSCAN RESULTS\n" + "=" * 50)
    if oversold:
        print(Fore.GREEN + "\n📈 OVERSOLD (RSI < 30):" + Style.RESET_ALL)
        for s, r, p in sorted(oversold, key=lambda x: x[1]):
            print(f"  {s}: RSI={r:.1f}, ${p:.2f}")
    if overbought:
        print(Fore.RED + "\n📉 OVERBOUGHT (RSI > 70):" + Style.RESET_ALL)
        for s, r, p in sorted(overbought, key=lambda x: -x[1]):
            print(f"  {s}: RSI={r:.1f}, ${p:.2f}")
    if golden_cross:
        print(Fore.YELLOW + "\n⭐ GOLDEN CROSS (Recent):" + Style.RESET_ALL)
        for s, p in golden_cross:
            print(f"  {s}: ${p:.2f}")
    if not oversold and not overbought and not golden_cross:
        print(Fore.YELLOW + "\nNo significant signals found")


# =============================================================================
# HELP & MAIN
# =============================================================================

def show_help():
    print(f"""
{Fore.CYAN}{'='*60}
STOCK TICKER APP - COMMAND REFERENCE
{'='*60}{Style.RESET_ALL}

{Fore.GREEN}ANALYSIS{Style.RESET_ALL}
  AAPL              Analyze stock (chart + technicals)
  compare A B C     Compare multiple stocks
  scan              Scan default stocks for signals
  scan AAPL MSFT    Scan specific stocks

{Fore.GREEN}PORTFOLIO{Style.RESET_ALL}
  portfolio         View holdings with live P&L
  buy AAPL 10 150   Add stock (symbol, shares, cost)
  sell AAPL         Remove stock position
  cash 65000        Set cash balance
  export ~/p.csv    Export portfolio to CSV

{Fore.GREEN}OPTIONS{Style.RESET_ALL}
  option buy AAPL call 175 2024-03-15 2 3.50
  option sell TSLA put 200 2024-04-21 1 5.25
  option close OPTION_ID

{Fore.GREEN}IMPORT FROM BROKERAGES{Style.RESET_ALL}
  import ~/Downloads/file.csv              Auto-detect
  import schwab ~/Downloads/txn.csv        Schwab format
  import fidelity ~/Downloads/Portfolio.csv
  import etrade ~/Downloads/history.csv
  import robinhood ~/Downloads/report.csv
  import thinkorswim ~/Downloads/stmt.csv
  
  Supported: Schwab, Fidelity, E*TRADE, Robinhood, thinkorswim

{Fore.GREEN}WATCHLIST{Style.RESET_ALL}
  watch AAPL 200 150   Add with high/low alerts
  watch AAPL           Add without alerts
  unwatch AAPL         Remove from watchlist
  watchlist            Show watchlist

{Fore.GREEN}SETTINGS{Style.RESET_ALL}
  charts on/off     Toggle charts
  theme dark/light  Switch theme
  refresh           Clear price cache
  clear portfolio   Clear all positions
  clear stocks      Clear only stocks
  clear options     Clear only options

{Fore.GREEN}OTHER{Style.RESET_ALL}
  help              Show this menu
  quit / exit / q   Exit
""")


def main():
    print(Fore.CYAN + "\n" + "=" * 60)
    print("  STOCK TICKER APP - Multi-Brokerage Edition")
    print("  Supports: Schwab | Fidelity | E*TRADE | Robinhood | TOS")
    print("=" * 60 + Style.RESET_ALL)
    print(f"Data directory: {config.data_dir}")
    print("Type 'help' for commands\n")

    fetcher = DataFetcher()
    watchlist = WatchlistManager()
    portfolio = Portfolio()

    while True:
        try:
            inp = input(Fore.CYAN + "> " + Style.RESET_ALL).strip()
            if not inp:
                continue

            cmd = inp.lower()
            parts = inp.split()

            if cmd in ['quit', 'exit', 'q']:
                break
            elif cmd == 'help':
                show_help()
            elif cmd == 'portfolio':
                portfolio.display(fetcher)
            elif cmd == 'refresh':
                price_cache.clear()
                print(Fore.GREEN + "Price cache cleared")
            elif cmd == 'watchlist':
                watchlist.display(fetcher)
            elif cmd.startswith('watch '):
                p = parts[1:]
                if p:
                    watchlist.add(p[0], float(p[1]) if len(p) > 1 else None, float(p[2]) if len(p) > 2 else None)
            elif cmd.startswith('unwatch '):
                watchlist.remove(parts[1])
            elif cmd.startswith('buy '):
                if len(parts) >= 4:
                    portfolio.add_stock(parts[1], float(parts[2]), float(parts[3]))
                else:
                    print("Usage: buy SYMBOL SHARES PRICE")
            elif cmd.startswith('sell '):
                portfolio.remove_stock(parts[1])
            elif cmd.startswith('cash '):
                try:
                    portfolio.set_cash(float(parts[1]))
                except:
                    print("Usage: cash AMOUNT")
            elif cmd.startswith('export '):
                portfolio.export_csv(parts[1] if len(parts) > 1 else '~/portfolio_export.csv')
            elif cmd.startswith('import '):
                p = parts[1:]
                formats = ['schwab', 'fidelity', 'etrade', 'robinhood', 'thinkorswim', 'generic']
                if len(p) >= 2 and p[0].lower() in formats:
                    portfolio.import_csv(p[1], p[0])
                elif len(p) >= 1:
                    portfolio.import_csv(p[0])
                else:
                    print("Usage: import [format] /path/to/file.csv")
            elif cmd.startswith('clear'):
                what = cmd[5:].strip()
                if 'portfolio' in what or not what:
                    if input("Clear all positions? (yes/no): ").lower() == 'yes':
                        portfolio.clear()
                elif 'stock' in what:
                    portfolio.clear(stocks=True, options=False)
                elif 'option' in what:
                    portfolio.clear(stocks=False, options=True)
            elif cmd.startswith('option '):
                p = parts[1:]
                if len(p) >= 7 and p[0] in ['buy', 'sell']:
                    try:
                        portfolio.add_option(p[1], p[2], float(p[3]), p[4], int(p[5]), float(p[6]), 'long' if p[0] == 'buy' else 'short')
                    except Exception as e:
                        print(Fore.RED + f"Error: {e}")
                        print("Usage: option buy/sell SYMBOL call/put STRIKE EXPIRY CONTRACTS PREMIUM")
                elif len(p) >= 2 and p[0] == 'close':
                    portfolio.remove_option(p[1])
                else:
                    print("Usage: option buy/sell SYMBOL call/put STRIKE EXPIRY CONTRACTS PREMIUM")
            elif cmd.startswith('compare '):
                compare(parts[1:], fetcher)
            elif cmd == 'scan':
                scan(fetcher)
            elif cmd.startswith('scan '):
                scan(fetcher, parts[1:])
            elif cmd.startswith('charts '):
                config.show_charts = 'on' in cmd
                print(f"Charts: {'on' if config.show_charts else 'off'}")
            elif cmd.startswith('theme '):
                config.theme = 'dark' if 'dark' in cmd else 'light'
                print(f"Theme: {config.theme}")
            else:
                if parts[0].isalpha() or '.' in parts[0]:
                    analyze(parts[0], fetcher)
                else:
                    print(Fore.YELLOW + f"Unknown command: {inp}. Type 'help' for commands.")

        except KeyboardInterrupt:
            print("\nCancelled")
        except Exception as e:
            print(Fore.RED + f"Error: {e}" + Style.RESET_ALL)

    print(Fore.GREEN + "\nGoodbye!" + Style.RESET_ALL)


if __name__ == "__main__":
    main()