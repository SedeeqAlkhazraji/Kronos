import requests
import pandas as pd
import os
import time

BINANCE_BASE_URL = "https://api.binance.com"

POPULAR_SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
    'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'DOTUSDT', 'MATICUSDT',
    'LINKUSDT', 'LTCUSDT', 'ATOMUSDT', 'UNIUSDT', 'APTUSDT',
]

INTERVAL_MAP = {
    '1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m', '30m': '30m',
    '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h', '12h': '12h',
    '1d': '1d', '3d': '3d', '1w': '1w', '1M': '1M',
}


def fetch_klines(symbol, interval='1h', limit=500):
    """
    Fetch OHLCV candlestick data from Binance public API.

    Args:
        symbol: Trading pair, e.g. 'BTCUSDT'
        interval: Candle interval ('1m','5m','15m','1h','4h','1d', etc.)
        limit: Number of candles to fetch (max 1000)

    Returns:
        pd.DataFrame with columns: timestamps, open, high, low, close, volume, amount
    """
    url = f"{BINANCE_BASE_URL}/api/v3/klines"
    params = {
        'symbol': symbol.upper(),
        'interval': INTERVAL_MAP.get(interval, interval),
        'limit': min(limit, 1000),
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for candle in data:
        rows.append({
            'timestamps': pd.to_datetime(candle[0], unit='ms'),
            'open': float(candle[1]),
            'high': float(candle[2]),
            'low': float(candle[3]),
            'close': float(candle[4]),
            'volume': float(candle[5]),
            'amount': float(candle[7]),  # quote asset volume
        })

    df = pd.DataFrame(rows)
    return df


def fetch_klines_extended(symbol, interval='1h', total_limit=2000):
    """
    Fetch more than 1000 candles by making multiple paginated requests.
    """
    all_data = []
    end_time = None

    while len(all_data) < total_limit:
        batch_limit = min(1000, total_limit - len(all_data))
        url = f"{BINANCE_BASE_URL}/api/v3/klines"
        params = {
            'symbol': symbol.upper(),
            'interval': INTERVAL_MAP.get(interval, interval),
            'limit': batch_limit,
        }
        if end_time:
            params['endTime'] = end_time

        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            break

        for candle in data:
            all_data.append({
                'timestamps': pd.to_datetime(candle[0], unit='ms'),
                'open': float(candle[1]),
                'high': float(candle[2]),
                'low': float(candle[3]),
                'close': float(candle[4]),
                'volume': float(candle[5]),
                'amount': float(candle[7]),
            })

        end_time = int(data[0][0]) - 1
        if len(data) < batch_limit:
            break
        time.sleep(0.1)

    df = pd.DataFrame(all_data)
    if not df.empty:
        df = df.sort_values('timestamps').drop_duplicates(subset='timestamps').reset_index(drop=True)
    return df


def get_current_price(symbol):
    """Get the latest price for a symbol."""
    url = f"{BINANCE_BASE_URL}/api/v3/ticker/price"
    resp = requests.get(url, params={'symbol': symbol.upper()}, timeout=10)
    resp.raise_for_status()
    return float(resp.json()['price'])


def get_available_symbols():
    """Return list of popular trading pairs with metadata."""
    results = []
    for sym in POPULAR_SYMBOLS:
        results.append({
            'symbol': sym,
            'display_name': _format_display_name(sym),
        })
    return results


def save_klines_to_csv(df, symbol, interval, data_dir=None):
    """Save fetched klines as CSV in the data/ directory for Kronos to load."""
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    os.makedirs(data_dir, exist_ok=True)

    filename = f"{symbol}_{interval}_{len(df)}.csv"
    filepath = os.path.join(data_dir, filename)
    df.to_csv(filepath, index=False)
    return filepath


def _format_display_name(symbol):
    s = symbol.upper()
    for quote in ['USDT', 'BUSD', 'USDC', 'BTC', 'ETH', 'BNB']:
        if s.endswith(quote) and len(s) > len(quote):
            return f"{s[:-len(quote)]}/{quote}"
    return s
