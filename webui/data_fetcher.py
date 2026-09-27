import requests
import pandas as pd
import numpy as np
import os
import time

COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

SYMBOL_TO_CG_ID = {
    'BTCUSDT': 'bitcoin',
    'ETHUSDT': 'ethereum',
    'BNBUSDT': 'binancecoin',
    'SOLUSDT': 'solana',
    'XRPUSDT': 'ripple',
    'DOGEUSDT': 'dogecoin',
    'ADAUSDT': 'cardano',
    'AVAXUSDT': 'avalanche-2',
    'DOTUSDT': 'polkadot',
    'MATICUSDT': 'matic-network',
    'LINKUSDT': 'chainlink',
    'LTCUSDT': 'litecoin',
    'ATOMUSDT': 'cosmos',
    'UNIUSDT': 'uniswap',
    'APTUSDT': 'aptos',
}

POPULAR_SYMBOLS = list(SYMBOL_TO_CG_ID.keys())

INTERVAL_TO_DAYS = {
    '5m': 1,
    '15m': 3,
    '30m': 7,
    '1h': 30,
    '4h': 90,
    '1d': 365,
}

INTERVAL_TO_SECONDS = {
    '5m': 300,
    '15m': 900,
    '30m': 1800,
    '1h': 3600,
    '4h': 14400,
    '1d': 86400,
}


def fetch_klines(symbol, interval='1h', limit=500):
    """
    Fetch OHLCV data from CoinGecko /market_chart endpoint and build
    candles from the granular price+volume data.
    """
    cg_id = SYMBOL_TO_CG_ID.get(symbol.upper())
    if not cg_id:
        raise ValueError(f"Unknown symbol: {symbol}")

    days = INTERVAL_TO_DAYS.get(interval, 30)

    # Use market_chart for accurate price data
    url = f"{COINGECKO_BASE_URL}/coins/{cg_id}/market_chart"
    params = {'vs_currency': 'usd', 'days': days}

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    prices = data.get('prices', [])
    volumes = data.get('total_volumes', [])

    if not prices:
        return pd.DataFrame()

    # Build price DataFrame
    price_df = pd.DataFrame(prices, columns=['ts', 'price'])
    price_df['ts'] = pd.to_datetime(price_df['ts'], unit='ms')

    vol_df = pd.DataFrame(volumes, columns=['ts', 'volume'])
    vol_df['ts'] = pd.to_datetime(vol_df['ts'], unit='ms')

    # Resample into OHLCV candles at the requested interval
    freq_map = {'5m': '5min', '15m': '15min', '30m': '30min', '1h': '1h', '4h': '4h', '1d': '1D'}
    freq = freq_map.get(interval, '1h')

    price_df = price_df.set_index('ts')
    vol_df = vol_df.set_index('ts')

    ohlc = price_df['price'].resample(freq).ohlc()
    ohlc.columns = ['open', 'high', 'low', 'close']

    vol_resampled = vol_df['volume'].resample(freq).sum()

    df = ohlc.join(vol_resampled)
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    df = df.reset_index()
    df = df.rename(columns={'ts': 'timestamps'})
    df['amount'] = df['volume'] * df['close']

    # Trim to requested limit
    if len(df) > limit:
        df = df.tail(limit).reset_index(drop=True)

    return df


def fetch_klines_extended(symbol, interval='1h', total_limit=2000):
    """Fetch extended data by requesting more days."""
    cg_id = SYMBOL_TO_CG_ID.get(symbol.upper())
    if not cg_id:
        raise ValueError(f"Unknown symbol: {symbol}")

    days_map = {'5m': 1, '15m': 7, '30m': 14, '1h': 90, '4h': 180, '1d': 365}
    days = days_map.get(interval, 90)

    url = f"{COINGECKO_BASE_URL}/coins/{cg_id}/market_chart"
    params = {'vs_currency': 'usd', 'days': days}

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    prices = data.get('prices', [])
    volumes = data.get('total_volumes', [])

    if not prices:
        return pd.DataFrame()

    price_df = pd.DataFrame(prices, columns=['ts', 'price'])
    price_df['ts'] = pd.to_datetime(price_df['ts'], unit='ms')

    vol_df = pd.DataFrame(volumes, columns=['ts', 'volume'])
    vol_df['ts'] = pd.to_datetime(vol_df['ts'], unit='ms')

    freq_map = {'5m': '5min', '15m': '15min', '30m': '30min', '1h': '1h', '4h': '4h', '1d': '1D'}
    freq = freq_map.get(interval, '1h')

    price_df = price_df.set_index('ts')
    vol_df = vol_df.set_index('ts')

    ohlc = price_df['price'].resample(freq).ohlc()
    ohlc.columns = ['open', 'high', 'low', 'close']
    vol_resampled = vol_df['volume'].resample(freq).sum()

    df = ohlc.join(vol_resampled)
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    df = df.reset_index()
    df = df.rename(columns={'ts': 'timestamps'})
    df['amount'] = df['volume'] * df['close']

    return df


def get_current_price(symbol):
    """Get the latest verified price."""
    cg_id = SYMBOL_TO_CG_ID.get(symbol.upper())
    if not cg_id:
        raise ValueError(f"Unknown symbol: {symbol}")

    url = f"{COINGECKO_BASE_URL}/simple/price"
    params = {'ids': cg_id, 'vs_currencies': 'usd'}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return float(resp.json()[cg_id]['usd'])


def get_available_symbols():
    """Return list of popular trading pairs."""
    results = []
    for sym in POPULAR_SYMBOLS:
        results.append({
            'symbol': sym,
            'display_name': _format_display_name(sym),
        })
    return results


def save_klines_to_csv(df, symbol, interval, data_dir=None):
    """Save fetched klines as CSV."""
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
