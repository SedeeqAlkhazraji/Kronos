import requests
import pandas as pd
import os
import time

COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

# Map our symbols to CoinGecko IDs
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

# CoinGecko interval → days mapping for /market_chart endpoint
INTERVAL_TO_DAYS = {
    '5m': 1,        # 1 day  → ~288 points at 5min
    '15m': 3,       # 3 days → ~288 points at 15min (auto granularity)
    '30m': 7,       # 7 days → ~336 points at 30min
    '1h': 30,       # 30 days → ~720 points at hourly
    '4h': 90,       # 90 days → ~540 points at 4h (approximated)
    '1d': 365,      # 365 days → 365 points at daily
}


def fetch_klines(symbol, interval='1h', limit=500):
    """
    Fetch OHLCV candlestick data from CoinGecko OHLC endpoint.

    Args:
        symbol: Trading pair, e.g. 'BTCUSDT'
        interval: Candle interval ('5m','15m','1h','4h','1d')
        limit: Ignored for CoinGecko (data size determined by days param)

    Returns:
        pd.DataFrame with columns: timestamps, open, high, low, close, volume, amount
    """
    cg_id = SYMBOL_TO_CG_ID.get(symbol.upper())
    if not cg_id:
        raise ValueError(f"Unknown symbol: {symbol}. Supported: {list(SYMBOL_TO_CG_ID.keys())}")

    days = INTERVAL_TO_DAYS.get(interval, 30)

    # Use /ohlc endpoint for OHLC data
    url = f"{COINGECKO_BASE_URL}/coins/{cg_id}/ohlc"
    params = {
        'vs_currency': 'usd',
        'days': days,
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
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # CoinGecko OHLC doesn't provide volume, so fetch it from market_chart
    try:
        vol_url = f"{COINGECKO_BASE_URL}/coins/{cg_id}/market_chart"
        vol_params = {'vs_currency': 'usd', 'days': days}
        vol_resp = requests.get(vol_url, params=vol_params, timeout=30)
        vol_resp.raise_for_status()
        vol_data = vol_resp.json()

        if 'total_volumes' in vol_data and len(vol_data['total_volumes']) > 0:
            vol_df = pd.DataFrame(vol_data['total_volumes'], columns=['ts', 'volume'])
            vol_df['ts'] = pd.to_datetime(vol_df['ts'], unit='ms')

            # Merge volume by nearest timestamp
            df['volume'] = 0.0
            df['amount'] = 0.0
            for i, row in df.iterrows():
                diffs = (vol_df['ts'] - row['timestamps']).abs()
                nearest_idx = diffs.idxmin()
                df.at[i, 'volume'] = vol_df.at[nearest_idx, 'volume']
                df.at[i, 'amount'] = vol_df.at[nearest_idx, 'volume'] * row['close']
        else:
            df['volume'] = 0.0
            df['amount'] = 0.0
    except Exception:
        df['volume'] = 0.0
        df['amount'] = 0.0

    df = df.sort_values('timestamps').drop_duplicates(subset='timestamps').reset_index(drop=True)
    return df


def fetch_klines_extended(symbol, interval='1h', total_limit=2000):
    """
    Fetch extended data. CoinGecko doesn't paginate the same way,
    so we request more days to get more data points.
    """
    cg_id = SYMBOL_TO_CG_ID.get(symbol.upper())
    if not cg_id:
        raise ValueError(f"Unknown symbol: {symbol}")

    # Request max days to get as much data as possible
    days_map = {'5m': 1, '15m': 7, '30m': 14, '1h': 90, '4h': 180, '1d': 365}
    days = days_map.get(interval, 90)

    url = f"{COINGECKO_BASE_URL}/coins/{cg_id}/ohlc"
    params = {'vs_currency': 'usd', 'days': days}

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
            'volume': 0.0,
            'amount': 0.0,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('timestamps').drop_duplicates(subset='timestamps').reset_index(drop=True)
    return df


def get_current_price(symbol):
    """Get the latest price for a symbol."""
    cg_id = SYMBOL_TO_CG_ID.get(symbol.upper())
    if not cg_id:
        raise ValueError(f"Unknown symbol: {symbol}")

    url = f"{COINGECKO_BASE_URL}/simple/price"
    params = {'ids': cg_id, 'vs_currencies': 'usd'}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return float(resp.json()[cg_id]['usd'])


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
