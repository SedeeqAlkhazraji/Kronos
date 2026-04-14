import numpy as np


def generate_signal(current_close, predicted_candles, buy_threshold=0.02, sell_threshold=-0.02):
    """
    Analyze Kronos-predicted candles to produce a BUY / SELL / HOLD signal with confidence.

    Args:
        current_close: The last known close price (float).
        predicted_candles: np.ndarray or list of dicts with 'open','high','low','close' keys,
                          representing the predicted future candles.
        buy_threshold: Minimum predicted % change to trigger BUY (default 2%).
        sell_threshold: Maximum predicted % change to trigger SELL (default -2%).

    Returns:
        dict with keys:
            signal: 'BUY' | 'SELL' | 'HOLD'
            confidence: float 0.0 - 1.0
            predicted_close: float (mean predicted close)
            price_change_pct: float (percentage change from current)
            details: dict with sub-metrics
    """
    pred_closes = _extract_closes(predicted_candles)
    pred_highs = _extract_column(predicted_candles, 'high', 1)
    pred_lows = _extract_column(predicted_candles, 'low', 2)

    if len(pred_closes) == 0 or current_close <= 0:
        return _hold_signal(current_close)

    mean_pred_close = float(np.mean(pred_closes))
    last_pred_close = float(pred_closes[-1])
    price_change_pct = (mean_pred_close - current_close) / current_close

    # --- Factor 1: Trend consistency (40% weight) ---
    if price_change_pct > 0:
        trend_count = np.sum(pred_closes > current_close)
    else:
        trend_count = np.sum(pred_closes < current_close)
    trend_consistency = float(trend_count / len(pred_closes))

    # --- Factor 2: Magnitude above threshold (30% weight) ---
    abs_change = abs(price_change_pct)
    threshold = abs(buy_threshold)
    if abs_change <= threshold:
        magnitude_score = abs_change / threshold * 0.5
    else:
        magnitude_score = min(1.0, 0.5 + (abs_change - threshold) / (threshold * 4) * 0.5)

    # --- Factor 3: Volatility penalty (30% weight) ---
    pred_ranges = pred_highs - pred_lows
    mean_range = float(np.mean(pred_ranges)) if len(pred_ranges) > 0 else 0
    relative_vol = mean_range / current_close if current_close > 0 else 0
    vol_penalty = max(0.0, 1.0 - relative_vol * 10)

    # --- Combined confidence ---
    confidence = (
        0.40 * trend_consistency +
        0.30 * magnitude_score +
        0.30 * vol_penalty
    )
    confidence = round(max(0.0, min(1.0, confidence)), 4)

    # --- Signal classification ---
    if price_change_pct > buy_threshold:
        signal = 'BUY'
    elif price_change_pct < sell_threshold:
        signal = 'SELL'
    else:
        signal = 'HOLD'

    return {
        'signal': signal,
        'confidence': confidence,
        'predicted_close': round(mean_pred_close, 6),
        'last_predicted_close': round(last_pred_close, 6),
        'price_change_pct': round(price_change_pct * 100, 4),
        'details': {
            'trend_consistency': round(trend_consistency, 4),
            'magnitude_score': round(magnitude_score, 4),
            'volatility_penalty': round(vol_penalty, 4),
            'mean_predicted_high': round(float(np.mean(pred_highs)), 6) if len(pred_highs) > 0 else None,
            'mean_predicted_low': round(float(np.mean(pred_lows)), 6) if len(pred_lows) > 0 else None,
            'predicted_range_pct': round(relative_vol * 100, 4),
            'candles_above_current': int(np.sum(pred_closes > current_close)),
            'candles_below_current': int(np.sum(pred_closes < current_close)),
            'total_predicted_candles': len(pred_closes),
            'buy_threshold_pct': round(buy_threshold * 100, 2),
            'sell_threshold_pct': round(sell_threshold * 100, 2),
        }
    }


def _extract_closes(predicted_candles):
    """Extract close prices from predicted candles (DataFrame, dict list, or ndarray)."""
    if hasattr(predicted_candles, 'values'):
        # DataFrame
        if 'close' in predicted_candles.columns:
            return predicted_candles['close'].values.astype(float)
        else:
            return predicted_candles.iloc[:, 3].values.astype(float)
    elif isinstance(predicted_candles, np.ndarray):
        if predicted_candles.ndim == 2:
            return predicted_candles[:, 3].astype(float)
        return predicted_candles.astype(float)
    elif isinstance(predicted_candles, list) and len(predicted_candles) > 0:
        if isinstance(predicted_candles[0], dict):
            return np.array([c['close'] for c in predicted_candles], dtype=float)
    return np.array([], dtype=float)


def _extract_column(predicted_candles, col_name, col_idx):
    """Extract a named column from predicted candles."""
    if hasattr(predicted_candles, 'values'):
        if col_name in predicted_candles.columns:
            return predicted_candles[col_name].values.astype(float)
        elif predicted_candles.shape[1] > col_idx:
            return predicted_candles.iloc[:, col_idx].values.astype(float)
    elif isinstance(predicted_candles, np.ndarray):
        if predicted_candles.ndim == 2 and predicted_candles.shape[1] > col_idx:
            return predicted_candles[:, col_idx].astype(float)
    elif isinstance(predicted_candles, list) and len(predicted_candles) > 0:
        if isinstance(predicted_candles[0], dict) and col_name in predicted_candles[0]:
            return np.array([c[col_name] for c in predicted_candles], dtype=float)
    return np.array([], dtype=float)


def _hold_signal(current_close):
    return {
        'signal': 'HOLD',
        'confidence': 0.0,
        'predicted_close': current_close,
        'last_predicted_close': current_close,
        'price_change_pct': 0.0,
        'details': {}
    }
