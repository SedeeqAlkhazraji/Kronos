import numpy as np
import pandas as pd
from signal_engine import generate_signal


def run_backtest(historical_df, predictor, pred_len=120, lookback=400,
                 strategy='threshold', initial_capital=10000.0,
                 buy_threshold=0.02, sell_threshold=-0.02, min_confidence=0.5,
                 temperature=1.0, top_p=0.9, sample_count=1, step_size=None):
    """
    Walk-forward backtest using Kronos predictions on historical crypto data.

    Args:
        historical_df: DataFrame with timestamps, open, high, low, close, volume, amount.
        predictor: KronosPredictor instance (already loaded with model).
        pred_len: Number of candles to predict at each step.
        lookback: Number of historical candles to feed the model.
        strategy: 'threshold' | 'always_follow' | 'conservative'.
        initial_capital: Starting capital in USD.
        buy_threshold: % change threshold for buy signal.
        sell_threshold: % change threshold for sell signal.
        min_confidence: Minimum confidence to execute a trade.
        temperature: Sampling temperature for model.
        top_p: Nucleus sampling parameter.
        sample_count: Number of samples to average.
        step_size: How many candles to advance between predictions. Defaults to pred_len.

    Returns:
        dict with backtest results and metrics.
    """
    if step_size is None:
        step_size = pred_len

    if strategy == 'conservative':
        min_confidence = max(min_confidence, 0.8)
    elif strategy == 'always_follow':
        min_confidence = 0.0

    df = historical_df.copy().reset_index(drop=True)
    total_rows = len(df)

    if total_rows < lookback + pred_len:
        return {'error': f'Insufficient data: need {lookback + pred_len} rows, have {total_rows}'}

    capital = initial_capital
    position = None  # None = no position, dict = active position
    trades = []
    equity_curve = [{'index': 0, 'capital': capital, 'timestamp': str(df['timestamps'].iloc[0])}]

    i = lookback
    while i + pred_len <= total_rows:
        x_df = df.iloc[i - lookback:i]
        x_timestamps = df['timestamps'].iloc[i - lookback:i]
        y_timestamps = df['timestamps'].iloc[i:i + pred_len]

        current_close = float(df['close'].iloc[i - 1])

        try:
            pred_df = predictor.predict(
                df=x_df[['open', 'high', 'low', 'close', 'volume', 'amount']],
                x_timestamp=pd.Series(x_timestamps.values, name='timestamps'),
                y_timestamp=pd.Series(y_timestamps.values, name='timestamps'),
                pred_len=pred_len,
                T=temperature,
                top_p=top_p,
                sample_count=sample_count,
            )
        except Exception as e:
            i += step_size
            continue

        signal_result = generate_signal(current_close, pred_df, buy_threshold, sell_threshold)
        signal_type = signal_result['signal']
        confidence = signal_result['confidence']

        # Determine the actual price at the end of the prediction window
        actual_future_close = float(df['close'].iloc[min(i + pred_len - 1, total_rows - 1)])

        # --- Trading logic ---
        if position is None:
            # No position: look for BUY signal
            if signal_type == 'BUY' and confidence >= min_confidence:
                position = {
                    'entry_price': current_close,
                    'entry_index': i,
                    'entry_timestamp': str(df['timestamps'].iloc[i - 1]),
                    'signal_confidence': confidence,
                }
        else:
            # Have position: look for SELL signal or exit
            if signal_type == 'SELL' and confidence >= min_confidence:
                exit_price = current_close
                pnl = (exit_price - position['entry_price']) / position['entry_price']
                trade_return = capital * pnl
                capital += trade_return

                trades.append({
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'entry_timestamp': position['entry_timestamp'],
                    'exit_timestamp': str(df['timestamps'].iloc[i - 1]),
                    'pnl_pct': round(pnl * 100, 4),
                    'pnl_usd': round(trade_return, 2),
                    'confidence': position['signal_confidence'],
                })
                position = None

        equity_curve.append({
            'index': i,
            'capital': round(capital, 2),
            'timestamp': str(df['timestamps'].iloc[min(i, total_rows - 1)]),
        })

        i += step_size

    # Close any remaining position at the end
    if position is not None:
        exit_price = float(df['close'].iloc[-1])
        pnl = (exit_price - position['entry_price']) / position['entry_price']
        trade_return = capital * pnl
        capital += trade_return
        trades.append({
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'entry_timestamp': position['entry_timestamp'],
            'exit_timestamp': str(df['timestamps'].iloc[-1]),
            'pnl_pct': round(pnl * 100, 4),
            'pnl_usd': round(trade_return, 2),
            'confidence': position['signal_confidence'],
        })

    return _calculate_metrics(initial_capital, capital, trades, equity_curve, df)


def _calculate_metrics(initial_capital, final_capital, trades, equity_curve, df):
    """Calculate backtest performance metrics."""
    total_return_pct = ((final_capital - initial_capital) / initial_capital) * 100
    total_trades = len(trades)

    if total_trades > 0:
        pnl_values = [t['pnl_pct'] for t in trades]
        winning_trades = [p for p in pnl_values if p > 0]
        win_rate = len(winning_trades) / total_trades

        gross_profit = sum(p for p in pnl_values if p > 0)
        gross_loss = abs(sum(p for p in pnl_values if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    else:
        win_rate = 0.0
        profit_factor = 0.0

    # Sharpe ratio (from equity curve)
    capitals = [e['capital'] for e in equity_curve]
    if len(capitals) > 1:
        returns = np.diff(capitals) / capitals[:-1]
        sharpe_ratio = float(np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252))
    else:
        sharpe_ratio = 0.0

    # Max drawdown
    peak = capitals[0]
    max_drawdown = 0.0
    for c in capitals:
        if c > peak:
            peak = c
        drawdown = (peak - c) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    return {
        'initial_capital': initial_capital,
        'final_capital': round(final_capital, 2),
        'total_return_pct': round(total_return_pct, 4),
        'sharpe_ratio': round(sharpe_ratio, 4),
        'max_drawdown_pct': round(max_drawdown * 100, 4),
        'win_rate': round(win_rate, 4),
        'total_trades': total_trades,
        'profit_factor': round(profit_factor, 4),
        'trades': trades,
        'equity_curve': equity_curve,
        'start_date': str(df['timestamps'].iloc[0]) if 'timestamps' in df.columns else None,
        'end_date': str(df['timestamps'].iloc[-1]) if 'timestamps' in df.columns else None,
    }
