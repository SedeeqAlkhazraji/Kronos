import json
import threading
from apscheduler.schedulers.background import BackgroundScheduler

import database as db
import data_fetcher

_scheduler = None
_lock = threading.Lock()


def start_scheduler():
    global _scheduler
    with _lock:
        if _scheduler is None or not _scheduler.running:
            _scheduler = BackgroundScheduler(daemon=True)
            _scheduler.add_job(_check_alerts, 'interval', minutes=5, id='alert_checker', replace_existing=True)
            _scheduler.start()


def stop_scheduler():
    global _scheduler
    with _lock:
        if _scheduler and _scheduler.running:
            _scheduler.shutdown(wait=False)
            _scheduler = None


def _check_alerts():
    """Periodic job: evaluate all active alerts against current market data."""
    alerts = db.get_alerts(active_only=True)
    for alert in alerts:
        try:
            _evaluate_alert(alert)
        except Exception as e:
            print(f"Alert check failed for alert {alert['id']}: {e}")


def _evaluate_alert(alert):
    """Evaluate a single alert and trigger notification if conditions are met."""
    alert_type = alert['alert_type']
    condition = json.loads(alert['condition']) if isinstance(alert['condition'], str) else (alert['condition'] or {})
    symbol = alert['symbol']

    if alert_type == 'price_threshold':
        _check_price_threshold(alert, symbol, condition)
    elif alert_type == 'signal_change':
        _check_signal_change(alert, symbol, condition)


def _check_price_threshold(alert, symbol, condition):
    """Check if price has crossed a threshold."""
    try:
        current_price = data_fetcher.get_current_price(symbol)
    except Exception:
        return

    direction = condition.get('direction', 'above')
    target_price = float(condition.get('target_price', 0))

    triggered = False
    if direction == 'above' and current_price >= target_price:
        triggered = True
        message = f"{symbol} price ${current_price:.2f} is above ${target_price:.2f}"
    elif direction == 'below' and current_price <= target_price:
        triggered = True
        message = f"{symbol} price ${current_price:.2f} is below ${target_price:.2f}"

    if triggered:
        db.trigger_alert(alert['id'], message)


def _check_signal_change(alert, symbol, condition):
    """Check if the latest signal matches the alert condition."""
    latest_signal = db.get_latest_signal(symbol)
    if not latest_signal:
        return

    target_direction = condition.get('direction', '').upper()
    min_confidence = float(condition.get('min_confidence', 0.5))

    if latest_signal['signal_type'] == target_direction and latest_signal['confidence'] >= min_confidence:
        message = (
            f"{symbol} signal: {latest_signal['signal_type']} "
            f"(confidence: {latest_signal['confidence']:.0%}, "
            f"predicted change: {latest_signal['price_change_pct']:+.2f}%)"
        )
        # Avoid duplicate notifications for the same signal
        recent = db.get_notifications(unseen_only=False, limit=5)
        for n in recent:
            if n.get('signal_id') == latest_signal['id']:
                return
        db.trigger_alert(alert['id'], message, signal_id=latest_signal['id'])
