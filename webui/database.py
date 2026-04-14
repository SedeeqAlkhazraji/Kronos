import sqlite3
import os
import json
import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kronos_invest.db')


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL UNIQUE,
            display_name TEXT,
            interval TEXT DEFAULT '1h',
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity REAL NOT NULL,
            entry_price REAL NOT NULL,
            entry_date TIMESTAMP NOT NULL,
            exit_price REAL,
            exit_date TIMESTAMP,
            status TEXT DEFAULT 'open',
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            confidence REAL,
            current_price REAL,
            predicted_price REAL,
            price_change_pct REAL,
            pred_len INTEGER,
            model_used TEXT,
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            alert_type TEXT NOT NULL,
            condition TEXT,
            is_active INTEGER DEFAULT 1,
            last_triggered TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS alert_notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_id INTEGER REFERENCES alerts(id),
            message TEXT NOT NULL,
            signal_id INTEGER,
            seen INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS backtests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            strategy TEXT NOT NULL,
            start_date TIMESTAMP,
            end_date TIMESTAMP,
            initial_capital REAL DEFAULT 10000,
            final_capital REAL,
            total_return_pct REAL,
            sharpe_ratio REAL,
            max_drawdown_pct REAL,
            win_rate REAL,
            total_trades INTEGER,
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    conn.commit()
    conn.close()


# --------------- Watchlist CRUD ---------------

def add_to_watchlist(symbol, display_name=None, interval='1h'):
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO watchlist (symbol, display_name, interval) VALUES (?, ?, ?)",
            (symbol.upper(), display_name or _format_display_name(symbol), interval)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def remove_from_watchlist(watchlist_id):
    conn = get_connection()
    conn.execute("DELETE FROM watchlist WHERE id = ?", (watchlist_id,))
    conn.commit()
    conn.close()


def get_watchlist():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM watchlist ORDER BY added_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# --------------- Positions CRUD ---------------

def add_position(symbol, side, quantity, entry_price, entry_date, notes=None):
    conn = get_connection()
    conn.execute(
        "INSERT INTO positions (symbol, side, quantity, entry_price, entry_date, notes) VALUES (?, ?, ?, ?, ?, ?)",
        (symbol.upper(), side.lower(), quantity, entry_price, entry_date, notes)
    )
    conn.commit()
    conn.close()


def get_positions(status=None):
    conn = get_connection()
    if status:
        rows = conn.execute("SELECT * FROM positions WHERE status = ? ORDER BY created_at DESC", (status,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM positions ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_position(position_id, exit_price, exit_date):
    conn = get_connection()
    conn.execute(
        "UPDATE positions SET exit_price = ?, exit_date = ?, status = 'closed' WHERE id = ?",
        (exit_price, exit_date, position_id)
    )
    conn.commit()
    conn.close()


def delete_position(position_id):
    conn = get_connection()
    conn.execute("DELETE FROM positions WHERE id = ?", (position_id,))
    conn.commit()
    conn.close()


# --------------- Signals CRUD ---------------

def save_signal(symbol, signal_type, confidence, current_price, predicted_price,
                price_change_pct, pred_len, model_used, details=None):
    conn = get_connection()
    conn.execute(
        """INSERT INTO signals (symbol, signal_type, confidence, current_price, predicted_price,
           price_change_pct, pred_len, model_used, details) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (symbol.upper(), signal_type, confidence, current_price, predicted_price,
         price_change_pct, pred_len, model_used, json.dumps(details) if details else None)
    )
    conn.commit()
    conn.close()


def get_signals(symbol=None, limit=50):
    conn = get_connection()
    if symbol:
        rows = conn.execute(
            "SELECT * FROM signals WHERE symbol = ? ORDER BY created_at DESC LIMIT ?",
            (symbol.upper(), limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM signals ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_latest_signal(symbol):
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM signals WHERE symbol = ? ORDER BY created_at DESC LIMIT 1",
        (symbol.upper(),)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


# --------------- Alerts CRUD ---------------

def create_alert(symbol, alert_type, condition):
    conn = get_connection()
    conn.execute(
        "INSERT INTO alerts (symbol, alert_type, condition) VALUES (?, ?, ?)",
        (symbol.upper(), alert_type, json.dumps(condition) if isinstance(condition, dict) else condition)
    )
    conn.commit()
    conn.close()


def get_alerts(active_only=False):
    conn = get_connection()
    if active_only:
        rows = conn.execute("SELECT * FROM alerts WHERE is_active = 1 ORDER BY created_at DESC").fetchall()
    else:
        rows = conn.execute("SELECT * FROM alerts ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_alert(alert_id, is_active):
    conn = get_connection()
    conn.execute("UPDATE alerts SET is_active = ? WHERE id = ?", (is_active, alert_id))
    conn.commit()
    conn.close()


def delete_alert(alert_id):
    conn = get_connection()
    conn.execute("DELETE FROM alert_notifications WHERE alert_id = ?", (alert_id,))
    conn.execute("DELETE FROM alerts WHERE id = ?", (alert_id,))
    conn.commit()
    conn.close()


def trigger_alert(alert_id, message, signal_id=None):
    conn = get_connection()
    conn.execute(
        "INSERT INTO alert_notifications (alert_id, message, signal_id) VALUES (?, ?, ?)",
        (alert_id, message, signal_id)
    )
    conn.execute(
        "UPDATE alerts SET last_triggered = ? WHERE id = ?",
        (datetime.datetime.now().isoformat(), alert_id)
    )
    conn.commit()
    conn.close()


def get_notifications(unseen_only=False, limit=50):
    conn = get_connection()
    if unseen_only:
        rows = conn.execute(
            "SELECT n.*, a.symbol, a.alert_type FROM alert_notifications n JOIN alerts a ON n.alert_id = a.id WHERE n.seen = 0 ORDER BY n.created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT n.*, a.symbol, a.alert_type FROM alert_notifications n JOIN alerts a ON n.alert_id = a.id ORDER BY n.created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def mark_notification_seen(notification_id):
    conn = get_connection()
    conn.execute("UPDATE alert_notifications SET seen = 1 WHERE id = ?", (notification_id,))
    conn.commit()
    conn.close()


# --------------- Backtests CRUD ---------------

def save_backtest(symbol, strategy, start_date, end_date, initial_capital,
                  final_capital, total_return_pct, sharpe_ratio, max_drawdown_pct,
                  win_rate, total_trades, details=None):
    conn = get_connection()
    conn.execute(
        """INSERT INTO backtests (symbol, strategy, start_date, end_date, initial_capital,
           final_capital, total_return_pct, sharpe_ratio, max_drawdown_pct, win_rate,
           total_trades, details) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (symbol.upper(), strategy, start_date, end_date, initial_capital,
         final_capital, total_return_pct, sharpe_ratio, max_drawdown_pct,
         win_rate, total_trades, json.dumps(details) if details else None)
    )
    conn.commit()
    conn.close()


def get_backtests(symbol=None, limit=20):
    conn = get_connection()
    if symbol:
        rows = conn.execute(
            "SELECT * FROM backtests WHERE symbol = ? ORDER BY created_at DESC LIMIT ?",
            (symbol.upper(), limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM backtests ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_backtest_by_id(backtest_id):
    conn = get_connection()
    row = conn.execute("SELECT * FROM backtests WHERE id = ?", (backtest_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


# --------------- Helpers ---------------

def _format_display_name(symbol):
    s = symbol.upper()
    for quote in ['USDT', 'BUSD', 'USDC', 'BTC', 'ETH', 'BNB']:
        if s.endswith(quote) and len(s) > len(quote):
            return f"{s[:-len(quote)]}/{quote}"
    return s
