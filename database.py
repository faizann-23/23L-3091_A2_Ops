"""Task 5: permanent audit trail of every prediction.

Two back-ends, chosen with the DB_BACKEND environment variable:
  * sqlite   (default) - zero-setup, used for local `python app.py` runs and tests
  * postgres           - used by the Docker Compose stack, where the web
                         container reaches the `db` service by its service name
"""
import os
import sqlite3
import time
from datetime import datetime, timezone

DB_BACKEND = os.getenv("DB_BACKEND", "sqlite").strip().lower()
DATABASE_PATH = os.getenv("DATABASE_PATH", "visionguard.db")

DB_HOST = os.getenv("DB_HOST", "db")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_USER = os.getenv("DB_USER", "")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "")

IS_POSTGRES = DB_BACKEND == "postgres"
# sqlite uses "?" placeholders, psycopg2 uses "%s"
PH = "%s" if IS_POSTGRES else "?"

COLUMNS = ("id", "timestamp", "filename", "verdict", "confidence",
           "raw_score", "model_name", "threshold")


def get_db_connection():
    if IS_POSTGRES:
        import psycopg2  # imported lazily so sqlite users do not need it
        return psycopg2.connect(
            host=DB_HOST, port=DB_PORT, user=DB_USER,
            password=DB_PASSWORD, dbname=DB_NAME, connect_timeout=5,
        )
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _create_table_sql():
    pk = "SERIAL PRIMARY KEY" if IS_POSTGRES else "INTEGER PRIMARY KEY AUTOINCREMENT"
    real = "DOUBLE PRECISION" if IS_POSTGRES else "REAL"
    return f"""
        CREATE TABLE IF NOT EXISTS inferences (
            id {pk},
            timestamp TEXT NOT NULL,
            filename TEXT NOT NULL,
            verdict TEXT NOT NULL,
            confidence {real} NOT NULL,
            raw_score {real} NOT NULL,
            model_name TEXT NOT NULL,
            threshold {real} NOT NULL
        )
    """


def init_db(retries=10, delay=2.0):
    """Create the schema. Retries so the web container survives a slow database start."""
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            conn = get_db_connection()
            try:
                cur = conn.cursor()
                cur.execute(_create_table_sql())
                conn.commit()
            finally:
                conn.close()
            where = f"{DB_HOST}:{DB_PORT}/{DB_NAME}" if IS_POSTGRES else DATABASE_PATH
            print(f"[+] Database ready (backend={DB_BACKEND}, target={where})")
            return
        except Exception as e:  # noqa: BLE001 - we re-raise after the last attempt
            last_error = e
            print(f"[!] Database not ready (attempt {attempt}/{retries}): {e}")
            if not IS_POSTGRES:
                break
            time.sleep(delay)
    raise RuntimeError(f"Could not initialise database: {last_error}")


def log_inference(filename, verdict, confidence, raw_score, model_name, threshold=0.5):
    """Write one row: what was decided, when, by which model, how confidently."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            f"INSERT INTO inferences (timestamp, filename, verdict, confidence, "
            f"raw_score, model_name, threshold) VALUES ({PH}, {PH}, {PH}, {PH}, {PH}, {PH}, {PH})",
            (timestamp, filename, verdict, confidence, raw_score, model_name, threshold),
        )
        conn.commit()
    finally:
        conn.close()


def get_all_inferences(limit=100):
    """Newest first, as plain dicts so templates behave the same on both back-ends."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            f"SELECT {', '.join(COLUMNS)} FROM inferences ORDER BY id DESC LIMIT {PH}",
            (limit,),
        )
        return [dict(zip(COLUMNS, row)) for row in cur.fetchall()]
    finally:
        conn.close()


def check_db():
    """Used by /health: True if the database answers a trivial query."""
    try:
        conn = get_db_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.fetchone()
        finally:
            conn.close()
        return True
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    init_db()
