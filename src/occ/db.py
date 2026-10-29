import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from occ.config import load_config, get_root_dir

def get_db_path(cfg: Optional[Dict[str, Any]] = None) -> Path:
    if cfg is None:
        cfg = load_config()
    db_path = Path(cfg["paths"]["db_path"])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path

def get_connection(cfg: Optional[Dict[str, Any]] = None) -> sqlite3.Connection:
    path = get_db_path(cfg)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db(cfg: Optional[Dict[str, Any]] = None) -> None:
    """Initialize database tables according to schema.sql."""
    schema_path = get_root_dir() / "db" / "schema.sql"
    if not schema_path.is_file():
        raise FileNotFoundError(f"Schema file not found at {schema_path}")
    
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()
    
    with get_connection(cfg) as conn:
        conn.executescript(schema_sql)
        conn.commit()

def insert_run(meta: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> None:
    """Insert or replace a run record in the runs table."""
    with get_connection(cfg) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO runs (
                run_id, day_id, date, mode, origin, seed,
                scenario_family, door_state, furniture_config,
                tx_device, notes, meta_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                meta.get("run_id"),
                meta.get("day_id"),
                meta.get("date"),
                meta.get("mode", "collect"),
                meta.get("origin", "real"),
                meta.get("seed"),
                meta.get("scenario_family"),
                meta.get("door_state", "closed"),
                meta.get("furniture_config", "F1"),
                meta.get("tx_device", "ESP32"),
                meta.get("notes", ""),
                json.dumps(meta)
            )
        )
        conn.commit()

def insert_raw_rssi_df(df: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None) -> None:
    """Bulk insert raw RSSI time-series DataFrame into SQLite."""
    with get_connection(cfg) as conn:
        df.to_sql("raw_rssi", conn, if_exists="append", index=False)

def insert_features_df(df: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None) -> None:
    """Bulk insert extracted features DataFrame into SQLite."""
    with get_connection(cfg) as conn:
        df.to_sql("features", conn, if_exists="append", index=False)

def log_prediction(
    run_id: str,
    timestamp: float,
    predicted_label: str,
    confidence: float,
    voted_label: str,
    latency_ms: float,
    origin: str = "real",
    cfg: Optional[Dict[str, Any]] = None
) -> None:
    """Log single window prediction event."""
    with get_connection(cfg) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO runs (run_id, origin, mode)
            VALUES (?, ?, 'live')
            """,
            (run_id, origin)
        )
        conn.execute(
            """
            INSERT INTO predictions (
                timestamp, run_id, predicted_label, confidence,
                voted_label, latency_ms, origin
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (timestamp, run_id, predicted_label, confidence, voted_label, latency_ms, origin)
        )
        conn.commit()

def fetch_runs(mode: Optional[str] = None, origin: Optional[str] = None, cfg: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Fetch runs metadata as a DataFrame."""
    query = "SELECT * FROM runs WHERE 1=1"
    params = []
    if mode:
        query += " AND mode = ?"
        params.append(mode)
    if origin:
        query += " AND origin = ?"
        params.append(origin)
    query += " ORDER BY rowid DESC"
    with get_connection(cfg) as conn:
        return pd.read_sql_query(query, conn, params=params)

def fetch_predictions(run_id: Optional[str] = None, limit: int = 100, cfg: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Fetch recent predictions for monitoring."""
    query = "SELECT * FROM predictions"
    params = []
    if run_id:
        query += " WHERE run_id = ?"
        params.append(run_id)
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    with get_connection(cfg) as conn:
        return pd.read_sql_query(query, conn, params=params)
