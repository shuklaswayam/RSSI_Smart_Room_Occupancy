-- SQLite schema for Wi-Fi RSSI Room Occupancy System
-- Supports both real-hardware runs and calibrated NS2 simulation traces

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    day_id TEXT,
    date TEXT,
    mode TEXT CHECK (mode IN ('collect', 'live', 'replay', 'sim')),
    origin TEXT CHECK (origin IN ('real', 'sim')) DEFAULT 'real',
    seed INTEGER,
    scenario_family TEXT,
    door_state TEXT,
    furniture_config TEXT,
    tx_device TEXT,
    notes TEXT,
    meta_json TEXT
);

CREATE TABLE IF NOT EXISTS raw_rssi (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    t_host REAL,
    receiver_id TEXT,
    source_id TEXT,
    technology TEXT DEFAULT 'wifi',
    channel INTEGER,
    rssi_dbm REAL,
    esp_ms INTEGER,
    origin TEXT DEFAULT 'real',
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS features (
    window_id TEXT PRIMARY KEY,
    run_id TEXT,
    t_start REAL,
    t_end REAL,
    origin TEXT DEFAULT 'real',
    seed INTEGER,
    label TEXT,
    -- Receiver 1 single link
    R1_mean REAL,
    R1_std REAL,
    R1_min REAL,
    R1_max REAL,
    R1_iqr REAL,
    R1_slope REAL,
    R1_zcr REAL,
    -- Receiver 2 single link
    R2_mean REAL,
    R2_std REAL,
    R2_min REAL,
    R2_max REAL,
    R2_iqr REAL,
    R2_slope REAL,
    R2_zcr REAL,
    -- Cross-receiver differential
    R1R2_mean_diff REAL,
    R1R2_std_diff REAL,
    -- Baseline-relative shifts
    R1_delta_mean REAL,
    R2_delta_mean REAL,
    R1_delta_min REAL,
    R2_delta_min REAL,
    R1_delta_max REAL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL,
    run_id TEXT,
    predicted_label TEXT,
    confidence REAL,
    voted_label TEXT,
    latency_ms REAL,
    origin TEXT DEFAULT 'real',
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_raw_rssi_run ON raw_rssi(run_id);
CREATE INDEX IF NOT EXISTS idx_features_run ON features(run_id);
CREATE INDEX IF NOT EXISTS idx_features_origin ON features(origin);
CREATE INDEX IF NOT EXISTS idx_predictions_run ON predictions(run_id);
