# Wi-Fi Signal Strength–Based Smart Room Occupancy Detector

A camera-free, privacy-preserving smart room occupancy sensing system using RF signal processing, an **NS2 (Network Simulator 2)** discrete-event network simulation testbed, lightweight machine learning, and an adaptive calibration engine.

---

## 🌟 The Three Hero Products

1. **HERO 1 · Live RSSI Occupancy Detector**
   - Fixed sniffers ($R_1, R_2$) log commodity 2.4 GHz signal strength (dBm) at 10 Hz.
   - Signal cleaning pipeline (10 Hz grid resampling, 3-sample median filter, EMA $\alpha=0.25$, gap interpolation).
   - Sliding 5.0 s windows (50% overlap) extracting 21 statistical features.
   - Primary Random Forest classifier ($>90\%$ accuracy) with temporal majority voting.
   - Real-time interactive Streamlit dashboard with live state, signal traces, and historical replay.
   - Guaranteed end-to-end latency $\le 2\text{ s}$ from window close to dashboard.

2. **HERO 2 · NS2 Simulation & Network Testbed**
   - 802.11 MAC/PHY room model synchronized with the physical layout coordinates.
   - C++ `Propagation/BodyShadow` first-Fresnel-zone excess loss ($L_{\text{body}}$) and variance ($\sigma_{\text{body}}$) extension.
   - Patched physical layer (`WirelessPhy`) writing per-packet received power traces.
   - Sim-to-Real calibration from physical $D_0$ runs, evaluated with Kolmogorov–Smirnov and Wasserstein-1 distance metrics.
   - 802.11 collection network scaling study (PDR, delay p95, throughput) evaluating network bottlenecks.
   - Simulation-assisted learning comparing 4 regimes (Real-only, Sim-only, Pre-train + Fine-tune, Mixed) on held-out real data.

3. **HERO 3 · Adaptive Calibration Engine**
   - Automatic baseline update from confirmed-empty periods ($P(\text{Empty}) \ge 0.90$) to eliminate environmental and thermal drift.
   - Anti-poisoning variance guards prevent absorbing motionless human occupants into the baseline.

---

## 📁 Repository Structure

```text
├── cisco_packet_tracer/               # Cisco Packet Tracer IoT Digital Twin, IOS configs & bridge
├── configs/
│   ├── config.yaml                    # Master parameters (sampling, windowing, sim, paths)
│   └── protocols/                     # Experiment protocols (state_run, drill, nearfar)
├── db/
│   └── schema.sql                     # SQLite schema (runs, raw_rssi, features, predictions)
├── ns2/
│   ├── Dockerfile                     # Pinned Ubuntu 16.04 + gcc-4.8 container for NS2 2.35
│   ├── patches/                       # C++ BodyShadow & WirelessPhy trace patches
│   └── tcl/                           # Base Tcl templates (room_base.tcl, net_study.tcl)
├── src/occ/
│   ├── collect/                       # ESP32 USB serial and synthetic mock collectors
│   ├── preprocess/                    # Signal cleaning, grid resampling, median & EMA filtering
│   ├── features/                      # Windowing and 21-feature extractor
│   ├── models/                        # Threshold, LogReg, KNN, SVM, and Random Forest models
│   ├── calibration/                   # Adaptive baseline tracking & drift engine
│   ├── sim/                           # NS2 scenario generator, trace parser, fidelity, and netstudy
│   ├── inference/                     # Live sliding-window streaming service and majority voter
│   └── dashboard/                     # Streamlit web application
├── scripts/
│   ├── check_run.py                   # Data integrity and packet rate validator
│   ├── calibrate_ns2.py               # Channel & body fitting from physical D0 runs
│   ├── build_dataset.py               # Feature compiler across raw and sim directories
│   ├── validate.py                    # 5-model Leave-One-Session-Out benchmark runner
│   ├── run_simreal.py                 # O11 transfer learning evaluation (4 training regimes)
│   ├── run_netstudy.py                # O10 802.11 collection network capacity sweep runner
│   └── run_ns2_batch.sh               # NS2 batch scenario execution script
└── tests/                             # Full unit, integration, and leakage prevention test suite
```

---

## 🚀 Quickstart & Setup

### 1. Installation
```bash
# Clone repository and enter directory
cd /Users/swayamshukla/RSSI_Smart_Room_Occupancy

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Run Automated Verification Tests
```bash
pytest tests/ -v
```

### 3. Generate Simulated & Mock Physical Datasets
```bash
# Generate synthetic hardware runs for development
python3 -c "from occ.collect.mock_collector import generate_mock_run; generate_mock_run('RUN_DEV_01'); generate_mock_run('RUN_DEV_02')"

# Generate NS2 synthetic scenarios S1-S5 across seeds
python3 -m occ.sim.gen_scenarios --scenarios S1,S2,S3,S4 --seeds 5

# Compile feature datasets across all runs
python3 scripts/build_dataset.py
```

### 4. Run Model Benchmarks (O4, E4)
```bash
# Run Leave-One-Session-Out cross-validation on real data
python3 scripts/validate.py
```

### 5. Run Sim-to-Real Calibration & Transfer Learning (O9, O11)
```bash
# Calibrate NS2 parameters
python3 scripts/calibrate_ns2.py

# Evaluate distribution fidelity (KS & Wasserstein distance)
python3 -m occ.sim.fidelity

# Run 4-regime transfer learning study
python3 scripts/run_simreal.py

# Run 802.11 network scaling study
python3 scripts/run_netstudy.py
```

### 6. Launch Interactive Dashboard (Hero 1)
```bash
streamlit run src/occ/dashboard/app.py
```
Open your browser to `http://localhost:8501`.

---

## 🐳 Running NS2 inside Docker Container
For running discrete-event simulations in the pinned legacy environment:
```bash
cd ns2
docker build -t ns2-hero .
docker run -it -v $(pwd)/..:/work ns2-hero bash
bash scripts/run_ns2_batch.sh
```

---

## 🔒 Security & Data Leakage Rules
1. **Zero Simulated Windows in Test Sets**: Every evaluation fold strictly asserts `(df_test.origin == 'real').all()`.
2. **Session-Wise Splitting (LOSO/LODO)**: Overlapping sliding windows from the same session never cross train/test splits.
3. **Sealed Calibration Data**: Calibration runs (`D0_dist`, `D0_body`) are strictly excluded from evaluation folds.
