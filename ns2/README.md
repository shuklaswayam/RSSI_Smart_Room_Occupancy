# NS2 (Network Simulator 2 v2.35) Simulation Testbed (Hero 2)

## 1. Overview
Hero 2 provides a discrete-event network simulation testbed that reproduces the smart room geometry, wireless nodes, human movement, and 802.11 network traffic.

It includes:
- **`BodyShadow` C++ Propagation Extension**: Evaluates the first Fresnel zone clearance for any transmitter-receiver link and applies geometric human body attenuation and variance fluctuation.
- **`WirelessPhy` RSSI Logging Patch**: Hooks into the physical layer receive path to log exact received signal power (dBm), timestamp, and node IDs for accepted packets.
- **Trace Parser (`nsparse.py`)**: Converts simulation traces into the standard dataset schema (`rssi.csv`, `events.csv`, `meta.json`).

## 2. Docker Setup & Build
NS2 2.35 requires a legacy GCC toolchain (pinned in Ubuntu 16.04).

To build and run:
```bash
cd ns2
docker build -t ns2-hero .
docker run -it -v $(pwd)/..:/work ns2-hero bash
```

## 3. Running Scenarios Inside Container
```bash
# Generate scenarios S1-S5 across 10 seeds
python3 -m occ.sim.gen_scenarios --scenarios S1,S2,S3,S4,S5 --seeds 10

# Run batch simulation
bash scripts/run_ns2_batch.sh
```

## 4. Pure Python Simulation Fallback
If Docker is unavailable on the host machine during development, `src/occ/sim/gen_scenarios.py` provides an analytical `BodyShadow` propagation engine that executes on the host without container dependencies:
```bash
python3 -m occ.sim.gen_scenarios --engine python --scenarios S1,S2,S3,S4,S5 --seeds 10
```
