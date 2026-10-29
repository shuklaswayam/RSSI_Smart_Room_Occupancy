# Experiment Registry & Provenance Tracking

This registry tracks reproducibility metadata, environment states, calibrated parameters, and results for all project experiments (E1–E17).

## System Environment & Image Fingerprint
- **Host OS**: macOS / Linux
- **Python Version**: 3.9+
- **NS2 Version**: ns-allinone-2.35 (pinned Docker / WSL image `ns2-hero:latest`)
- **Git Commit**: `HEAD`
- **Active Calibration Runs**: `D0_dist`, `D0_body`, `D0_walk`

---

## Active Simulation Parameters (`sim_params`)
```yaml
channel:
  pathloss_exp: 2.70
  shadow_std_db: 4.00
  tx_power_w: 0.1000
body:
  loss_db: 6.00
  extra_std_db: 2.00
  fresnel_scale: 1.00
seeds: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

---

## Experiment Matrix (E1 – E17)

| ID | Objective | Name | Target Condition / Regimes | Output Artifact | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **E1** | O3 | Window Length & Overlap Sweep | $W \in [2, 5, 10]\text{ s}$, overlap $\in [0, 50, 75]\%$ | `results/tables/e1_window_ablation.csv` | Ready |
| **E2** | O3 | Receiver Count Ablation | Single Link ($R_1$) vs Dual Link ($R_1, R_2$) | `results/tables/e2_receiver_ablation.csv` | Ready |
| **E3** | O3 | Feature Subset Ablation | Levels vs Spreads vs Slopes vs ZCR | `results/tables/e3_feature_ablation.csv` | Ready |
| **E4** | O4 | Five-Model LOSO Benchmark | Threshold, LogReg, KNN, SVM, Random Forest | `results/tables/e4_model_comparison.csv` | Active |
| **E5** | O5 | Majority Voter Window Tuning | $K \in [1, 3, 5, 7]$ windows | `results/tables/e5_voter_tuning.csv` | Ready |
| **E6** | O6 | Door State Robustness | Door Open vs Door Closed | `results/tables/e6_door_robustness.csv` | Ready |
| **E7** | O6 | Furniture Movement Robustness | Baseline Layout vs Modified Layout | `results/tables/e7_furniture_robustness.csv` | Ready |
| **E8** | O6 | Unseen Device / Carrier Test | Laptop vs Smartphone vs ESP32 Tx | `results/tables/e8_device_robustness.csv` | Ready |
| **E9** | O7 | Static vs Adaptive Calibration | Drift Injection: Ramp, Step (+3 dB), Variance | `results/tables/e9_adaptive_drift.csv` | Active |
| **E10**| O5 | Pipeline Latency Breakdown | Guard + Preprocess + Features + Inference | `results/tables/e10_latency_breakdown.csv` | Active |
| **E11**| O6 | Final Untouched Test Evaluation| Sealed evaluation on Day D5 (Run once) | `results/tables/e11_final_d5_test.csv` | Sealed |
| **E12**| O1 | Device-Based Near/Far Proximity | Near ($< 1.5\text{ m}$) vs Far ($> 3.0\text{ m}$) | `results/tables/e12_nearfar.csv` | Ready |
| **E13**| O1 | Multi-Modal Fusion (Optional) | Wi-Fi Only vs BLE Only vs Wi-Fi + BLE | `results/tables/e13_multimodal.csv` | Optional |
| **E14**| O9 | Simulation Fidelity Evaluation | KS statistic & Wasserstein-1 distance (Before/After) | `results/tables/e14_fidelity.csv` | Active |
| **E15**| O10| Collection Network Capacity | Sweeps across rates, receivers, background devices | `results/tables/e15_network_study.csv` | Active |
| **E16**| O11| Sim-to-Real Transfer Regimes | Real-only, Sim-only, Pretrain+FT, Mixed ($n \in [1..15]$) | `results/tables/e16_simreal_learning_curves.csv` | Active |
| **E17**| O8,11| Sim vs Real Design Correlation | Spearman rank correlation of ablations | `results/tables/e17_ablation_correlation.csv` | Ready |
