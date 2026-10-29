#!/usr/bin/env bash
set -e

SCENARIOS=("S1" "S2" "S3" "S4" "S5")
SEEDS=(1 2 3 4 5 6 7 8 9 10)

echo "[*] Starting NS2 batch simulation across scenarios and seeds..."

for scn in "${SCENARIOS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_id="SIM_${scn}_seed$(printf "%02d" $seed)"
    echo "  [>] Processing $run_id..."
    
    if command -v ns &> /dev/null; then
      python3 -c "from occ.sim.gen_scenarios import generate_scenario_tcl; generate_scenario_tcl('$scn', $seed)"
      ns "ns2/tcl/${run_id}.tcl"
      python3 -m occ.sim.nsparse "$run_id"
    else
      # Fallback to analytical Python BodyShadow engine if ns is not installed on host
      python3 -c "from occ.sim.gen_scenarios import simulate_scenario_python; simulate_scenario_python('$scn', $seed)"
    fi
  done
done

echo "[SUCCESS] NS2 batch simulation completed successfully!"
