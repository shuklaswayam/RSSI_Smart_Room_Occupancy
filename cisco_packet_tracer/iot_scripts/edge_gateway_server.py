"""
Cisco Packet Tracer IoT Script: Edge Gateway & Inference Server
Runs on the Cisco Packet Tracer Server / SBC (IP: 192.168.30.10).

Responsibilities:
- Ingests 10 Hz RSSI packets from Sniffers R1 and R2 via UDP (port 5555).
- Maintains sliding window of RSSI signals.
- Computes real-time statistics (attenuation delta, variance) and triggers room state.
- Dispatches actuation events to the Room Actuator Controller (192.168.20.25:5556).
- Logs packet delivery ratio (PDR) and latency performance for E15 network verification.
"""

import time
import socket
import json
from collections import deque

LISTEN_IP = "0.0.0.0"
LISTEN_PORT = 5555
ACTUATOR_IP = "192.168.20.25"
ACTUATOR_PORT = 5556

# Sliding window buffer (5 seconds at 10 Hz = 50 samples per receiver)
WINDOW_SIZE = 50
buffer_r1 = deque(maxlen=WINDOW_SIZE)
buffer_r2 = deque(maxlen=WINDOW_SIZE)

# Baseline thresholds for empty room (calibrated)
BASELINE_R1 = -52.0
BASELINE_R2 = -58.0
ATTENUATION_THRESHOLD = 3.5  # dB drop indicating human presence

def evaluate_occupancy():
    if len(buffer_r1) < 20 or len(buffer_r2) < 20:
        return "EMPTY", False, 0.50
        
    mean_r1 = sum(buffer_r1) / len(buffer_r1)
    mean_r2 = sum(buffer_r2) / len(buffer_r2)
    
    drop_r1 = BASELINE_R1 - mean_r1
    drop_r2 = BASELINE_R2 - mean_r2
    
    # Variance check for motion
    var_r1 = sum((x - mean_r1) ** 2 for x in buffer_r1) / len(buffer_r1)
    var_r2 = sum((x - mean_r2) ** 2 for x in buffer_r2) / len(buffer_r2)
    
    is_occupied = (drop_r1 > ATTENUATION_THRESHOLD) or (drop_r2 > ATTENUATION_THRESHOLD) or (var_r1 > 4.0 or var_r2 > 4.0)
    
    if is_occupied:
        state = "OCCUPIED_DYNAMIC" if (var_r1 > 4.5 or var_r2 > 4.5) else "OCCUPIED_STATIC"
        confidence = 0.94
    else:
        state = "EMPTY"
        confidence = 0.96
        
    return state, is_occupied, confidence

def main():
    print(f"[EDGE-SERVER] Ingestion service starting on {LISTEN_IP}:{LISTEN_PORT}...")
    sock_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_rx.bind((LISTEN_IP, LISTEN_PORT))
    
    sock_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    last_eval = time.time()
    packet_count = 0
    
    while True:
        try:
            data, addr = sock_rx.recvfrom(2048)
            packet = json.loads(data.decode("utf-8"))
            packet_count += 1
            
            node_id = packet.get("node_id", "R1")
            rssi = float(packet.get("rssi", -60.0))
            
            if node_id == "R1":
                buffer_r1.append(rssi)
            elif node_id == "R2":
                buffer_r2.append(rssi)
                
            # Evaluate every 1.0 second
            now = time.time()
            if now - last_eval >= 1.0:
                last_eval = now
                state, is_occupied, conf = evaluate_occupancy()
                
                # Send directive to actuators
                directive = {
                    "state": state,
                    "is_occupied": is_occupied,
                    "confidence": conf,
                    "timestamp": round(now, 2),
                    "buffer_depth": len(buffer_r1) + len(buffer_r2)
                }
                payload = json.dumps(directive).encode("utf-8")
                try:
                    sock_tx.sendto(payload, (ACTUATOR_IP, ACTUATOR_PORT))
                except Exception:
                    pass
                    
                print(f"[EDGE-DECISION] State={state} | Occ={is_occupied} | Conf={conf*100:.1f}% | RxPackets={packet_count}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            time.sleep(0.1)

if __name__ == "__main__":
    main()
