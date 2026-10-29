"""
Host-to-Cisco Packet Tracer Bridge Service.

Bridges the local Python ML Inference Engine / Mock Data Generator to the
Cisco Packet Tracer virtual network simulation.

Features:
1. RSSI Streamer: Sends real/synthetic 10 Hz RSSI data packets to the Packet Tracer Edge Server.
2. Actuator Bridge: Relays predictions from the primary Random Forest model directly
   to the Packet Tracer Room Actuators (Light, HVAC, Indicator).
3. Test Demo Mode: Generates sequential occupancy scenario events to demonstrate smart room automation.
"""

import socket
import json
import time
import argparse
from typing import Optional, Dict, Any

class PacketTracerBridge:
    def __init__(self, pt_host: str = "127.0.0.1", edge_port: int = 5555, actuator_port: int = 5556):
        self.pt_host = pt_host
        self.edge_port = edge_port
        self.actuator_port = actuator_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
    def send_rssi_sample(self, node_id: str, seq: int, rssi: float, timestamp: Optional[float] = None) -> bool:
        """Send an individual 10 Hz RSSI packet into the Packet Tracer network."""
        payload = {
            "node_id": node_id,
            "seq": seq,
            "timestamp": timestamp or round(time.time(), 3),
            "rssi": round(rssi, 2),
            "channel": 6,
            "rate_hz": 10.0
        }
        try:
            raw = json.dumps(payload).encode("utf-8")
            self.sock.sendto(raw, (self.pt_host, self.edge_port))
            return True
        except Exception as e:
            print(f"[BRIDGE ERROR] Failed to send RSSI sample: {e}")
            return False

    def send_occupancy_state(self, state: str, confidence: float = 0.95, is_occupied: Optional[bool] = None) -> bool:
        """Directly command the Cisco Packet Tracer room actuators."""
        if is_occupied is None:
            is_occupied = (state.upper() != "EMPTY")
            
        payload = {
            "state": state.upper(),
            "is_occupied": is_occupied,
            "confidence": round(confidence, 3),
            "timestamp": round(time.time(), 3)
        }
        try:
            raw = json.dumps(payload).encode("utf-8")
            self.sock.sendto(raw, (self.pt_host, self.actuator_port))
            return True
        except Exception as e:
            print(f"[BRIDGE ERROR] Failed to send actuator state: {e}")
            return False

    def run_demo_sequence(self, interval_s: float = 3.0):
        """Run an automated scenario cycle (EMPTY -> STATIC -> DYNAMIC -> EMPTY)."""
        scenarios = [
            ("EMPTY", 0.98, False, "Room is vacant. All lights & HVAC turned OFF (energy-saving)."),
            ("OCCUPIED_STATIC", 0.91, True, "Occupant seated at desk. Turning ON lights & HVAC."),
            ("OCCUPIED_DYNAMIC", 0.97, True, "Occupant moving across room. Keeping all systems active."),
            ("EMPTY", 0.96, False, "Occupant left room. Initiating shutdown cycle.")
        ]
        print(f"[*] Starting Packet Tracer Live Bridge Demo -> Target: {self.pt_host}...")
        for state, conf, occ, desc in scenarios:
            print(f"\n[DEMO] >> State: {state} | Conf: {conf*100:.1f}%")
            print(f"       Action: {desc}")
            self.send_occupancy_state(state, conf, occ)
            time.sleep(interval_s)
        print("\n[SUCCESS] Demo sequence completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cisco Packet Tracer Smart Room Bridge")
    parser.add_argument("--host", default="127.0.0.1", help="Target Packet Tracer host IP")
    parser.add_argument("--demo", action="store_true", help="Run scenario sequence demo")
    args = parser.parse_args()
    
    bridge = PacketTracerBridge(pt_host=args.host)
    if args.demo:
        bridge.run_demo_sequence()
    else:
        print("[*] Bridge ready. Use as a module or pass --demo to run scenario cycle.")
