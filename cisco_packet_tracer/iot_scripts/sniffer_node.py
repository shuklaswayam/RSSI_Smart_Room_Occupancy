"""
Cisco Packet Tracer IoT Script: Sniffer Node (R1 / R2)
Runs directly inside Cisco Packet Tracer MCU or Single Board Computer (SBC).

Responsibilities:
- Reads or simulates 2.4 GHz RF RSSI samples at 10 Hz (interval 0.1s).
- Transmits telemetry packets (node_id, sequence, timestamp, rssi) to the Edge Gateway.
- Uses UDP socket (port 5555) or MQTT client for low-overhead telemetry.
- Toggles onboard status LED (GPIO 0) on transmission.
"""

import time
import json
import random
import math

try:
    import socket
    SOCKET_AVAILABLE = True
except (ImportError, NotImplementedError, Exception):
    SOCKET_AVAILABLE = False

# Try importing Packet Tracer GPIO library (available in PT Python environment)
try:
    from gpio import pinMode, digitalRead, digitalWrite, OUT, IN, HIGH, LOW
    PT_GPIO_AVAILABLE = True
except (ImportError, NotImplementedError, Exception):
    PT_GPIO_AVAILABLE = False
    OUT = 1
    IN = 0
    HIGH = 1
    LOW = 0
    def pinMode(pin, mode): pass
    def digitalWrite(pin, value): pass
    def digitalRead(pin): return 0

# Node Configuration
NODE_ID = "R2"  # "R1" or "R2"
SERVER_IP = "192.168.30.10"
SERVER_PORT = 5555
SAMPLING_INTERVAL_S = 0.1  # 10 Hz rate
LED_PIN = 0

# Baseline RF properties for simulated sniffer
BASELINE_RSSI = -52.0 if NODE_ID == "R1" else -58.0
NOISE_STD = 1.2

def setup():
    if PT_GPIO_AVAILABLE:
        pinMode(LED_PIN, OUT)
        digitalWrite(LED_PIN, LOW)
    print(f"[{NODE_ID}] Sniffer node initialized. Target Edge: {SERVER_IP}:{SERVER_PORT}")

def generate_sample(seq: int, occupied: bool = False, motion: bool = False) -> dict:
    """Generate a realistic 10 Hz Wi-Fi RSSI packet with multipath and body attenuation."""
    t = time.time()
    rssi = BASELINE_RSSI + random.gauss(0, NOISE_STD)
    
    if occupied:
        # Body shadow attenuation (3 to 8 dB drop)
        rssi -= random.uniform(3.5, 7.5)
        if motion:
            # Increased multipath variance during human walking/movement
            rssi += 4.0 * math.sin(seq * 0.4) + random.gauss(0, 2.5)

    return {
        "node_id": NODE_ID,
        "seq": seq,
        "timestamp": round(t, 3),
        "rssi": round(rssi, 2),
        "channel": 6,
        "rate_hz": 10.0
    }

def main():
    setup()
    sock = None
    if SOCKET_AVAILABLE:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except Exception:
            sock = None

    seq = 0
    print(f"[{NODE_ID}] Starting 10 Hz RSSI transmission loop...")
    while True:
        try:
            # Generate 10 Hz Wi-Fi packet
            packet = generate_sample(seq)
            payload = json.dumps(packet).encode("utf-8")
            
            # Send to edge server if socket is present
            if sock is not None:
                try:
                    sock.sendto(payload, (SERVER_IP, SERVER_PORT))
                except Exception:
                    pass
            elif seq % 10 == 0:
                print(f"[{NODE_ID}] Generated RSSI={packet['rssi']} dBm | Seq={seq}")
            
            # Blink LED
            if PT_GPIO_AVAILABLE and seq % 10 == 0:
                digitalWrite(LED_PIN, HIGH)
                time.sleep(0.02)
                digitalWrite(LED_PIN, LOW)
                
            seq += 1
            time.sleep(SAMPLING_INTERVAL_S)
        except KeyboardInterrupt:
            break
        except Exception:
            time.sleep(1.0)

if __name__ == "__main__":
    main()
