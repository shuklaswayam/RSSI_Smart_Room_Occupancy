"""
Cisco Packet Tracer IoT Script: Room Actuators Controller
Runs on the Smart Room Actuator MCU / SBC inside Cisco Packet Tracer.

Controls:
- GPIO 0: Occupancy Indicator (0=Vacant/Green, 1=Occupied/Red)
- GPIO 1: Smart Room Lighting Relay (0=OFF, 1=ON)
- GPIO 2: Smart HVAC / Ceiling Fan (0=OFF, 1=ON)
- GPIO 3: Smart Door Lock / Energy Save Solenoid (0=Normal, 1=Active)

Listens for state directives from the Edge Gateway on UDP port 5556.
"""

import time
import json
import random

try:
    import socket
    SOCKET_AVAILABLE = True
except (ImportError, NotImplementedError, Exception):
    SOCKET_AVAILABLE = False

try:
    from gpio import pinMode, digitalWrite, digitalRead, customWrite, OUT, IN, HIGH, LOW
    PT_GPIO_AVAILABLE = True
except (ImportError, NotImplementedError, Exception):
    PT_GPIO_AVAILABLE = False
    OUT = 1
    IN = 0
    HIGH = 1
    LOW = 0
    def pinMode(pin, mode): pass
    def digitalWrite(pin, value): pass
    def customWrite(pin, value): pass
    def digitalRead(pin): return 0

PIN_OCCUPANCY_INDICATOR = 0
PIN_LIGHTING = 1
PIN_HVAC = 2
PIN_DOOR_LOCK = 3

LISTEN_IP = "0.0.0.0"
LISTEN_PORT = 5556

def setup():
    if PT_GPIO_AVAILABLE:
        pinMode(PIN_OCCUPANCY_INDICATOR, OUT)
        pinMode(PIN_LIGHTING, OUT)
        pinMode(PIN_HVAC, OUT)
        pinMode(PIN_DOOR_LOCK, OUT)
        
        # Default: Vacant room state (lights & HVAC off to save power)
        try:
            customWrite(PIN_LIGHTING, "0")
            customWrite(PIN_HVAC, "0")
            customWrite(PIN_OCCUPANCY_INDICATOR, "0")
        except Exception:
            pass
        digitalWrite(PIN_OCCUPANCY_INDICATOR, LOW)
        digitalWrite(PIN_LIGHTING, LOW)
        digitalWrite(PIN_HVAC, LOW)
        digitalWrite(PIN_DOOR_LOCK, LOW)
        
    print("[ACTUATORS] Controller initialized. Ready for room actuation.")

def apply_room_state(is_occupied: bool, state_str: str, confidence: float):
    print(f"[ACTUATORS] Applying state: {state_str} (Occupied={is_occupied}, Conf={confidence:.2f})")
    if PT_GPIO_AVAILABLE:
        if is_occupied:
            # Turn ON lights (Level 2), HVAC Fan (Level 2), and set indicator to OCCUPIED
            try:
                customWrite(PIN_LIGHTING, "2")
                customWrite(PIN_HVAC, "2")
                customWrite(PIN_OCCUPANCY_INDICATOR, "1")
            except Exception:
                pass
            digitalWrite(PIN_OCCUPANCY_INDICATOR, HIGH)
            digitalWrite(PIN_LIGHTING, HIGH)
            digitalWrite(PIN_HVAC, HIGH)
            digitalWrite(PIN_DOOR_LOCK, HIGH)
        else:
            # Power-saving mode: Turn OFF lights (Level 0) and HVAC (Level 0)
            try:
                customWrite(PIN_LIGHTING, "0")
                customWrite(PIN_HVAC, "0")
                customWrite(PIN_OCCUPANCY_INDICATOR, "0")
            except Exception:
                pass
            digitalWrite(PIN_OCCUPANCY_INDICATOR, LOW)
            digitalWrite(PIN_LIGHTING, LOW)
            digitalWrite(PIN_HVAC, LOW)
            digitalWrite(PIN_DOOR_LOCK, LOW)

def main():
    setup()
    
    if SOCKET_AVAILABLE:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind((LISTEN_IP, LISTEN_PORT))
            print(f"[ACTUATORS] Listening for Edge Server UDP packets on {LISTEN_IP}:{LISTEN_PORT}...")
            while True:
                try:
                    data, addr = sock.recvfrom(2048)
                    payload = json.loads(data.decode("utf-8"))
                    state = payload.get("state", "EMPTY").upper()
                    is_occupied = payload.get("is_occupied", "OCCUPIED" in state)
                    conf = float(payload.get("confidence", 1.0))
                    apply_room_state(is_occupied, state, conf)
                except Exception:
                    time.sleep(0.5)
        except Exception as e:
            print(f"[ACTUATORS] Socket binding error: {e}. Switching to autonomous animation mode.")
            run_autonomous_loop()
    else:
        # Fallback for MCU-PT: Skulpt environment does not implement BSD sockets
        print("[ACTUATORS] Running in MCU Skulpt mode: Autonomous Occupancy Animation Cycle active!")
        run_autonomous_loop()

def run_autonomous_loop():
    cycle = 0
    while True:
        if cycle % 2 == 0:
            apply_room_state(True, "OCCUPIED_DYNAMIC", 0.95)
        else:
            apply_room_state(False, "VACANT_SAVINGS", 0.98)
        cycle += 1
        time.sleep(4)

if __name__ == "__main__":
    main()
