import csv
import logging
import time
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

class SerialCollector:
    """Collects RSSI packets from ESP32 serial ports."""
    
    def __init__(self, port: str, baudrate: int = 115200, rx_id: str = "R1"):
        self.port = port
        self.baudrate = baudrate
        self.rx_id = rx_id
        self.is_running = False
        self.serial_inst = None

    def start(self, output_csv: Path, callback: Optional[Callable] = None):
        try:
            import serial
        except ImportError:
            logger.error("pyserial is not installed. Use mock collector or install requirements.")
            return

        self.is_running = True
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        write_header = not output_csv.exists()

        with open(output_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["t_host", "receiver_id", "source_id", "technology", "channel", "rssi_dbm", "esp_ms"])

            logger.info("Opening serial port %s at %d baud...", self.port, self.baudrate)
            try:
                self.serial_inst = serial.Serial(self.port, self.baudrate, timeout=1.0)
            except Exception as e:
                logger.error("Failed to open serial port %s: %s", self.port, e)
                self.is_running = False
                return

            while self.is_running:
                line = self.serial_inst.readline().decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                # Expected format: esp_ms,tx_id,channel,rssi_dbm or comma-separated tokens
                parts = line.split(",")
                t_host = round(time.time(), 3)
                if len(parts) >= 4:
                    try:
                        esp_ms = int(parts[0].strip())
                        source_id = parts[1].strip()
                        channel = int(parts[2].strip())
                        rssi_dbm = float(parts[3].strip())
                        row = [t_host, self.rx_id, source_id, "wifi", channel, rssi_dbm, esp_ms]
                        writer.writerow(row)
                        f.flush()
                        if callback:
                            callback(row)
                    except ValueError:
                        continue

    def stop(self):
        self.is_running = False
        if self.serial_inst and self.serial_inst.is_open:
            self.serial_inst.close()
