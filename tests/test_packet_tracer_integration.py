"""
Tests for Cisco Packet Tracer IoT Integration and Telemetry Bridge.
"""

import pytest
import json
import socket
from cisco_packet_tracer.bridge.pt_bridge import PacketTracerBridge
from cisco_packet_tracer.iot_scripts.sniffer_node import generate_sample
from cisco_packet_tracer.iot_scripts.edge_gateway_server import evaluate_occupancy, buffer_r1, buffer_r2

def test_sniffer_sample_generation():
    """Verify generated sniffer packets conform to 10 Hz schema and physical ranges."""
    sample = generate_sample(seq=1, occupied=False, motion=False)
    assert sample["node_id"] in ["R1", "R2"]
    assert sample["rate_hz"] == 10.0
    assert -90.0 <= sample["rssi"] <= -30.0
    assert sample["seq"] == 1

def test_sniffer_body_shadow_drop():
    """Verify that occupied state introduces RF attenuation."""
    empty_samples = [generate_sample(seq=i, occupied=False)["rssi"] for i in range(50)]
    occ_samples = [generate_sample(seq=i, occupied=True)["rssi"] for i in range(50)]
    
    avg_empty = sum(empty_samples) / len(empty_samples)
    avg_occ = sum(occ_samples) / len(occ_samples)
    
    # Occupied average should be lower (more negative dBm) than empty average
    assert avg_occ < avg_empty

def test_pt_bridge_serialization(monkeypatch):
    """Verify PacketTracerBridge sends valid UDP JSON payloads."""
    sent_packets = []
    
    def fake_sendto(self, data, addr):
        sent_packets.append((data, addr))
        return len(data)
        
    monkeypatch.setattr(socket.socket, "sendto", fake_sendto)
    
    bridge = PacketTracerBridge(pt_host="192.168.30.10", edge_port=5555, actuator_port=5556)
    
    # 1. Test sending RSSI sample
    ok = bridge.send_rssi_sample(node_id="R1", seq=42, rssi=-53.4)
    assert ok is True
    assert len(sent_packets) == 1
    data_rssi, addr_rssi = sent_packets[0]
    assert addr_rssi == ("192.168.30.10", 5555)
    parsed_rssi = json.loads(data_rssi.decode("utf-8"))
    assert parsed_rssi["node_id"] == "R1"
    assert parsed_rssi["seq"] == 42
    assert parsed_rssi["rssi"] == -53.4

    # 2. Test sending Actuator command
    ok_act = bridge.send_occupancy_state("OCCUPIED_STATIC", confidence=0.92, is_occupied=True)
    assert ok_act is True
    assert len(sent_packets) == 2
    data_act, addr_act = sent_packets[1]
    assert addr_act == ("192.168.30.10", 5556)
    parsed_act = json.loads(data_act.decode("utf-8"))
    assert parsed_act["state"] == "OCCUPIED_STATIC"
    assert parsed_act["is_occupied"] is True
    assert parsed_act["confidence"] == 0.92

def test_edge_gateway_evaluation_logic():
    """Verify Edge Gateway sliding window evaluation for EMPTY vs OCCUPIED."""
    buffer_r1.clear()
    buffer_r2.clear()
    
    # Empty baseline buffers (-52 dBm for R1, -58 dBm for R2)
    for _ in range(30):
        buffer_r1.append(-52.0)
        buffer_r2.append(-58.0)
        
    state, is_occ, conf = evaluate_occupancy()
    assert is_occ is False
    assert state == "EMPTY"
    assert conf >= 0.90
    
    # Ingest severe human body attenuation (-62 dBm for R1, -68 dBm for R2)
    for _ in range(30):
        buffer_r1.append(-63.0)
        buffer_r2.append(-69.0)
        
    state_occ, is_occ_flag, conf_occ = evaluate_occupancy()
    assert is_occ_flag is True
    assert "OCCUPIED" in state_occ
    assert conf_occ >= 0.90
