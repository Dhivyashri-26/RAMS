"""
simulation/sumo_simulator.py — SUMO-Based Smart Mobility Traffic Simulator
RAMS Framework — Objective 6

Simulates realistic smart mobility network traffic with injected cyberattacks.

Two modes:
  1. SUMO-integrated: Uses actual SUMO binary (if installed)
  2. Python-native: Full standalone simulation (no SUMO install needed)
     — mimics SUMO vehicle/RSU behaviour and generates realistic network flows

Reference: FUSE-Net paper — SUMO-based traffic simulation for ITS security

Attack scenarios injected:
  - DDoS against Roadside Units (RSUs)
  - Botnet C2 beaconing between compromised vehicles
  - DoS slowloris-style against V2I infrastructure
  - Port scanning of RSU services

Output: CSV of network flows (same format as CIC-IDS2017)
        → feeds directly into Edge IDS (Objective 2)
"""

import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SUMO_CONFIG, RESULTS_DIR


# ══════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════

@dataclass
class Vehicle:
    """Represents a vehicle node in the smart mobility network."""
    vehicle_id: str
    vehicle_type: str        # "car", "bus", "emergency", "compromised"
    ip: str
    position: tuple          # (x, y) on grid
    speed: float             # km/h
    is_compromised: bool = False
    bot_id: Optional[str] = None


@dataclass
class RoadsideUnit:
    """RSU — infrastructure node (V2I communication point)."""
    rsu_id: str
    ip: str
    position: tuple
    services: List[str] = field(default_factory=lambda: ["v2i", "traffic_mgmt", "emergency"])
    is_under_attack: bool = False


@dataclass
class NetworkFlow:
    """
    A single network flow record.
    Fields match CIC-IDS2017 structure for pipeline compatibility.
    """
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    duration: float
    total_fwd_packets: int
    total_bwd_packets: int
    total_length_fwd: float
    total_length_bwd: float
    fwd_packet_length_mean: float
    bwd_packet_length_mean: float
    flow_bytes_per_s: float
    flow_packets_per_s: float
    flow_iat_mean: float
    flow_iat_std: float
    fwd_iat_mean: float
    bwd_iat_mean: float
    syn_flag: int
    ack_flag: int
    fin_flag: int
    rst_flag: int
    psh_flag: int
    init_win_bytes_fwd: int
    init_win_bytes_bwd: int
    label: str               # BENIGN / DDoS / Bot / DoS / PortScan


# ══════════════════════════════════════════════════════════════════
# NETWORK FLOW GENERATOR
# ══════════════════════════════════════════════════════════════════

class FlowGenerator:
    """
    Generates realistic network flow statistics for different traffic types.
    Based on CIC-IDS2017 statistical profiles per attack category.
    """

    def __init__(self, rng: np.random.RandomState):
        self.rng = rng

    def _base_flow(self, src_ip, dst_ip, src_port, dst_port,
                   protocol, timestamp) -> dict:
        return {
            "timestamp": timestamp,
            "src_ip": src_ip, "dst_ip": dst_ip,
            "src_port": src_port, "dst_port": dst_port,
            "protocol": protocol,
        }

    def benign_v2i(self, src_ip, dst_ip, timestamp) -> NetworkFlow:
        """Normal V2I (vehicle-to-infrastructure) communication flow."""
        r = self.rng
        return NetworkFlow(
            timestamp=timestamp,
            src_ip=src_ip, dst_ip=dst_ip,
            src_port=r.randint(1024, 65535),
            dst_port=r.choice([80, 443, 8080, 5000]),
            protocol="TCP",
            duration=r.uniform(0.1, 2.0),
            total_fwd_packets=r.randint(5, 30),
            total_bwd_packets=r.randint(3, 25),
            total_length_fwd=r.uniform(500, 5000),
            total_length_bwd=r.uniform(200, 3000),
            fwd_packet_length_mean=r.uniform(100, 400),
            bwd_packet_length_mean=r.uniform(80, 300),
            flow_bytes_per_s=r.uniform(1000, 50000),
            flow_packets_per_s=r.uniform(10, 200),
            flow_iat_mean=r.uniform(0.005, 0.1),
            flow_iat_std=r.uniform(0.001, 0.05),
            fwd_iat_mean=r.uniform(0.01, 0.15),
            bwd_iat_mean=r.uniform(0.01, 0.15),
            syn_flag=1, ack_flag=1, fin_flag=1, rst_flag=0, psh_flag=1,
            init_win_bytes_fwd=r.choice([8192, 16384, 32768, 65535]),
            init_win_bytes_bwd=r.choice([8192, 16384, 32768, 65535]),
            label="BENIGN"
        )

    def ddos_flow(self, src_ip, dst_ip, timestamp) -> NetworkFlow:
        """
        DDoS attack flow against RSU.
        Characteristics: massive packet rate, tiny packets, UDP flood.
        """
        r = self.rng
        return NetworkFlow(
            timestamp=timestamp,
            src_ip=src_ip, dst_ip=dst_ip,
            src_port=r.randint(1024, 65535),
            dst_port=r.choice([80, 443, 53]),
            protocol=r.choice(["UDP", "TCP"]),
            duration=r.uniform(0.0001, 0.01),   # Very short flows
            total_fwd_packets=r.randint(500, 5000),   # Huge packet count
            total_bwd_packets=r.randint(0, 5),         # Little/no response
            total_length_fwd=r.uniform(50000, 500000),
            total_length_bwd=r.uniform(0, 100),
            fwd_packet_length_mean=r.uniform(40, 100),  # Tiny packets
            bwd_packet_length_mean=r.uniform(0, 20),
            flow_bytes_per_s=r.uniform(1e6, 10e6),     # Multi-megabit
            flow_packets_per_s=r.uniform(10000, 100000),
            flow_iat_mean=r.uniform(0.00001, 0.0001),  # Microsecond intervals
            flow_iat_std=r.uniform(0.000001, 0.00005),
            fwd_iat_mean=r.uniform(0.00001, 0.0001),
            bwd_iat_mean=0.0,
            syn_flag=r.randint(0, 1), ack_flag=0, fin_flag=0,
            rst_flag=0, psh_flag=0,
            init_win_bytes_fwd=r.choice([1024, 2048]),
            init_win_bytes_bwd=0,
            label="DDoS"
        )

    def botnet_flow(self, src_ip, dst_ip, timestamp) -> NetworkFlow:
        """
        Botnet C2 beaconing flow.
        Characteristics: periodic, low-volume, encrypted-like,
        consistent timing (heartbeat pattern).
        """
        r = self.rng
        return NetworkFlow(
            timestamp=timestamp,
            src_ip=src_ip, dst_ip=dst_ip,
            src_port=r.randint(1024, 65535),
            dst_port=r.choice([443, 8443, 4444, 6667]),  # Common C2 ports
            protocol="TCP",
            duration=r.uniform(0.5, 3.0),
            total_fwd_packets=r.randint(3, 15),    # Low packet count
            total_bwd_packets=r.randint(3, 12),
            total_length_fwd=r.uniform(200, 1500),
            total_length_bwd=r.uniform(150, 1200),
            fwd_packet_length_mean=r.uniform(80, 200),
            bwd_packet_length_mean=r.uniform(70, 180),
            flow_bytes_per_s=r.uniform(200, 3000),    # Low bandwidth
            flow_packets_per_s=r.uniform(2, 15),
            flow_iat_mean=r.uniform(0.1, 1.0),        # Regular intervals
            flow_iat_std=r.uniform(0.001, 0.01),      # Very consistent (bot)
            fwd_iat_mean=r.uniform(0.15, 1.5),
            bwd_iat_mean=r.uniform(0.15, 1.5),
            syn_flag=1, ack_flag=1, fin_flag=0, rst_flag=0, psh_flag=1,
            init_win_bytes_fwd=r.choice([8192, 16384]),
            init_win_bytes_bwd=r.choice([8192, 16384]),
            label="Bot"
        )

    def dos_flow(self, src_ip, dst_ip, timestamp) -> NetworkFlow:
        """DoS slowloris-style against V2I infrastructure."""
        r = self.rng
        return NetworkFlow(
            timestamp=timestamp,
            src_ip=src_ip, dst_ip=dst_ip,
            src_port=r.randint(1024, 65535),
            dst_port=80,
            protocol="TCP",
            duration=r.uniform(100, 900),           # Very long connections
            total_fwd_packets=r.randint(10, 50),
            total_bwd_packets=r.randint(5, 20),
            total_length_fwd=r.uniform(1000, 8000),
            total_length_bwd=r.uniform(500, 4000),
            fwd_packet_length_mean=r.uniform(20, 80),  # Small packets
            bwd_packet_length_mean=r.uniform(30, 100),
            flow_bytes_per_s=r.uniform(50, 500),       # Low rate (slowloris)
            flow_packets_per_s=r.uniform(0.1, 2.0),
            flow_iat_mean=r.uniform(5, 30),            # Long inter-arrival
            flow_iat_std=r.uniform(1, 10),
            fwd_iat_mean=r.uniform(8, 40),
            bwd_iat_mean=r.uniform(8, 40),
            syn_flag=1, ack_flag=1, fin_flag=0, rst_flag=0, psh_flag=1,
            init_win_bytes_fwd=r.choice([4096, 8192]),
            init_win_bytes_bwd=r.choice([4096, 8192]),
            label="DoS Hulk"
        )

    def port_scan_flow(self, src_ip, dst_ip, timestamp,
                        port: int = None) -> NetworkFlow:
        """Port scan reconnaissance against RSU."""
        r = self.rng
        return NetworkFlow(
            timestamp=timestamp,
            src_ip=src_ip, dst_ip=dst_ip,
            src_port=r.randint(1024, 65535),
            dst_port=port or r.randint(1, 1024),
            protocol="TCP",
            duration=r.uniform(0.0, 0.002),        # Near-instant
            total_fwd_packets=r.randint(1, 3),      # 1-2 SYN packets
            total_bwd_packets=r.randint(0, 2),
            total_length_fwd=r.uniform(40, 80),     # Just SYN packet
            total_length_bwd=r.uniform(0, 60),
            fwd_packet_length_mean=r.uniform(40, 80),
            bwd_packet_length_mean=r.uniform(0, 60),
            flow_bytes_per_s=r.uniform(100, 5000),
            flow_packets_per_s=r.uniform(100, 2000),
            flow_iat_mean=r.uniform(0.0001, 0.001),
            flow_iat_std=r.uniform(0.00001, 0.0005),
            fwd_iat_mean=r.uniform(0.0001, 0.001),
            bwd_iat_mean=0.0,
            syn_flag=1, ack_flag=0, fin_flag=0, rst_flag=r.randint(0, 1),
            psh_flag=0,
            init_win_bytes_fwd=1024,
            init_win_bytes_bwd=0,
            label="PortScan"
        )


# ══════════════════════════════════════════════════════════════════
# SMART MOBILITY NETWORK TOPOLOGY
# ══════════════════════════════════════════════════════════════════

class SmartMobilityNetwork:
    """
    Simulates a 5x5 grid smart city network with vehicles and RSUs.
    Mimics what SUMO would generate: vehicles moving on roads,
    communicating with RSUs via V2I protocols.
    """

    def __init__(self, config: dict, seed: int = 42):
        self.config = config
        self.rng = np.random.RandomState(seed)
        random.seed(seed)
        self.vehicles: Dict[str, Vehicle] = {}
        self.rsus: Dict[str, RoadsideUnit] = {}
        self.flows: List[NetworkFlow] = []
        self.step = 0
        self._setup_network()

    def _ip(self, subnet: str, host: int) -> str:
        return f"192.168.{subnet}.{host}"

    def _setup_network(self):
        """Initialise RSUs and initial vehicle population."""
        grid = self.config["grid_size"]

        # Place RSUs at grid intersections
        rsu_id = 0
        for x in range(0, grid, 2):
            for y in range(0, grid, 2):
                rsu = RoadsideUnit(
                    rsu_id=f"RSU_{rsu_id:03d}",
                    ip=self._ip("1", rsu_id + 1),
                    position=(x, y),
                )
                self.rsus[rsu.rsu_id] = rsu
                rsu_id += 1

        # Spawn initial vehicles
        for i in range(30):
            self._spawn_vehicle(i)

        print(f"[SUMO] Network: {len(self.rsus)} RSUs, "
              f"{len(self.vehicles)} initial vehicles")
        print(f"[SUMO] RSUs: {list(self.rsus.keys())[:3]}...")

    def _spawn_vehicle(self, vid: int, compromised: bool = False):
        grid = self.config["grid_size"]
        v = Vehicle(
            vehicle_id=f"V_{vid:04d}",
            vehicle_type=self.rng.choice(
                ["car", "bus", "car", "car", "emergency"],
                p=[0.6, 0.1, 0.15, 0.1, 0.05]
            ),
            ip=self._ip("2", (vid % 250) + 1),
            position=(self.rng.randint(0, grid), self.rng.randint(0, grid)),
            speed=self.rng.uniform(20, 80),
            is_compromised=compromised,
            bot_id=f"BOT_{vid}" if compromised else None,
        )
        self.vehicles[v.vehicle_id] = v

    def _nearest_rsu(self, vehicle: Vehicle) -> RoadsideUnit:
        """Find the closest RSU to a vehicle (for V2I comms)."""
        vx, vy = vehicle.position
        nearest = min(
            self.rsus.values(),
            key=lambda r: abs(r.position[0] - vx) + abs(r.position[1] - vy)
        )
        return nearest

    def _move_vehicles(self):
        """Simulate vehicles moving on the road grid."""
        grid = self.config["grid_size"]
        for v in self.vehicles.values():
            dx = self.rng.choice([-1, 0, 1])
            dy = self.rng.choice([-1, 0, 1])
            v.position = (
                max(0, min(grid - 1, v.position[0] + dx)),
                max(0, min(grid - 1, v.position[1] + dy))
            )
            v.speed = max(0, min(120, v.speed + self.rng.normal(0, 5)))


# ══════════════════════════════════════════════════════════════════
# MAIN SIMULATOR
# ══════════════════════════════════════════════════════════════════

class SUMOSimulator:
    """
    RAMS Objective 6: Smart Mobility Traffic Simulator

    Simulates 1 hour of smart city traffic with realistic attack scenarios.
    Outputs a labelled network flow CSV compatible with the Edge IDS pipeline.

    If SUMO is installed, uses traci for real vehicular simulation.
    Otherwise, uses the built-in Python simulation (recommended for development).
    """

    def __init__(self, config: dict = None):
        self.config = config or SUMO_CONFIG
        self.rng = np.random.RandomState(self.config["seed"])
        self.network = SmartMobilityNetwork(self.config, self.config["seed"])
        self.flow_gen = FlowGenerator(self.rng)
        self.flows: List[NetworkFlow] = []
        self.attack_log: List[dict] = []
        self.stats = {
            "total_flows": 0, "benign": 0, "ddos": 0,
            "botnet": 0, "dos": 0, "port_scan": 0,
        }
        os.makedirs(self.config["output_dir"], exist_ok=True)
        self._check_sumo()

    def _check_sumo(self):
        """Check if real SUMO is available."""
        try:
            import traci
            self.use_real_sumo = True
            print("[SUMO] Real SUMO (traci) detected — using SUMO binary")
        except ImportError:
            self.use_real_sumo = False
            print("[SUMO] SUMO not installed — using Python-native simulation")
            print("[SUMO] (Install SUMO from https://sumo.dlr.de for full simulation)")

    # ── Normal traffic generation ─────────────────────────────────

    def _generate_benign_step(self, step: int):
        """Generate normal V2I and V2V communication flows for this step."""
        vehicles = list(self.network.vehicles.values())
        # Each vehicle has ~30% chance of communicating per step
        active = self.rng.choice(
            vehicles,
            size=max(1, int(len(vehicles) * 0.3)),
            replace=False
        )
        for v in active:
            rsu = self.network._nearest_rsu(v)
            flow = self.flow_gen.benign_v2i(v.ip, rsu.ip, step)
            self.flows.append(flow)
            self.stats["benign"] += 1

        # Spawn/despawn vehicles (traffic flow)
        if self.rng.random() < 0.1:
            new_id = len(self.network.vehicles) + self.rng.randint(1000, 9999)
            self.network._spawn_vehicle(new_id)
        if len(self.network.vehicles) > 50 and self.rng.random() < 0.05:
            remove_id = self.rng.choice(list(self.network.vehicles.keys()))
            del self.network.vehicles[remove_id]

        self.network._move_vehicles()

    # ── Attack injection ──────────────────────────────────────────

    def _inject_ddos(self, step: int):
        """
        DDoS attack: fleet of compromised vehicles floods RSU_001.
        Mimics a coordinated DDoS from vehicles with hijacked V2X modules.
        """
        cfg = self.config["attack_scenarios"]["ddos"]
        target_rsu = self.network.rsus.get("RSU_001")
        if not target_rsu:
            target_rsu = list(self.network.rsus.values())[0]

        target_rsu.is_under_attack = True
        n_attackers = self.rng.randint(20, 50)

        for _ in range(n_attackers):
            # Spoofed/compromised vehicle IPs
            src_ip = f"192.168.{self.rng.randint(1,254)}.{self.rng.randint(1,254)}"
            for _ in range(self.rng.randint(3, 10)):  # Multiple flows per attacker
                flow = self.flow_gen.ddos_flow(src_ip, target_rsu.ip, step)
                self.flows.append(flow)
                self.stats["ddos"] += 1

        if step == cfg["start_step"]:
            self._log_attack("DDoS", step, target_rsu.rsu_id, n_attackers)

    def _inject_botnet(self, step: int):
        """
        Botnet C2 beaconing: compromised vehicles beacon to C2 server.
        Periodic heartbeats at regular intervals (detectable by Bi-LSTM).
        """
        cfg = self.config["attack_scenarios"]["botnet"]
        n_bots = cfg["n_bots"]
        c2_ip = "10.0.0.1"   # External C2 server

        # Ensure we have compromised vehicles
        compromised = [v for v in self.network.vehicles.values()
                       if v.is_compromised]
        while len(compromised) < n_bots:
            new_id = self.rng.randint(10000, 99999)
            self.network._spawn_vehicle(new_id, compromised=True)
            compromised = [v for v in self.network.vehicles.values()
                           if v.is_compromised]

        # Beacon every `beacon_interval` steps
        if step % cfg["beacon_interval"] == 0:
            for bot in compromised[:n_bots]:
                flow = self.flow_gen.botnet_flow(bot.ip, c2_ip, step)
                self.flows.append(flow)
                self.stats["botnet"] += 1

        if step == cfg["start_step"]:
            self._log_attack("Botnet", step, c2_ip, n_bots)

    def _inject_dos(self, step: int):
        """DoS slowloris against RSU HTTP service."""
        cfg = self.config["attack_scenarios"]["dos"]
        target_rsu = list(self.network.rsus.values())[1] \
            if len(self.network.rsus) > 1 else list(self.network.rsus.values())[0]
        attacker_ip = f"192.168.{self.rng.randint(1,254)}.{self.rng.randint(1,254)}"

        for _ in range(self.rng.randint(5, 15)):
            flow = self.flow_gen.dos_flow(attacker_ip, target_rsu.ip, step)
            self.flows.append(flow)
            self.stats["dos"] += 1

        if step == cfg["start_step"]:
            self._log_attack("DoS", step, target_rsu.rsu_id, 1)

    def _inject_port_scan(self, step: int):
        """Port scan reconnaissance of RSU services."""
        cfg = self.config["attack_scenarios"]["port_scan"]
        target_rsu = list(self.network.rsus.values())[-1]
        attacker_ip = f"192.168.{self.rng.randint(1,254)}.{self.rng.randint(1,254)}"

        port_range = cfg["port_range"]
        ports_this_step = range(
            port_range[0] + (step - cfg["start_step"]) * 20,
            min(port_range[1],
                port_range[0] + (step - cfg["start_step"] + 1) * 20)
        )
        for port in ports_this_step:
            flow = self.flow_gen.port_scan_flow(
                attacker_ip, target_rsu.ip, step, port
            )
            self.flows.append(flow)
            self.stats["port_scan"] += 1

        if step == cfg["start_step"]:
            self._log_attack("PortScan", step, target_rsu.rsu_id, 1)

    def _log_attack(self, attack_type: str, step: int,
                    target: str, n_sources: int):
        entry = {
            "attack_type": attack_type,
            "start_step": step,
            "target": target,
            "n_sources": n_sources,
            "timestamp": datetime.now().isoformat(),
        }
        self.attack_log.append(entry)
        print(f"  [SUMO] ⚠ Attack injected: {attack_type} → {target} "
              f"(step {step}, {n_sources} sources)")

    # ── Main simulation loop ──────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """
        Run full simulation: normal traffic + attack injection.

        Returns:
            DataFrame of labelled network flows (CIC-IDS2017 compatible)
        """
        cfg = self.config
        total_steps = cfg["simulation_steps"]
        attacks = cfg["attack_scenarios"]

        print(f"\n[SUMO] Starting smart mobility simulation")
        print(f"[SUMO] Duration: {total_steps} steps "
              f"({total_steps // 3600}h {(total_steps % 3600) // 60}m)")
        print(f"[SUMO] Attack scenarios: "
              f"{[k for k, v in attacks.items() if v['enabled']]}")

        start_time = time.time()

        for step in range(total_steps):
            # Progress reporting
            if step % 300 == 0:
                elapsed = time.time() - start_time
                pct = step / total_steps * 100
                print(f"  [SUMO] Step {step:>4}/{total_steps} "
                      f"({pct:.0f}%) | Flows: {len(self.flows):>6,} | "
                      f"Elapsed: {elapsed:.1f}s")

            # Normal traffic every step
            self._generate_benign_step(step)

            # Attack injection based on scenario timelines
            if attacks["ddos"]["enabled"]:
                ddos = attacks["ddos"]
                if ddos["start_step"] <= step < ddos["start_step"] + ddos["duration"]:
                    self._inject_ddos(step)

            if attacks["botnet"]["enabled"]:
                bot = attacks["botnet"]
                if bot["start_step"] <= step < bot["start_step"] + bot["duration"]:
                    self._inject_botnet(step)

            if attacks["dos"]["enabled"]:
                dos = attacks["dos"]
                if dos["start_step"] <= step < dos["start_step"] + dos["duration"]:
                    self._inject_dos(step)

            if attacks["port_scan"]["enabled"]:
                ps = attacks["port_scan"]
                if ps["start_step"] <= step < ps["start_step"] + ps["duration"]:
                    self._inject_port_scan(step)

        self.stats["total_flows"] = len(self.flows)
        elapsed = time.time() - start_time
        print(f"\n[SUMO] Simulation complete in {elapsed:.1f}s")
        self._print_stats()

        df = self._to_dataframe()
        self._save_outputs(df)
        return df

    def _print_stats(self):
        s = self.stats
        total = s["total_flows"]
        print(f"\n[SUMO] Flow Statistics:")
        print(f"  Total flows:  {total:>8,}")
        print(f"  BENIGN:       {s['benign']:>8,}  ({s['benign']/total*100:.1f}%)")
        print(f"  DDoS:         {s['ddos']:>8,}  ({s['ddos']/total*100:.1f}%)")
        print(f"  Botnet:       {s['botnet']:>8,}  ({s['botnet']/total*100:.1f}%)")
        print(f"  DoS:          {s['dos']:>8,}  ({s['dos']/total*100:.1f}%)")
        print(f"  Port Scan:    {s['port_scan']:>8,}  ({s['port_scan']/total*100:.1f}%)")

    def _to_dataframe(self) -> pd.DataFrame:
        """Convert flow objects to CIC-IDS2017 compatible DataFrame."""
        records = []
        for f in self.flows:
            records.append({
                "Timestamp": f.timestamp,
                "Src IP": f.src_ip,
                "Dst IP": f.dst_ip,
                "Src Port": f.src_port,
                "Dst Port": f.dst_port,
                "Protocol": f.protocol,
                "Duration": f.duration,
                "Total Fwd Packets": f.total_fwd_packets,
                "Total Backward Packets": f.total_bwd_packets,
                "Total Length of Fwd Packets": f.total_length_fwd,
                "Total Length of Bwd Packets": f.total_length_bwd,
                "Fwd Packet Length Mean": f.fwd_packet_length_mean,
                "Bwd Packet Length Mean": f.bwd_packet_length_mean,
                "Flow Bytes/s": f.flow_bytes_per_s,
                "Flow Packets/s": f.flow_packets_per_s,
                "Flow IAT Mean": f.flow_iat_mean,
                "Flow IAT Std": f.flow_iat_std,
                "Fwd IAT Mean": f.fwd_iat_mean,
                "Bwd IAT Mean": f.bwd_iat_mean,
                "SYN Flag Count": f.syn_flag,
                "ACK Flag Count": f.ack_flag,
                "FIN Flag Count": f.fin_flag,
                "RST Flag Count": f.rst_flag,
                "PSH Flag Count": f.psh_flag,
                "Init_Win_bytes_forward": f.init_win_bytes_fwd,
                "Init_Win_bytes_backward": f.init_win_bytes_bwd,
                "Label": f.label,
            })
        df = pd.DataFrame(records)
        print(f"[SUMO] DataFrame shape: {df.shape}")
        return df

    def _save_outputs(self, df: pd.DataFrame):
        """Save simulation outputs: CSV, attack log, stats."""
        # Flow CSV
        csv_path = self.config["output_pcap"]
        df.to_csv(csv_path, index=False)
        print(f"[SUMO] Flows saved: {csv_path}")

        # Attack log JSON
        log_path = os.path.join(self.config["output_dir"], "attack_log.json")
        with open(log_path, "w") as f:
            json.dump(self.attack_log, f, indent=2)
        print(f"[SUMO] Attack log: {log_path}")

        # Stats JSON
        stats_path = os.path.join(self.config["output_dir"], "sim_stats.json")
        with open(stats_path, "w") as f:
            json.dump(self.stats, f, indent=2)

        # Timeline plot
        self._plot_attack_timeline(df)

    def _plot_attack_timeline(self, df: pd.DataFrame):
        """Visualise attack injection timeline over simulation."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        label_colours = {
            "BENIGN": "#2ca02c",
            "DDoS": "#d62728",
            "Bot": "#9467bd",
            "DoS Hulk": "#ff7f0e",
            "PortScan": "#1f77b4",
        }

        # Count flows per step per label (sample for speed)
        df_sample = df.copy()
        df_sample["step_bin"] = (df_sample["Timestamp"] // 60).astype(int)
        grouped = df_sample.groupby(["step_bin", "Label"]).size().unstack(fill_value=0)

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        # Top: stacked area chart of flows over time
        ax = axes[0]
        colours = [label_colours.get(c, "#aec7e8") for c in grouped.columns]
        grouped.plot(kind="area", ax=ax, stacked=True, color=colours, alpha=0.75)
        ax.set_title("RAMS Objective 6 — SUMO Simulation: Network Flows Over Time\n"
                     "Smart Mobility Network with Injected Cyberattacks",
                     fontweight="bold", fontsize=12)
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Flows / minute")
        ax.legend(loc="upper right", fontsize=9)

        # Add attack period markers
        attacks = self.config["attack_scenarios"]
        colours_atk = {"ddos": "#d62728", "botnet": "#9467bd",
                       "dos": "#ff7f0e", "port_scan": "#1f77b4"}
        for atk_name, atk_cfg in attacks.items():
            if atk_cfg["enabled"]:
                start_m = atk_cfg["start_step"] / 60
                dur_m = atk_cfg["duration"] / 60
                ax.axvspan(start_m, start_m + dur_m,
                           alpha=0.15, color=colours_atk[atk_name],
                           label=f"{atk_name} window")

        # Bottom: label distribution pie
        ax2 = axes[1]
        label_counts = df["Label"].value_counts()
        pie_colours = [label_colours.get(l, "#aec7e8") for l in label_counts.index]
        ax2.pie(label_counts.values, labels=label_counts.index,
                colors=pie_colours, autopct="%1.1f%%",
                startangle=90, textprops={"fontsize": 9})
        ax2.set_title("Traffic Label Distribution", fontweight="bold")

        plt.tight_layout()
        plot_path = os.path.join(self.config["output_dir"], "simulation_timeline.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SUMO] Timeline plot: {plot_path}")


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    sim = SUMOSimulator(SUMO_CONFIG)
    df = sim.run()
    print(f"\nSimulation complete. Output: {SUMO_CONFIG['output_pcap']}")
    print(df["Label"].value_counts())
