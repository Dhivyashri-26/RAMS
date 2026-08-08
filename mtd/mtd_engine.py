"""
mtd/mtd_engine.py — Moving Target Defense Engine
RAMS Framework — Objective 5

Implements autonomous cyber defense mechanisms triggered by threat alerts
from the Hybrid ML Engine (Objective 3).

MTD Actions (from project document):
  1. IP Shuffling       — reassign virtual IPs to confuse attackers
  2. Port Hopping       — rotate service ports on a schedule/trigger
  3. Container Mutation — swap container images to break persistence
  4. Pod Recreation     — destroy + recreate pods to evict active threats

Design:
  - No real Kubernetes/AWS required — simulates the full MTD lifecycle
  - Pluggable: swap SimulatedKubernetes with real kubectl/boto3 calls
  - Maintains a state machine per protected service
  - Logs every action for the feedback loop (Obj system learning)
  - Evaluates MTD effectiveness: measures attack success before/after

Reference: "Enhancing Ransomware Resilience in Cloud-Based HR Systems
            through Moving Target Defense"
"""

import os
import sys
import json
import time
import random
import hashlib
import ipaddress
import threading
from enum import Enum
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR, THREAT_SEVERITY


# ══════════════════════════════════════════════════════════════════
# ENUMS & DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════

class MTDAction(Enum):
    NONE            = "none"
    IP_SHUFFLING    = "ip_shuffling"
    PORT_HOPPING    = "port_hopping"
    CONTAINER_MUT   = "container_mutation"
    POD_RECREATION  = "pod_recreation"


class ServiceState(Enum):
    NORMAL      = "normal"
    UNDER_ATTACK = "under_attack"
    MUTATING    = "mutating"
    RECOVERED   = "recovered"


class MTDTriggerReason(Enum):
    DDOS        = "DDoS detected"
    BOTNET      = "Botnet C2 detected"
    INFILTRATION = "Infiltration detected"
    PORT_SCAN   = "Port scan detected"
    HIGH_CONF   = "High-confidence threat"
    SCHEDULED   = "Scheduled rotation"
    MANUAL      = "Manual trigger"


@dataclass
class ProtectedService:
    """Represents a cloud service protected by MTD."""
    service_id: str
    service_name: str
    current_ip: str
    current_port: int
    container_image: str
    pod_name: str
    state: ServiceState = ServiceState.NORMAL
    mtd_history: List[dict] = field(default_factory=list)
    attack_count: int = 0
    last_mtd_time: Optional[datetime] = None
    # Metrics
    uptime_start: datetime = field(default_factory=datetime.now)
    total_mtd_actions: int = 0


@dataclass
class ThreatAlert:
    """Incoming threat alert from the Hybrid ML Engine."""
    alert_id: str
    timestamp: datetime
    threat_type: str
    severity: str          # none / low / medium / high / critical
    confidence: float
    source_ip: str
    target_service_id: str
    mtd_action: MTDAction
    shap_top_feature: Optional[str] = None


@dataclass
class MTDEvent:
    """Records a single MTD action execution."""
    event_id: str
    timestamp: datetime
    service_id: str
    action: MTDAction
    trigger_reason: MTDTriggerReason
    old_state: dict           # Before: {ip, port, container, pod}
    new_state: dict           # After:  {ip, port, container, pod}
    execution_time_ms: float
    success: bool
    attack_neutralised: bool = False


# ══════════════════════════════════════════════════════════════════
# SIMULATED INFRASTRUCTURE CONTROLLER
# ══════════════════════════════════════════════════════════════════

class SimulatedInfrastructure:
    """
    Simulates Kubernetes/Terraform operations.
    In production: replace each method with real kubectl / boto3 / terraform calls.

    How to connect to real Kubernetes:
      pip install kubernetes
      from kubernetes import client, config
      config.load_kube_config()   # uses ~/.kube/config
      v1 = client.CoreV1Api()
      v1.patch_namespaced_pod(name=pod_name, namespace="default", body={...})
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        # Available IP pool (simulated VPC subnet)
        self._ip_pool = [
            str(ipaddress.IPv4Address("10.0.1.1") + i)
            for i in range(1, 255)
        ]
        self._used_ips = set()
        # Available port ranges per service type
        self._port_pools = {
            "web":    list(range(8000, 8100)),
            "api":    list(range(9000, 9100)),
            "db":     list(range(5432, 5532)),
            "cache":  list(range(6379, 6479)),
            "default": list(range(10000, 11000)),
        }
        # Container image registry (simulated)
        self._image_registry = {
            "nginx":   ["nginx:1.24", "nginx:1.25", "nginx:alpine", "nginx:1.26"],
            "flask":   ["flask-app:v1.0", "flask-app:v1.1", "flask-app:v1.2"],
            "fastapi": ["fastapi-app:v2.0", "fastapi-app:v2.1", "fastapi-app:v2.2"],
            "default": ["service:v1.0", "service:v1.1", "service:v2.0"],
        }

    def allocate_new_ip(self, exclude: str = None) -> str:
        """Assign a new IP from the pool (IP Shuffling action)."""
        available = [ip for ip in self._ip_pool
                     if ip not in self._used_ips and ip != exclude]
        if not available:
            # Recycle oldest IPs
            self._used_ips.clear()
            available = [ip for ip in self._ip_pool if ip != exclude]
        new_ip = self.rng.choice(available)
        self._used_ips.add(new_ip)
        if exclude:
            self._used_ips.discard(exclude)
        # Simulate API call latency
        time.sleep(self.rng.uniform(0.01, 0.05))
        return new_ip

    def allocate_new_port(self, service_type: str = "default",
                           exclude: int = None) -> int:
        """Rotate to a new port (Port Hopping action)."""
        pool = self._port_pools.get(service_type,
                                     self._port_pools["default"])
        available = [p for p in pool if p != exclude]
        new_port = self.rng.choice(available)
        time.sleep(self.rng.uniform(0.005, 0.02))
        return new_port

    def pull_new_image(self, current_image: str) -> str:
        """
        Pull a different container image (Container Mutation).
        Breaks attacker's knowledge of the running software stack.
        """
        service_type = current_image.split(":")[0].split("-")[0]
        images = self._image_registry.get(
            service_type, self._image_registry["default"]
        )
        available = [img for img in images if img != current_image]
        new_image = self.rng.choice(available) if available else images[0]
        # Simulate docker pull latency
        time.sleep(self.rng.uniform(0.05, 0.2))
        return new_image

    def recreate_pod(self, pod_name: str, image: str) -> str:
        """
        Delete + recreate pod (Pod Recreation).
        Evicts any malware/rootkit that persisted in the running container.

        Real implementation:
          kubectl delete pod <pod_name> --namespace=default
          kubectl run <new_pod_name> --image=<image> --restart=Never
        """
        # Generate new pod name with timestamp hash
        ts_hash = hashlib.md5(
            datetime.now().isoformat().encode()
        ).hexdigest()[:6]
        base_name = pod_name.split("-")[0]
        new_pod_name = f"{base_name}-{ts_hash}"
        # Simulate pod termination + startup
        time.sleep(self.rng.uniform(0.1, 0.5))
        return new_pod_name

    def update_dns(self, service_id: str, new_ip: str, new_port: int):
        """Update service discovery / DNS record after IP/port change."""
        time.sleep(self.rng.uniform(0.01, 0.03))
        return True

    def update_firewall(self, old_port: int, new_port: int,
                         service_id: str) -> bool:
        """Update firewall rules after port hop."""
        time.sleep(self.rng.uniform(0.005, 0.015))
        return True


# ══════════════════════════════════════════════════════════════════
# MTD ACTION HANDLERS
# ══════════════════════════════════════════════════════════════════

class MTDActionExecutor:
    """
    Executes the four MTD actions against a protected service.
    Each action is logged, timed, and returns an MTDEvent.
    """

    def __init__(self, infra: SimulatedInfrastructure):
        self.infra = infra

    def _make_event(self, service: ProtectedService,
                     action: MTDAction,
                     reason: MTDTriggerReason,
                     old_state: dict, new_state: dict,
                     exec_time: float, success: bool) -> MTDEvent:
        return MTDEvent(
            event_id=f"MTD-{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            timestamp=datetime.now(),
            service_id=service.service_id,
            action=action,
            trigger_reason=reason,
            old_state=old_state,
            new_state=new_state,
            execution_time_ms=exec_time,
            success=success,
        )

    def ip_shuffling(self, service: ProtectedService,
                      reason: MTDTriggerReason) -> MTDEvent:
        """
        IP Shuffling: reassign virtual IP to break active DDoS targeting.
        Attacker's flood traffic hits the old IP, which is now unused.
        """
        print(f"    [MTD] IP Shuffling → {service.service_id}")
        old_state = {"ip": service.current_ip, "port": service.current_port,
                     "container": service.container_image,
                     "pod": service.pod_name}
        t0 = time.time()
        try:
            new_ip = self.infra.allocate_new_ip(exclude=service.current_ip)
            self.infra.update_dns(service.service_id, new_ip,
                                   service.current_port)
            service.current_ip = new_ip
            service.state = ServiceState.RECOVERED
            success = True
            print(f"      Old IP: {old_state['ip']} → New IP: {new_ip}")
        except Exception as e:
            print(f"      [MTD] IP Shuffling failed: {e}")
            success = False
        exec_time = (time.time() - t0) * 1000
        new_state = {"ip": service.current_ip, "port": service.current_port,
                     "container": service.container_image,
                     "pod": service.pod_name}
        return self._make_event(service, MTDAction.IP_SHUFFLING, reason,
                                 old_state, new_state, exec_time, success)

    def port_hopping(self, service: ProtectedService,
                      reason: MTDTriggerReason,
                      service_type: str = "default") -> MTDEvent:
        """
        Port Hopping: rotate service port to evade port-scan targeting.
        Attackers lose visibility of the service endpoint.
        """
        print(f"    [MTD] Port Hopping → {service.service_id}")
        old_state = {"ip": service.current_ip, "port": service.current_port,
                     "container": service.container_image,
                     "pod": service.pod_name}
        t0 = time.time()
        try:
            new_port = self.infra.allocate_new_port(
                service_type, exclude=service.current_port
            )
            self.infra.update_firewall(service.current_port, new_port,
                                        service.service_id)
            service.current_port = new_port
            service.state = ServiceState.RECOVERED
            success = True
            print(f"      Old Port: {old_state['port']} → New Port: {new_port}")
        except Exception as e:
            print(f"      [MTD] Port Hopping failed: {e}")
            success = False
        exec_time = (time.time() - t0) * 1000
        new_state = {"ip": service.current_ip, "port": service.current_port,
                     "container": service.container_image,
                     "pod": service.pod_name}
        return self._make_event(service, MTDAction.PORT_HOPPING, reason,
                                 old_state, new_state, exec_time, success)

    def container_mutation(self, service: ProtectedService,
                            reason: MTDTriggerReason) -> MTDEvent:
        """
        Container Mutation: swap container image + restart.
        Breaks attacker's fingerprinting of running software/versions.
        Also rotates IP to compound the defense.
        """
        print(f"    [MTD] Container Mutation → {service.service_id}")
        old_state = {"ip": service.current_ip, "port": service.current_port,
                     "container": service.container_image,
                     "pod": service.pod_name}
        service.state = ServiceState.MUTATING
        t0 = time.time()
        try:
            new_image = self.infra.pull_new_image(service.container_image)
            new_ip = self.infra.allocate_new_ip(exclude=service.current_ip)
            new_pod = self.infra.recreate_pod(service.pod_name, new_image)
            self.infra.update_dns(service.service_id, new_ip,
                                   service.current_port)
            service.container_image = new_image
            service.current_ip = new_ip
            service.pod_name = new_pod
            service.state = ServiceState.RECOVERED
            success = True
            print(f"      Old Image: {old_state['container']} → {new_image}")
            print(f"      Old IP:    {old_state['ip']} → {new_ip}")
        except Exception as e:
            print(f"      [MTD] Container Mutation failed: {e}")
            service.state = ServiceState.UNDER_ATTACK
            success = False
        exec_time = (time.time() - t0) * 1000
        new_state = {"ip": service.current_ip, "port": service.current_port,
                     "container": service.container_image,
                     "pod": service.pod_name}
        return self._make_event(service, MTDAction.CONTAINER_MUT, reason,
                                 old_state, new_state, exec_time, success)

    def pod_recreation(self, service: ProtectedService,
                        reason: MTDTriggerReason) -> MTDEvent:
        """
        Pod Recreation: full destroy + recreate with new IP, port, image.
        Most aggressive action — evicts rootkits, breaks all persistence.
        Used for: Infiltration, Heartbleed, SQL Injection (critical severity).
        """
        print(f"    [MTD] Pod Recreation → {service.service_id} "
              f"(FULL RESET)")
        old_state = {"ip": service.current_ip, "port": service.current_port,
                     "container": service.container_image,
                     "pod": service.pod_name}
        service.state = ServiceState.MUTATING
        t0 = time.time()
        try:
            new_image = self.infra.pull_new_image(service.container_image)
            new_ip = self.infra.allocate_new_ip(exclude=service.current_ip)
            new_port = self.infra.allocate_new_port(
                exclude=service.current_port
            )
            new_pod = self.infra.recreate_pod(service.pod_name, new_image)
            self.infra.update_dns(service.service_id, new_ip, new_port)
            self.infra.update_firewall(service.current_port, new_port,
                                        service.service_id)
            service.current_ip = new_ip
            service.current_port = new_port
            service.container_image = new_image
            service.pod_name = new_pod
            service.state = ServiceState.RECOVERED
            success = True
            print(f"      New IP:    {new_ip}")
            print(f"      New Port:  {new_port}")
            print(f"      New Image: {new_image}")
            print(f"      New Pod:   {new_pod}")
        except Exception as e:
            print(f"      [MTD] Pod Recreation failed: {e}")
            service.state = ServiceState.UNDER_ATTACK
            success = False
        exec_time = (time.time() - t0) * 1000
        new_state = {"ip": service.current_ip, "port": service.current_port,
                     "container": service.container_image,
                     "pod": service.pod_name}
        return self._make_event(service, MTDAction.POD_RECREATION, reason,
                                 old_state, new_state, exec_time, success)


# ══════════════════════════════════════════════════════════════════
# MTD POLICY ENGINE
# ══════════════════════════════════════════════════════════════════

class MTDPolicy:
    """
    Decides WHICH MTD action to apply based on:
      - Threat type
      - Severity
      - Confidence score
      - Service state history
      - Cooldown periods (avoid thrashing)

    Policy table from project document + MTD reference paper:
    ┌─────────────────────┬──────────────┬────────────────────┐
    │ Threat              │ Severity     │ MTD Action         │
    ├─────────────────────┼──────────────┼────────────────────┤
    │ DDoS                │ critical     │ IP Shuffling       │
    │ PortScan            │ low/medium   │ Port Hopping       │
    │ Bot / Botnet        │ critical     │ Container Mutation  │
    │ Infiltration        │ critical     │ Pod Recreation     │
    │ Heartbleed          │ critical     │ Pod Recreation     │
    │ SQL Injection       │ critical     │ Pod Recreation     │
    │ XSS                 │ high         │ Container Mutation  │
    │ DoS variants        │ medium       │ IP Shuffling       │
    │ Brute Force         │ high         │ Port Hopping       │
    └─────────────────────┴──────────────┴────────────────────┘
    """

    # Minimum seconds between MTD actions per service (avoid thrashing)
    COOLDOWN = {
        MTDAction.IP_SHUFFLING:   30,
        MTDAction.PORT_HOPPING:   20,
        MTDAction.CONTAINER_MUT:  60,
        MTDAction.POD_RECREATION: 120,
    }

    # Minimum confidence to trigger MTD
    MIN_CONFIDENCE = {
        "critical": 0.65,
        "high":     0.75,
        "medium":   0.85,
        "low":      0.95,
    }

    THREAT_TO_ACTION = {
        "DDoS":                          MTDAction.IP_SHUFFLING,
        "PortScan":                      MTDAction.PORT_HOPPING,
        "Bot":                           MTDAction.CONTAINER_MUT,
        "Infiltration":                  MTDAction.POD_RECREATION,
        "Heartbleed":                    MTDAction.POD_RECREATION,
        "Web Attack \x96 Sql Injection": MTDAction.POD_RECREATION,
        "Web Attack \x96 XSS":           MTDAction.CONTAINER_MUT,
        "Web Attack \x96 Brute Force":   MTDAction.PORT_HOPPING,
        "DoS Hulk":                      MTDAction.IP_SHUFFLING,
        "DoS GoldenEye":                 MTDAction.IP_SHUFFLING,
        "DoS slowloris":                 MTDAction.IP_SHUFFLING,
        "DoS Slowhttptest":              MTDAction.IP_SHUFFLING,
        "FTP-Patator":                   MTDAction.PORT_HOPPING,
        "SSH-Patator":                   MTDAction.PORT_HOPPING,
    }

    def decide(self, alert: ThreatAlert,
                service: ProtectedService) -> Optional[MTDAction]:
        """
        Decide the MTD action for a given alert + service state.
        Returns None if action should be skipped (cooldown / low confidence).
        """
        # Skip if BENIGN or no action needed
        if alert.severity == "none" or alert.threat_type == "BENIGN":
            return None

        # Confidence gate
        min_conf = self.MIN_CONFIDENCE.get(alert.severity, 0.90)
        if alert.confidence < min_conf:
            return None

        # Cooldown check
        action = self.THREAT_TO_ACTION.get(
            alert.threat_type, MTDAction.IP_SHUFFLING
        )
        if service.last_mtd_time:
            elapsed = (datetime.now() - service.last_mtd_time).total_seconds()
            if elapsed < self.COOLDOWN.get(action, 30):
                return None   # Still in cooldown

        # Escalate if service is repeatedly attacked
        if service.attack_count >= 3:
            # Escalate to more aggressive action
            if action == MTDAction.IP_SHUFFLING:
                action = MTDAction.CONTAINER_MUT
            elif action in (MTDAction.PORT_HOPPING, MTDAction.CONTAINER_MUT):
                action = MTDAction.POD_RECREATION

        return action

    def get_reason(self, threat_type: str) -> MTDTriggerReason:
        mapping = {
            "DDoS": MTDTriggerReason.DDOS,
            "Bot": MTDTriggerReason.BOTNET,
            "Infiltration": MTDTriggerReason.INFILTRATION,
            "PortScan": MTDTriggerReason.PORT_SCAN,
        }
        for key, reason in mapping.items():
            if key in threat_type:
                return reason
        return MTDTriggerReason.HIGH_CONF


# ══════════════════════════════════════════════════════════════════
# MAIN MTD ENGINE
# ══════════════════════════════════════════════════════════════════

class MTDEngine:
    """
    RAMS Objective 5 — Moving Target Defense Engine.

    Receives threat alerts from the Hybrid ML Engine,
    applies MTD policy decisions, executes defense actions,
    and logs everything for the feedback loop.

    Integration with Obj 3:
      hybrid_engine.generate_alert() → MTDEngine.process_alert()

    Scheduled rotation thread runs independently (proactive MTD).
    """

    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or os.path.join(RESULTS_DIR, "mtd")
        os.makedirs(self.output_dir, exist_ok=True)

        self.infra = SimulatedInfrastructure()
        self.executor = MTDActionExecutor(self.infra)
        self.policy = MTDPolicy()

        self.services: Dict[str, ProtectedService] = {}
        self.event_log: List[MTDEvent] = []
        self.alert_log: List[dict] = []

        self._scheduled_rotation_running = False
        self._rotation_thread = None

        self._setup_default_services()
        print("[MTD] Moving Target Defense Engine initialised (Objective 5)")
        print(f"[MTD] Protected services: {list(self.services.keys())}")

    def _setup_default_services(self):
        """Register the cloud services that MTD protects."""
        defaults = [
            ("svc-web",   "Web Gateway",    "10.0.1.10", 8080,
             "nginx:1.24",    "web-pod-abc123"),
            ("svc-api",   "REST API",       "10.0.1.11", 9000,
             "fastapi-app:v2.0", "api-pod-def456"),
            ("svc-db",    "Database Proxy", "10.0.1.12", 5432,
             "pgproxy:v1.0",  "db-pod-ghi789"),
            ("svc-cache", "Cache Layer",    "10.0.1.13", 6379,
             "flask-app:v1.1", "cache-pod-jkl012"),
        ]
        for sid, name, ip, port, image, pod in defaults:
            self.services[sid] = ProtectedService(
                service_id=sid, service_name=name,
                current_ip=ip, current_port=port,
                container_image=image, pod_name=pod,
            )

    def register_service(self, service_id: str, service_name: str,
                          ip: str, port: int,
                          container_image: str, pod_name: str):
        """Register a new service for MTD protection."""
        self.services[service_id] = ProtectedService(
            service_id=service_id, service_name=service_name,
            current_ip=ip, current_port=port,
            container_image=container_image, pod_name=pod_name,
        )
        print(f"[MTD] Registered service: {service_id} ({service_name})")

    # ── Core: Process incoming alert ─────────────────────────────

    def process_alert(self, alert: ThreatAlert) -> Optional[MTDEvent]:
        """
        Main entry point: receive a threat alert, decide + execute MTD.

        Args:
            alert: ThreatAlert from Hybrid ML Engine (Obj 3)

        Returns:
            MTDEvent if action was taken, None otherwise
        """
        self.alert_log.append({
            "alert_id": alert.alert_id,
            "timestamp": alert.timestamp.isoformat(),
            "threat_type": alert.threat_type,
            "severity": alert.severity,
            "confidence": alert.confidence,
            "target": alert.target_service_id,
        })

        # Get target service
        service = self.services.get(alert.target_service_id)
        if service is None:
            # Default to web service if target unknown
            service = list(self.services.values())[0]

        service.attack_count += 1
        service.state = ServiceState.UNDER_ATTACK

        # Policy decision
        action = self.policy.decide(alert, service)
        if action is None:
            return None   # Cooldown or below confidence threshold

        reason = self.policy.get_reason(alert.threat_type)

        print(f"\n  [MTD] Alert received: {alert.threat_type} "
              f"(sev={alert.severity}, conf={alert.confidence:.2f})")
        print(f"  [MTD] Action selected: {action.value}")

        # Execute action
        event = self._execute_action(service, action, reason)

        if event and event.success:
            service.last_mtd_time = datetime.now()
            service.total_mtd_actions += 1
            event.attack_neutralised = True
            self.event_log.append(event)
            print(f"  [MTD] ✓ Action complete in {event.execution_time_ms:.1f}ms")

        return event

    def _execute_action(self, service: ProtectedService,
                         action: MTDAction,
                         reason: MTDTriggerReason) -> Optional[MTDEvent]:
        """Dispatch to the correct action handler."""
        if action == MTDAction.IP_SHUFFLING:
            return self.executor.ip_shuffling(service, reason)
        elif action == MTDAction.PORT_HOPPING:
            return self.executor.port_hopping(service, reason)
        elif action == MTDAction.CONTAINER_MUT:
            return self.executor.container_mutation(service, reason)
        elif action == MTDAction.POD_RECREATION:
            return self.executor.pod_recreation(service, reason)
        return None

    # ── Batch processing from pipeline alerts ────────────────────

    def process_alert_batch(self, alerts: List[dict]) -> dict:
        """
        Process a batch of alert dicts from the streaming pipeline.
        Converts dicts → ThreatAlert objects and processes each.
        """
        print(f"\n[MTD] Processing {len(alerts)} alerts...")
        events = []
        actions_taken = {}

        for a in alerts:
            # Map alert dict to ThreatAlert
            threat = a.get("threat_type", "Unknown")
            sev = a.get("severity", "medium")

            # Assign to appropriate service based on threat type
            target = self._assign_service(threat)

            alert_obj = ThreatAlert(
                alert_id=a.get("alert_id", "UNKNOWN"),
                timestamp=datetime.now(),
                threat_type=threat,
                severity=sev,
                confidence=a.get("confidence", 0.8),
                source_ip=a.get("source_flow", {}).get("Src IP", "0.0.0.0"),
                target_service_id=target,
                mtd_action=MTDAction(a.get("recommended_mtd_action", "none")
                                     if a.get("recommended_mtd_action", "none")
                                     in [m.value for m in MTDAction]
                                     else "none"),
            )
            event = self.process_alert(alert_obj)
            if event:
                events.append(event)
                act = event.action.value
                actions_taken[act] = actions_taken.get(act, 0) + 1

        summary = {
            "total_alerts": len(alerts),
            "actions_executed": len(events),
            "actions_skipped": len(alerts) - len(events),
            "actions_by_type": actions_taken,
            "success_rate": (sum(1 for e in events if e.success) /
                             len(events) if events else 0),
        }
        print(f"\n[MTD] Batch complete: {len(events)} actions executed")
        return summary

    def _assign_service(self, threat_type: str) -> str:
        """Map threat type to most likely target service."""
        if any(t in threat_type for t in ["SQL", "XSS", "Web"]):
            return "svc-web"
        elif any(t in threat_type for t in ["DDoS", "DoS"]):
            return "svc-api"
        elif any(t in threat_type for t in ["Bot", "Infiltration"]):
            return "svc-db"
        elif any(t in threat_type for t in ["Port", "Scan"]):
            return "svc-cache"
        return "svc-web"

    # ── Scheduled (proactive) rotation ───────────────────────────

    def start_scheduled_rotation(self, interval_seconds: int = 300):
        """
        Proactive MTD: rotate IPs and ports on a schedule
        regardless of detected attacks. Makes the attack surface
        unpredictable even without a detected threat.
        """
        print(f"[MTD] Starting scheduled rotation every {interval_seconds}s...")
        self._scheduled_rotation_running = True

        def _rotate():
            while self._scheduled_rotation_running:
                time.sleep(interval_seconds)
                if not self._scheduled_rotation_running:
                    break
                print("\n[MTD] Scheduled rotation triggered...")
                for service in self.services.values():
                    if service.state != ServiceState.MUTATING:
                        event = self.executor.port_hopping(
                            service, MTDTriggerReason.SCHEDULED
                        )
                        if event:
                            self.event_log.append(event)

        self._rotation_thread = threading.Thread(
            target=_rotate, daemon=True
        )
        self._rotation_thread.start()

    def stop_scheduled_rotation(self):
        self._scheduled_rotation_running = False

    # ── Evaluation & Reporting ───────────────────────────────────

    def evaluate_effectiveness(self) -> dict:
        """
        Compute MTD effectiveness metrics:
          - Mean time to respond (MTTR)
          - Action success rate
          - Attack neutralisation rate
          - Mean execution time per action type
          - Service availability (uptime during attacks)
        """
        if not self.event_log:
            return {"message": "No MTD events recorded yet"}

        total = len(self.event_log)
        successful = sum(1 for e in self.event_log if e.success)
        neutralised = sum(1 for e in self.event_log if e.attack_neutralised)

        exec_times = [e.execution_time_ms for e in self.event_log]
        by_action = {}
        for e in self.event_log:
            k = e.action.value
            if k not in by_action:
                by_action[k] = {"count": 0, "success": 0,
                                  "total_ms": 0.0}
            by_action[k]["count"] += 1
            by_action[k]["success"] += int(e.success)
            by_action[k]["total_ms"] += e.execution_time_ms

        action_stats = {
            k: {
                "count": v["count"],
                "success_rate": v["success"] / v["count"],
                "avg_exec_ms": v["total_ms"] / v["count"],
            }
            for k, v in by_action.items()
        }

        metrics = {
            "total_mtd_actions": total,
            "successful_actions": successful,
            "success_rate": successful / total,
            "attacks_neutralised": neutralised,
            "neutralisation_rate": neutralised / max(len(self.alert_log), 1),
            "mean_execution_ms": float(np.mean(exec_times)),
            "max_execution_ms": float(np.max(exec_times)),
            "min_execution_ms": float(np.min(exec_times)),
            "actions_by_type": action_stats,
            "services": {
                sid: {
                    "total_actions": s.total_mtd_actions,
                    "attack_count": s.attack_count,
                    "current_state": s.state.value,
                    "current_ip": s.current_ip,
                    "current_port": s.current_port,
                    "current_image": s.container_image,
                }
                for sid, s in self.services.items()
            },
        }

        # Print summary
        print(f"\n{'═'*60}")
        print(f"  RAMS — Objective 5: MTD Effectiveness Report")
        print(f"{'═'*60}")
        print(f"  Total MTD actions:      {total:>8}")
        print(f"  Success rate:           {successful/total:>8.1%}")
        print(f"  Attacks neutralised:    {neutralised:>8}")
        print(f"  Mean response time:     {np.mean(exec_times):>8.1f}ms")
        print(f"\n  Actions by type:")
        for atype, stats in action_stats.items():
            print(f"    {atype:<25} count={stats['count']:>4}  "
                  f"success={stats['success_rate']:.0%}  "
                  f"avg={stats['avg_exec_ms']:.1f}ms")
        print(f"{'═'*60}")

        return metrics

    def save_report(self) -> str:
        """Save full MTD event log and effectiveness report."""
        metrics = self.evaluate_effectiveness()

        report = {
            "timestamp": datetime.now().isoformat(),
            "framework": "RAMS — Objective 5: MTD",
            "effectiveness": metrics,
            "event_log": [
                {
                    "event_id": e.event_id,
                    "timestamp": e.timestamp.isoformat(),
                    "service_id": e.service_id,
                    "action": e.action.value,
                    "reason": e.trigger_reason.value,
                    "old_state": e.old_state,
                    "new_state": e.new_state,
                    "execution_ms": e.execution_time_ms,
                    "success": e.success,
                    "neutralised": e.attack_neutralised,
                }
                for e in self.event_log
            ],
            "alert_log": self.alert_log,
        }

        path = os.path.join(self.output_dir, "mtd_report.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[MTD] Report saved: {path}")
        return path

    def plot_results(self):
        """Generate MTD evaluation visualisations."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.event_log:
            print("[MTD] No events to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # ── Plot 1: Actions over time ──────────────────────────────
        ax = axes[0, 0]
        action_colours = {
            "ip_shuffling": "#1f77b4",
            "port_hopping": "#ff7f0e",
            "container_mutation": "#2ca02c",
            "pod_recreation": "#d62728",
        }
        times = [e.timestamp for e in self.event_log]
        actions = [e.action.value for e in self.event_log]
        for i, (t, a) in enumerate(zip(times, actions)):
            ax.scatter(i, list(action_colours.keys()).index(a)
                       if a in action_colours else 0,
                       c=action_colours.get(a, "#aec7e8"),
                       s=100, zorder=3)
        ax.set_yticks(range(len(action_colours)))
        ax.set_yticklabels(list(action_colours.keys()), fontsize=9)
        ax.set_xlabel("Event #")
        ax.set_title("MTD Actions Timeline", fontweight="bold")
        ax.grid(alpha=0.3)

        # ── Plot 2: Action type distribution ──────────────────────
        ax2 = axes[0, 1]
        from collections import Counter
        action_counts = Counter(e.action.value for e in self.event_log)
        bars = ax2.bar(action_counts.keys(), action_counts.values(),
                       color=[action_colours.get(k, "#aec7e8")
                              for k in action_counts.keys()],
                       alpha=0.85)
        for bar, val in zip(bars, action_counts.values()):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.05, str(val),
                     ha="center", fontweight="bold")
        ax2.set_title("MTD Action Distribution", fontweight="bold")
        ax2.set_xlabel("Action Type")
        ax2.set_ylabel("Count")
        plt.sca(ax2)
        plt.xticks(rotation=20, ha="right", fontsize=8)
        ax2.grid(axis="y", alpha=0.3)

        # ── Plot 3: Execution times ────────────────────────────────
        ax3 = axes[1, 0]
        exec_by_action = {}
        for e in self.event_log:
            exec_by_action.setdefault(e.action.value, []).append(
                e.execution_time_ms
            )
        labels = list(exec_by_action.keys())
        data = [exec_by_action[l] for l in labels]
        # FIXED
        bp = ax3.boxplot(data, tick_labels=labels, patch_artist=True)
        for patch, label in zip(bp["boxes"], labels):
            patch.set_facecolor(action_colours.get(label, "#aec7e8"))
            patch.set_alpha(0.7)
        ax3.set_ylabel("Execution Time (ms)")
        ax3.set_title("MTD Response Time Distribution",
                       fontweight="bold")
        plt.sca(ax3)
        plt.xticks(rotation=20, ha="right", fontsize=8)
        ax3.grid(axis="y", alpha=0.3)

        # ── Plot 4: Service state summary ─────────────────────────
        ax4 = axes[1, 1]
        states = [s.state.value for s in self.services.values()]
        svc_names = [s.service_name for s in self.services.values()]
        state_colours = {
            "normal": "#2ca02c", "under_attack": "#d62728",
            "mutating": "#ff7f0e", "recovered": "#1f77b4",
        }
        clrs = [state_colours.get(st, "#aec7e8") for st in states]
        bars4 = ax4.barh(svc_names,
                          [s.total_mtd_actions for s in self.services.values()],
                          color=clrs, alpha=0.85)
        for bar, svc in zip(bars4, self.services.values()):
            ax4.text(bar.get_width() + 0.05,
                     bar.get_y() + bar.get_height()/2,
                     f"{svc.state.value} | {svc.total_mtd_actions} actions",
                     va="center", fontsize=8)
        ax4.set_xlabel("Total MTD Actions")
        ax4.set_title("Service MTD Activity",
                       fontweight="bold")
        ax4.grid(axis="x", alpha=0.3)

        plt.suptitle("RAMS Framework — Objective 5: Moving Target Defense\n"
                     "Autonomous Response Evaluation",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(self.output_dir, "mtd_evaluation.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[MTD] Evaluation plot saved: {path}")
        return path
