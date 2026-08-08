"""
streaming/pipeline.py — Edge-to-Cloud Streaming Pipeline
RAMS Framework — Tier 2 → Tier 3 Data Flow

Simulates Kafka-style streaming of suspicious flows from Edge IDS
to the Cloud detection engine (Obj 3). No real Kafka needed —
uses an in-memory queue that mirrors the real Kafka API.

In production: replace InMemoryQueue with kafka-python producer/consumer.
"""

import os
import sys
import time
import json
import queue
import threading
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Callable, Optional
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR


# ══════════════════════════════════════════════════════════════════
# IN-MEMORY QUEUE (Kafka substitute)
# ══════════════════════════════════════════════════════════════════

class InMemoryQueue:
    """
    Simulates a Kafka topic with producer/consumer semantics.
    Drop-in replacement — swap with kafka-python for production.

    Topics:
      - 'suspicious_flows': Edge → Cloud (forwarded flows for deep analysis)
      - 'alerts':           Cloud → Dashboard (threat alerts)
      - 'mtd_triggers':     Cloud → MTD engine (response actions)
    """

    def __init__(self):
        self.topics = {
            "suspicious_flows": queue.Queue(),
            "alerts": queue.Queue(),
            "mtd_triggers": queue.Queue(),
        }
        self.stats = {t: {"produced": 0, "consumed": 0}
                      for t in self.topics}

    def produce(self, topic: str, message: dict):
        if topic not in self.topics:
            self.topics[topic] = queue.Queue()
            self.stats[topic] = {"produced": 0, "consumed": 0}
        self.topics[topic].put(message)
        self.stats[topic]["produced"] += 1

    def consume(self, topic: str, timeout: float = 0.1) -> Optional[dict]:
        try:
            msg = self.topics[topic].get(timeout=timeout)
            self.stats[topic]["consumed"] += 1
            return msg
        except queue.Empty:
            return None

    def consume_batch(self, topic: str, max_batch: int = 100) -> list:
        batch = []
        for _ in range(max_batch):
            msg = self.consume(topic, timeout=0.01)
            if msg is None:
                break
            batch.append(msg)
        return batch

    def size(self, topic: str) -> int:
        return self.topics.get(topic, queue.Queue()).qsize()

    def print_stats(self):
        print("\n[Pipeline] Queue Statistics:")
        for topic, stat in self.stats.items():
            print(f"  {topic:<25} produced={stat['produced']:>6,} "
                  f"consumed={stat['consumed']:>6,} "
                  f"pending={self.size(topic):>6,}")


# ══════════════════════════════════════════════════════════════════
# EDGE PRODUCER
# ══════════════════════════════════════════════════════════════════

class EdgeProducer:
    """
    Runs on the Edge node.
    Filters traffic via EdgeIDS, then publishes suspicious flows
    to the 'suspicious_flows' Kafka topic for Cloud processing.
    """

    def __init__(self, edge_ids, message_queue: InMemoryQueue,
                 batch_size: int = 64):
        self.edge_ids = edge_ids
        self.queue = message_queue
        self.batch_size = batch_size
        self.total_produced = 0
        self.total_filtered = 0

    def process_batch(self, df_batch: pd.DataFrame) -> dict:
        """
        Filter a batch of raw flows at the edge.
        Forward only suspicious ones to cloud topic.
        """
        filter_result = self.edge_ids.filter_traffic(df_batch)

        suspicious_df = filter_result["suspicious_flows"]
        self.total_filtered += filter_result["benign_blocked"]
        self.total_produced += filter_result["suspicious_count"]

        # Publish suspicious flows to queue
        for _, row in suspicious_df.iterrows():
            msg = {
                "timestamp": datetime.now().isoformat(),
                "source": "edge_ids",
                "flow": row.to_dict(),
                "suspicion_score": float(
                    filter_result["suspicion_scores"][
                        suspicious_df.index.get_loc(row.name)
                    ] if hasattr(suspicious_df.index, 'get_loc') else 0.9
                ),
            }
            self.queue.produce("suspicious_flows", msg)

        return {
            "batch_size": len(df_batch),
            "forwarded": filter_result["suspicious_count"],
            "blocked": filter_result["benign_blocked"],
            "bandwidth_saved_pct": filter_result["bandwidth_saved_pct"],
            "per_flow_ms": filter_result["per_flow_inference_ms"],
        }

    def stream_dataset(self, df: pd.DataFrame) -> dict:
        """
        Stream an entire dataset through the edge pipeline in batches.
        Simulates real-time ingestion from network interface.
        """
        print(f"\n[EdgeProducer] Streaming {len(df):,} flows in "
              f"batches of {self.batch_size}...")

        total_forwarded = 0
        total_blocked = 0
        batch_results = []

        for start in range(0, len(df), self.batch_size):
            batch = df.iloc[start: start + self.batch_size].copy()
            result = self.process_batch(batch)
            total_forwarded += result["forwarded"]
            total_blocked += result["blocked"]
            batch_results.append(result)

            if (start // self.batch_size) % 50 == 0:
                pct = start / len(df) * 100
                print(f"  [EdgeProducer] {pct:.0f}% | "
                      f"Forwarded: {total_forwarded:,} | "
                      f"Blocked: {total_blocked:,}")

        total = total_forwarded + total_blocked
        summary = {
            "total_flows": total,
            "total_forwarded": total_forwarded,
            "total_blocked": total_blocked,
            "overall_filter_rate": total_blocked / total if total > 0 else 0,
            "bandwidth_saved_pct": total_blocked / total * 100 if total > 0 else 0,
            "queue_depth": self.queue.size("suspicious_flows"),
        }

        print(f"\n[EdgeProducer] Streaming complete:")
        print(f"  Forwarded to Cloud: {total_forwarded:,} "
              f"({total_forwarded/total*100:.1f}%)")
        print(f"  Blocked at Edge:    {total_blocked:,} "
              f"({total_blocked/total*100:.1f}%)")
        print(f"  Queue depth:        {summary['queue_depth']:,}")

        return summary


# ══════════════════════════════════════════════════════════════════
# CLOUD CONSUMER
# ══════════════════════════════════════════════════════════════════

class CloudConsumer:
    """
    Runs on Cloud (Tier 3).
    Consumes suspicious flows from Kafka, runs Hybrid ML engine,
    publishes alerts and MTD triggers back to queue.
    """

    def __init__(self, hybrid_engine, message_queue: InMemoryQueue,
                 preprocessor, bilstm_config: dict):
        self.engine = hybrid_engine
        self.queue = message_queue
        self.preprocessor = preprocessor
        self.bilstm_config = bilstm_config
        self.alerts_generated = 0
        self.mtd_triggers = 0

    def _flows_to_array(self, flows: list) -> Optional[np.ndarray]:
        """Convert consumed flow messages to numpy array for ML inference."""
        if not flows:
            return None, None
        try:
            records = [msg["flow"] for msg in flows]
            df = pd.DataFrame(records)

            # Drop non-numeric cols
            X = df.select_dtypes(include=[np.number])
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

            # Align feature count to what the model expects
            expected = self.preprocessor.selector.get_support().sum() \
                if self.preprocessor.selector else 60
            if X.shape[1] > expected:
                X = X.iloc[:, :expected]
            elif X.shape[1] < expected:
                pad = pd.DataFrame(
                    np.zeros((len(X), expected - X.shape[1])),
                    columns=[f"pad_{i}" for i in range(expected - X.shape[1])]
                )
                X = pd.concat([X.reset_index(drop=True), pad], axis=1)

            return self.preprocessor.scaler.transform(X.values), df
        except Exception as e:
            print(f"[CloudConsumer] Flow conversion error: {e}")
            return None, None

    def process_queue(self, max_batches: int = 100) -> dict:
        """
        Consume all pending suspicious flows, run hybrid inference,
        generate alerts and MTD triggers.
        """
        print(f"\n[CloudConsumer] Processing suspicious flows queue...")
        all_alerts = []
        batches_processed = 0

        while batches_processed < max_batches:
            batch = self.queue.consume_batch("suspicious_flows", max_batch=128)
            if not batch:
                break

            X_flat, df_batch = self._flows_to_array(batch)
            if X_flat is None or len(X_flat) == 0:
                batches_processed += 1
                continue

            # Run XGBoost inference (fast path for streaming)
            try:
                probs = self.engine.xgboost.predict_proba(X_flat)
                preds = probs.argmax(axis=1)
                confs = probs.max(axis=1)
            except Exception as e:
                print(f"[CloudConsumer] Inference error: {e}")
                batches_processed += 1
                continue

            class_names = self.engine.class_names or []

            # Generate alert for each detected threat
            for i, (pred, conf) in enumerate(zip(preds, confs)):
                if pred == 0:
                    continue  # BENIGN — no alert

                label = class_names[pred] if pred < len(class_names) else str(pred)
                from config import THREAT_SEVERITY
                severity = THREAT_SEVERITY.get(int(pred), "unknown")

                alert = {
                    "alert_id": f"RAMS-{datetime.now().strftime('%f')}",
                    "timestamp": datetime.now().isoformat(),
                    "threat_type": label,
                    "severity": severity,
                    "confidence": float(conf),
                    "source_flow": batch[i].get("flow", {}),
                    "mtd_required": severity in ("critical", "high"),
                }
                self.queue.produce("alerts", alert)
                all_alerts.append(alert)
                self.alerts_generated += 1

                # MTD trigger for critical/high severity
                if alert["mtd_required"]:
                    mtd_action = self.engine._get_mtd_action(label, severity)
                    mtd_trigger = {
                        "trigger_id": alert["alert_id"],
                        "timestamp": alert["timestamp"],
                        "threat_type": label,
                        "severity": severity,
                        "action": mtd_action,
                        "confidence": float(conf),
                    }
                    self.queue.produce("mtd_triggers", mtd_trigger)
                    self.mtd_triggers += 1

            batches_processed += 1

        print(f"[CloudConsumer] Processed {batches_processed} batches")
        print(f"  Alerts generated:  {self.alerts_generated:,}")
        print(f"  MTD triggers:      {self.mtd_triggers:,}")

        return {
            "batches_processed": batches_processed,
            "alerts_generated": self.alerts_generated,
            "mtd_triggers": self.mtd_triggers,
            "alerts": all_alerts,
        }


# ══════════════════════════════════════════════════════════════════
# FULL PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════

class RAMSPipeline:
    """
    End-to-end RAMS pipeline orchestrator.
    Connects: SUMO → Edge IDS → Kafka Queue → Cloud ML → Alerts → MTD
    """

    def __init__(self, edge_ids, hybrid_engine, preprocessor,
                 bilstm_config: dict):
        self.edge_ids = edge_ids
        self.hybrid_engine = hybrid_engine
        self.preprocessor = preprocessor
        self.bilstm_config = bilstm_config

        self.queue = InMemoryQueue()
        self.producer = EdgeProducer(edge_ids, self.queue, batch_size=128)
        self.consumer = CloudConsumer(hybrid_engine, self.queue,
                                      preprocessor, bilstm_config)
        self.pipeline_stats = {}

    def run(self, df: pd.DataFrame) -> dict:
        """
        Run the full Edge→Cloud pipeline on a dataset.

        Args:
            df: Raw network flows (from SUMO simulation or real capture)

        Returns:
            Complete pipeline statistics
        """
        print("\n" + "="*60)
        print(" RAMS — Full Pipeline: Edge → Cloud")
        print(" Obj 2 (Edge IDS) → Kafka → Obj 3 (Hybrid ML)")
        print("="*60)

        t_start = time.time()

        # Phase 1: Edge filtering + streaming
        print("\n[Pipeline] Phase 1: Edge IDS Filtering...")
        edge_summary = self.producer.stream_dataset(df)

        # Phase 2: Cloud processing
        print("\n[Pipeline] Phase 2: Cloud ML Inference...")
        cloud_summary = self.consumer.process_queue()

        # Phase 3: Drain alert queue
        alerts = []
        while True:
            alert = self.queue.consume("alerts", timeout=0.01)
            if alert is None:
                break
            alerts.append(alert)

        mtd_actions = []
        while True:
            trigger = self.queue.consume("mtd_triggers", timeout=0.01)
            if trigger is None:
                break
            mtd_actions.append(trigger)

        elapsed = time.time() - t_start

        self.pipeline_stats = {
            "edge": edge_summary,
            "cloud": cloud_summary,
            "total_alerts": len(alerts),
            "total_mtd_triggers": len(mtd_actions),
            "pipeline_time_s": elapsed,
            "alerts": alerts[:20],       # First 20 for report
            "mtd_actions": mtd_actions[:10],
        }

        self._print_pipeline_summary()
        self._save_pipeline_report(alerts, mtd_actions)
        self.queue.print_stats()

        return self.pipeline_stats

    def _print_pipeline_summary(self):
        s = self.pipeline_stats
        e = s["edge"]
        print(f"\n{'═'*60}")
        print(f"  RAMS PIPELINE SUMMARY")
        print(f"{'═'*60}")
        print(f"  Total flows ingested:     {e['total_flows']:>10,}")
        print(f"  Blocked at Edge:          {e['total_blocked']:>10,} "
              f"({e['bandwidth_saved_pct']:.1f}% bandwidth saved)")
        print(f"  Forwarded to Cloud:       {e['total_forwarded']:>10,}")
        print(f"  Alerts generated:         {s['total_alerts']:>10,}")
        print(f"  MTD triggers:             {s['total_mtd_triggers']:>10,}")
        print(f"  Total pipeline time:      {s['pipeline_time_s']:>10.1f}s")
        print(f"{'═'*60}")

        if s["mtd_actions"]:
            print(f"\n  MTD Actions (sample):")
            seen = set()
            for action in s["mtd_actions"][:5]:
                key = (action["threat_type"], action["action"])
                if key not in seen:
                    print(f"    {action['threat_type']:<20} → "
                          f"{action['action']:<20} "
                          f"(conf={action['confidence']:.2f})")
                    seen.add(key)

    def _save_pipeline_report(self, alerts: list, mtd_actions: list):
        """Save pipeline execution report as JSON."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "pipeline_stats": {
                k: v for k, v in self.pipeline_stats.items()
                if k not in ("alerts", "mtd_actions")
            },
            "alert_summary": {
                "total": len(alerts),
                "by_severity": {},
                "by_type": {},
            },
            "mtd_summary": {
                "total_triggers": len(mtd_actions),
                "by_action": {},
            },
        }

        for alert in alerts:
            sev = alert.get("severity", "unknown")
            typ = alert.get("threat_type", "unknown")
            report["alert_summary"]["by_severity"][sev] = \
                report["alert_summary"]["by_severity"].get(sev, 0) + 1
            report["alert_summary"]["by_type"][typ] = \
                report["alert_summary"]["by_type"].get(typ, 0) + 1

        for trigger in mtd_actions:
            action = trigger.get("action", "unknown")
            report["mtd_summary"]["by_action"][action] = \
                report["mtd_summary"]["by_action"].get(action, 0) + 1

        path = os.path.join(RESULTS_DIR, "pipeline_report.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[Pipeline] Report saved: {path}")

        self._plot_pipeline_summary(report)

    def _plot_pipeline_summary(self, report: dict):
        """Visualise the full pipeline flow and alert distribution."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot 1: Pipeline flow (Sankey-style bar)
        ax = axes[0]
        e = self.pipeline_stats["edge"]
        stages = ["Raw Flows\n(SUMO)", "Edge Filtered\n(Blocked)",
                  "Cloud Processed\n(Forwarded)", "Alerts\nGenerated"]
        values = [
            e["total_flows"],
            e["total_blocked"],
            e["total_forwarded"],
            self.pipeline_stats["total_alerts"],
        ]
        colours = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
        bars = ax.bar(stages, values, color=colours, alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.01,
                    f"{val:,}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
        ax.set_title("Pipeline Flow\n(Flows at Each Stage)",
                     fontweight="bold")
        ax.set_ylabel("Flow Count")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)

        # Plot 2: Alert severity distribution
        ax2 = axes[1]
        sev_data = report["alert_summary"]["by_severity"]
        if sev_data:
            sev_colours = {"critical": "#d62728", "high": "#ff7f0e",
                           "medium": "#ffdd57", "low": "#2ca02c",
                           "none": "#aec7e8"}
            labels = list(sev_data.keys())
            vals = list(sev_data.values())
            clrs = [sev_colours.get(l, "#aec7e8") for l in labels]
            ax2.pie(vals, labels=labels, colors=clrs,
                    autopct="%1.1f%%", startangle=90,
                    textprops={"fontsize": 9})
            ax2.set_title("Alert Severity Distribution",
                          fontweight="bold")
        else:
            ax2.text(0.5, 0.5, "No alerts generated",
                     ha="center", va="center", transform=ax2.transAxes)
            ax2.set_title("Alert Severity Distribution", fontweight="bold")

        # Plot 3: MTD actions
        ax3 = axes[2]
        mtd_data = report["mtd_summary"]["by_action"]
        if mtd_data:
            mtd_colours = {
                "ip_shuffling": "#1f77b4",
                "container_mutation": "#ff7f0e",
                "pod_recreation": "#d62728",
                "port_hopping": "#9467bd",
                "none": "#aec7e8",
            }
            labels = list(mtd_data.keys())
            vals = list(mtd_data.values())
            clrs = [mtd_colours.get(l, "#aec7e8") for l in labels]
            ax3.barh(labels, vals, color=clrs, alpha=0.85)
            for i, val in enumerate(vals):
                ax3.text(val + 0.1, i, str(val),
                         va="center", fontsize=9, fontweight="bold")
            ax3.set_xlabel("Trigger Count")
            ax3.set_title("MTD Actions Triggered\n(Obj 5 Preview)",
                          fontweight="bold")
            ax3.grid(axis="x", alpha=0.3)
        else:
            ax3.text(0.5, 0.5, "No MTD triggers",
                     ha="center", va="center", transform=ax3.transAxes)
            ax3.set_title("MTD Actions Triggered", fontweight="bold")

        plt.suptitle("RAMS Framework — Full Pipeline Summary\n"
                     "Obj 2 (Edge) → Kafka → Obj 3 (Cloud) → Alerts → MTD",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, "pipeline_summary.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Pipeline] Summary plot saved: {path}")
