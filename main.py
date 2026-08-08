"""
main.py — RAMS Framework Full Pipeline
=======================================
Objectives: 2 (Edge IDS) | 3 (Hybrid ML) | 4 (XAI) | 5 (MTD) | 6 (SUMO)

Usage:
  python main.py --dataset synthetic --n_samples 8000 --epochs 5
  python main.py --dataset cicids2017 --data_path ./data/cicids2017/
  python main.py --dataset cicids2017 --data_path ./data/cicids2017/ --skip_bilstm --skip_xai
"""

import os, sys, argparse, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (BILSTM_CONFIG, XGBOOST_CONFIG, HYBRID_CONFIG,
                    SHAP_CONFIG, EDGE_CONFIG, SUMO_CONFIG, RESULTS_DIR, MODELS_DIR)
from data.data_loader import load_dataset, RAMSDataPreprocessor
from models.bilstm_model import BiLSTMTrainer
from models.xgboost_model import XGBoostDetector
from models.hybrid_engine import HybridDetectionEngine
from explainability.shap_explainer import RAMSExplainer
from edge.edge_ids import EdgeIDS
from simulation.sumo_simulator import SUMOSimulator
from streaming.pipeline import RAMSPipeline
from mtd.mtd_engine import MTDEngine, ThreatAlert, MTDAction
from utils.metrics import (plot_confusion_matrix, plot_training_history,
                           plot_model_comparison, plot_class_distribution,
                           print_metrics_table)
from datetime import datetime


def parse_args():
    p = argparse.ArgumentParser(description="RAMS Framework")
    p.add_argument("--dataset", default="synthetic",
                   choices=["cicids2017","toniot","synthetic"])
    p.add_argument("--data_path", default=None)
    p.add_argument("--n_samples", type=int, default=8000)
    p.add_argument("--n_features", type=int, default=60)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--skip_simulation", action="store_true")
    p.add_argument("--skip_bilstm", action="store_true")
    p.add_argument("--skip_xai", action="store_true")
    p.add_argument("--skip_pipeline", action="store_true")
    p.add_argument("--skip_mtd", action="store_true")
    return p.parse_args()


def banner(text):
    print(f"\n{'▓'*65}\n  {text}\n{'▓'*65}")


def main():
    args = parse_args()
    t_total = time.time()

    print("\n" + "█"*65)
    print("  RAMS — RESILIENT AUTONOMOUS MULTI-TIER SECURITY")
    print("  Obj 2: Edge IDS | Obj 3: Hybrid ML | Obj 4: XAI")
    print("  Obj 5: MTD      | Obj 6: SUMO Simulation")
    print("█"*65)

    os.makedirs(os.path.join(RESULTS_DIR, "simulation"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "mtd"), exist_ok=True)

    # ── OBJ 6: SUMO Simulation ────────────────────────────────────
    banner("OBJECTIVE 6: SUMO Smart Mobility Simulation")
    sumo_df = None
    if not args.skip_simulation:
        sim = SUMOSimulator(SUMO_CONFIG)
        sumo_df = sim.run()
    else:
        print("▶ Skipping simulation (--skip_simulation)")
        if os.path.exists(SUMO_CONFIG["output_pcap"]):
            import pandas as pd
            sumo_df = pd.read_csv(SUMO_CONFIG["output_pcap"])
            print(f"  Loaded existing: {len(sumo_df):,} flows")

    # ── Load Dataset ──────────────────────────────────────────────
    banner("Loading Main Dataset")
    df = load_dataset(args.dataset, args.data_path, args.n_samples)
    print(f"  Loaded: {len(df):,} rows | Classes: {df['Label'].nunique()}")

    # ── OBJ 2: Edge IDS ──────────────────────────────────────────
    banner("OBJECTIVE 2: Edge IDS (BoT-EnsIDS)")
    edge_ids = EdgeIDS(config=EDGE_CONFIG, save_dir=MODELS_DIR)
    edge_results = edge_ids.train(df.copy())
    edge_ids.plot_results(edge_results, output_dir=RESULTS_DIR)
    print(f"\n  Ensemble F1: {edge_results.get('ensemble',{}).get('f1_weighted',0):.4f}")

    # ── Preprocess for Cloud ──────────────────────────────────────
    banner("Preprocessing for Cloud Engine")
    preprocessor = RAMSDataPreprocessor(
        n_features=args.n_features,
        sequence_len=BILSTM_CONFIG["sequence_len"],
        use_smote=True,
    )
    data = preprocessor.fit_transform(df)
    n_classes = data["n_classes"]
    class_names = list(data["classes"])
    n_features = data["n_features"]
    selected_features = preprocessor.selected_features or \
        [f"f{i}" for i in range(n_features)]
    print(f"  Classes ({n_classes}): {class_names}")
    plot_class_distribution(data["y_train"], class_names,
        "RAMS — Class Distribution",
        os.path.join(RESULTS_DIR, "class_distribution.png"))

    # ── OBJ 3: XGBoost ───────────────────────────────────────────
    banner("OBJECTIVE 3: XGBoost Detector")
    xgb = XGBoostDetector(config=XGBOOST_CONFIG, n_classes=n_classes,
                          save_path=os.path.join(MODELS_DIR, "xgboost_model.joblib"))
    xgb.train(data["X_train"], data["y_train"], data["X_val"], data["y_val"])

    # ── OBJ 3: Bi-LSTM ───────────────────────────────────────────
    bilstm = None
    if not args.skip_bilstm:
        banner("OBJECTIVE 3: Bi-LSTM Detector")
        bcfg = {**BILSTM_CONFIG, "input_size": n_features,
                "num_classes": n_classes}
        if args.epochs: bcfg["epochs"] = args.epochs
        bilstm = BiLSTMTrainer(config=bcfg, n_classes=n_classes,
                               save_path=os.path.join(MODELS_DIR, "bilstm_best.pt"))
        history = bilstm.train(data["X_train_seq"], data["y_train_seq"],
                               data["X_val_seq"], data["y_val_seq"])
        plot_training_history(history,
            os.path.join(RESULTS_DIR, "bilstm_training_history.png"))
    else:
        print("▶ Bi-LSTM skipped (--skip_bilstm)")

    # ── OBJ 3: Hybrid Evaluation ──────────────────────────────────
    banner("OBJECTIVE 3: Hybrid Ensemble Evaluation")
    engine = HybridDetectionEngine(bilstm_trainer=bilstm,
                                   xgboost_detector=xgb,
                                   config=HYBRID_CONFIG,
                                   class_names=class_names)
    if bilstm:
        results = engine.evaluate(data)
        ens_preds = results["ensemble"]["predictions"]
        ens_conf  = results["ensemble"]["confidence"]
        y_aligned = results["ensemble"]["labels"]
    else:
        xr = xgb.evaluate(data["X_test"], data["y_test"], class_names)
        results = {"xgboost": xr,
                   "bilstm": {"f1_weighted":0,"f1_macro":0,"accuracy":0},
                   "ensemble": {**xr,
                       "confidence": xr["probabilities"].max(axis=1)}}
        ens_preds = xr["predictions"]
        ens_conf  = xr["probabilities"].max(axis=1)
        y_aligned = data["y_test"]

    print_metrics_table(results)
    plot_confusion_matrix(y_aligned, ens_preds, class_names,
        os.path.join(RESULTS_DIR, "confusion_matrix.png"), "RAMS Hybrid")
    plot_model_comparison(results,
        os.path.join(RESULTS_DIR, "model_comparison.png"))

    # ── OBJ 4: SHAP XAI ──────────────────────────────────────────
    if not args.skip_xai:
        banner("OBJECTIVE 4: Explainable AI — SHAP")
        exp = RAMSExplainer(xgboost_model=xgb, bilstm_trainer=bilstm,
                            feature_names=selected_features,
                            class_names=class_names, config=SHAP_CONFIG,
                            output_dir=RESULTS_DIR)
        bg = (data["X_test_seq"][:50]
              if bilstm and len(data["X_test_seq"]) > 0 else None)
        exp.run_full_xai_pipeline(data["X_test"], y_aligned,
                                   ens_preds, ens_conf, bg)
    else:
        print("▶ XAI skipped (--skip_xai)")

    # ── Full Pipeline ─────────────────────────────────────────────
    pipeline_stats = {"alerts": [], "mtd_actions": []}
    if not args.skip_pipeline:
        banner("Full Pipeline: Edge → Kafka → Cloud → Alerts")
        if sumo_df is not None and len(sumo_df) > 0:
            pipe_df = sumo_df.copy()
        else:
            import pandas as pd
            X_demo = data["X_test"][:2000]
            y_demo = data["y_test"][:2000]
            pipe_df = pd.DataFrame(X_demo,
                columns=[f"feat_{i}" for i in range(X_demo.shape[1])])
            pipe_df["Label"] = [class_names[i] for i in y_demo]

        rams_pipeline = RAMSPipeline(
            edge_ids=edge_ids, hybrid_engine=engine,
            preprocessor=preprocessor, bilstm_config=BILSTM_CONFIG)
        pipeline_stats = rams_pipeline.run(pipe_df)

    # ── OBJ 5: Moving Target Defense ─────────────────────────────
    if not args.skip_mtd:
        banner("OBJECTIVE 5: Moving Target Defense (MTD)")
        mtd_engine = MTDEngine(
            output_dir=os.path.join(RESULTS_DIR, "mtd"))

        alerts = pipeline_stats.get("alerts", [])
        if alerts:
            # Process real alerts from pipeline
            mtd_engine.process_alert_batch(alerts)
        else:
            # Demo mode: synthesise alerts from detection results
            print("[MTD] Demo mode — synthesising alerts from detections...")
            detected_threats = []
            if len(ens_preds) > 0:
                from collections import Counter
                threat_counts = Counter(
                    class_names[p] for p in ens_preds if p != 0
                )
                for threat, count in threat_counts.most_common(6):
                    from config import THREAT_SEVERITY
                    # Find class index
                    cls_idx = class_names.index(threat) \
                        if threat in class_names else 1
                    # Get mean confidence for this class
                    mask = ens_preds == cls_idx
                    conf = float(ens_conf[mask].mean()) if mask.sum() > 0 else 0.85
                    detected_threats.append((threat, conf))

            if not detected_threats:
                detected_threats = [
                    ("DDoS", 0.94), ("Bot", 0.89),
                    ("PortScan", 0.92), ("Infiltration", 0.88),
                    ("DoS Hulk", 0.91), ("SSH-Patator", 0.87),
                ]

            from config import THREAT_SEVERITY
            for threat, conf in detected_threats:
                # Get severity
                cls_idx = (class_names.index(threat)
                           if threat in class_names else 1)
                sev = THREAT_SEVERITY.get(cls_idx, "medium")
                target = mtd_engine._assign_service(threat)

                alert = ThreatAlert(
                    alert_id=f"DET-{threat[:4].upper()}-001",
                    timestamp=datetime.now(),
                    threat_type=threat,
                    severity=sev,
                    confidence=conf,
                    source_ip="192.168.x.x",
                    target_service_id=target,
                    mtd_action=MTDAction.NONE,
                )
                event = mtd_engine.process_alert(alert)

        mtd_engine.save_report()
        mtd_engine.plot_results()
        mtd_engine.evaluate_effectiveness()
    else:
        print("▶ MTD skipped (--skip_mtd)")

    # ── Final Summary ─────────────────────────────────────────────
    elapsed = time.time() - t_total
    banner(f"RAMS COMPLETE — Total time: {elapsed:.1f}s")

    print("\n  All output files:")
    for root, dirs, files in os.walk(RESULTS_DIR):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, RESULTS_DIR)
            size = os.path.getsize(fpath) / 1024
            print(f"    {rel:<55} {size:>7.1f} KB")
    print(f"\n{'█'*65}\n")


if __name__ == "__main__":
    main()
