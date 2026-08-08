# RAMS Framework — Full Project
## Resilient Autonomous Multi-Tier Security for Smart Mobility Networks

---

## Objectives Implemented

| Obj | Description | Key Files |
|-----|-------------|-----------|
| **2** | Edge IDS — Lightweight DT + RF ensemble (BoT-EnsIDS) | `edge/edge_ids.py` |
| **3** | Hybrid ML Engine — Bi-LSTM + XGBoost | `models/bilstm_model.py`, `models/xgboost_model.py`, `models/hybrid_engine.py` |
| **4** | Explainable AI — SHAP (TreeExplainer + DeepExplainer) | `explainability/shap_explainer.py` |
| **6** | SUMO Simulation — Smart mobility traffic + attack injection | `simulation/sumo_simulator.py` |

---

## Project Structure

```
rams_project/
├── main.py                          ← Full pipeline entry point
├── config.py                        ← All hyperparameters + paths
├── requirements.txt
│
├── simulation/                      ← Objective 6
│   └── sumo_simulator.py            ← SUMO smart mobility + attack injection
│
├── edge/                            ← Objective 2
│   └── edge_ids.py                  ← Lightweight DT + RF Edge IDS
│
├── models/                          ← Objective 3
│   ├── bilstm_model.py              ← Bidirectional LSTM
│   ├── xgboost_model.py             ← XGBoost classifier
│   └── hybrid_engine.py             ← Weighted ensemble fusion
│
├── explainability/                  ← Objective 4
│   └── shap_explainer.py            ← SHAP XAI (global + local)
│
├── streaming/                       ← Edge → Cloud bridge
│   └── pipeline.py                  ← Kafka-style streaming pipeline
│
├── data/
│   ├── data_loader.py               ← CIC-IDS2017 / TON_IoT / SUMO / synthetic
│   └── cicids2017/                  ← ← Place your CSV files HERE
│
├── utils/
│   └── metrics.py                   ← Plots + evaluation tables
│
└── results/                         ← All outputs (auto-created)
    ├── saved_models/                ← Trained model files
    ├── simulation/                  ← SUMO plots + logs
    ├── *.png                        ← All evaluation plots
    ├── *.json                       ← Alert logs, pipeline reports
    └── xai_report.txt               ← Human-readable SHAP report
```

---

## Dataset Setup

Place CIC-IDS2017 CSVs in `data/cicids2017/`:
```
data/cicids2017/
├── Monday-WorkingHours.pcap_ISCX.csv
├── Tuesday-WorkingHours.pcap_ISCX.csv
├── Wednesday-workingHours.pcap_ISCX.csv
├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
├── Friday-WorkingHours-Morning.pcap_ISCX.csv
├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
└── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
```

---

## Quick Start (VS Code)

```bash
# 1. Activate virtual environment (Windows)
rams_env\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Quick demo — synthetic data, all objectives, ~5 min
python main.py --dataset synthetic --n_samples 8000 --epochs 5

# 4. Full run with CIC-IDS2017
python main.py --dataset cicids2017 --data_path ./data/cicids2017/

# 5. Skip slow steps for faster testing
python main.py --dataset cicids2017 --data_path ./data/cicids2017/ --skip_bilstm --skip_xai
```

---

## Pipeline Flow

```
[SUMO Simulation] (Obj 6)
       ↓ labelled network flows (DDoS, Bot, DoS, PortScan, BENIGN)
[Edge IDS] (Obj 2)
       ↓ suspicious flows only (90% BENIGN filtered locally)
[Kafka Queue] (streaming/pipeline.py)
       ↓ batched suspicious flows
[Hybrid ML Engine] (Obj 3)
  ├── XGBoost (tabular features)
  └── Bi-LSTM (sequential patterns)
       ↓ weighted probability fusion
[Threat Classification + Alert]
       ↓
[SHAP Explanation] (Obj 4)        [MTD Trigger] (Obj 5 — next)
  global + local explanations       ip_shuffling / container_mutation
```

---

## Expected Results (CIC-IDS2017)

| Model | Accuracy | F1-Weighted |
|-------|----------|-------------|
| Edge DT | ~98% | ~0.98 |
| Edge RF | ~99% | ~0.99 |
| XGBoost | ~99.5% | ~0.995 |
| Bi-LSTM | ~98% | ~0.982 |
| **Hybrid Ensemble** | **~99.5%** | **~0.995** |

Edge bandwidth saved: ~85–92% (only suspicious flows forwarded to cloud)
