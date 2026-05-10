# RAMS Framework — Objectives 3 & 4
## Hybrid ML Detection Engine + Explainable AI (XAI)

### Project: Resilient Autonomous Multi-Tier Security (RAMS) Framework for Smart Mobility Networks

---

## Objectives Covered

| Objective | Description |
|-----------|-------------|
| **Obj 3** | Hybrid ML Detection Engine: Bi-LSTM (sequence) + XGBoost (tabular) for advanced threat detection |
| **Obj 4** | Explainable AI (XAI): SHAP-based feature importance for transparent decision-making |

---

## Recommended Datasets

| Dataset | Why Use It | Download |
|---------|-----------|----------|
| **CIC-IDS2017** ✅ PRIMARY | 78 features, covers DDoS, Botnet, DoS, Brute Force, Web Attacks | https://www.unb.ca/cic/datasets/ids-2017.html |
| **TON_IoT** | IoT-specific, 43 attack categories, smart mobility relevant | https://research.unsw.edu.au/projects/toniot-datasets |
| **Bot-IoT** | Botnet + DDoS in IoT env | https://research.unsw.edu.au/projects/bot-iot-dataset |
| **UNSW-NB15** | 49 features, modern attacks | https://research.unsw.edu.au/projects/unsw-nb15-dataset |
| **Edge-IIoTset** | Industrial IoT, edge-cloud relevant | https://ieee-dataport.org/documents/edge-iiotset |

**Recommended for this project**: CIC-IDS2017 (primary) + TON_IoT (supplementary for IoT context)

---

## File Structure

```
rams_obj3_obj4/
├── README.md
├── requirements.txt
├── config.py                        # Global configuration
├── data/
│   └── data_loader.py               # Dataset loading + preprocessing
├── models/
│   ├── bilstm_model.py              # Bi-LSTM deep learning model
│   ├── xgboost_model.py             # XGBoost classifier
│   └── hybrid_engine.py             # Hybrid ensemble (Bi-LSTM + XGBoost)
├── explainability/
│   ├── shap_explainer.py            # SHAP-based XAI module
│   └── visualizer.py                # SHAP plots + threat reports
├── utils/
│   ├── feature_engineering.py       # Flow-based feature extraction
│   ├── metrics.py                   # Evaluation metrics
│   └── threat_classifier.py         # Multi-class threat categorization
├── results/                         # Output plots, reports, models
├── main.py                          # Entry point — full pipeline
└── demo_synthetic.py                # Quick demo with synthetic data (no download needed)
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run demo with synthetic data (no dataset download needed)
python demo_synthetic.py

# 3. Full pipeline with real dataset (place CIC-IDS2017 CSVs in data/ folder)
python main.py --dataset cicids2017 --data_path ./data/cicids2017/

# 4. With TON_IoT dataset
python main.py --dataset toniot --data_path ./data/toniot/
```

---

## Reference Papers Implemented

1. **META** (encrypted traffic anomaly) → Bi-LSTM sequential analysis module
2. **BoT-EnsIDS** → Ensemble feature selection + hybrid deep learning
3. **FUSE-Net** → Hybrid ensemble for cloud-based ITS
4. **XAI Healthcare paper** → SHAP-based explainability module (adapted for cybersecurity)
5. **MTD Ransomware paper** → Threat classification that feeds into MTD trigger (Obj 5)
