# Real-Time Time-Series Anomaly Detection System

An end-to-end unsupervised/semi-supervised machine learning pipeline and real-time REST inference server designed to detect anomalies in multivariate time-series data (such as **Credit Card Fraud** and **Network Intrusion Detection**).

---

## 1. Problem Statement & Approach

In real-world financial transaction streams and network traffic logs:
* **Extreme Class Imbalance:** Anomalies represent less than **1%–5%** of total activity, rendering standard accuracy metrics misleading.
* **Zero-Day / Unseen Threats:** Attack vectors evolve continuously, making supervised classifiers trained only on historical attack labels ineffective.
* **Temporal & Cross-Channel Dependencies:** Many intrusions (e.g., stealth data exfiltration or flatlined sensors) only manifest across sliding time windows or through broken correlations between metrics.

To address this, the system trains **exclusively on normal operational behavior** using a sliding-window feature extractor and scores incoming streams via an **ECDF Percentile-Calibrated Ensemble** without any test-label leakage.

---

## 2. System Architecture

```text
[Raw 3-Channel Stream] ──► (Transaction Amount / Rate / Latency)
         │
         ▼
[Sliding Windowing (W=24)] ──► Detrended Relative Windows + Volatility + Cyclic Time + Cross-Correlation (85 Features)
         │
         ▼
[RobustScaler Normalization] ──► Fit strictly on training data
         │
         ├──► [One-Class SVM (RBF Kernel)] ──► Boundary Distance Score
         │
         └──► [Deep Autoencoder (85→64→16→64→85)] ──► Reconstruction Error (MSE)
         │
         ▼
[Train-ECDF Percentile Calibration] ──► Weighted Fusion (0.55 OCSVM + 0.45 Autoencoder)
         │
         ▼
[Zero-Leakage Threshold (99th Pct)] ──► Real-Time REST API Alert (NORMAL / WARNING / CRITICAL)
```

---

## 3. Dataset & Anomaly Taxonomy

The pipeline generates a **60,000-step multivariate time-series dataset** (5-minute intervals, 288 steps/day) simulating daily seasonality, trend drift, and cross-channel correlation, split chronologically (50% train, 50% test).

Four distinct anomaly archetypes are injected into the test stream:
1. **Point Spikes:** Sudden high-magnitude bursts (simulating DDoS spikes or fraudulent transaction bursts).
2. **Contextual Anomalies:** Values within global bounds that violate time-of-day expectations.
3. **Collective Anomalies:** Sustained sensor flatlines or variance bursts across contiguous steps.
4. **Correlation Breaks:** Decoupling of normally correlated channels (simulating stealth exfiltration).

---

## 4. Experimental Results

All thresholds are fixed at the **99th percentile of training anomaly scores** (zero test-label leakage).

### V1 Baseline vs. V2 Feature-Engineered Pipeline

| Pipeline | Model | Precision | Recall | F1-Score | PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **V1 Baseline** | Isolation Forest | 0.869 | 0.565 | 0.684 | 0.758 |
| **V1 Baseline** | One-Class SVM | 0.877 | 0.777 | 0.824 | 0.851 |
| **V1 Baseline** | Autoencoder | 0.904 | 0.767 | 0.830 | 0.830 |
| **V2 Upgraded** | **One-Class SVM** | **0.910** | **0.786** | **0.843** | **0.866** |
| **V2 Upgraded** | **Autoencoder** | **0.919** | **0.775** | **0.841** | **0.839** |
| **V2 Upgraded** | **Weighted Ensemble** | **0.907** | **0.740** | **0.815** | **0.843** |

### Key Takeaways
* **Detrended Window Normalization** eliminated false positives caused by slow upward baseline drift.
* **Peak-to-Peak (`ptp`) & Flatline Indicators** enabled real-time detection of collective sensor freeze anomalies.
* **Autoencoder + One-Class SVM** achieved **99.6%–100% recall** on sudden point spikes while maintaining **>91% precision**.

---

## 5. Project Structure

```text
anomaly-detection-project/
├── anomaly_detection.py     # V1 baseline experiment & visualization generator
├── upgrade_pipeline.py      # V2 detrended feature pipeline & model bundle exporter
├── app.py                   # Offline-compatible real-time HTTP inference server
├── test_stream.py           # Live streaming client simulator (Normal & Attack cases)
├── requirements.txt         # Python dependencies
├── models/
│   └── anomaly_bundle.joblib # Serialized scaler, OCSVM, Autoencoder & ECDF distributions
├── outputs/                 # V1 PR curves & timeline plots
└── outputs_v2/              # V2 evaluation metric tables (CSV)
```

---

## 6. Quickstart & Usage

### 1. Environment Setup
```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. Train & Export Model Bundle
```bash
python upgrade_pipeline.py
```

### 3. Start Real-Time Inference Server
```bash
python app.py
# Server starts at http://127.0.0.1:8000
```

### 4. Run Live Stream Simulation (Second Terminal)
```bash
venv\Scripts\activate
python test_stream.py
```

#### Sample API Response (`POST /predict`)
```json
{
  "is_anomaly": true,
  "severity": "CRITICAL",
  "ensemble_score": 0.9982,
  "threshold": 0.985,
  "model_breakdown": {
    "OneClassSVM_percentile": 0.9973,
    "Autoencoder_percentile": 0.9993
  }
}
```