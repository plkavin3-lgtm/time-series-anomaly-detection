"""
Real-Time Time-Series Anomaly Detection API (FastAPI)
Loads trained OneClassSVM + Autoencoder bundle and scores incoming 24-step windows.
Run: uvicorn app:app --reload
"""
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

app = FastAPI(
    title="Time-Series Anomaly Detection Engine",
    description="Real-time intrusion & fraud detection using One-Class SVM + Autoencoder",
    version="2.0"
)

# Load saved model bundle
BUNDLE = joblib.load("models/anomaly_bundle.joblib")
SCALER = BUNDLE["scaler"]
OCS = BUNDLE["ocs"]
AE = BUNDLE["ae"]
SORTED_TR = BUNDLE["sorted_tr"]
W = BUNDLE["window_size"]

# Pure top-2 fusion (drops noisy Isolation Forest)
WEIGHTS = {"OneClassSVM": 0.55, "Autoencoder": 0.45}
# Calibrated 99th percentile threshold for the pure OCSVM+AE fusion
ENSEMBLE_THRESHOLD = 0.985


class WindowRequest(BaseModel):
    window: List[List[float]] = Field(
        ...,
        description=f"2D array of shape ({W}, 3) representing [channel_0, channel_1, channel_2] over {W} steps"
    )
    tod_phase: float = Field(
        ...,
        description="Time-of-day phase in radians [0, 2*pi]"
    )


def extract_features(raw_win: np.ndarray, tod_phase: float) -> np.ndarray:
    """Replicates exact V2 feature engineering for a single (24, 3) window."""
    rel_win = (raw_win - raw_win[0:1, :]).reshape(1, -1)
    diffs = np.abs(np.diff(raw_win, axis=0)).mean(axis=0, keepdims=True)
    stds = raw_win.std(axis=0, keepdims=True)
    ptp = np.ptp(raw_win, axis=0, keepdims=True)
    flatline_flag = (stds[:, 0:1] < 1e-4).astype(float)
    tod = np.array([[np.sin(tod_phase), np.cos(tod_phase)]])

    with np.errstate(invalid="ignore", divide="ignore"):
        corr_val = np.corrcoef(raw_win[:, 0], raw_win[:, 1])[0, 1]
    corr = np.array([[0.0 if np.isnan(corr_val) else corr_val]])
    if np.isnan(corr_val):
        corr = np.array([[-2.0]])

    return np.c_[rel_win, diffs, stds, ptp, flatline_flag, tod, corr]


def to_percentile(model_name: str, score: float) -> float:
    arr = SORTED_TR[model_name]
    return float(np.searchsorted(arr, score) / len(arr))


@app.get("/")
def health_check():
    return {"status": "online", "window_size": W, "active_models": list(WEIGHTS.keys())}


@app.post("/predict")
def predict_anomaly(payload: WindowRequest):
    arr = np.array(payload.window, dtype=float)
    if arr.shape != (W, 3):
        raise HTTPException(
            status_code=400,
            detail=f"Expected window shape ({W}, 3), got {arr.shape}"
        )

    # 1. Extract & scale features
    feats = extract_features(arr, payload.tod_phase)
    feats_s = SCALER.transform(feats)

    # 2. Compute raw anomaly scores
    ocs_raw = float(-OCS.decision_function(feats_s)[0])
    ae_raw = float(((feats_s - AE.predict(feats_s)) ** 2).mean())

    # 3. Convert to training ECDF percentiles & fuse
    ocs_pct = to_percentile("OneClassSVM", ocs_raw)
    ae_pct = to_percentile("Autoencoder", ae_raw)
    ensemble_score = WEIGHTS["OneClassSVM"] * ocs_pct + WEIGHTS["Autoencoder"] * ae_pct

    is_anomaly = ensemble_score > ENSEMBLE_THRESHOLD

    return {
        "is_anomaly": bool(is_anomaly),
        "severity": "CRITICAL" if ensemble_score > 0.995 else ("WARNING" if is_anomaly else "NORMAL"),
        "ensemble_score": round(ensemble_score, 4),
        "threshold": ENSEMBLE_THRESHOLD,
        "model_breakdown": {
            "OneClassSVM_percentile": round(ocs_pct, 4),
            "Autoencoder_percentile": round(ae_pct, 4)
        }
    }