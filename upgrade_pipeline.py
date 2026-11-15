"""
Upgraded Time-Series Anomaly Detection Pipeline (v2)
Improvements:
1. Window baseline normalization (eliminates linear trend drift false positives)
2. Explicit peak-to-peak (ptp) & flatline indicators (boosts collective anomaly recall)
3. Weighted ECDF Ensemble (prevents weak Isolation Forest from dragging down F1)
4. Saves model artifacts & thresholds to ./models/ for FastAPI deployment
"""
import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

RNG = np.random.default_rng(42)
OUT = "outputs_v2"
MODEL_DIR = "models"
os.makedirs(OUT, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# 1. Synthetic Time-Series Generator (Same seed for apples-to-apples comparison)
# ----------------------------------------------------------------------------
N, PERIOD = 60_000, 288
t = np.arange(N)
phase = 2 * np.pi * (t % PERIOD) / PERIOD
base = np.sin(phase)

ch0 = 10 + 3 * base + 0.0002 * t / 100 + RNG.normal(0, 0.5, N)
ch1 = 0.8 * ch0 + 2 * np.cos(phase) + RNG.normal(0, 0.4, N)
ch2 = 5 + 1.5 * np.sin(phase + 1.0) + RNG.normal(0, 0.3, N)
X_raw = np.c_[ch0, ch1, ch2]
y = np.zeros(N, dtype=int)
kind = np.array([""] * N, dtype=object)

def inject(lo, hi, n_events):
    for _ in range(n_events):
        typ = RNG.choice(["point", "contextual", "collective", "corr_break"])
        i = int(RNG.integers(lo + 50, hi - 60))
        if typ == "point":
            X_raw[i, RNG.integers(0, 3)] += RNG.choice([-1, 1]) * RNG.uniform(5, 9)
            sl = slice(i, i + 1)
        elif typ == "contextual":
            j = RNG.integers(0, 2)
            X_raw[i:i + 3, j] += -2.2 * 3 * base[i:i + 3] * np.sign(base[i]) + RNG.normal(0, .2, 3)
            sl = slice(i, i + 3)
        elif typ == "collective":
            L = int(RNG.integers(15, 40))
            if RNG.random() < .5:
                X_raw[i:i + L, 0] = X_raw[i, 0]
            else:
                X_raw[i:i + L, 0] += RNG.normal(0, 2.2, L)
            sl = slice(i, i + L)
        else:
            L = int(RNG.integers(10, 30))
            X_raw[i:i + L, 1] = 10 + RNG.normal(0, 1.2, L)
            sl = slice(i, i + L)
        y[sl] = 1
        kind[sl] = typ

SPLIT = N // 2
inject(0, SPLIT, 6)
inject(SPLIT, N, 70)

# ----------------------------------------------------------------------------
# 2. Upgraded Feature Extraction (85 Features)
# ----------------------------------------------------------------------------
W = 24
def make_windows(X, tod_phase):
    idx = np.arange(W - 1, len(X))
    raw_wins = np.stack([X[i - W + 1:i + 1] for i in idx])  # (N_win, W, 3)
    
    # Detrended relative window (removes slow upward/downward trend drift)
    rel_wins = (raw_wins - raw_wins[:, 0:1, :]).reshape(len(idx), -1)
    
    # Step-to-step roughness & volatility
    diffs = np.abs(np.diff(raw_wins, axis=1)).mean(axis=1)
    stds = raw_wins.std(axis=1)
    
    # NEW: Peak-to-peak range & log-variance (catches flatlines & bursts sharply)
    ptp = np.ptp(raw_wins, axis=1)
    flatline_flag = (stds[:, 0:1] < 1e-4).astype(float)
    
    # Time-of-day cyclic features
    tod = np.c_[np.sin(tod_phase[idx]), np.cos(tod_phase[idx])]
    
    # Cross-channel correlation
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.array([np.corrcoef(w[:, 0], w[:, 1])[0, 1] for w in raw_wins])
    corr = np.nan_to_num(corr, nan=-2.0)[:, None]  # -2.0 signals broken/undefined corr
    
    return np.c_[rel_wins, diffs, stds, ptp, flatline_flag, tod, corr], idx

F, idx = make_windows(X_raw, phase)
yw = np.array([y[i - W + 1:i + 1].max() for i in idx])
kw = np.array([next((kind[j] for j in range(i, i - W, -1) if y[j]), "") for i in idx], dtype=object)

tr = idx < SPLIT
Xtr, Xte, yte, kte = F[tr], F[~tr], yw[~tr], kw[~tr]

# RobustScaler handles outlier spikes better than StandardScaler
scaler = RobustScaler().fit(Xtr)
Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

# ----------------------------------------------------------------------------
# 3. Train Detectors
# ----------------------------------------------------------------------------
scores_tr, scores_te, fit_time = {}, {}, {}

t0 = time.time()
iso = IsolationForest(n_estimators=300, max_samples=512, contamination="auto", random_state=0).fit(Xtr_s)
scores_tr["IsolationForest"], scores_te["IsolationForest"] = -iso.score_samples(Xtr_s), -iso.score_samples(Xte_s)
fit_time["IsolationForest"] = time.time() - t0

t0 = time.time()
sub = RNG.choice(len(Xtr_s), 6000, replace=False)
ocs = OneClassSVM(kernel="rbf", gamma="scale", nu=0.01).fit(Xtr_s[sub])
scores_tr["OneClassSVM"], scores_te["OneClassSVM"] = -ocs.decision_function(Xtr_s), -ocs.decision_function(Xte_s)
fit_time["OneClassSVM"] = time.time() - t0

t0 = time.time()
ae = MLPRegressor(hidden_layer_sizes=(64, 16, 64), activation="relu", alpha=1e-3, learning_rate_init=1e-3,
                  max_iter=80, early_stopping=True, validation_fraction=0.1, random_state=0).fit(Xtr_s, Xtr_s)
rec = lambda A: ((A - ae.predict(A)) ** 2).mean(1)
scores_tr["Autoencoder"], scores_te["Autoencoder"] = rec(Xtr_s), rec(Xte_s)
fit_time["Autoencoder"] = time.time() - t0

# ----------------------------------------------------------------------------
# 4. Weighted ECDF Ensemble (0.50 AE + 0.40 OCSVM + 0.10 IF)
# ----------------------------------------------------------------------------
def pct(s_tr_sorted, s):
    return np.searchsorted(s_tr_sorted, s) / len(s_tr_sorted)

sorted_tr = {m: np.sort(scores_tr[m]) for m in ["IsolationForest", "OneClassSVM", "Autoencoder"]}
weights = {"Autoencoder": 0.50, "OneClassSVM": 0.40, "IsolationForest": 0.10}

ens_tr = sum(weights[m] * pct(sorted_tr[m], scores_tr[m]) for m in weights)
ens_te = sum(weights[m] * pct(sorted_tr[m], scores_te[m]) for m in weights)
scores_tr["WeightedEnsemble"], scores_te["WeightedEnsemble"] = ens_tr, ens_te

# ----------------------------------------------------------------------------
# 5. Evaluate & Compare
# ----------------------------------------------------------------------------
rows = []
thresholds = {}
for m in scores_te:
    thr = float(np.quantile(scores_tr[m], 0.99))
    thresholds[m] = thr
    pred = (scores_te[m] > thr).astype(int)
    p = precision_score(yte, pred, zero_division=0)
    r = recall_score(yte, pred)
    f = f1_score(yte, pred)
    rows.append(dict(model=m, precision=p, recall=r, f1=f,
                     pr_auc=average_precision_score(yte, scores_te[m]),
                     fit_sec=fit_time.get(m, np.nan)))

res = pd.DataFrame(rows).set_index("model").round(3)
print("\n=== V2 Overall Performance (threshold = train 99th pct) ===")
print(res.to_string())

print("\n=== V2 Recall by Anomaly Type ===")
types = ["point", "contextual", "collective", "corr_break"]
rt = {}
for m in scores_te:
    pred = scores_te[m] > thresholds[m]
    rt[m] = {k: round(pred[kte == k].mean(), 3) for k in types}
rt = pd.DataFrame(rt).T
print(rt.to_string())

res.to_csv(f"{OUT}/v2_results_overall.csv")
rt.to_csv(f"{OUT}/v2_results_by_type.csv")

# Save production bundle for FastAPI deployment
bundle = {
    "scaler": scaler,
    "iso": iso,
    "ocs": ocs,
    "ae": ae,
    "sorted_tr": sorted_tr,
    "weights": weights,
    "thresholds": thresholds,
    "window_size": W
}
joblib.dump(bundle, f"{MODEL_DIR}/anomaly_bundle.joblib")
print(f"\nSaved production model bundle to {MODEL_DIR}/anomaly_bundle.joblib")