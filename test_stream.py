"""
Real-Time Stream Simulator Client for Anomaly Detection API
Tests Normal Traffic, Point Spike, Correlation Break, and Collective Flatline.
"""
import json
import urllib.request
import numpy as np

API_URL = "http://127.0.0.1:8000/predict"
RNG = np.random.default_rng(99)


def send_window(name: str, window: np.ndarray, tod_phase: float = 1.5):
    payload = json.dumps({
        "window": window.tolist(),
        "tod_phase": tod_phase
    }).encode("utf-8")

    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        print(f"\n[{name}]")
        print(f"  Status   : {result['severity']} (Anomaly: {result['is_anomaly']})")
        print(f"  Score    : {result['ensemble_score']} (Threshold: {result['threshold']})")
        print(f"  Breakdown: OCSVM={result['model_breakdown']['OneClassSVM_percentile']} | AE={result['model_breakdown']['Autoencoder_percentile']}")


# Generate a realistic 24-step baseline normal window
t = np.arange(24)
phase = 1.5 + 2 * np.pi * t / 288
base = np.sin(phase)
ch0 = 10 + 3 * base + RNG.normal(0, 0.4, 24)
ch1 = 0.8 * ch0 + 2 * np.cos(phase) + RNG.normal(0, 0.3, 24)
ch2 = 5 + 1.5 * np.sin(phase + 1.0) + RNG.normal(0, 0.25, 24)
normal_win = np.c_[ch0, ch1, ch2]

# 1. Normal Traffic
send_window("1. Normal Traffic", normal_win)

# 2. Point Spike (DDoS / Fraud Burst)
spike_win = normal_win.copy()
spike_win[-1, 0] += 8.5
send_window("2. Sudden Point Spike", spike_win)

# 3. Correlation Break (Stealth Exfiltration)
corr_win = normal_win.copy()
corr_win[-12:, 1] = 10.0 + RNG.normal(0, 1.1, 12)
send_window("3. Correlation Break", corr_win)

# 4. Collective Flatline (Sensor Freeze)
flat_win = normal_win.copy()
flat_win[:, 0] = flat_win[0, 0]
send_window("4. Collective Flatline", flat_win)