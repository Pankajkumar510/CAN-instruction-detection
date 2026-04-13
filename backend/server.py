"""
server.py — Flask API for TinyCNNCANNet Live Demo
Wraps the trained artifact (artifacts/can_ids_model.pkl) and
exposes a /api/predict endpoint consumed by the frontend.

Run:
    python server.py

Then open: http://localhost:5000
"""

import os
import re
import time
import pickle
import traceback

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ─── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_PATH = os.path.join(BASE_DIR, "artifacts", "can_ids_model.pkl")
FRONTEND_DIR  = os.path.join(BASE_DIR, "frontend")

# ─── Class labels ──────────────────────────────────────────────────────────
DEFAULT_LABELS = {0: "Normal", 1: "DoS", 2: "Fuzzy", 3: "Gear", 4: "RPM"}

# ─── Load artifact once at startup ─────────────────────────────────────────
print("[server] Loading model artifact ...", end=" ", flush=True)
with open(ARTIFACT_PATH, "rb") as f:
    ARTIFACT = pickle.load(f)
print("OK")
print(f"[server] Features : {ARTIFACT['features']}")
print(f"[server] Model    : {type(ARTIFACT['model']).__name__}")

MODEL         = ARTIFACT["model"]
SCALER        = ARTIFACT.get("scaler")
FEATURES      = ARTIFACT["features"]
ID_FREQ_MAP   = ARTIFACT.get("id_freq_map", {})
BURST_THRESH  = ARTIFACT.get("burst_threshold", 0.001)
LABEL_NAMES   = ARTIFACT.get("label_names", DEFAULT_LABELS)

# ─── Flask app ──────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app)   # allow cross-origin requests from file:// or localhost:7823


# ─── Helper: hex str → int ──────────────────────────────────────────────────
def hex_to_int(x):
    try:
        return int(str(x), 16)
    except Exception:
        return 0


# ─── Helper: build feature row from a single CAN frame dict ─────────────────
def build_single_frame_features(frame: dict, prev_timestamp: float) -> pd.DataFrame:
    """
    frame keys: timestamp (float), id (str hex), dlc (int), d0-d7 (str hex)
    Returns a 1-row DataFrame with the same feature columns as training.
    """
    ts    = float(frame.get("timestamp", 0))
    can_id = hex_to_int(frame.get("id", "0"))
    dlc   = min(max(int(frame.get("dlc", 0)), 0), 8)
    bytes_ = [hex_to_int(frame.get(f"d{i}", "00")) for i in range(8)]

    time_diff      = max(ts - prev_timestamp, 0)
    byte_sum       = sum(bytes_)
    byte_std       = float(np.std(bytes_))
    byte_change    = 0.0          # no previous frame to diff against
    id_change      = 0.0
    byte_zero_ratio = bytes_.count(0) / 8
    byte_max       = max(bytes_)
    id_freq        = ID_FREQ_MAP.get(can_id, 0)
    burst          = int(time_diff < BURST_THRESH)

    row = {
        "timestamp":       ts,
        "ID":              can_id,
        "DLC":             dlc,
        "D0": bytes_[0], "D1": bytes_[1], "D2": bytes_[2], "D3": bytes_[3],
        "D4": bytes_[4], "D5": bytes_[5], "D6": bytes_[6], "D7": bytes_[7],
        "time_diff":       time_diff,
        "byte_sum":        byte_sum,
        "byte_std":        byte_std,
        "byte_change":     byte_change,
        "id_change":       id_change,
        "byte_zero_ratio": byte_zero_ratio,
        "byte_max":        byte_max,
        "id_freq":         id_freq,
        "burst":           burst,
    }

    # Fill anything extra the model needs with 0
    for col in FEATURES:
        if col not in row:
            row[col] = 0

    return pd.DataFrame([row])[FEATURES]


# ══════════════════════════════════════════════════════════════════
# API Routes
# ══════════════════════════════════════════════════════════════════

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    POST /api/predict
    Body (JSON):
        {
          "timestamp": 0.001200,
          "id":        "0244",
          "dlc":       8,
          "d0": "00", "d1": "00", ..., "d7": "00",
          "prev_timestamp": 0.0      // optional
        }
    Response (JSON):
        {
          "label":     "Normal",
          "is_attack": false,
          "class_id":  0,
          "confidence": 0.977,
          "probs": { "Normal": 0.977, "DoS": 0.009, ... },
          "inference_ms": 0.41
        }
    """
    try:
        data = request.get_json(force=True)
        prev_ts = float(data.get("prev_timestamp", 0.0))

        t0  = time.perf_counter()
        X   = build_single_frame_features(data, prev_ts)
        X_input = SCALER.transform(X) if SCALER is not None else X.values

        pred        = int(MODEL.predict(X_input)[0])
        label_name  = LABEL_NAMES.get(pred, str(pred))
        is_attack   = pred != 0

        # Probabilities (if the model supports predict_proba)
        probs = {}
        confidence = 1.0
        if hasattr(MODEL, "predict_proba"):
            raw = MODEL.predict_proba(X_input)[0]
            classes = MODEL.classes_
            probs = {
                LABEL_NAMES.get(int(c), str(c)): round(float(p), 4)
                for c, p in zip(classes, raw)
            }
            confidence = round(float(max(raw)), 4)
        else:
            # Models without predict_proba (e.g. SVM)
            probs = {LABEL_NAMES.get(k, str(k)): (1.0 if k == pred else 0.0)
                     for k in range(len(LABEL_NAMES))}

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        return jsonify({
            "label":        label_name,
            "is_attack":    is_attack,
            "class_id":     pred,
            "confidence":   confidence,
            "probs":        probs,
            "inference_ms": elapsed_ms,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/batch_predict", methods=["POST"])
def batch_predict():
    """
    POST /api/batch_predict
    Body (JSON):
        { "frames": [ <frame>, <frame>, ... ] }
    Each frame has the same shape as /api/predict.
    Returns a list of predictions in order.
    """
    try:
        data   = request.get_json(force=True)
        frames = data.get("frames", [])
        if not frames:
            return jsonify({"error": "No frames provided"}), 400

        rows = []
        prev_ts = 0.0
        for frame in frames:
            rows.append(build_single_frame_features(frame, prev_ts))
            prev_ts = float(frame.get("timestamp", 0))

        X = pd.concat(rows, ignore_index=True)
        X_input = SCALER.transform(X) if SCALER is not None else X.values

        t0    = time.perf_counter()
        preds = MODEL.predict(X_input)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        results = []
        for pred in preds:
            pred = int(pred)
            results.append({
                "label":    LABEL_NAMES.get(pred, str(pred)),
                "is_attack": pred != 0,
                "class_id": pred,
            })

        return jsonify({"predictions": results, "total_ms": elapsed_ms})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/model_info", methods=["GET"])
def model_info():
    """GET /api/model_info — returns metadata about the loaded model."""
    return jsonify({
        "model_type":  type(MODEL).__name__,
        "features":    FEATURES,
        "num_classes": len(LABEL_NAMES),
        "labels":      LABEL_NAMES,
        "has_scaler":  SCALER is not None,
        "burst_threshold": BURST_THRESH,
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": type(MODEL).__name__})


# ══════════════════════════════════════════════════════════════════
# Serve Frontend
# ══════════════════════════════════════════════════════════════════

@app.route("/", defaults={"path": "index.html"})
@app.route("/<path:path>")
def serve_frontend(path):
    """Serve the static frontend files."""
    return send_from_directory(FRONTEND_DIR, path)


# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    sep = "-" * 55
    print("\n" + sep)
    print("  TinyCNNCANNet -- API Server")
    print("  http://localhost:5000         <- Frontend + API")
    print("  http://localhost:5000/api/health")
    print("  http://localhost:5000/api/model_info")
    print(sep + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
