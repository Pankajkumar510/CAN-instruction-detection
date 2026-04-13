import argparse
import os
import pickle
import re
from typing import List, Optional

import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_PATH = os.path.join(BASE_DIR, "artifacts", "can_ids_model.pkl")


def parse_normal_line(line: str) -> Optional[List[str]]:
    m = re.search(
        r"Timestamp:\s*([0-9]+\.[0-9]+)\s+ID:\s*([0-9A-Fa-f]+)\s+[0-9A-Fa-f]+\s+DLC:\s*(\d+)\s+([0-9A-Fa-f\s]+)$",
        line,
    )
    if not m:
        return None

    timestamp, can_id, dlc, payload = m.groups()
    dlc_val = max(0, min(int(dlc), 8))
    bytes_raw = re.findall(r"[0-9A-Fa-f]{2}", payload)
    bytes_fixed = (bytes_raw[:dlc_val] + ["00"] * (8 - dlc_val))[:8]
    return [timestamp, can_id, str(dlc_val)] + bytes_fixed + ["R"]


def hex_to_int(x: object) -> int:
    try:
        return int(str(x), 16)
    except Exception:
        return 0


def load_input_file(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    rows = []

    if ext == ".txt":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parsed = parse_normal_line(line)
                if parsed is not None:
                    rows.append(parsed)
    else:
        df = pd.read_csv(path, header=None, sep=",", engine="python", on_bad_lines="skip")
        for _, row in df.iloc[:, :12].iterrows():
            rows.append(list(row.values))

    if not rows:
        raise ValueError("No valid rows were parsed from the input file.")

    df = pd.DataFrame(rows)
    df.columns = [
        "timestamp",
        "ID",
        "DLC",
        "D0",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6",
        "D7",
        "flag",
    ]
    return df


def build_features(df: pd.DataFrame, id_freq_map: dict, burst_threshold: float, features: List[str]) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    df["ID"] = df["ID"].apply(hex_to_int)
    df["DLC"] = pd.to_numeric(df["DLC"], errors="coerce").fillna(0).clip(0, 8)

    for i in range(8):
        df[f"D{i}"] = df[f"D{i}"].apply(hex_to_int)

    df = df.sort_values("timestamp").copy()
    byte_cols = [f"D{i}" for i in range(8)]

    df["time_diff"] = df["timestamp"].diff().clip(lower=0).fillna(0)
    df["byte_sum"] = df[byte_cols].sum(axis=1)
    df["byte_std"] = df[byte_cols].std(axis=1)
    df["byte_change"] = df[byte_cols].diff().abs().sum(axis=1).fillna(0)
    df["id_change"] = df["ID"].diff().abs().fillna(0)
    df["byte_zero_ratio"] = (df[byte_cols] == 0).mean(axis=1)
    df["byte_max"] = df[byte_cols].max(axis=1)
    df["id_freq"] = df["ID"].map(id_freq_map).fillna(0)
    df["burst"] = (df["time_diff"] < burst_threshold).astype(int)

    for col in features:
        if col not in df.columns:
            df[col] = 0

    return df[features]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict CAN bus classes using the trained artifact.")
    parser.add_argument("--input", required=True, help="Path to a CAN CSV or normal text log")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path")
    args = parser.parse_args()

    with open(ARTIFACT_PATH, "rb") as f:
        artifact = pickle.load(f)

    df = load_input_file(args.input)
    features = artifact["features"]
    id_freq_map = artifact["id_freq_map"]
    burst_threshold = artifact["burst_threshold"]

    X = build_features(df, id_freq_map, burst_threshold, features)

    model = artifact["model"]
    scaler = artifact["scaler"]
    if scaler is not None:
        X_input = scaler.transform(X)
    else:
        X_input = X

    predictions = model.predict(X_input)
    label_names = artifact.get("label_names", {})
    pred_names = [label_names.get(int(p), str(int(p))) for p in predictions]

    output_df = df.iloc[: len(predictions)].copy()
    output_df["prediction"] = predictions
    output_df["prediction_name"] = pred_names
    output_df.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
