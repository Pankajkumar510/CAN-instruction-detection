import os
import pickle
import re
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

Input = Dense = Dropout = Sequential = EarlyStopping = ReduceLROnPlateau = None

try:
    from tensorflow.keras import Input  # type: ignore[import-not-found]
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore[import-not-found]
    from tensorflow.keras.layers import Dense, Dropout  # type: ignore[import-not-found]
    from tensorflow.keras.models import Sequential  # type: ignore[import-not-found]

    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False


# ============================================
# CONFIG
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = BASE_DIR
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_ARTIFACT_PATH = os.path.join(ARTIFACT_DIR, "can_ids_model.pkl")

RANDOM_STATE = 42
MAX_ATTACK_ROWS_PER_CLASS = 40000
MAX_NORMAL_ROWS = 40000
TEST_SIZE = 0.2
SEARCH_SAMPLE_SIZE = 50000
SEARCH_ITERATIONS = 8
MLP_EPOCHS = 25
MLP_BATCH_SIZE = 128
SAVE_TRAIN_TEST_SPLITS = True

LABEL_NAMES = {
    0: "Normal",
    1: "DoS",
    2: "Fuzzy",
    3: "Gear",
    4: "RPM",
}


def print_section(title: str) -> None:
    print(f"\n{'=' * 20} {title} {'=' * 20}")


def check_dataset_files() -> None:
    required_files = [
        "DoS_dataset.csv",
        "Fuzzy_dataset.csv",
        "gear_dataset.csv",
        "RPM_dataset.csv",
        "normal_run_data.txt",
    ]
    missing = [f for f in required_files if not os.path.exists(os.path.join(DATA_PATH, f))]
    if missing:
        raise FileNotFoundError(f"Missing dataset files: {missing}")
    print("Dataset files found:")
    for file_name in required_files:
        print(f" - {file_name}")


# ============================================
# LOAD ATTACK DATA
# ============================================
def load_attack_file(path: str, label: int, max_rows: int = 40000) -> pd.DataFrame:
    chunks: List[pd.DataFrame] = []
    total = 0
    raw_rows = 0
    valid_rows = 0

    for chunk in pd.read_csv(
        path,
        header=None,
        sep=",",
        engine="python",
        on_bad_lines="skip",
        chunksize=20000,
    ):
        raw_rows += len(chunk)
        chunk12 = chunk.iloc[:, :12]
        chunk12 = chunk12.dropna(subset=list(range(12)))
        valid_rows += len(chunk12)

        chunk12["label"] = label
        chunks.append(chunk12)

        total += len(chunk12)
        if total >= max_rows:
            break

    if not chunks:
        return pd.DataFrame(columns=list(range(12)) + ["label"])

    df = pd.concat(chunks, ignore_index=True).head(max_rows)
    print(
        f"[ATTACK] {os.path.basename(path)} | raw_rows={raw_rows} | valid_rows={valid_rows} | used_rows={len(df)}"
    )
    return df


def load_attack_data() -> pd.DataFrame:
    files = {
        "DoS_dataset.csv": 1,
        "Fuzzy_dataset.csv": 2,
        "gear_dataset.csv": 3,
        "RPM_dataset.csv": 4,
    }

    dfs = []
    for file_name, label in files.items():
        print(f"Loading {file_name}")
        dfs.append(
            load_attack_file(
                os.path.join(DATA_PATH, file_name),
                label,
                max_rows=MAX_ATTACK_ROWS_PER_CLASS,
            )
        )

    return pd.concat(dfs, ignore_index=True)


# ============================================
# LOAD NORMAL DATA
# ============================================
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


def load_normal_data(max_rows: int = 40000) -> pd.DataFrame:
    path = os.path.join(DATA_PATH, "normal_run_data.txt")
    rows: List[List[str]] = []
    total_lines = 0
    parsed_lines = 0

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            total_lines += 1
            parsed = parse_normal_line(line)
            if parsed is not None:
                parsed_lines += 1
                rows.append(parsed)

            if len(rows) >= max_rows:
                break

    print(f"[NORMAL] parsed_lines={parsed_lines}/{total_lines} | used_rows={len(rows)}")
    df = pd.DataFrame(rows)
    df["label"] = 0
    return df


# ============================================
# PREPROCESSING
# ============================================
def hex_to_int(x: object) -> int:
    try:
        return int(str(x), 16)
    except Exception:
        return 0


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
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
        "label",
    ]
    df.columns = columns

    df = df.dropna().copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    df["ID"] = df["ID"].apply(hex_to_int)
    df["DLC"] = pd.to_numeric(df["DLC"], errors="coerce").fillna(0).clip(0, 8)

    for i in range(8):
        df[f"D{i}"] = df[f"D{i}"].apply(hex_to_int)

    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    return df


def time_based_split_per_class(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    test_parts = []

    for _, group in df.groupby("label"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        if len(group) < 2:
            train_parts.append(group)
            continue

        split_idx = int(len(group) * (1 - test_size))
        split_idx = min(max(split_idx, 1), len(group) - 1)
        train_parts.append(group.iloc[:split_idx].copy())
        test_parts.append(group.iloc[split_idx:].copy())

    train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    test_df = pd.concat(test_parts, ignore_index=True).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return train_df, test_df


def add_features(
    df: pd.DataFrame,
    id_freq_map: Optional[pd.Series] = None,
    burst_threshold: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.Series, float]:
    df = df.sort_values("timestamp").copy()
    byte_cols = [f"D{i}" for i in range(8)]

    df["time_diff"] = df["timestamp"].diff().clip(lower=0).fillna(0)
    df["byte_sum"] = df[byte_cols].sum(axis=1)
    df["byte_std"] = df[byte_cols].std(axis=1)
    df["byte_change"] = df[byte_cols].diff().abs().sum(axis=1).fillna(0)
    df["id_change"] = df["ID"].diff().abs().fillna(0)
    df["byte_zero_ratio"] = (df[byte_cols] == 0).mean(axis=1)
    df["byte_max"] = df[byte_cols].max(axis=1)

    if id_freq_map is None:
        id_freq_map = df["ID"].value_counts()
    df["id_freq"] = df["ID"].map(id_freq_map).fillna(0)

    if burst_threshold is None:
        burst_threshold = float(df["time_diff"].quantile(0.1))
    df["burst"] = (df["time_diff"] < burst_threshold).astype(int)

    return df, id_freq_map, burst_threshold


def evaluate_model(name: str, y_true: pd.Series, y_pred: np.ndarray) -> float:
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    print(f"\n===== {name} =====")
    print(f"Macro F1: {macro_f1:.4f}")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    return macro_f1


def tune_model(name: str, estimator: Any, param_distributions: Dict, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    sample_size = min(SEARCH_SAMPLE_SIZE, len(X_train))
    if sample_size < len(X_train):
        X_sample, _, y_sample, _ = train_test_split(
            X_train,
            y_train,
            train_size=sample_size,
            stratify=y_train,
            random_state=RANDOM_STATE,
        )
    else:
        X_sample, y_sample = X_train, y_train

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=SEARCH_ITERATIONS,
        scoring="f1_macro",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    print(f"\nTuning {name} on {len(X_sample)} rows...")
    search.fit(X_sample, y_sample)
    print(f"Best {name} params: {search.best_params_}")
    best_estimator: Any = search.best_estimator_
    best_estimator.fit(X_train, y_train)
    return best_estimator


def save_artifact(artifact: dict, path: str = MODEL_ARTIFACT_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(artifact, f)


def main() -> None:
    print_section("CONFIG")
    print(f"DATA_PATH: {DATA_PATH}")
    print(f"MAX_ATTACK_ROWS_PER_CLASS: {MAX_ATTACK_ROWS_PER_CLASS}")
    print(f"MAX_NORMAL_ROWS: {MAX_NORMAL_ROWS}")
    print(f"TEST_SIZE: {TEST_SIZE}")
    print(f"SEARCH_SAMPLE_SIZE: {SEARCH_SAMPLE_SIZE}")
    print(f"SEARCH_ITERATIONS: {SEARCH_ITERATIONS}")
    print(f"MLP_EPOCHS: {MLP_EPOCHS}")
    print(f"MLP_BATCH_SIZE: {MLP_BATCH_SIZE}")

    print_section("DATASET CHECK")
    check_dataset_files()

    print_section("LOADING DATA")
    attack_df = load_attack_data()
    normal_df = load_normal_data(max_rows=MAX_NORMAL_ROWS)
    df = pd.concat([attack_df, normal_df], ignore_index=True)

    print_section("CLEANING")
    df = clean_dataframe(df)
    print("Class Distribution (full):")
    print(df["label"].value_counts().sort_index())

    print_section("TRAIN TEST SPLIT")
    train_df, test_df = time_based_split_per_class(df, test_size=TEST_SIZE)
    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")
    print("Train class distribution:")
    print(train_df["label"].value_counts().sort_index())
    print("Test class distribution:")
    print(test_df["label"].value_counts().sort_index())

    if SAVE_TRAIN_TEST_SPLITS:
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        train_path = os.path.join(ARTIFACT_DIR, "train_dataset.csv")
        test_path = os.path.join(ARTIFACT_DIR, "test_dataset.csv")
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        print(f"Saved train dataset: {train_path}")
        print(f"Saved test dataset: {test_path}")

    train_df, id_freq_map, burst_threshold = add_features(train_df)
    test_df, _, _ = add_features(test_df, id_freq_map=id_freq_map, burst_threshold=burst_threshold)

    features = [
        "ID",
        "DLC",
        "time_diff",
        "byte_sum",
        "byte_std",
        "id_freq",
        "byte_change",
        "burst",
        "id_change",
        "byte_zero_ratio",
        "byte_max",
    ] + [f"D{i}" for i in range(8)]
    print("Features used for training:")
    print(features)

    X_train = train_df[features]
    X_test = test_df[features]
    y_train = train_df["label"]
    y_test = test_df["label"]
    n_classes = y_train.nunique()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    scores: Dict[str, float] = {}
    trained_models: Dict[str, Any] = {}

    lr = LogisticRegression(max_iter=700, class_weight="balanced", random_state=RANDOM_STATE)
    print(f"Logistic Regression params: {lr.get_params()}")
    lr.fit(X_train_scaled, y_train)
    trained_models["Logistic Regression"] = lr
    scores["Logistic Regression"] = evaluate_model("LOGISTIC REGRESSION", y_test, lr.predict(X_test_scaled))

    rf_base = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf_params = {
        "n_estimators": [200, 300, 400, 500],
        "max_depth": [None, 12, 18, 24],
        "min_samples_leaf": [1, 2, 4],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2", 0.5],
    }
    rf = tune_model("Random Forest", rf_base, rf_params, X_train, y_train)
    print(f"Random Forest final params: {rf.get_params()}")
    trained_models["Random Forest"] = rf
    scores["Random Forest"] = evaluate_model("RANDOM FOREST", y_test, rf.predict(X_test))

    xgb_base = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb_params = {
        "n_estimators": [200, 300, 400],
        "max_depth": [6, 8, 10],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.2],
    }
    xgb_model = tune_model("XGBoost", xgb_base, xgb_params, X_train, y_train)
    print(f"XGBoost final params: {xgb_model.get_params()}")
    trained_models["XGBoost"] = xgb_model
    scores["XGBoost"] = evaluate_model("XGBOOST", y_test, xgb_model.predict(X_test))

    lgb_base = lgb.LGBMClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    lgb_params = {
        "n_estimators": [200, 300, 400, 500],
        "learning_rate": [0.02, 0.03, 0.05],
        "num_leaves": [31, 63, 127],
        "max_depth": [-1, 10, 14, 18],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "min_child_samples": [10, 20, 30],
    }
    lgb_model = tune_model("LightGBM", lgb_base, lgb_params, X_train, y_train)
    print(f"LightGBM final params: {lgb_model.get_params()}")
    trained_models["LightGBM"] = lgb_model
    lgb_pred = lgb_model.predict(X_test)
    scores["LightGBM"] = evaluate_model("LIGHTGBM", y_test, lgb_pred)
    print("\nConfusion Matrix (LightGBM):\n", confusion_matrix(y_test, lgb_pred))

    if TENSORFLOW_AVAILABLE:
        assert (
            Input is not None
            and Dense is not None
            and Dropout is not None
            and Sequential is not None
            and EarlyStopping is not None
            and ReduceLROnPlateau is not None
        )
        mlp = Sequential(
            [
                Input(shape=(X_train.shape[1],)),
                Dense(128, activation="relu"),
                Dropout(0.3),
                Dense(64, activation="relu"),
                Dense(n_classes, activation="softmax"),
            ]
        )
        mlp.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        print(
            f"MLP training started with epochs={MLP_EPOCHS}, batch_size={MLP_BATCH_SIZE}, validation_split=0.2"
        )
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
        ]
        mlp.fit(
            X_train_scaled,
            y_train,
            epochs=MLP_EPOCHS,
            batch_size=MLP_BATCH_SIZE,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1,
        )
        mlp_pred = np.argmax(mlp.predict(X_test_scaled, verbose=0), axis=1)
        trained_models["MLP"] = mlp
        scores["MLP"] = evaluate_model("MLP", y_test, mlp_pred)
    else:
        print("\nTensorFlow not installed. Skipping MLP model.")

    best_model_name = max(scores.items(), key=lambda item: item[1])[0]
    best_model = trained_models[best_model_name]

    artifact = {
        "model_name": best_model_name,
        "model": best_model,
        "scaler": scaler if best_model_name in {"Logistic Regression", "MLP"} else None,
        "features": features,
        "label_names": LABEL_NAMES,
        "id_freq_map": id_freq_map.to_dict(),
        "burst_threshold": burst_threshold,
        "class_count": n_classes,
    }
    save_artifact(artifact)

    print("\n===== SUMMARY =====")
    for model_name, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        print(f"{model_name}: macro_f1={score:.4f}")
    print(f"Best model: {best_model_name}")
    print(f"Saved artifact: {MODEL_ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
