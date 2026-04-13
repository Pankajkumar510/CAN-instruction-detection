# CAN Bus Intrusion Detection (Internship)

This project trains and compares multiple ML models to classify CAN traffic into 5 classes:
- `0`: Normal
- `1`: DoS
- `2`: Fuzzy
- `3`: Gear
- `4`: RPM

## What Was Fixed

The original pipeline had critical issues that reduced reliability:
- Normal data parsing was incorrect and could drop normal samples.
- `DATA_PATH` did not match the current workspace structure.
- MLP class count was hardcoded.
- A leakage-prone feature (`id_freq`) was computed on the full dataset before split.

The updated `1.py` now:
- Parses `normal_run_data.txt` using strict field extraction.
- Uses the current file directory as dataset path.
- Uses a time-based split per class for a more realistic evaluation.
- Splits train/test before leakage-sensitive feature generation.
- Uses dynamic class count for neural model output.
- Tunes Random Forest, XGBoost, and LightGBM with randomized search.
- Saves the best model artifact for later inference.

## Dataset Files

Place these files in the same folder as `1.py`:
- `DoS_dataset.csv`
- `Fuzzy_dataset.csv`
- `gear_dataset.csv`
- `RPM_dataset.csv`
- `normal_run_data.txt`

## Environment Setup

Use the workspace virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install pandas numpy scikit-learn xgboost lightgbm
```

Optional deep learning model:

```powershell
pip install tensorflow
```

## Run

```powershell
python 1.py
```

## Predict on New Data

After training, the best model is saved to `artifacts/can_ids_model.pkl`.

```powershell
python predict.py --input normal_run_data.txt --output predictions.csv
```

You can also provide a CSV file in the same CAN row format.

## Current Training Configuration

- Attack rows loaded per class: `40,000`
- Normal rows loaded: `40,000`
- Total rows used: `200,000`
- Train/test split: `80/20` time-based per class
- Hyperparameter search: randomized search on a training subset

## Latest Results (This Machine)

From the latest tuned run with the temporal split:

- Logistic Regression: `macro_f1 = 0.3509`
- Random Forest: `macro_f1 = 0.7427`
- XGBoost: `macro_f1 = 0.7496`  <- best
- LightGBM: `macro_f1 = 0.7388`
- MLP: skipped (TensorFlow not installed)

### LightGBM Confusion Matrix

```text
[[6216  222 1287  154  121]
 [ 221 5911 1292  397  179]
 [ 175  551 6543  458  273]
 [ 207  435 1344 5587  427]
 [ 200  253 1131  581 5835]]
```

## How To Improve Accuracy Further

1. Increase training rows per class
- Raise `MAX_ATTACK_ROWS_PER_CLASS` and `MAX_NORMAL_ROWS` (for example to 80k or 120k) if memory allows.

2. Increase hyperparameter search budget
- Raise `SEARCH_ITERATIONS` and `SEARCH_SAMPLE_SIZE` if you want a deeper search.

3. Add stronger sequence-aware features
- Rolling window stats per `ID` (mean/std of time gap, byte entropy, change rate).
- Frequency of same `ID` in recent window.

4. Increase training rows per class
- Raise `MAX_ATTACK_ROWS_PER_CLASS` and `MAX_NORMAL_ROWS` if memory allows.

5. Class-specific thresholding / calibration
- Inspect probability outputs and optimize per-class trade-offs.

6. Ensemble top tree models
- Soft-vote or stack RandomForest + LightGBM + XGBoost for a small gain.

## Notes

- TensorFlow is optional. If unavailable, the script skips MLP and continues.
- The current best model in this run is **XGBoost**.
- The best trained artifact is saved to `artifacts/can_ids_model.pkl`.
