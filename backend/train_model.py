"""Train the first SIGNBRIDGE static-sign classifier from sign_landmarks.csv."""
from pathlib import Path
import csv
import sys

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "dataset" / "sign_landmarks.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "sign_model.joblib"
FEATURE_COLUMNS = [
    f"{hand}_{axis}{i}"
    for hand in ("left", "right")
    for i in range(21)
    for axis in ("x", "y", "z")
]
MIN_SAMPLES_PER_SIGN = 8


def main():
    if not CSV_PATH.exists():
        print(f"Dataset not found: {CSV_PATH}")
        return 1

    X, y = [], []
    with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in ["label", *FEATURE_COLUMNS] if name not in (reader.fieldnames or [])]
        if missing:
            print("CSV format does not match the expected 126-landmark format.")
            print("Missing columns:", ", ".join(missing[:8]))
            return 1
        for row in reader:
            label = (row.get("label") or "").strip().upper()
            if not label:
                continue
            try:
                values = [float(row[column]) for column in FEATURE_COLUMNS]
            except (TypeError, ValueError):
                continue
            X.append(values)
            y.append(label)

    if not X:
        print("No valid samples found.")
        return 1

    labels, counts = np.unique(y, return_counts=True)
    print("Dataset summary:")
    for label, count in zip(labels, counts):
        print(f"  {label}: {count} samples")

    if len(labels) < 2:
        print("\nTraining stopped: at least TWO different sign labels are required.")
        return 1
    if int(counts.min()) < MIN_SAMPLES_PER_SIGN:
        print(f"\nTraining stopped: collect at least {MIN_SAMPLES_PER_SIGN} samples for EVERY sign first.")
        return 1

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            min_samples_leaf=2,
        )),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, predictions))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Held-out test accuracy: {accuracy:.1%}")
    print("Important: this score is only a small-dataset prototype metric, not proof of real-world ISL accuracy.")
    print("Restart the FastAPI server after training so it reloads the model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
