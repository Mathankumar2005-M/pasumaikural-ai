"""
train_sensor_model.py
----------------------
One-time script (run locally, not on Vercel) that builds a synthetic-but-
realistic sensor dataset using agronomy rules of thumb + noise, then trains
a RandomForestClassifier on it and saves it to model/sensor_model.pkl.

Run:
    python model/train_sensor_model.py

Classes:
    Healthy, Thirsty, Pest Risk, Nutrient Deficient
"""
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

RNG = np.random.default_rng(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(BASE_DIR, "sensor_model.pkl")


def build_dataset(n_per_class=500):
    """
    Generates synthetic (soil_moisture, temperature, humidity, light) rows
    labelled using simple agronomy rules + gaussian noise, so the model has
    to learn a real decision boundary instead of memorizing a lookup table.

    Ranges:
        soil_moisture: 0-100 (%)
        temperature:   15-40 (deg C)
        humidity:      20-100 (%)
        light:         0.0-1.0 (normalized lux)
    """
    rows = []
    labels = []

    def add(soil, temp, hum, light, label):
        rows.append([soil, temp, hum, light])
        labels.append(label)

    for _ in range(n_per_class):
        # Healthy: moderate moisture, moderate temp/humidity, decent light
        add(
            RNG.normal(60, 8),
            RNG.normal(27, 2),
            RNG.normal(55, 8),
            np.clip(RNG.normal(0.65, 0.1), 0, 1),
            "Healthy",
        )
        # Thirsty: low soil moisture, higher temp
        add(
            RNG.normal(18, 6),
            RNG.normal(33, 3),
            RNG.normal(40, 10),
            np.clip(RNG.normal(0.7, 0.15), 0, 1),
            "Thirsty",
        )
        # Pest Risk: high humidity + high temperature (classic pest breeding conditions)
        add(
            RNG.normal(55, 10),
            RNG.normal(32, 2),
            RNG.normal(85, 6),
            np.clip(RNG.normal(0.5, 0.15), 0, 1),
            "Pest Risk",
        )
        # Nutrient Deficient: moisture ok-ish but low light, cooler temps
        # (mimics stunted growth conditions where nutrient uptake suffers)
        add(
            RNG.normal(45, 10),
            RNG.normal(22, 3),
            RNG.normal(50, 10),
            np.clip(RNG.normal(0.25, 0.1), 0, 1),
            "Nutrient Deficient",
        )

    X = np.array(rows, dtype=np.float32)
    y = np.array(labels)

    # clip to physically valid ranges
    X[:, 0] = np.clip(X[:, 0], 0, 100)    # soil_moisture
    X[:, 1] = np.clip(X[:, 1], 10, 45)    # temperature
    X[:, 2] = np.clip(X[:, 2], 10, 100)   # humidity
    X[:, 3] = np.clip(X[:, 3], 0, 1)      # light

    return X, y


def main():
    X, y = build_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"Sensor model — train accuracy: {train_acc:.3f}, test accuracy: {test_acc:.3f}")

    bundle = {
        "model": model,
        "feature_names": ["soil_moisture", "temperature", "humidity", "light"],
        "classes": sorted(set(y)),
    }
    joblib.dump(bundle, OUT_PATH)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
