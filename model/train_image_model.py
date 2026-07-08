"""
train_image_model.py
---------------------
One-time script (run locally, not on Vercel) that builds a synthetic-but-
realistic dataset of the 4 color features used by
ai_engine._extract_image_features (green_ratio, yellow_brown_ratio,
dark_spot_ratio, brightness), then trains a LogisticRegression classifier
and saves it to model/image_model.pkl.

Why synthetic features instead of real photos: this keeps the addon fully
self-contained and runnable offline for the demo. Step 7 of NOTES.md
explains how to swap this out for a real labelled-photo dataset later —
just replace build_dataset() with real feature vectors extracted via
ai_engine._extract_image_features on your own leaf photos.

Run:
    python model/train_image_model.py

Classes:
    Healthy, Nutrient Deficient, Disease Spots, Pest Damage
"""
import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

RNG = np.random.default_rng(7)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(BASE_DIR, "image_model.pkl")


def _clip01(a):
    return np.clip(a, 0.0, 1.0)


def build_dataset(n_per_class=500):
    """
    Feature order: [green_ratio, yellow_brown_ratio, dark_spot_ratio, brightness]
    """
    rows = []
    labels = []

    def add(green, yellow_brown, dark_spot, brightness, label):
        rows.append([green, yellow_brown, dark_spot, brightness])
        labels.append(label)

    for _ in range(n_per_class):
        # Healthy: mostly green, very little yellow/brown or dark spots, good brightness
        add(
            _clip01(RNG.normal(0.75, 0.08)),
            _clip01(RNG.normal(0.05, 0.03)),
            _clip01(RNG.normal(0.03, 0.02)),
            _clip01(RNG.normal(0.55, 0.08)),
            "Healthy",
        )
        # Nutrient Deficient: noticeably more yellow/brown, green ratio drops
        add(
            _clip01(RNG.normal(0.45, 0.1)),
            _clip01(RNG.normal(0.35, 0.08)),
            _clip01(RNG.normal(0.05, 0.03)),
            _clip01(RNG.normal(0.5, 0.08)),
            "Nutrient Deficient",
        )
        # Disease Spots: moderate green, elevated dark-spot ratio (lesions), dimmer
        add(
            _clip01(RNG.normal(0.55, 0.1)),
            _clip01(RNG.normal(0.1, 0.05)),
            _clip01(RNG.normal(0.25, 0.07)),
            _clip01(RNG.normal(0.4, 0.08)),
            "Disease Spots",
        )
        # Pest Damage: lower green (chewed/missing leaf area), high dark-spot ratio, low brightness
        add(
            _clip01(RNG.normal(0.35, 0.1)),
            _clip01(RNG.normal(0.15, 0.06)),
            _clip01(RNG.normal(0.35, 0.08)),
            _clip01(RNG.normal(0.35, 0.08)),
            "Pest Damage",
        )

    X = np.array(rows, dtype=np.float32)
    y = np.array(labels)
    return X, y


def main():
    X, y = build_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=7, stratify=y
    )

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"Image model — train accuracy: {train_acc:.3f}, test accuracy: {test_acc:.3f}")

    bundle = {
        "model": model,
        "feature_names": ["green_ratio", "yellow_brown_ratio", "dark_spot_ratio", "brightness"],
        "classes": sorted(set(y)),
    }
    joblib.dump(bundle, OUT_PATH)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
