"""Train both models and save artifacts using sklearn GradientBoosting."""

import os
import json
import pickle
import numpy as np

from seismic_interpretation.data_generator import generate_synthetic_data
from seismic_interpretation.utils.preprocessor import prepare_data
from seismic_interpretation.models.horizon_tracker import HorizonTracker
from seismic_interpretation.models.facies_classifier import FaciesClassifier

OUTPUT_DIR = os.path.join("outputs", "models")


def main():
    print("=" * 60)
    print("  Seismic Interpretation - Model Training (sklearn GradientBoosting)")
    print("=" * 60)

    print("\n[1/5] Generating synthetic seismic data...")
    df = generate_synthetic_data(n_samples=2000, random_state=42)
    print(f"  Generated {len(df)} samples")
    print(f"  Features: {list(df.columns)}")

    print("\n[2/5] Preprocessing data...")
    data = prepare_data(df, test_size=0.2, random_state=42)
    print(f"  Train: {len(data['X_train'])} | Test: {len(data['X_test'])}")

    print("\n[3/5] Training HorizonTracker (GradientBoosting)...")
    tracker = HorizonTracker(in_features=6, lr=0.001, epochs=100, batch_size=64)
    tracker.train(data["X_train"], data["y_reg_train"])
    reg_metrics = tracker.evaluate(data["X_test"], data["y_reg_test"])
    print(f"  MSE:  {reg_metrics['mse']:.6f}")
    print(f"  RMSE: {reg_metrics['rmse']:.6f}")
    print(f"  MAE:  {reg_metrics['mae']:.6f}")
    print(f"  R2:   {reg_metrics['r2']:.4f}")

    print("\n[4/5] Training FaciesClassifier (GradientBoosting)...")
    classifier = FaciesClassifier(in_features=6, num_classes=5, lr=0.001, epochs=100, batch_size=64)
    classifier.train(data["X_train"], data["y_clf_train"])
    clf_metrics = classifier.evaluate(data["X_test"], data["y_clf_test"])
    print(f"  Accuracy: {clf_metrics['accuracy']:.4f}")
    print(f"  F1 Macro: {clf_metrics['f1_macro']:.4f}")

    print("\n[5/5] Saving models and metadata...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tracker.save(os.path.join(OUTPUT_DIR, "horizon_tracker.joblib"))
    classifier.save(os.path.join(OUTPUT_DIR, "facies_classifier.joblib"))

    with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(data["scaler"], f)

    metadata = {
        "horizon_tracker": {"model": "GradientBoosting (HorizonTracker)", "metrics": reg_metrics},
        "facies_classifier": {"model": "GradientBoosting (FaciesClassifier)", "metrics": clf_metrics},
        "training_samples": len(data["X_train"]),
        "test_samples": len(data["X_test"]),
        "features": data["feature_cols"],
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"\n  Models saved to: {OUTPUT_DIR}/")
    print(f"  - horizon_tracker.joblib")
    print(f"  - facies_classifier.joblib")
    print(f"  - scaler.pkl")
    print(f"  - metadata.json")
    print()


if __name__ == "__main__":
    main()
