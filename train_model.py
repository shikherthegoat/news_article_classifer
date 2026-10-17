"""
train_model.py — Main Training Script
=======================================
Run this script to train all 3 models, evaluate them, and save to disk.

Usage:
    python train_model.py
"""

import time
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.data_loader import load_data, CATEGORIES
from src.model import get_models, train_model, evaluate_model, save_model
from src.utils import get_classification_metrics, get_confusion_matrix, get_top_features


def main():
    print("=" * 60)
    print("📰  News Article Classifier — Training Pipeline")
    print("=" * 60)

    # ── Step 1: Load Data ──────────────────────────────────────
    print("\n📦 Loading 20 Newsgroups dataset...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"   Training samples : {len(X_train):,}")
    print(f"   Testing samples  : {len(X_test):,}")
    print(f"   Categories       : {CATEGORIES}")

    # ── Step 2: Train & Evaluate Models ────────────────────────
    models = get_models()
    results = {}

    for name, pipeline in models.items():
        print(f"\n🔧 Training {name}...")
        start = time.time()

        trained = train_model(pipeline, X_train, y_train)
        accuracy = evaluate_model(trained, X_test, y_test)
        elapsed = time.time() - start

        print(f"   ✅ Accuracy : {accuracy * 100:.2f}%")
        print(f"   ⏱️  Time    : {elapsed:.1f}s")

        # Save model
        path = save_model(trained, name)
        print(f"   💾 Saved to : {path}")

        # Collect predictions for metrics
        y_pred = trained.predict(X_test)
        metrics_df = get_classification_metrics(y_test, y_pred, CATEGORIES)
        cm_df = get_confusion_matrix(y_test, y_pred, CATEGORIES)

        results[name] = {
            "accuracy": round(accuracy * 100, 2),
            "time": round(elapsed, 1),
            "metrics": metrics_df.to_dict(),
            "confusion_matrix": cm_df.values.tolist(),
        }

        # Print per-category metrics
        print(f"\n   📊 Classification Report for {name}:")
        print(metrics_df.to_string())

    # ── Step 3: Save evaluation results as JSON ────────────────
    results_path = Path("models") / "evaluation_results.json"
    # Convert for JSON serialization
    json_results = {}
    for name, data in results.items():
        json_results[name] = {
            "accuracy": data["accuracy"],
            "training_time_seconds": data["time"],
        }
    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\n📄 Evaluation results saved to {results_path}")

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("🏆  Model Comparison Summary")
    print("=" * 60)
    for name, data in results.items():
        print(f"   {name:25s} → {data['accuracy']:.2f}% accuracy ({data['time']:.1f}s)")
    
    best = max(results, key=lambda k: results[k]["accuracy"])
    print(f"\n   🥇 Best Model: {best} ({results[best]['accuracy']:.2f}%)")
    print("\n✅ All models trained and saved! Run `streamlit run app.py` to launch the app.\n")


if __name__ == "__main__":
    main()
