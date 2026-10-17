"""
model.py — Model Training, Evaluation & Persistence
=====================================================
Trains three classifiers and provides functions to save/load them.

Models:
    1. Multinomial Naive Bayes  — fast, probabilistic baseline
    2. Logistic Regression      — strong linear classifier
    3. Linear SVM (SGD)         — scalable, excellent for text
"""

import joblib
from pathlib import Path
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from src.preprocessor import build_vectorizer

# Directory to save trained models
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def get_models() -> dict:
    """
    Return a dictionary of named model instances.

    Each model is wrapped in a Pipeline with a TF-IDF vectorizer
    so prediction is end-to-end (raw text → category).
    """
    return {
        "Naive Bayes": Pipeline([
            ("tfidf", build_vectorizer()),
            ("clf", MultinomialNB(alpha=0.1)),
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", build_vectorizer()),
            ("clf", LogisticRegression(
                max_iter=1000,
                C=5.0,
                solver="lbfgs",
                random_state=42,
            )),
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", build_vectorizer()),
            ("clf", CalibratedClassifierCV(
                SGDClassifier(
                    loss="hinge",
                    alpha=1e-4,
                    max_iter=1000,
                    random_state=42,
                ),
                cv=3,
            )),
        ]),
    }


def train_model(pipeline: Pipeline, X_train, y_train) -> Pipeline:
    """Fit a pipeline on training data and return it."""
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(pipeline: Pipeline, X_test, y_test) -> float:
    """Return the accuracy of a fitted pipeline on test data."""
    y_pred = pipeline.predict(X_test)
    return accuracy_score(y_test, y_pred)


def save_model(pipeline: Pipeline, name: str) -> Path:
    """Save a trained pipeline to disk as a .pkl file."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    filename = name.lower().replace(" ", "_") + ".pkl"
    path = MODELS_DIR / filename
    joblib.dump(pipeline, path)
    return path


def load_model(name: str) -> Pipeline:
    """Load a saved pipeline from disk."""
    filename = name.lower().replace(" ", "_") + ".pkl"
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Model '{name}' not found at {path}. Run train_model.py first."
        )
    return joblib.load(path)


def load_all_models() -> dict[str, Pipeline]:
    """Load all three trained models. Returns {name: pipeline}."""
    names = ["Naive Bayes", "Logistic Regression", "Linear SVM"]
    return {name: load_model(name) for name in names}
