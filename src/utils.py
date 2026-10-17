"""
utils.py — Helper Functions
=============================
Utility functions for visualization data preparation and formatting.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)


def get_classification_metrics(y_true, y_pred, categories: list[str]) -> pd.DataFrame:
    """
    Generate a clean classification report as a DataFrame.

    Returns a DataFrame with Precision, Recall, F1-Score, and Support
    for each category, plus macro/weighted averages.
    """
    report = classification_report(
        y_true, y_pred, target_names=categories, output_dict=True, zero_division=0
    )
    df = pd.DataFrame(report).T
    df = df.rename(
        columns={
            "precision": "Precision",
            "recall": "Recall",
            "f1-score": "F1-Score",
            "support": "Support",
        }
    )
    # Format percentages nicely
    for col in ["Precision", "Recall", "F1-Score"]:
        df[col] = df[col].apply(lambda x: round(x, 3))
    df["Support"] = df["Support"].astype(int)
    return df


def get_confusion_matrix(y_true, y_pred, categories: list[str]) -> pd.DataFrame:
    """Return confusion matrix as a labeled DataFrame."""
    cm = confusion_matrix(y_true, y_pred, labels=categories)
    return pd.DataFrame(cm, index=categories, columns=categories)


def get_accuracy(y_true, y_pred) -> float:
    """Return accuracy score as a percentage."""
    return round(accuracy_score(y_true, y_pred) * 100, 2)


def get_top_features(vectorizer, model, categories: list[str], n: int = 15) -> dict:
    """
    Extract the top-N most important words for each category.

    Works for linear models that have a .coef_ attribute
    (Logistic Regression, Linear SVM) and Naive Bayes (.feature_log_prob_).
    Also handles CalibratedClassifierCV wrappers.
    """
    from sklearn.calibration import CalibratedClassifierCV

    feature_names = np.array(vectorizer.get_feature_names_out())
    top_features = {}

    coef_matrix = None
    if isinstance(model, CalibratedClassifierCV) and hasattr(model, "calibrated_classifiers_"):
        # Average the weights across cross-validated calibrated estimators
        fold_coefs = [
            cc.estimator.coef_ for cc in model.calibrated_classifiers_
            if hasattr(cc, "estimator") and hasattr(cc.estimator, "coef_")
        ]
        if fold_coefs:
            coef_matrix = np.mean(fold_coefs, axis=0)
    elif hasattr(model, "coef_"):
        coef_matrix = model.coef_

    if coef_matrix is not None:
        for i, category in enumerate(categories):
            top_indices = coef_matrix[i].argsort()[-n:][::-1]
            top_features[category] = list(feature_names[top_indices])
    elif hasattr(model, "feature_log_prob_"):
        # Naive Bayes
        for i, category in enumerate(categories):
            top_indices = model.feature_log_prob_[i].argsort()[-n:][::-1]
            top_features[category] = list(feature_names[top_indices])

    return top_features
