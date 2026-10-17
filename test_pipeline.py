"""
test_pipeline.py — Project Verification Test Suite
==================================================
Runs unit & integration checks on data loading, preprocessing,
model inference, and metrics computation.
"""

import unittest
from src.preprocessor import clean_text, build_vectorizer
from src.data_loader import CATEGORIES, get_sample_articles
from src.model import load_all_models
from src.utils import get_top_features


class TestNewsClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = load_all_models()
        cls.samples = get_sample_articles()

    def test_text_cleaning(self):
        sample = "Contact us at info@example.com! Visit http://news.org for 100% details."
        cleaned = clean_text(sample)
        self.assertNotIn("info@example.com", cleaned)
        self.assertNotIn("http://news.org", cleaned)
        self.assertNotIn("100", cleaned)
        self.assertNotIn("!", cleaned)
        self.assertEqual(cleaned, "contact us at visit for details")

    def test_models_loaded(self):
        expected_models = {"Naive Bayes", "Logistic Regression", "Linear SVM"}
        self.assertEqual(set(self.models.keys()), expected_models)

    def test_sample_predictions(self):
        for name, model in self.models.items():
            for sample_title, text in self.samples.items():
                pred = model.predict([text])[0]
                proba = model.predict_proba([text])[0]
                self.assertIn(pred, CATEGORIES, f"Prediction {pred} not in categories for model {name}")
                self.assertEqual(len(proba), len(CATEGORIES), f"Probabilities length mismatch for model {name}")
                self.assertAlmostEqual(float(sum(proba)), 1.0, places=2)

    def test_top_features_extraction(self):
        for name, model in self.models.items():
            vec = model.named_steps["tfidf"]
            clf = model.named_steps["clf"]
            feats = get_top_features(vec, clf, CATEGORIES, n=5)
            self.assertIsInstance(feats, dict)
            for cat in CATEGORIES:
                self.assertIn(cat, feats)
                self.assertEqual(len(feats[cat]), 5)


if __name__ == "__main__":
    unittest.main()
