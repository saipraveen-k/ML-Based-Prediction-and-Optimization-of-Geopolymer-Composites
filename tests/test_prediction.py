"""
Test Suite: Prediction Pipeline & Model Inference
=================================================
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import load_cached_models
from src.ui.formatting import FEATURE_DEFAULTS
from src.ui.pages import FEATURE_NAMES


class TestPredictionPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.models, cls.scaler = load_cached_models()

    def test_model_loading(self):
        """Verify that at least one model is loaded."""
        self.assertTrue(len(self.models) > 0, "No trained models found")
        self.assertIn('XGBoost', self.models, "XGBoost model missing")

    def test_prediction_vector(self):
        """Test feature input vector formatting, scaling, and inference."""
        sample_input = {feat: FEATURE_DEFAULTS.get(feat, 0.0) for feat in FEATURE_NAMES}
        input_df = pd.DataFrame([sample_input])[FEATURE_NAMES]

        # Verify exact 21 features in correct order
        self.assertEqual(list(input_df.columns), FEATURE_NAMES)

        if self.scaler is not None:
            input_scaled = self.scaler.transform(input_df)
        else:
            input_scaled = input_df.values

        for m_name, m_obj in self.models.items():
            pred = m_obj.predict(input_scaled)[0]
            self.assertIsInstance(float(pred), float)
            self.assertGreater(pred, 0.0, f"Predicted strength for {m_name} should be positive")
            self.assertLess(pred, 200.0, f"Predicted strength for {m_name} should be realistic")


if __name__ == '__main__':
    unittest.main()
