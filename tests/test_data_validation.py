"""
Test Suite: Data & Output Validation Module
============================================
"""

import sys
import os
import unittest
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ui.data_validation import (
    validate_dataset, validate_models, validate_scaler,
    validate_metrics, validate_shap_outputs, validate_monte_carlo_outputs,
    validate_cv_outputs, get_system_health_status
)


class TestDataValidation(unittest.TestCase):

    def test_validate_dataset(self):
        sample_df = pd.DataFrame({"A": range(20), "Compressive_Strength_MPa": range(20)})
        res = validate_dataset(sample_df)
        self.assertTrue(res["valid"])
        self.assertEqual(res["row_count"], 20)

        empty_res = validate_dataset(None)
        self.assertFalse(empty_res["valid"])

    def test_validate_models(self):
        mock_models = {"XGBoost": object(), "Random Forest": object(), "SVR": object()}
        res = validate_models(mock_models)
        self.assertTrue(res["valid"])
        self.assertEqual(len(res["loaded"]), 3)

        partial_models = {"XGBoost": object()}
        p_res = validate_models(partial_models)
        self.assertTrue(p_res["valid"])
        self.assertIn("Random Forest", p_res["missing"])

    def test_system_health_status(self):
        sample_df = pd.DataFrame({"A": range(20)})
        mock_models = {"XGBoost": object(), "Random Forest": object(), "SVR": object()}
        scaler = type("Scaler", (), {"transform": lambda self, x: x})()
        metrics = pd.DataFrame({"Model": ["XGBoost"], "R2": [0.98], "MAE": [2.3], "RMSE": [3.4], "MSE": [11.5]})

        status = get_system_health_status(sample_df, mock_models, scaler, metrics)
        self.assertIn(status["overall_status"], ["System Operational", "Partial Analysis Available"])


if __name__ == '__main__':
    unittest.main()
