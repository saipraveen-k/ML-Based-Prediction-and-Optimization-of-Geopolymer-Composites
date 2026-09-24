"""
Test Suite: Metric Normalization Utilities
===========================================
"""

import sys
import os
import unittest
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ui.formatting import (
    normalize_metric_name, normalize_metrics_dataframe,
    get_metric_display_name, METRIC_DISPLAY_NAMES
)


class TestMetricNormalization(unittest.TestCase):
    
    def test_normalize_metric_name_variations(self):
        """Test mapped canonical metric strings."""
        r2_variations = ['R2', 'r2', 'R²', 'R2_Score', 'R² Score', 'R2 Score', 'r2_scores', 'Mean R2', 'Mean_R2']
        for v in r2_variations:
            self.assertEqual(normalize_metric_name(v), 'R2', f"Failed for variant: {v}")

        self.assertEqual(normalize_metric_name('Std R2'), 'Std R2')
        self.assertEqual(normalize_metric_name('Std MAE'), 'Std MAE')

        mae_variations = ['MAE', 'mae', 'Mean Absolute Error', 'Mean MAE']
        for v in mae_variations:
            self.assertEqual(normalize_metric_name(v), 'MAE', f"Failed for variant: {v}")

        rmse_variations = ['RMSE', 'rmse', 'Root Mean Squared Error', 'Mean RMSE']
        for v in rmse_variations:
            self.assertEqual(normalize_metric_name(v), 'RMSE', f"Failed for variant: {v}")

        mse_variations = ['MSE', 'mse', 'Mean Squared Error', 'Mean MSE']
        for v in mse_variations:
            self.assertEqual(normalize_metric_name(v), 'MSE', f"Failed for variant: {v}")

        model_variations = ['Model', 'model', 'MODEL', 'Model_Name']
        for v in model_variations:
            self.assertEqual(normalize_metric_name(v), 'Model', f"Failed for variant: {v}")

    def test_normalize_metrics_dataframe_columns(self):
        """Test column renaming on DataFrames and duplicate prevention."""
        raw_data = {
            'Algorithm': ['XGBoost', 'RF'],
            'R² Score': [0.98, 0.96],
            'MAE': [2.3, 3.0],
            'RMSE': [3.4, 4.3]
        }
        df = pd.DataFrame(raw_data)
        norm_df = normalize_metrics_dataframe(df)
        
        self.assertIn('Model', norm_df.columns)
        self.assertIn('R2', norm_df.columns)
        self.assertIn('MAE', norm_df.columns)
        self.assertIn('RMSE', norm_df.columns)

    def test_prevent_duplicate_column_names(self):
        """Test that DataFrames with Mean and Std metrics do not produce duplicate column names."""
        cv_data = {
            'Model': ['XGBoost', 'RF'],
            'Mean R2': [0.98, 0.96],
            'Std R2': [0.007, 0.009],
            'Mean MAE': [2.3, 3.0],
            'Std MAE': [0.2, 0.3]
        }
        cv_df = pd.DataFrame(cv_data)
        norm_cv_df = normalize_metrics_dataframe(cv_df)
        
        # Verify columns are unique
        self.assertEqual(len(norm_cv_df.columns), len(set(norm_cv_df.columns)))
        self.assertIn('R2', norm_cv_df.columns)
        self.assertIn('Std R2', norm_cv_df.columns)
        self.assertIn('MAE', norm_cv_df.columns)
        self.assertIn('Std MAE', norm_cv_df.columns)

    def test_display_names(self):
        """Test display name mapping."""
        self.assertEqual(get_metric_display_name('R2'), 'R²')
        self.assertEqual(get_metric_display_name('MAE'), 'MAE')
        self.assertEqual(get_metric_display_name('RMSE'), 'RMSE')
        self.assertEqual(get_metric_display_name('Std R2'), 'Std R²')


if __name__ == '__main__':
    unittest.main()
