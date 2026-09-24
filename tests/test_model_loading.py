"""
Test Suite: Model & Scaler Artifact Loading
============================================
"""

import sys
import os
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import load_cached_models


class TestModelLoading(unittest.TestCase):

    def test_load_models_and_scaler(self):
        models, scaler = load_cached_models()
        self.assertIsInstance(models, dict)
        self.assertIn('XGBoost', models)
        self.assertIn('Random Forest', models)
        self.assertIn('SVR', models)
        self.assertIsNotNone(scaler)
        self.assertTrue(hasattr(scaler, 'transform'))


if __name__ == '__main__':
    unittest.main()
