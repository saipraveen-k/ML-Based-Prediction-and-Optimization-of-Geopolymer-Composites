"""
Test Suite: Gemini AI Service Integration & Fallback Behavior
==============================================================
"""

import sys
import os
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.services.gemini_service import get_gemini_api_key, generate_engineering_explanation


class TestGeminiService(unittest.TestCase):

    def test_missing_api_key_handling(self):
        """Verify that missing API key returns a structured failure without throwing exceptions."""
        # Temporarily clear env key for test
        old_key = os.environ.get("GEMINI_API_KEY")
        if "GEMINI_API_KEY" in os.environ:
            del os.environ["GEMINI_API_KEY"]

        res = generate_engineering_explanation(
            predicted_strength=55.4,
            model_name="XGBoost",
            top_pos_contributors=[{"Feature": "Silica Fume", "Impact": 3.2}],
            top_neg_contributors=[{"Feature": "Water/Binder Ratio", "Impact": -2.1}]
        )

        self.assertFalse(res["success"])
        self.assertIn("AI explanation is temporarily unavailable", res["explanation"])

        # Restore env key if existed
        if old_key is not None:
            os.environ["GEMINI_API_KEY"] = old_key


if __name__ == '__main__':
    unittest.main()
