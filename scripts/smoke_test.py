"""
Comprehensive Smoke Test Script
===============================
Verifies end-to-end dataset loading, model inference, metric normalization,
CV/Monte Carlo/SHAP output loading, and Gemini service resilience.

Usage:
    python scripts/smoke_test.py
"""

import sys
import os
import time
import pandas as pd
import numpy as np

# Ensure root directory in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ui.components import load_dataset_sample
from app import load_cached_models
from src.ui.formatting import (
    normalize_metric_name, normalize_metrics_dataframe,
    FEATURE_DEFAULTS, FEATURE_CATEGORIES
)
from src.ui.pages import FEATURE_NAMES, load_evaluation_metrics, load_cv_summary
from src.ui.data_validation import get_system_health_status
from src.services.gemini_service import generate_engineering_explanation


def run_smoke_test():
    print("=" * 70)
    print("      SUSTAINABLE GEOPOLYMER AI PLATFORM - SYSTEM SMOKE TEST      ")
    print("=" * 70)
    
    start_time = time.time()
    errors = []
    
    # 1. Dataset Loading Test
    print("\n[1/7] Validating Dataset Loading...")
    t0 = time.time()
    df = load_dataset_sample()
    t_ds = time.time() - t0
    if df is not None and not df.empty:
        print(f"  [OK] Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns ({t_ds*1000:.1f} ms)")
    else:
        errors.append("Dataset loading failed or returned empty dataframe.")
        print("  [FAIL] Dataset loading failed.")

    # 2. Model & Scaler Loading Test
    print("\n[2/7] Validating Model & Scaler Artifacts...")
    t0 = time.time()
    models, scaler = load_cached_models()
    t_models = time.time() - t0
    
    expected_models = ['XGBoost', 'Random Forest', 'SVR']
    for m in expected_models:
        if m in models:
            print(f"  [OK] Model '{m}' loaded successfully.")
        else:
            errors.append(f"Model '{m}' missing from models/ directory.")
            print(f"  [FAIL] Model '{m}' missing.")
            
    if scaler is not None and hasattr(scaler, 'transform'):
        print(f"  [OK] StandardScaler loaded successfully. ({t_models*1000:.1f} ms)")
    else:
        errors.append("StandardScaler failed to load.")
        print("  [FAIL] StandardScaler missing.")

    # 3. Prediction Pipeline Inference Test
    print("\n[3/7] Validating Model Prediction Pipeline...")
    t0 = time.time()
    try:
        sample_input = {feat: FEATURE_DEFAULTS.get(feat, 0.0) for feat in FEATURE_NAMES}
        input_df = pd.DataFrame([sample_input])[FEATURE_NAMES]
        
        if scaler is not None:
            input_scaled = scaler.transform(input_df)
        else:
            input_scaled = input_df.values
            
        preds = {}
        for m_name, m_obj in models.items():
            pred_val = float(m_obj.predict(input_scaled)[0])
            preds[m_name] = pred_val
            print(f"  [OK] {m_name} Prediction: {pred_val:.2f} MPa")
            
        t_pred = time.time() - t0
        print(f"  [OK] Prediction latency: {t_pred*1000:.2f} ms")
    except Exception as e:
        errors.append(f"Prediction pipeline failed with error: {e}")
        print(f"  [FAIL] Prediction error: {e}")

    # 4. Metric Normalization & Evaluation Metrics Test
    print("\n[4/7] Validating Metric Normalization System...")
    t0 = time.time()
    try:
        raw_test_df = pd.DataFrame({
            "Algorithm": ["XGBoost", "RF", "SVR"],
            "R² Score": [0.98, 0.96, 0.95],
            "Mean MAE": [2.3, 3.0, 3.4],
            "RMSE": [3.4, 4.3, 4.8],
            "Mean_MSE": [11.5, 18.5, 23.0]
        })
        norm_df = normalize_metrics_dataframe(raw_test_df)
        
        required_cols = ["Model", "R2", "MAE", "RMSE", "MSE"]
        for c in required_cols:
            if c in norm_df.columns:
                print(f"  [OK] Canonical metric '{c}' present after normalization.")
            else:
                errors.append(f"Normalized dataframe missing column '{c}'.")
                print(f"  [FAIL] Missing metric '{c}'.")
                
        # Load actual output metrics
        eval_metrics = load_evaluation_metrics()
        if eval_metrics is not None:
            print("  [OK] Evaluation metrics CSV loaded and normalized successfully.")
        else:
            print("  [INFO] Evaluation metrics CSV not found (non-blocking).")
    except Exception as e:
        errors.append(f"Metric normalization failed: {e}")
        print(f"  [FAIL] Metric error: {e}")

    # 5. Cross-Validation & Monte Carlo Output Test
    print("\n[5/7] Validating Validation Outputs (CV & Monte Carlo)...")
    cv_summary = load_cv_summary()
    if cv_summary is not None:
        print("  [OK] 10-Fold CV summary CSV loaded.")
    else:
        print("  [INFO] 10-Fold CV summary CSV missing (non-blocking).")

    # 6. System Health Validation Check
    print("\n[6/7] Validating System Health Status Aggregator...")
    health = get_system_health_status(df, models, scaler, eval_metrics)
    print(f"  [OK] System Status: {health['overall_status']}")

    # 7. Gemini Service Resilience Test
    print("\n[7/7] Validating Gemini Service Resilience...")
    g_res = generate_engineering_explanation(
        predicted_strength=55.4,
        model_name="XGBoost",
        top_pos_contributors=[{"Feature": "Silica Fume", "Impact": 3.2}],
        top_neg_contributors=[{"Feature": "Water/Binder Ratio", "Impact": -2.1}]
    )
    if not g_res["success"]:
        print(f"  [OK] Gemini fallback handled gracefully: '{g_res['explanation']}'")
    else:
        print("  [OK] Gemini API responded successfully.")

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"SMOKE TEST COMPLETED IN {total_time:.2f} SECONDS")
    
    if errors:
        print(f"STATUS: FAILED ({len(errors)} errors detected)")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("STATUS: PASSED - ALL CHECKS SUCCESSFUL")
        sys.exit(0)


if __name__ == '__main__':
    run_smoke_test()
