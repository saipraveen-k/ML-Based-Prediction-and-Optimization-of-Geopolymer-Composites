"""
Data & System Validation Module
================================
Provides lightweight validation functions for datasets, models, scalers,
metrics, and precomputed analysis outputs. Prevents application crashes.
"""

import os
import glob
import pandas as pd
import numpy as np


def validate_dataset(df):
    """Validate dataset existence and structural integrity."""
    status = {"valid": False, "missing": [], "warnings": [], "row_count": 0, "col_count": 0}
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        status["missing"].append("dataset.csv file or content")
        return status
        
    status["row_count"] = len(df)
    status["col_count"] = len(df.columns)
    
    if len(df) < 10:
        status["warnings"].append(f"Small dataset size: {len(df)} rows")
        
    status["valid"] = True
    return status


def validate_models(models):
    """Validate loaded trained ML models."""
    status = {"valid": False, "loaded": [], "missing": [], "warnings": []}
    expected_models = ['XGBoost', 'Random Forest', 'SVR']
    
    if not isinstance(models, dict) or not models:
        status["missing"] = expected_models
        return status
        
    for m in expected_models:
        if m in models and models[m] is not None:
            status["loaded"].append(m)
        else:
            status["missing"].append(m)
            
    if status["loaded"]:
        status["valid"] = True
        
    if status["missing"]:
        status["warnings"].append(f"Missing models: {', '.join(status['missing'])}")
        
    return status


def validate_scaler(scaler):
    """Validate fitted StandardScaler."""
    status = {"valid": False, "warnings": []}
    if scaler is not None and hasattr(scaler, "transform"):
        status["valid"] = True
    else:
        status["warnings"].append("StandardScaler not loaded. Unscaled inputs will be used.")
    return status


def validate_metrics(metrics_df):
    """Validate model evaluation metrics DataFrame."""
    status = {"valid": False, "available_metrics": [], "missing_metrics": [], "warnings": []}
    if metrics_df is None or not isinstance(metrics_df, pd.DataFrame) or metrics_df.empty:
        status["warnings"].append("Metrics summary file unavailable.")
        return status
        
    for req in ["R2", "MAE", "RMSE", "MSE"]:
        if req in metrics_df.columns:
            status["available_metrics"].append(req)
        else:
            status["missing_metrics"].append(req)
            
    if status["available_metrics"]:
        status["valid"] = True
    return status


def validate_shap_outputs():
    """Validate precomputed SHAP analysis artifacts."""
    status = {"valid": False, "existing_files": [], "missing_files": []}
    expected = [
        'outputs/shap/shap_beeswarm.png',
        'outputs/shap/shap_summary.png',
        'outputs/shap/shap_bar.png',
        'outputs/shap/Explainable_AI_Analysis.md'
    ]
    
    for f in expected:
        if os.path.exists(f):
            status["existing_files"].append(f)
        else:
            status["missing_files"].append(f)
            
    if status["existing_files"]:
        status["valid"] = True
    return status


def validate_monte_carlo_outputs():
    """Validate precomputed Monte Carlo simulation artifacts."""
    status = {"valid": False, "reports": [], "graphs": []}
    status["reports"] = glob.glob('outputs/reports/monte_carlo_report_*.txt')
    status["graphs"] = glob.glob('outputs/graphs/monte_carlo_*.png')
    
    if status["reports"] or status["graphs"]:
        status["valid"] = True
    return status


def validate_cv_outputs():
    """Validate 10-Fold Cross-Validation artifacts."""
    status = {"valid": False, "files": []}
    for f in ['outputs/cross_validation/cv_summary.csv', 'outputs/cross_validation/validation_comparison.csv']:
        if os.path.exists(f):
            status["files"].append(f)
            
    if status["files"]:
        status["valid"] = True
    return status


def get_system_health_status(df, models, scaler, metrics_df):
    """Aggregate lightweight validation statuses for startup check."""
    d_val = validate_dataset(df)
    m_val = validate_models(models)
    s_val = validate_scaler(scaler)
    met_val = validate_metrics(metrics_df)
    shap_val = validate_shap_outputs()
    mc_val = validate_monte_carlo_outputs()
    cv_val = validate_cv_outputs()
    
    is_fully_operational = (
        d_val["valid"] and m_val["valid"] and not m_val["missing"] and
        s_val["valid"] and met_val["valid"] and shap_val["valid"]
    )
    
    overall_status = "System Operational" if is_fully_operational else "Partial Analysis Available"
    
    return {
        "overall_status": overall_status,
        "dataset": d_val,
        "models": m_val,
        "scaler": s_val,
        "metrics": met_val,
        "shap": shap_val,
        "monte_carlo": mc_val,
        "cv": cv_val
    }
