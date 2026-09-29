"""
Formatting & Metric Normalization Utilities for Streamlit UI
============================================================
Provides centralized metric normalization, feature categories, 
units, and UI display labels to guarantee error-free data flow.
"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import format_feature_name, format_metric_name

# Internal Canonical Metric Names: "Model", "R2", "MAE", "RMSE", "MSE"
METRIC_DISPLAY_NAMES = {
    "R2": "R²",
    "MAE": "MAE",
    "RMSE": "RMSE",
    "MSE": "MSE",
    "Std R2": "Std R²",
    "Std MAE": "Std MAE",
    "Std RMSE": "Std RMSE",
    "Std MSE": "Std MSE"
}

FEATURE_CATEGORIES = {
    "01 Binder System": [
        'RHA_kg_m3',
        'POFA_kg_m3',
        'GGBS_kg_m3',
        'Silica_Fume_kg_m3',
        'Fly_Ash_kg_m3',
        'Metakaolin_kg_m3',
        'Cement_kg_m3'
    ],
    "02 Activator": [
        'NaOH_Content_kg_m3',
        'Na2SiO3_Content_kg_m3',
        'KOH_Content_kg_m3',
        'Activator_Molarity_M',
        'Extra_Water_kg_m3'
    ],
    "03 Mix Design": [
        'Water_kg_m3',
        'Water_Binder_Ratio',
        'Fine_Sand_kg_m3',
        'Superplasticizer_kg_m3'
    ],
    "04 Reinforcement": [
        'Polypropylene_Fiber_Content_%',
        'PP_Fiber_kg_m3',
        'Fiber_Length_mm'
    ],
    "05 Curing": [
        'Curing_Temperature_C',
        'Curing_Duration_days'
    ]
}

FEATURE_DEFAULTS = {
    'Cement_kg_m3': 780.0,
    'Fly_Ash_kg_m3': 120.0,
    'Silica_Fume_kg_m3': 50.0,
    'Metakaolin_kg_m3': 0.0,
    'GGBS_kg_m3': 0.0,
    'RHA_kg_m3': 0.0,
    'POFA_kg_m3': 0.0,
    'Fine_Sand_kg_m3': 0.0,
    'Water_kg_m3': 0.0,
    'Extra_Water_kg_m3': 0.0,
    'Water_Binder_Ratio': 0.22,
    'Na2SiO3_Content_kg_m3': 120.0,
    'NaOH_Content_kg_m3': 40.0,
    'KOH_Content_kg_m3': 0.0,
    'Activator_Molarity_M': 12.0,
    'Superplasticizer_kg_m3': 15.0,
    'Polypropylene_Fiber_Content_%': 0.5,
    'PP_Fiber_kg_m3': 4.5,
    'Fiber_Length_mm': 12.0,
    'Curing_Temperature_C': 60.0,
    'Curing_Duration_days': 28.0
}

FEATURE_RANGES = {
    'Cement_kg_m3': (0.0, 2000.0, 10.0),
    'Fly_Ash_kg_m3': (0.0, 1500.0, 10.0),
    'Silica_Fume_kg_m3': (0.0, 500.0, 5.0),
    'Metakaolin_kg_m3': (0.0, 1500.0, 5.0),
    'GGBS_kg_m3': (0.0, 1500.0, 10.0),
    'RHA_kg_m3': (0.0, 500.0, 5.0),
    'POFA_kg_m3': (0.0, 500.0, 5.0),
    'Fine_Sand_kg_m3': (0.0, 2500.0, 10.0),
    'Water_kg_m3': (0.0, 800.0, 5.0),
    'Extra_Water_kg_m3': (0.0, 300.0, 5.0),
    'Water_Binder_Ratio': (0.0, 2.0, 0.01),
    'Na2SiO3_Content_kg_m3': (0.0, 500.0, 5.0),
    'NaOH_Content_kg_m3': (0.0, 300.0, 5.0),
    'KOH_Content_kg_m3': (0.0, 300.0, 1.0),
    'Activator_Molarity_M': (0.0, 30.0, 0.5),
    'Superplasticizer_kg_m3': (0.0, 100.0, 0.5),
    'Polypropylene_Fiber_Content_%': (0.0, 10.0, 0.05),
    'PP_Fiber_kg_m3': (0.0, 50.0, 0.5),
    'Fiber_Length_mm': (0.0, 100.0, 1.0),
    'Curing_Temperature_C': (0.0, 150.0, 1.0),
    'Curing_Duration_days': (0.0, 365.0, 1.0)
}


def deduplicate_columns(df):
    """Ensure DataFrame column names are strictly unique to prevent PyArrow Table.from_pandas duplicate column ValueError."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
        
    seen = {}
    new_cols = []
    for col in df.columns:
        col_str = str(col)
        if col_str in seen:
            seen[col_str] += 1
            new_cols.append(f"{col_str} ({seen[col_str]})")
        else:
            seen[col_str] = 0
            new_cols.append(col_str)
            
    df_copy = df.copy()
    df_copy.columns = new_cols
    return df_copy


def normalize_metric_name(name):
    """
    Map any raw metric column or string to canonical internal key (R2, MAE, RMSE, MSE, Model).
    Preserves Std qualifiers when present to differentiate standard deviation from mean metrics.
    """
    if not isinstance(name, str):
        return name
        
    cleaned = name.strip()
    lower = cleaned.lower()
    
    # Check Model
    if lower in ['model', 'models', 'model_name', 'algorithm']:
        return 'Model'
        
    is_std = 'std' in lower or 'sd' in lower
    
    # Base metric recognition
    base_metric = None
    if any(p in lower for p in ['r2', 'r²', 'r_2', 'r-2', 'rsquared', 'r_squared']):
        base_metric = 'R2'
    elif 'mae' in lower or 'mean absolute error' in lower:
        base_metric = 'MAE'
    elif 'rmse' in lower or 'root mean' in lower:
        base_metric = 'RMSE'
    elif 'mse' in lower or 'mean squared error' in lower:
        base_metric = 'MSE'
        
    if base_metric:
        if is_std:
            return f"Std {base_metric}"
        return base_metric
        
    return name


def normalize_metrics_dataframe(df):
    """
    Safely normalize metric column names or metric values in a DataFrame.
    Guarantees unique column names to prevent PyArrow ValueError on duplicates.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
        
    normalized_df = df.copy()
    
    # 1. Rename columns while guaranteeing unique column names
    new_columns = {}
    seen = set()
    for col in normalized_df.columns:
        canonical = normalize_metric_name(str(col))
        
        # If canonical causes duplicate, fallback to original column name
        if canonical in seen:
            canonical = str(col)
            
        seen.add(canonical)
        new_columns[col] = canonical
        
    normalized_df = normalized_df.rename(columns=new_columns)
    
    # 2. If 'Metric' is a column (long format), normalize string values
    if 'Metric' in normalized_df.columns:
        normalized_df['Metric'] = normalized_df['Metric'].apply(normalize_metric_name)
        
    return deduplicate_columns(normalized_df)


def get_metric_display_name(canonical_key):
    """Return publication UI display label for a canonical metric key."""
    if not isinstance(canonical_key, str):
        return canonical_key
    if canonical_key in METRIC_DISPLAY_NAMES:
        return METRIC_DISPLAY_NAMES[canonical_key]
    return canonical_key.replace("R2", "R²")


def get_clean_label(feature_name):
    """Return publication-ready formatted label for any feature name."""
    return format_feature_name(feature_name)
