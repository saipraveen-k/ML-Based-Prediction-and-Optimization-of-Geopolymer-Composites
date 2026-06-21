"""
Utility Functions for UHPGC Compressive Strength Prediction
============================================================
This module contains common utility functions used across the project.
"""

import os
import pickle
import joblib
import pandas as pd
import numpy as np
from datetime import datetime


def create_directory(directory_path):
    """
    Create a directory if it doesn't exist.
    
    Parameters:
    -----------
    directory_path : str
        Path to the directory to create
    
    Returns:
    --------
    None
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")


def save_model(model, model_path):
    """
    Save a trained model using joblib.
    
    Parameters:
    -----------
    model : object
        Trained model object
    model_path : str
        Path to save the model
    
    Returns:
    --------
    None
    """
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")


def load_model(model_path):
    """
    Load a trained model using joblib.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model
    
    Returns:
    --------
    model : object
        Loaded model object
    """
    model = joblib.load(model_path)
    print(f"Model loaded from: {model_path}")
    return model


def save_scaler(scaler, scaler_path):
    """
    Save a fitted scaler using joblib.
    
    Parameters:
    -----------
    scaler : object
        Fitted scaler object
    scaler_path : str
        Path to save the scaler
    
    Returns:
    --------
    None
    """
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")


def load_scaler(scaler_path):
    """
    Load a fitted scaler using joblib.
    
    Parameters:
    -----------
    scaler_path : str
        Path to the saved scaler
    
    Returns:
    --------
    scaler : object
        Loaded scaler object
    """
    scaler = joblib.load(scaler_path)
    print(f"Scaler loaded from: {scaler_path}")
    return scaler


def save_dataframe(df, file_path, file_format='csv'):
    """
    Save a DataFrame to file.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to save
    file_path : str
        Path to save the file
    file_format : str
        Format to save ('csv' or 'xlsx')
    
    Returns:
    --------
    None
    """
    if file_format == 'csv':
        df.to_csv(file_path, index=False)
    elif file_format == 'xlsx':
        df.to_excel(file_path, index=False, engine='openpyxl')
    print(f"DataFrame saved to: {file_path}")


def load_dataframe(file_path):
    """
    Load a DataFrame from file.
    
    Parameters:
    -----------
    file_path : str
        Path to the file
    
    Returns:
    --------
    df : pandas.DataFrame
        Loaded DataFrame
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path, engine='openpyxl')
    else:
        raise ValueError("Unsupported file format. Use .csv or .xlsx")
    print(f"DataFrame loaded from: {file_path}")
    return df


def get_timestamp():
    """
    Get current timestamp as a string.
    
    Returns:
    --------
    timestamp : str
        Current timestamp in format: YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def print_separator(char='=', length=80):
    """
    Print a separator line.
    
    Parameters:
    -----------
    char : str
        Character to use for separator
    length : int
        Length of the separator
    
    Returns:
    --------
    None
    """
    print(char * length)


def print_section(title):
    """
    Print a section header.
    
    Parameters:
    -----------
    title : str
        Title of the section
    
    Returns:
    --------
    None
    """
    print_separator()
    print(f"{title.center(80)}")
    print_separator()


def calculate_statistics(y_true, y_pred):
    """
    Calculate basic statistics for predictions.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    
    Returns:
    --------
    stats : dict
        Dictionary containing statistics
    """
    errors = y_true - y_pred
    stats = {
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'min_error': np.min(errors),
        'max_error': np.max(errors),
        'mean_true': np.mean(y_true),
        'mean_pred': np.mean(y_pred),
        'std_true': np.std(y_true),
        'std_pred': np.std(y_pred)
    }
    return stats


def validate_input_data(input_dict, expected_features):
    """
    Validate input data for prediction.
    
    Parameters:
    -----------
    input_dict : dict
        Dictionary of input features
    expected_features : list
        List of expected feature names
    
    Returns:
    --------
    is_valid : bool
        True if input is valid
    message : str
        Validation message
    """
    missing_features = set(expected_features) - set(input_dict.keys())
    extra_features = set(input_dict.keys()) - set(expected_features)
    
    if missing_features:
        return False, f"Missing features: {missing_features}"
    
    if extra_features:
        return False, f"Unexpected features: {extra_features}"
    
    for key, value in input_dict.items():
        if not isinstance(value, (int, float)):
            return False, f"Feature '{key}' must be numeric"
    
    return True, "Input validation successful"


def format_number(value, decimals=4):
    """
    Format a number to specified decimal places.
    
    Parameters:
    -----------
    value : float
        Number to format
    decimals : int
        Number of decimal places
    
    Returns:
    --------
    formatted : str
        Formatted number string
    """
    return f"{value:.{decimals}f}"


def create_results_dict(model_name, metrics, predictions=None):
    """
    Create a results dictionary for a model.
    
    Parameters:
    -----------
    model_name : str
        Name of the model
    metrics : dict
        Dictionary of evaluation metrics
    predictions : array-like, optional
        Predictions array
    
    Returns:
    --------
    results : dict
        Results dictionary
    """
    results = {
        'model_name': model_name,
        'metrics': metrics,
        'timestamp': get_timestamp()
    }
    if predictions is not None:
        results['predictions'] = predictions
    return results


def format_feature_name(name):
    """
    Format a raw feature/variable name to a publication-quality label with proper units.
    """
    if not isinstance(name, str):
        return name
        
    # Specific known feature mapping overrides for clean titles and units
    known_mappings = {
        'Cement_kg_m3': 'Cement (kg/m³)',
        'Fly_Ash_kg_m3': 'Fly Ash (kg/m³)',
        'Silica_Fume_kg_m3': 'Silica Fume (kg/m³)',
        'Metakaolin_kg_m3': 'Metakaolin (kg/m³)',
        'GGBS_kg_m3': 'GGBS (kg/m³)',
        'RHA_kg_m3': 'Rice Husk Ash (kg/m³)',
        'POFA_kg_m3': 'POFA (kg/m³)',
        'Fine_Sand_kg_m3': 'Fine Sand (kg/m³)',
        'Water_kg_m3': 'Water (kg/m³)',
        'Extra_Water_kg_m3': 'Extra Water (kg/m³)',
        'Water_Binder_Ratio': 'Water/Binder Ratio',
        'Na2SiO3_Content_kg_m3': 'Na₂SiO₃ (kg/m³)',
        'NaOH_Content_kg_m3': 'NaOH (kg/m³)',
        'KOH_Content_kg_m3': 'KOH (kg/m³)',
        'Activator_Molarity_M': 'Activator Molarity (M)',
        'Superplasticizer_kg_m3': 'Superplasticizer (kg/m³)',
        'Polypropylene_Fiber_Content_%': 'Polypropylene Fiber Content (%)',
        'PP_Fiber_kg_m3': 'PP Fiber (kg/m³)',
        'Fiber_Length_mm': 'Fiber Length (mm)',
        'Curing_Temperature_C': 'Curing Temperature (°C)',
        'Curing_Duration_days': 'Curing Duration (days)',
        'Compressive_Strength_MPa': 'Compressive Strength (MPa)',
        'Flexural_Strength_MPa': 'Flexural Strength (MPa)',
        'Tensile_Strength_MPa': 'Tensile Strength (MPa)',
        'Elastic_Modulus_GPa': 'Elastic Modulus (GPa)',
        'Interlayer_Bond_Strength_MPa': 'Interlayer Bond Strength (MPa)'
    }
    
    if name in known_mappings:
        return known_mappings[name]
        
    formatted = name
    
    # Replace unit suffixes like kgperm3, kg_per_m3, kg_m3
    unit_map = {
        'kgperm3': ' (kg/m³)',
        'kg_per_m3': ' (kg/m³)',
        'kg_m3': ' (kg/m³)',
        'MPa': ' (MPa)',
        'GPa': ' (GPa)',
        '_C': ' (°C)',
        '_mm': ' (mm)',
        '_days': ' (days)',
        '_%': ' (%)',
        '_M': ' (M)'
    }
    
    # Handle Water_Binder_Ratio -> Water/Binder Ratio
    if 'Water_Binder' in formatted:
        formatted = formatted.replace('Water_Binder', 'Water/Binder')
    
    # Apply replacements of units
    for raw_unit, clean_unit in unit_map.items():
        if formatted.endswith(f'_{raw_unit}'):
            formatted = formatted[:-len(raw_unit)-1] + clean_unit
            break
        elif formatted.endswith(raw_unit):
            formatted = formatted[:-len(raw_unit)] + clean_unit
            break
            
    # Clean up underscores and strip double spaces
    formatted = formatted.replace('_', ' ')
    formatted = ' '.join(formatted.split())
    
    return formatted


def format_metric_name(metric):
    """
    Format a metric name to publication standard.
    """
    if not isinstance(metric, str):
        return metric
        
    mapping = {
        'r2_scores': 'R² Score',
        'r2': 'R² Score',
        'mean r2': 'Mean R² Score',
        'std r2': 'Std R² Score',
        'mae_scores': 'MAE (MPa)',
        'mae': 'MAE (MPa)',
        'mean mae': 'Mean MAE (MPa)',
        'std mae': 'Std MAE (MPa)',
        'rmse_scores': 'RMSE (MPa)',
        'rmse': 'RMSE (MPa)',
        'mean rmse': 'Mean RMSE (MPa)',
        'std rmse': 'Std RMSE (MPa)',
        'mse_scores': 'MSE (MPa²)',
        'mse': 'MSE (MPa²)',
        'mean mse': 'Mean MSE (MPa²)',
        'std mse': 'Std MSE (MPa²)'
    }
    
    return mapping.get(metric.lower(), metric)

