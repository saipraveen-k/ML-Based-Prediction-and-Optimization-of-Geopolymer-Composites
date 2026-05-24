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
