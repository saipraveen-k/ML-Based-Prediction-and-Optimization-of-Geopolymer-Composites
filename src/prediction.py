"""
Prediction Module for UHPGC Compressive Strength Prediction
=============================================================
This module handles prediction functionality for trained models.
"""

import numpy as np
import pandas as pd
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import print_section, print_separator, load_model, load_scaler, validate_input_data


class PredictionSystem:
    """
    A class for making predictions using trained models.
    """
    
    def __init__(self, models_dir='models', scaler_path='models/scaler.pkl'):
        """
        Initialize the PredictionSystem.
        
        Parameters:
        -----------
        models_dir : str
            Directory containing trained models
        scaler_path : str
            Path to the saved scaler
        """
        self.models_dir = models_dir
        self.scaler_path = scaler_path
        self.models = {}
        self.scaler = None
        self.feature_names = None
        
    def load_models(self):
        """
        Load all trained models and scaler.
        
        Returns:
        --------
        None
        """
        print_section("LOADING MODELS")
        
        # Load scaler
        if os.path.exists(self.scaler_path):
            self.scaler = load_scaler(self.scaler_path)
        else:
            print("Warning: Scaler not found. Predictions will not be scaled.")
        
        # Load models
        model_files = {
            'Random Forest': 'random_forest.pkl',
            'SVR': 'svr.pkl',
            'XGBoost': 'xgboost.pkl'
        }
        
        for model_name, filename in model_files.items():
            model_path = os.path.join(self.models_dir, filename)
            if os.path.exists(model_path):
                self.models[model_name] = load_model(model_path)
                print(f"Loaded: {model_name}")
            else:
                print(f"Warning: {model_name} not found at {model_path}")
        
        print(f"Total models loaded: {len(self.models)}")
    
    def set_feature_names(self, feature_names):
        """
        Set the feature names for prediction.
        
        Parameters:
        -----------
        feature_names : list
            List of feature names
        
        Returns:
        --------
        None
        """
        self.feature_names = feature_names
        print(f"Feature names set: {self.feature_names}")
    
    def predict_single(self, input_dict, model_name='XGBoost'):
        """
        Make a single prediction.
        
        Parameters:
        -----------
        input_dict : dict
            Dictionary of feature values
        model_name : str
            Name of the model to use
        
        Returns:
        --------
        prediction : float
            Predicted compressive strength
        """
        print_section("MAKING SINGLE PREDICTION")
        
        # Validate input
        if self.feature_names:
            is_valid, message = validate_input_data(input_dict, self.feature_names)
            if not is_valid:
                raise ValueError(message)
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])
        
        # Ensure correct column order
        if self.feature_names:
            input_df = input_df[self.feature_names]
        
        print("Input values:")
        print_separator('-')
        for key, value in input_dict.items():
            print(f"{key}: {value}")
        
        # Scale if scaler is available
        if self.scaler:
            input_scaled = self.scaler.transform(input_df)
        else:
            input_scaled = input_df.values
        
        # Get model
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not loaded. Available models: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        print_separator('-')
        print(f"Model used: {model_name}")
        print(f"Predicted Compressive Strength: {prediction:.4f} MPa")
        
        return prediction
    
    def predict_batch(self, input_df, model_name='XGBoost'):
        """
        Make batch predictions.
        
        Parameters:
        -----------
        input_df : pandas.DataFrame
            DataFrame of input features
        model_name : str
            Name of the model to use
        
        Returns:
        --------
        predictions : numpy.ndarray
            Array of predictions
        """
        print_section("MAKING BATCH PREDICTIONS")
        
        print(f"Number of samples: {len(input_df)}")
        
        # Ensure correct column order
        if self.feature_names:
            input_df = input_df[self.feature_names]
        
        # Scale if scaler is available
        if self.scaler:
            input_scaled = self.scaler.transform(input_df)
        else:
            input_scaled = input_df.values
        
        # Get model
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not loaded. Available models: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        # Make predictions
        predictions = model.predict(input_scaled)
        
        print(f"Predictions completed using {model_name}")
        
        return predictions
    
    def predict_with_all_models(self, input_dict):
        """
        Make predictions using all available models.
        
        Parameters:
        -----------
        input_dict : dict
            Dictionary of feature values
        
        Returns:
        --------
        all_predictions : dict
            Dictionary of predictions from each model
        """
        print_section("PREDICTING WITH ALL MODELS")
        
        # Validate input
        if self.feature_names:
            is_valid, message = validate_input_data(input_dict, self.feature_names)
            if not is_valid:
                raise ValueError(message)
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])
        
        # Ensure correct column order
        if self.feature_names:
            input_df = input_df[self.feature_names]
        
        print("Input values:")
        print_separator('-')
        for key, value in input_dict.items():
            print(f"{key}: {value}")
        
        # Scale if scaler is available
        if self.scaler:
            input_scaled = self.scaler.transform(input_df)
        else:
            input_scaled = input_df.values
        
        # Make predictions with all models
        all_predictions = {}
        print_separator('-')
        for model_name, model in self.models.items():
            prediction = model.predict(input_scaled)[0]
            all_predictions[model_name] = prediction
            print(f"{model_name}: {prediction:.4f} MPa")
        
        # Calculate average prediction
        avg_prediction = np.mean(list(all_predictions.values()))
        print_separator('-')
        print(f"Average Prediction: {avg_prediction:.4f} MPa")
        
        return all_predictions
    
    def get_model_list(self):
        """
        Get list of available models.
        
        Returns:
        --------
        model_list : list
            List of available model names
        """
        return list(self.models.keys())


def interactive_prediction():
    """
    Interactive prediction function for command-line use.
    
    Returns:
    --------
    None
    """
    print_section("INTERACTIVE PREDICTION SYSTEM")
    
    # Initialize prediction system
    predictor = PredictionSystem()
    predictor.load_models()
    
    if not predictor.models:
        print("No models loaded. Please train models first.")
        return
    
    # Get feature names from user (or use default)
    print("\nAvailable models:", predictor.get_model_list())
    
    # Example feature names (should be loaded from saved metadata)
    default_features = [
        'Cement_kg_m3',
        'Fly_Ash_kg_m3',
        'Silica_Fume_kg_m3',
        'Metakaolin_kg_m3',
        'GGBS_kg_m3',
        'RHA_kg_m3',
        'POFA_kg_m3',
        'Fine_Sand_kg_m3',
        'Water_kg_m3',
        'Extra_Water_kg_m3',
        'Water_Binder_Ratio',
        'Na2SiO3_Content_kg_m3',
        'NaOH_Content_kg_m3',
        'KOH_Content_kg_m3',
        'Activator_Molarity_M',
        'Superplasticizer_kg_m3',
        'Polypropylene_Fiber_Content_%',
        'PP_Fiber_kg_m3',
        'Fiber_Length_mm',
        'Curing_Temperature_C',
        'Curing_Duration_days'
    ]
    
    print(f"\nExpected features: {default_features}")
    predictor.set_feature_names(default_features)
    
    # Get input values
    print("\nEnter feature values:")
    input_dict = {}
    for feature in default_features:
        while True:
            try:
                value = float(input(f"{feature}: "))
                input_dict[feature] = value
                break
            except ValueError:
                print("Please enter a valid number.")
    
    # Select model
    print("\nSelect model:")
    for idx, model_name in enumerate(predictor.get_model_list(), 1):
        print(f"{idx}. {model_name}")
    print(f"{len(predictor.get_model_list()) + 1}. All models")
    
    choice = int(input("Enter choice: "))
    
    if choice == len(predictor.get_model_list()) + 1:
        # Predict with all models
        predictions = predictor.predict_with_all_models(input_dict)
    else:
        # Predict with selected model
        model_name = predictor.get_model_list()[choice - 1]
        prediction = predictor.predict_single(input_dict, model_name)


if __name__ == "__main__":
    # Example usage
    print("Prediction Module")
    print("This module provides prediction functionality.")
    print("Import and use the PredictionSystem class in your main script.")
    print("\nFor interactive prediction, run: interactive_prediction()")
