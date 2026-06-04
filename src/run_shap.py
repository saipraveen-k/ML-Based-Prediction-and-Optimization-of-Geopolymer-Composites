"""
Script to Run SHAP Analysis Workflow
====================================
This script loads the preprocessed data and trained XGBoost model, 
runs the SHAPAnalyzer, and saves all outputs.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

# Add project root directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import DataPreprocessor
from src.shap_analyzer import SHAPAnalyzer
from src.utils import print_section, load_model, load_scaler

def main():
    print_section("RUNNING SHAP EXPLAINABILITY PIPELINE")
    
    # Paths
    data_path = 'data/dataset.csv'
    model_path = 'models/xgboost.pkl'
    scaler_path = 'models/scaler.pkl'
    output_dir = 'outputs'
    
    # Validate file existence
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        sys.exit(1)
        
    if not os.path.exists(model_path):
        print(f"Error: Trained XGBoost model not found at {model_path}")
        print("Please train the models first by running the training scripts or notebook.")
        sys.exit(1)
        
    if not os.path.exists(scaler_path):
        print(f"Error: Fitted scaler not found at {scaler_path}")
        sys.exit(1)
        
    # 1. Preprocess the dataset
    print("Loading and preprocessing dataset...")
    preprocessor = DataPreprocessor(data_path=data_path, target_column='Compressive_Strength_MPa')
    
    # Run the pipeline (loads, cleans, separates features, scales)
    # We use the same parameters as the main training pipeline
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
        test_size=0.3,
        random_state=42,
        scaling_method='standard'
    )
    
    # Get scaled feature matrix (X) and feature names
    X_scaled = preprocessor.X
    feature_names = preprocessor.original_feature_names
    
    # 2. Load trained XGBoost model and scaler
    print("Loading XGBoost model and fitted scaler...")
    xgb_model = load_model(model_path)
    scaler = load_scaler(scaler_path)
    
    # Reconstruct original features from scaled features
    X_original = pd.DataFrame(scaler.inverse_transform(X_scaled), columns=feature_names)
    
    # 3. Initialize and run SHAP analysis
    analyzer = SHAPAnalyzer(
        model=xgb_model,
        X_scaled=X_scaled,
        feature_names=feature_names,
        scaler=scaler,
        X_original=X_original,
        output_dir=output_dir
    )
    
    analyzer.run_all_analysis()
    
    print("\nSHAP pipeline completed successfully!")

if __name__ == '__main__':
    main()
