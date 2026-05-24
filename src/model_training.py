"""
Model Training Module for UHPGC Compressive Strength Prediction
================================================================
This module handles training of Random Forest, SVR, and XGBoost models
with hyperparameter tuning and cross-validation.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import print_section, print_separator, save_model, get_timestamp


class ModelTrainer:
    """
    A class for training and tuning ML models for UHPGC compressive strength prediction.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the ModelTrainer.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_params = {}
        self.training_history = {}
        
    def train_random_forest(self, X_train, y_train, tune_hyperparameters=True):
        """
        Train Random Forest Regressor with optional hyperparameter tuning.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        tune_hyperparameters : bool
            Whether to perform hyperparameter tuning
        
        Returns:
        --------
        model : RandomForestRegressor
            Trained Random Forest model
        """
        print_section("RANDOM FOREST REGRESSOR")
        
        if tune_hyperparameters:
            print("Performing hyperparameter tuning...")
            
            # Define parameter grid
            param_grid = {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            
            # Initialize Random Forest
            rf = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
            
            # Perform Randomized Search (faster than Grid Search)
            random_search = RandomizedSearchCV(
                rf, param_distributions=param_grid,
                n_iter=50, cv=5, scoring='r2',
                random_state=self.random_state, n_jobs=-1, verbose=1
            )
            
            random_search.fit(X_train, y_train)
            
            # Get best model
            model = random_search.best_estimator_
            self.best_params['Random Forest'] = random_search.best_params_
            
            print(f"Best Parameters: {random_search.best_params_}")
            print(f"Best CV Score: {random_search.best_score_:.4f}")
        else:
            print("Training with default parameters...")
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=self.random_state,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            self.best_params['Random Forest'] = model.get_params()
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        print(f"Cross-Validation R² Scores: {cv_scores}")
        print(f"Mean CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        self.models['Random Forest'] = model
        self.training_history['Random Forest'] = {
            'cv_scores': cv_scores,
            'mean_cv_score': cv_scores.mean(),
            'std_cv_score': cv_scores.std()
        }
        
        print("Random Forest training completed!")
        return model
    
    def train_svr(self, X_train, y_train, tune_hyperparameters=True):
        """
        Train Support Vector Regressor with optional hyperparameter tuning.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        tune_hyperparameters : bool
            Whether to perform hyperparameter tuning
        
        Returns:
        --------
        model : SVR
            Trained SVR model
        """
        print_section("SUPPORT VECTOR REGRESSION")
        
        if tune_hyperparameters:
            print("Performing hyperparameter tuning...")
            
            # Define parameter grid
            param_grid = {
                'C': [0.1, 1, 10, 100, 1000],
                'epsilon': [0.01, 0.1, 0.5, 1.0],
                'kernel': ['rbf', 'linear', 'poly'],
                'gamma': ['scale', 'auto', 0.01, 0.1, 1]
            }
            
            # Initialize SVR
            svr = SVR()
            
            # Perform Randomized Search
            random_search = RandomizedSearchCV(
                svr, param_distributions=param_grid,
                n_iter=50, cv=5, scoring='r2',
                random_state=self.random_state, n_jobs=-1, verbose=1
            )
            
            random_search.fit(X_train, y_train)
            
            # Get best model
            model = random_search.best_estimator_
            self.best_params['SVR'] = random_search.best_params_
            
            print(f"Best Parameters: {random_search.best_params_}")
            print(f"Best CV Score: {random_search.best_score_:.4f}")
        else:
            print("Training with default parameters...")
            model = SVR(
                C=100,
                epsilon=0.1,
                kernel='rbf',
                gamma='scale'
            )
            model.fit(X_train, y_train)
            self.best_params['SVR'] = model.get_params()
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        print(f"Cross-Validation R² Scores: {cv_scores}")
        print(f"Mean CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        self.models['SVR'] = model
        self.training_history['SVR'] = {
            'cv_scores': cv_scores,
            'mean_cv_score': cv_scores.mean(),
            'std_cv_score': cv_scores.std()
        }
        
        print("SVR training completed!")
        return model
    
    def train_xgboost(self, X_train, y_train, tune_hyperparameters=True):
        """
        Train XGBoost Regressor with optional hyperparameter tuning.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        tune_hyperparameters : bool
            Whether to perform hyperparameter tuning
        
        Returns:
        --------
        model : XGBRegressor
            Trained XGBoost model
        """
        print_section("XGBOOST REGRESSOR")
        
        if tune_hyperparameters:
            print("Performing hyperparameter tuning...")
            
            # Define parameter grid
            param_grid = {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'min_child_weight': [1, 3, 5],
                'gamma': [0, 0.1, 0.2],
                'reg_alpha': [0, 0.01, 0.1],
                'reg_lambda': [1, 1.5, 2]
            }
            
            # Initialize XGBoost
            xgb_model = xgb.XGBRegressor(
                random_state=self.random_state,
                objective='reg:squarederror',
                n_jobs=-1
            )
            
            # Perform Randomized Search
            random_search = RandomizedSearchCV(
                xgb_model, param_distributions=param_grid,
                n_iter=50, cv=5, scoring='r2',
                random_state=self.random_state, n_jobs=-1, verbose=1
            )
            
            random_search.fit(X_train, y_train)
            
            # Get best model
            model = random_search.best_estimator_
            self.best_params['XGBoost'] = random_search.best_params_
            
            print(f"Best Parameters: {random_search.best_params_}")
            print(f"Best CV Score: {random_search.best_score_:.4f}")
        else:
            print("Training with default parameters...")
            model = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=7,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0,
                reg_alpha=0.01,
                reg_lambda=1.5,
                random_state=self.random_state,
                objective='reg:squarederror',
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            self.best_params['XGBoost'] = model.get_params()
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        print(f"Cross-Validation R² Scores: {cv_scores}")
        print(f"Mean CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        self.models['XGBoost'] = model
        self.training_history['XGBoost'] = {
            'cv_scores': cv_scores,
            'mean_cv_score': cv_scores.mean(),
            'std_cv_score': cv_scores.std()
        }
        
        print("XGBoost training completed!")
        return model
    
    def train_all_models(self, X_train, y_train, tune_hyperparameters=True):
        """
        Train all models (Random Forest, SVR, XGBoost).
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        tune_hyperparameters : bool
            Whether to perform hyperparameter tuning
        
        Returns:
        --------
        models : dict
            Dictionary of trained models
        """
        print_section("TRAINING ALL MODELS")
        
        # Train Random Forest
        self.train_random_forest(X_train, y_train, tune_hyperparameters)
        
        # Train SVR
        self.train_svr(X_train, y_train, tune_hyperparameters)
        
        # Train XGBoost
        self.train_xgboost(X_train, y_train, tune_hyperparameters)
        
        print_section("ALL MODELS TRAINED")
        print(f"Models trained: {list(self.models.keys())}")
        
        return self.models
    
    def save_all_models(self, output_dir='models'):
        """
        Save all trained models to disk.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save models
        
        Returns:
        --------
        None
        """
        print_section("SAVING MODELS")
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            # Sanitize model name for filename
            filename = model_name.lower().replace(' ', '_')
            model_path = os.path.join(output_dir, f'{filename}.pkl')
            save_model(model, model_path)
        
        print("All models saved successfully!")
    
    def get_model(self, model_name):
        """
        Get a trained model by name.
        
        Parameters:
        -----------
        model_name : str
            Name of the model ('Random Forest', 'SVR', 'XGBoost')
        
        Returns:
        --------
        model : object
            Trained model
        """
        if model_name in self.models:
            return self.models[model_name]
        else:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
    
    def get_training_summary(self):
        """
        Get a summary of training results.
        
        Returns:
        --------
        summary : pandas.DataFrame
            Training summary DataFrame
        """
        summary_data = []
        for model_name, history in self.training_history.items():
            summary_data.append({
                'Model': model_name,
                'Mean CV R²': history['mean_cv_score'],
                'Std CV R²': history['std_cv_score']
            })
        
        summary_df = pd.DataFrame(summary_data)
        return summary_df


if __name__ == "__main__":
    # Example usage
    print("Model Training Module")
    print("This module provides model training functionality.")
    print("Import and use the ModelTrainer class in your main script.")
