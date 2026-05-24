"""
Data Preprocessing Module for UHPGC Compressive Strength Prediction
====================================================================
This module handles data loading, cleaning, preprocessing, and preparation
for machine learning models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import print_section, print_separator, create_directory


class DataPreprocessor:
    """
    A class for preprocessing UHPGC compressive strength data.
    """
    
    def __init__(self, data_path, target_column='Compressive_Strength_MPa'):
        """
        Initialize the DataPreprocessor.
        
        Parameters:
        -----------
        data_path : str
            Path to the dataset file (Excel or CSV)
        target_column : str
            Name of the target column (default: 'Compressive_Strength_MPa')
        """
        self.data_path = data_path
        self.target_column = target_column
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.original_feature_names = None
        
    def load_data(self):
        """
        Load data from Excel or CSV file.
        
        Returns:
        --------
        df : pandas.DataFrame
            Loaded dataset
        """
        print_section("DATA LOADING")
        
        if self.data_path.endswith('.xlsx') or self.data_path.endswith('.xls'):
            self.df = pd.read_excel(self.data_path, engine='openpyxl')
        elif self.data_path.endswith('.csv'):
            self.df = pd.read_csv(self.data_path)
        else:
            raise ValueError("Unsupported file format. Use .xlsx, .xls, or .csv")
        
        print(f"Dataset loaded successfully from: {self.data_path}")
        print(f"Dataset shape: {self.df.shape}")
        print(f"Number of samples: {self.df.shape[0]}")
        print(f"Number of features: {self.df.shape[1]}")
        
        return self.df
    
    def explore_data(self):
        """
        Perform initial data exploration.
        
        Returns:
        --------
        None
        """
        print_section("DATA EXPLORATION")
        
        print("\nDataset Information:")
        print(self.df.info())
        
        print("\n" + "="*80)
        print("First 5 rows:")
        print(self.df.head())
        
        print("\n" + "="*80)
        print("Statistical Summary:")
        print(self.df.describe())
        
        print("\n" + "="*80)
        print("Data Types:")
        print(self.df.dtypes)
        
        print("\n" + "="*80)
        print("Missing Values:")
        missing = self.df.isnull().sum()
        print(missing[missing > 0] if missing.sum() > 0 else "No missing values")
        
        print("\n" + "="*80)
        print("Duplicate Rows:")
        duplicates = self.df.duplicated().sum()
        print(f"Number of duplicate rows: {duplicates}")
        
    def clean_data(self, remove_duplicates=True, handle_missing=True):
        """
        Clean the dataset by handling missing values and duplicates.
        
        Parameters:
        -----------
        remove_duplicates : bool
            Whether to remove duplicate rows
        handle_missing : bool
            Whether to handle missing values
        
        Returns:
        --------
        df_cleaned : pandas.DataFrame
            Cleaned dataset
        """
        print_section("DATA CLEANING")
        
        initial_shape = self.df.shape
        
        # Remove duplicates
        if remove_duplicates:
            self.df = self.df.drop_duplicates()
            print(f"Removed {initial_shape[0] - self.df.shape[0]} duplicate rows")
        
        # Handle missing values
        if handle_missing:
            missing_before = self.df.isnull().sum().sum()
            if missing_before > 0:
                # For numerical columns, use median imputation
                numeric_columns = self.df.select_dtypes(include=[np.number]).columns
                imputer = SimpleImputer(strategy='median')
                self.df[numeric_columns] = imputer.fit_transform(self.df[numeric_columns])
                print(f"Imputed {missing_before} missing values using median strategy")
            else:
                print("No missing values to handle")
        
        print(f"Dataset shape after cleaning: {self.df.shape}")
        print(f"Samples retained: {self.df.shape[0]}")
        
        return self.df
    
    def detect_outliers(self, method='iqr', threshold=1.5):
        """
        Detect outliers in the dataset.
        
        Parameters:
        -----------
        method : str
            Method to use ('iqr' or 'zscore')
        threshold : float
            Threshold for outlier detection
        
        Returns:
        --------
        outliers : dict
            Dictionary containing outlier information for each column
        """
        print_section("OUTLIER DETECTION")
        
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        outliers = {}
        
        for col in numeric_columns:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                outliers[col] = {
                    'count': outlier_mask.sum(),
                    'percentage': (outlier_mask.sum() / len(self.df)) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
            elif method == 'zscore':
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                outlier_mask = z_scores > threshold
                outliers[col] = {
                    'count': outlier_mask.sum(),
                    'percentage': (outlier_mask.sum() / len(self.df)) * 100
                }
        
        # Print outlier summary
        print("\nOutlier Summary:")
        print_separator('-')
        for col, info in outliers.items():
            if info['count'] > 0:
                print(f"{col}: {info['count']} outliers ({info['percentage']:.2f}%)")
        
        return outliers
    
    def separate_features_target(self):
        """
        Separate features and target variable.
        
        Returns:
        --------
        X : pandas.DataFrame
            Feature matrix
        y : pandas.Series
            Target variable
        """
        print_section("SEPARATING FEATURES AND TARGET")
        
        # Check if target column exists
        if self.target_column not in self.df.columns:
            # Try to find the target column automatically
            possible_targets = ['CS', 'Compressive Strength', 'Compressive_Strength_MPa', 'compressive_strength', 'strength']
            for target in possible_targets:
                if target in self.df.columns:
                    self.target_column = target
                    print(f"Target column automatically detected: {self.target_column}")
                    break
            else:
                # Use last column as target
                self.target_column = self.df.columns[-1]
                print(f"Using last column as target: {self.target_column}")
        
        # Define relevant features for compressive strength prediction
        relevant_features = [
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
        
        # Filter to only include relevant features that exist in the dataset
        available_features = [f for f in relevant_features if f in self.df.columns]
        
        # Drop non-relevant columns (Mix_ID, other strength properties, metadata)
        columns_to_drop = [self.target_column, 'Mix_ID']
        
        # Also drop other strength properties and metadata columns
        metadata_columns = [
            'Flexural_Strength_MPa', 'Tensile_Strength_MPa', 'Elastic_Modulus_GPa',
            'Interlayer_Bond_Strength_MPa', 'RCPT', 'Water_Absorption_%', 'Sorptivity',
            'Gas_Permeability', 'Chloride_Penetration', 'Acid_Attack_Loss_%',
            'Sulphate_Resistance_%', 'Printing_Speed_mm_s', 'Layer_Height_mm',
            'Nozzle_Diameter_mm', 'Build_Direction', 'Print_Orientation',
            'Anisotropy_Index', 'Density_kg_m3', 'Porosity_%', 'SEM_Results',
            'XRD_Results', 'FTIR_Results', 'Heat_Curing_Conditions', 'Curing_Method',
            'Dataset_Distribution', 'Validation_Dataset', 'Testing_Dataset',
            'Statistical_Normalization', 'Missing_Data_Handled', 'Feature_Selection_Method'
        ]
        
        for col in metadata_columns:
            if col in self.df.columns:
                columns_to_drop.append(col)
        
        # Create feature matrix with only relevant features
        self.X = self.df[available_features].copy()
        self.y = self.df[self.target_column]
        
        self.original_feature_names = self.X.columns.tolist()
        self.feature_names = self.X.columns.tolist()
        
        print(f"Target variable: {self.target_column}")
        print(f"Number of features: {len(self.feature_names)}")
        print(f"Feature names: {self.feature_names}")
        
        return self.X, self.y
    
    def scale_features(self, scaling_method='standard'):
        """
        Scale the features.
        
        Parameters:
        -----------
        scaling_method : str
            Scaling method ('standard' or 'minmax')
        
        Returns:
        --------
        X_scaled : numpy.ndarray
            Scaled feature matrix
        """
        print_section("FEATURE SCALING")
        
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("scaling_method must be 'standard' or 'minmax'")
        
        X_scaled = self.scaler.fit_transform(self.X)
        print(f"Features scaled using {scaling_method} scaling")
        print(f"Scaling parameters saved")
        
        return X_scaled
    
    def split_data(self, test_size=0.3, random_state=42):
        """
        Split the data into training and testing sets.
        
        Parameters:
        -----------
        test_size : float
            Proportion of data to use for testing (default: 0.3)
        random_state : int
            Random seed for reproducibility
        
        Returns:
        --------
        X_train, X_test, y_train, y_test : tuple
            Split data arrays
        """
        print_section("TRAIN-TEST SPLIT")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set size: {self.X_train.shape[0]} samples ({(1-test_size)*100:.0f}%)")
        print(f"Testing set size: {self.X_test.shape[0]} samples ({test_size*100:.0f}%)")
        print(f"Random state: {random_state}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_correlation_matrix(self):
        """
        Calculate the correlation matrix.
        
        Returns:
        --------
        corr_matrix : pandas.DataFrame
            Correlation matrix
        """
        corr_matrix = self.df.corr()
        return corr_matrix
    
    def preprocess_pipeline(self, test_size=0.3, random_state=42, scaling_method='standard'):
        """
        Run the complete preprocessing pipeline.
        
        Parameters:
        -----------
        test_size : float
            Proportion of data to use for testing
        random_state : int
            Random seed for reproducibility
        scaling_method : str
            Scaling method ('standard' or 'minmax')
        
        Returns:
        --------
        X_train, X_test, y_train, y_test : tuple
            Preprocessed and split data
        """
        print_section("COMPLETE PREPROCESSING PIPELINE")
        
        # Load data
        self.load_data()
        
        # Explore data
        self.explore_data()
        
        # Clean data
        self.clean_data()
        
        # Detect outliers
        self.detect_outliers()
        
        # Separate features and target
        self.separate_features_target()
        
        # Scale features
        self.X = self.scale_features(scaling_method=scaling_method)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(
            test_size=test_size, random_state=random_state
        )
        
        print_section("PREPROCESSING COMPLETED")
        print("Data is ready for model training!")
        
        return self.X_train, self.X_test, self.y_train, self.y_test


if __name__ == "__main__":
    # Example usage
    print("Data Preprocessing Module")
    print("This module provides data preprocessing functionality.")
    print("Import and use the DataPreprocessor class in your main script.")
