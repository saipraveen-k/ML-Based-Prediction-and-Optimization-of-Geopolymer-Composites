"""
Feature Importance Analysis Module for UHPGC Compressive Strength Prediction
=============================================================================
This module analyzes and visualizes feature importance for trained models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import print_section, print_separator, create_directory, get_timestamp


class FeatureImportanceAnalyzer:
    """
    A class for analyzing feature importance of ML models.
    """
    
    def __init__(self, models, feature_names, output_dir='outputs'):
        """
        Initialize the FeatureImportanceAnalyzer.
        
        Parameters:
        -----------
        models : dict
            Dictionary of trained models
        feature_names : list
            List of feature names
        output_dir : str
            Directory to save outputs
        """
        self.models = models
        self.feature_names = feature_names
        self.output_dir = output_dir
        self.feature_importance = {}
        create_directory(os.path.join(output_dir, 'graphs'))
        
    def get_random_forest_importance(self, model):
        """
        Get feature importance from Random Forest model.
        
        Parameters:
        -----------
        model : RandomForestRegressor
            Trained Random Forest model
        
        Returns:
        --------
        importance : pandas.Series
            Feature importance values
        """
        importance = pd.Series(model.feature_importances_, index=self.feature_names)
        return importance.sort_values(ascending=False)
    
    def get_xgboost_importance(self, model):
        """
        Get feature importance from XGBoost model.
        
        Parameters:
        -----------
        model : XGBRegressor
            Trained XGBoost model
        
        Returns:
        --------
        importance : pandas.Series
            Feature importance values
        """
        importance = pd.Series(model.feature_importances_, index=self.feature_names)
        return importance.sort_values(ascending=False)
    
    def get_svr_importance(self, model, X, y):
        """
        Get feature importance from SVR model using permutation importance.
        
        Parameters:
        -----------
        model : SVR
            Trained SVR model
        X : array-like
            Feature matrix
        y : array-like
            Target variable
        
        Returns:
        --------
        importance : pandas.Series
            Feature importance values
        """
        from sklearn.inspection import permutation_importance
        
        # Calculate permutation importance
        perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
        
        importance = pd.Series(perm_importance.importances_mean, index=self.feature_names)
        return importance.sort_values(ascending=False)
    
    def analyze_all_models(self, X_train=None, y_train=None):
        """
        Analyze feature importance for all models.
        
        Parameters:
        -----------
        X_train : array-like, optional
            Training features (required for SVR)
        y_train : array-like, optional
            Training target (required for SVR)
        
        Returns:
        --------
        all_importance : dict
            Dictionary of feature importance for all models
        """
        print_section("FEATURE IMPORTANCE ANALYSIS")
        
        for model_name, model in self.models.items():
            print(f"\nAnalyzing {model_name}...")
            
            if model_name == 'Random Forest':
                importance = self.get_random_forest_importance(model)
            elif model_name == 'XGBoost':
                importance = self.get_xgboost_importance(model)
            elif model_name == 'SVR':
                if X_train is None or y_train is None:
                    print("Warning: X_train and y_train required for SVR importance. Skipping SVR.")
                    continue
                importance = self.get_svr_importance(model, X_train, y_train)
            else:
                print(f"Warning: Unknown model type {model_name}. Skipping.")
                continue
            
            self.feature_importance[model_name] = importance
            
            print(f"Top 5 important features for {model_name}:")
            print_separator('-')
            for idx, (feature, value) in enumerate(importance.head().items(), 1):
                print(f"{idx}. {feature}: {value:.4f}")
        
        return self.feature_importance
    
    def plot_feature_importance(self, model_name, top_n=10):
        """
        Plot feature importance for a specific model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        top_n : int
            Number of top features to display
        
        Returns:
        --------
        None
        """
        if model_name not in self.feature_importance:
            raise ValueError(f"No feature importance data for {model_name}")
        
        importance = self.feature_importance[model_name]
        top_features = importance.head(top_n)
        
        plt.figure(figsize=(12, 8))
        
        # Create horizontal bar plot
        bars = plt.barh(range(len(top_features)), top_features.values, color='steelblue', alpha=0.7, edgecolor='k')
        
        # Add value labels
        for i, (idx, bar) in enumerate(zip(top_features.index, bars)):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.4f}', ha='left', va='center', fontsize=10)
        
        plt.yticks(range(len(top_features)), top_features.index)
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title(f'Feature Importance - {model_name} (Top {top_n})', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_dir, 'graphs',
                                 f'feature_importance_{model_name}_{get_timestamp()}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot for {model_name} saved")
        plt.close()
    
    def plot_all_importances(self, top_n=10):
        """
        Plot feature importance for all models.
        
        Parameters:
        -----------
        top_n : int
            Number of top features to display
        
        Returns:
        --------
        None
        """
        print_section("GENERATING FEATURE IMPORTANCE PLOTS")
        
        for model_name in self.feature_importance.keys():
            self.plot_feature_importance(model_name, top_n)
        
        print_section("ALL FEATURE IMPORTANCE PLOTS GENERATED")
    
    def plot_comparison(self, top_n=10):
        """
        Plot feature importance comparison across all models.
        
        Parameters:
        -----------
        top_n : int
            Number of top features to display
        
        Returns:
        --------
        None
        """
        print_section("FEATURE IMPORTANCE COMPARISON")
        
        if len(self.feature_importance) < 2:
            print("Need at least 2 models for comparison")
            return
        
        # Get top features from the first model
        first_model = list(self.feature_importance.keys())[0]
        top_features = self.feature_importance[first_model].head(top_n).index.tolist()
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name in self.feature_importance.keys():
            for feature in top_features:
                if feature in self.feature_importance[model_name].index:
                    value = self.feature_importance[model_name][feature]
                else:
                    value = 0
                comparison_data.append({
                    'Model': model_name,
                    'Feature': feature,
                    'Importance': value
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Plot grouped bar chart
        plt.figure(figsize=(14, 8))
        
        # Pivot for plotting
        pivot_df = comparison_df.pivot(index='Feature', columns='Model', values='Importance')
        
        # Plot
        pivot_df.plot(kind='bar', figsize=(14, 8), alpha=0.7, edgecolor='k')
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Importance Score', fontsize=12)
        plt.title(f'Feature Importance Comparison Across Models (Top {top_n})', fontsize=14, fontweight='bold')
        plt.legend(title='Model', fontsize=10)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_dir, 'graphs',
                                 f'feature_importance_comparison_{get_timestamp()}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance comparison plot saved")
        plt.close()
    
    def generate_summary_report(self):
        """
        Generate a summary report of feature importance analysis.
        
        Returns:
        --------
        report : str
            Summary report text
        """
        print_section("GENERATING FEATURE IMPORTANCE REPORT")
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("FEATURE IMPORTANCE ANALYSIS REPORT".center(80))
        report_lines.append("="*80)
        report_lines.append(f"\nGenerated on: {get_timestamp()}")
        report_lines.append("\n" + "="*80)
        
        for model_name, importance in self.feature_importance.items():
            report_lines.append(f"\n{model_name}:")
            report_lines.append("-"*80)
            report_lines.append("\nTop 10 Most Important Features:")
            for idx, (feature, value) in enumerate(importance.head(10).items(), 1):
                report_lines.append(f"{idx:2d}. {feature:30s}: {value:.4f}")
            
            report_lines.append("\nBottom 5 Least Important Features:")
            for idx, (feature, value) in enumerate(importance.tail(5).items(), 1):
                report_lines.append(f"{idx:2d}. {feature:30s}: {value:.4f}")
        
        report_lines.append("\n" + "="*80)
        report_lines.append("\nINTERPRETATION:")
        report_lines.append("-"*80)
        report_lines.append("- Higher importance values indicate stronger influence on compressive strength")
        report_lines.append("- Top features are the most critical for accurate predictions")
        report_lines.append("- Features with low importance may be considered for removal")
        report_lines.append("- Different models may rank features differently")
        
        report_lines.append("\n" + "="*80)
        report_lines.append("END OF REPORT".center(80))
        report_lines.append("="*80)
        
        report = "\n".join(report_lines)
        
        # Save report
        output_path = os.path.join(self.output_dir, 'reports', f'feature_importance_report_{get_timestamp()}.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Feature importance report saved to: {output_path}")
        
        return report
    
    def get_feature_ranking(self):
        """
        Get average feature ranking across all models.
        
        Returns:
        --------
        ranking : pandas.DataFrame
            Average feature ranking
        """
        ranking_data = {}
        
        for feature in self.feature_names:
            ranks = []
            for model_name, importance in self.feature_importance.items():
                if feature in importance.index:
                    rank = importance.index.get_loc(feature) + 1
                    ranks.append(rank)
            
            if ranks:
                ranking_data[feature] = {
                    'Average Rank': np.mean(ranks),
                    'Std Rank': np.std(ranks),
                    'Min Rank': np.min(ranks),
                    'Max Rank': np.max(ranks)
                }
        
        ranking_df = pd.DataFrame.from_dict(ranking_data, orient='index')
        ranking_df = ranking_df.sort_values('Average Rank')
        
        return ranking_df


if __name__ == "__main__":
    # Example usage
    print("Feature Importance Analysis Module")
    print("This module provides feature importance analysis functionality.")
    print("Import and use the FeatureImportanceAnalyzer class in your main script.")
