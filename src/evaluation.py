"""
Model Evaluation Module for UHPGC Compressive Strength Prediction
================================================================
This module handles comprehensive evaluation of trained models including
metrics calculation, visualization, and comparison.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 20,
    'axes.labelsize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'figure.titlesize': 22
})

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import print_section, print_separator, create_directory, get_timestamp, save_dataframe, format_feature_name, format_metric_name


class ModelEvaluator:
    """
    A class for evaluating ML models for UHPGC compressive strength prediction.
    """
    
    def __init__(self, output_dir='outputs'):
        """
        Initialize the ModelEvaluator.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save evaluation outputs
        """
        self.output_dir = output_dir
        self.evaluation_results = {}
        create_directory(os.path.join(output_dir, 'graphs'))
        create_directory(os.path.join(output_dir, 'metrics'))
        create_directory(os.path.join(output_dir, 'reports'))
        
    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate evaluation metrics.
        
        Parameters:
        -----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values
        
        Returns:
        --------
        metrics : dict
            Dictionary of evaluation metrics
        """
        metrics = {
            'R²': r2_score(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MSE': mean_squared_error(y_true, y_pred)
        }
        return metrics
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """
        Evaluate a single model.
        
        Parameters:
        -----------
        model : object
            Trained model
        X_test : array-like
            Test features
        y_test : array-like
            Test target
        model_name : str
            Name of the model
        
        Returns:
        --------
        results : dict
            Evaluation results
        """
        print_section(f"EVALUATING {model_name.upper()}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred)
        
        # Print metrics
        print("Evaluation Metrics:")
        print_separator('-')
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
        
        # Store results
        results = {
            'model_name': model_name,
            'y_true': y_test,
            'y_pred': y_pred,
            'metrics': metrics
        }
        self.evaluation_results[model_name] = results
        
        return results
    
    def evaluate_all_models(self, models, X_test, y_test):
        """
        Evaluate all models.
        
        Parameters:
        -----------
        models : dict
            Dictionary of trained models
        X_test : array-like
            Test features
        y_test : array-like
            Test target
        
        Returns:
        --------
        all_results : dict
            Dictionary of all evaluation results
        """
        print_section("EVALUATING ALL MODELS")
        
        all_results = {}
        for model_name, model in models.items():
            results = self.evaluate_model(model, X_test, y_test, model_name)
            all_results[model_name] = results
        
        return all_results
    
    def plot_actual_vs_predicted(self, model_name=None):
        """
        Plot actual vs predicted values.
        
        Parameters:
        -----------
        model_name : str, optional
            Name of specific model to plot. If None, plots all models.
        
        Returns:
        --------
        None
        """
        print_section("ACTUAL VS PREDICTED PLOTS")
        
        if model_name:
            models_to_plot = [model_name]
        else:
            models_to_plot = list(self.evaluation_results.keys())
        
        for name in models_to_plot:
            if name not in self.evaluation_results:
                continue
            
            results = self.evaluation_results[name]
            y_true = results['y_true']
            y_pred = results['y_pred']
            
            plt.figure(figsize=(8, 8))
            
            # Scatter plot
            plt.scatter(y_true, y_pred, alpha=0.7, s=100, edgecolor='k', color='steelblue')
            
            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label='y = x (Perfect Prediction)')
            
            plt.xlabel('Actual Compressive Strength (MPa)', fontsize=16)
            plt.ylabel('Predicted Compressive Strength (MPa)', fontsize=16)
            plt.title(f'Actual vs Predicted Compressive Strength - {name}', fontsize=20, fontweight='bold', pad=15)
            plt.legend(fontsize=13, loc='lower right')
            plt.grid(True, linestyle='--', alpha=0.5)
            
            # Add stats (R², RMSE, MAE)
            r2 = results['metrics']['R²']
            rmse = results['metrics']['RMSE']
            mae = results['metrics']['MAE']
            stats_text = f"R² = {r2:.4f}\nRMSE = {rmse:.2f} MPa\nMAE = {mae:.2f} MPa"
            plt.text(0.05, 0.95, stats_text, 
                    transform=plt.gca().transAxes, fontsize=13,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
            
            plt.tight_layout()
            
            # Save plot
            output_path = os.path.join(self.output_dir, 'graphs', f'actual_vs_predicted_{name}_{get_timestamp()}.png')
            plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
            print(f"Actual vs Predicted plot for {name} saved")
            plt.close()
    
    def plot_residuals(self, model_name=None):
        """
        Plot residual plots.
        
        Parameters:
        -----------
        model_name : str, optional
            Name of specific model to plot. If None, plots all models.
        
        Returns:
        --------
        None
        """
        print_section("RESIDUAL PLOTS")
        
        if model_name:
            models_to_plot = [model_name]
        else:
            models_to_plot = list(self.evaluation_results.keys())
        
        for name in models_to_plot:
            if name not in self.evaluation_results:
                continue
            
            results = self.evaluation_results[name]
            y_true = results['y_true']
            y_pred = results['y_pred']
            residuals = y_true - y_pred
            
            plt.figure(figsize=(10, 6))
            
            # Residual vs Predicted
            plt.scatter(y_pred, residuals, alpha=0.7, s=80, edgecolor='k', color='steelblue')
            plt.axhline(y=0, color='red', linestyle='--', lw=3)
            plt.xlabel('Predicted Compressive Strength (MPa)', fontsize=16)
            plt.ylabel('Residuals (MPa)', fontsize=16)
            plt.title(f'Residuals vs Predicted Compressive Strength - {name}', fontsize=20, fontweight='bold', pad=15)
            plt.grid(True, linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            
            # Save plot
            output_path = os.path.join(self.output_dir, 'graphs', f'residuals_{name}_{get_timestamp()}.png')
            plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
            print(f"Residual plot for {name} saved")
            plt.close()
    
    def plot_error_distribution(self, model_name=None):
        """
        Plot error distribution.
        
        Parameters:
        -----------
        model_name : str, optional
            Name of specific model to plot. If None, plots all models.
        
        Returns:
        --------
        None
        """
        print_section("ERROR DISTRIBUTION PLOTS")
        
        if model_name:
            models_to_plot = [model_name]
        else:
            models_to_plot = list(self.evaluation_results.keys())
        
        for name in models_to_plot:
            if name not in self.evaluation_results:
                continue
            
            results = self.evaluation_results[name]
            y_true = results['y_true']
            y_pred = results['y_pred']
            errors = y_true - y_pred
            
            plt.figure(figsize=(10, 6))
            
            # Histogram with KDE
            sns.histplot(errors, kde=True, bins=30, color='darkgreen', alpha=0.7, edgecolor='black')
            plt.axvline(x=0, color='red', linestyle='--', lw=2)
            plt.axvline(x=errors.mean(), color='orange', linestyle='--', lw=2, label=f'Mean Error: {errors.mean():.4f}')
            
            plt.xlabel('Prediction Error (MPa)', fontsize=16)
            plt.ylabel('Frequency', fontsize=16)
            plt.title(f'Error Distribution - {name}', fontsize=20, fontweight='bold', pad=15)
            plt.legend(fontsize=13)
            plt.grid(True, linestyle='--', alpha=0.5, axis='y')
            
            plt.tight_layout()
            
            # Save plot
            output_path = os.path.join(self.output_dir, 'graphs', f'error_distribution_{name}_{get_timestamp()}.png')
            plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
            print(f"Error distribution plot for {name} saved")
            plt.close()
    
    def create_comparison_table(self):
        """
        Create a comparison table of all models.
        
        Returns:
        --------
        comparison_df : pandas.DataFrame
            Comparison DataFrame
        """
        print_section("MODEL COMPARISON TABLE")
        
        comparison_data = []
        for model_name, results in self.evaluation_results.items():
            metrics = results['metrics']
            comparison_data.append({
                'Model': model_name,
                'R² Score': metrics['R²'],
                'MAE': metrics['MAE'],
                'RMSE': metrics['RMSE'],
                'MSE': metrics['MSE']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('R² Score', ascending=False)
        
        print(comparison_df.to_string(index=False))
        
        # Save comparison table
        output_path = os.path.join(self.output_dir, 'metrics', f'model_comparison_{get_timestamp()}.csv')
        save_dataframe(comparison_df, output_path)
        print(f"\nComparison table saved to: {output_path}")
        
        return comparison_df
    
    def plot_model_comparison(self):
        """
        Plot model comparison bar charts.
        
        Returns:
        --------
        None
        """
        print_section("MODEL COMPARISON VISUALIZATION")
        
        comparison_df = self.create_comparison_table()
        
        # Create subplots for each metric
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        metrics = ['R² Score', 'MAE', 'RMSE', 'MSE']
        colors = ['steelblue', 'lightcoral', 'darkgreen', 'orange']
        
        for idx, (metric, color) in enumerate(zip(metrics, colors)):
            ax = axes[idx]
            bars = ax.bar(comparison_df['Model'], comparison_df[metric], color=color, alpha=0.75, edgecolor='k', width=0.4)
            
            # Format using format_metric_name helper
            clean_metric = format_metric_name(metric)
            ax.set_ylabel(clean_metric, fontsize=14)
            ax.set_title(f'{clean_metric} Comparison', fontsize=16, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.5, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height * 1.01,
                       f'{height:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=11)
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_dir, 'graphs', f'model_comparison_{get_timestamp()}.png')
        plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"Model comparison plot saved to: {output_path}")
        plt.close()
    
    def generate_evaluation_report(self):
        """
        Generate a comprehensive evaluation report.
        
        Returns:
        --------
        report : str
            Evaluation report text
        """
        print_section("GENERATING EVALUATION REPORT")
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("MODEL EVALUATION REPORT".center(80))
        report_lines.append("="*80)
        report_lines.append(f"\nGenerated on: {get_timestamp()}")
        report_lines.append("\n" + "="*80)
        
        # Model comparison
        comparison_df = self.create_comparison_table()
        report_lines.append("\nMODEL COMPARISON:")
        report_lines.append("-"*80)
        report_lines.append(comparison_df.to_string(index=False))
        
        # Best model
        best_model = comparison_df.iloc[0]
        report_lines.append("\n" + "="*80)
        report_lines.append("\nBEST MODEL:")
        report_lines.append("-"*80)
        report_lines.append(f"Model: {best_model['Model']}")
        report_lines.append(f"R² Score: {best_model['R² Score']:.4f}")
        report_lines.append(f"MAE: {best_model['MAE']:.4f}")
        report_lines.append(f"RMSE: {best_model['RMSE']:.4f}")
        
        # Individual model details
        report_lines.append("\n" + "="*80)
        report_lines.append("\nDETAILED MODEL RESULTS:")
        report_lines.append("-"*80)
        
        for model_name, results in self.evaluation_results.items():
            report_lines.append(f"\n{model_name}:")
            report_lines.append("-"*40)
            for metric, value in results['metrics'].items():
                report_lines.append(f"  {metric}: {value:.4f}")
        
        report_lines.append("\n" + "="*80)
        report_lines.append("END OF REPORT".center(80))
        report_lines.append("="*80)
        
        report = "\n".join(report_lines)
        
        # Save report
        output_path = os.path.join(self.output_dir, 'reports', f'evaluation_report_{get_timestamp()}.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Evaluation report saved to: {output_path}")
        
        return report
    
    def generate_all_plots(self):
        """
        Generate all evaluation plots.
        
        Returns:
        --------
        None
        """
        print_section("GENERATING ALL EVALUATION PLOTS")
        
        # Actual vs Predicted
        self.plot_actual_vs_predicted()
        
        # Residuals
        self.plot_residuals()
        
        # Error distribution
        self.plot_error_distribution()
        
        # Model comparison
        self.plot_model_comparison()
        
        print_section("ALL EVALUATION PLOTS GENERATED")


if __name__ == "__main__":
    from src.data_preprocessing import DataPreprocessor
    from src.utils import load_model
    
    data_path = 'data/dataset.csv'
    if os.path.exists(data_path):
        print(f"Loading dataset and running preprocessing pipeline...")
        preprocessor = DataPreprocessor(data_path=data_path, target_column='Compressive_Strength_MPa')
        X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(test_size=0.3, random_state=42)
        
        # Load trained models
        models = {}
        model_paths = {
            'Random Forest': 'models/random_forest.pkl',
            'SVR': 'models/svr.pkl',
            'XGBoost': 'models/xgboost.pkl'
        }
        for name, path in model_paths.items():
            if os.path.exists(path):
                models[name] = load_model(path)
            else:
                print(f"WARNING: Model {name} not found at {path}")
                
        if models:
            evaluator = ModelEvaluator()
            evaluator.evaluate_all_models(models, X_test, y_test)
            evaluator.generate_all_plots()
            evaluator.generate_evaluation_report()
            print("Model evaluation complete!")
        else:
            print("Error: No models loaded. Please train models first.")
    else:
        print(f"Error: Dataset not found at {data_path}")
