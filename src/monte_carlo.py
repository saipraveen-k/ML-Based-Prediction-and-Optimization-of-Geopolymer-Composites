"""
Monte Carlo Simulation Module for UHPGC Compressive Strength Prediction
========================================================================
This module performs Monte Carlo simulations to analyze model stability
and performance distribution across multiple random train-test splits.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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


class MonteCarloSimulation:
    """
    A class for performing Monte Carlo simulations on ML models.
    """
    
    def __init__(self, X, y, models, n_simulations=100, test_size=0.3, output_dir='outputs'):
        """
        Initialize the MonteCarloSimulation.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target variable
        models : dict
            Dictionary of trained models
        n_simulations : int
            Number of Monte Carlo simulations (default: 100)
        test_size : float
            Proportion of data to use for testing
        output_dir : str
            Directory to save outputs
        """
        self.X = X
        self.y = y
        self.models = models
        self.n_simulations = n_simulations
        self.test_size = test_size
        self.output_dir = output_dir
        self.simulation_results = {}
        create_directory(os.path.join(output_dir, 'graphs'))
        
    def run_simulation(self, model_name):
        """
        Run Monte Carlo simulation for a specific model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to simulate
        
        Returns:
        --------
        results : dict
            Simulation results
        """
        print_section(f"MONTE CARLO SIMULATION - {model_name.upper()}")
        print(f"Running {self.n_simulations} simulations...")
        
        model = self.models[model_name]
        
        # Initialize result arrays
        r2_scores = []
        mae_scores = []
        rmse_scores = []
        mse_scores = []
        
        for i in range(self.n_simulations):
            # Random train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=self.test_size, random_state=i
            )
            
            # Fit model on training data
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mse = mean_squared_error(y_test, y_pred)
            
            r2_scores.append(r2)
            mae_scores.append(mae)
            rmse_scores.append(rmse)
            mse_scores.append(mse)
            
            # Progress update
            if (i + 1) % 20 == 0:
                print(f"Completed {i + 1}/{self.n_simulations} simulations")
        
        # Calculate statistics
        results = {
            'r2_scores': np.array(r2_scores),
            'mae_scores': np.array(mae_scores),
            'rmse_scores': np.array(rmse_scores),
            'mse_scores': np.array(mse_scores),
            'r2_mean': np.mean(r2_scores),
            'r2_std': np.std(r2_scores),
            'r2_min': np.min(r2_scores),
            'r2_max': np.max(r2_scores),
            'mae_mean': np.mean(mae_scores),
            'mae_std': np.std(mae_scores),
            'rmse_mean': np.mean(rmse_scores),
            'rmse_std': np.std(rmse_scores),
            'mse_mean': np.mean(mse_scores),
            'mse_std': np.std(mse_scores)
        }
        
        self.simulation_results[model_name] = results
        
        # Print summary statistics
        print_separator('-')
        print(f"R² Score: {results['r2_mean']:.4f} ± {results['r2_std']:.4f}")
        print(f"R² Range: [{results['r2_min']:.4f}, {results['r2_max']:.4f}]")
        print(f"MAE: {results['mae_mean']:.4f} ± {results['mae_std']:.4f}")
        print(f"RMSE: {results['rmse_mean']:.4f} ± {results['rmse_std']:.4f}")
        
        return results
    
    def run_all_simulations(self):
        """
        Run Monte Carlo simulations for all models.
        
        Returns:
        --------
        all_results : dict
            Dictionary of all simulation results
        """
        print_section("RUNNING MONTE CARLO SIMULATIONS FOR ALL MODELS")
        
        for model_name in self.models.keys():
            self.run_simulation(model_name)
        
        print_section("ALL MONTE CARLO SIMULATIONS COMPLETED")
        
        return self.simulation_results
    
    def plot_distribution(self, model_name, metric='r2_scores'):
        """
        Plot distribution of a metric for a model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        metric : str
            Metric to plot ('r2_scores', 'mae_scores', 'rmse_scores', 'mse_scores')
        
        Returns:
        --------
        None
        """
        if model_name not in self.simulation_results:
            raise ValueError(f"No simulation results for {model_name}")
        
        results = self.simulation_results[model_name]
        values = results[metric]
        
        metric_label = metric.replace('_scores', '').upper()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(values, bins=30, edgecolor='k', color='steelblue', alpha=0.7, density=True)
        axes[0].axvline(x=results[f'{metric_label.lower()}_mean'], color='r', linestyle='--', lw=2, 
                       label=f'Mean: {results[f"{metric_label.lower()}_mean"]:.4f}')
        axes[0].set_xlabel(metric_label, fontsize=12)
        axes[0].set_ylabel('Density', fontsize=12)
        axes[0].set_title(f'Distribution of {metric_label} - {model_name}', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Box plot
        axes[1].boxplot(values, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightcoral', alpha=0.7))
        axes[1].set_ylabel(metric_label, fontsize=12)
        axes[1].set_title(f'Box Plot of {metric_label} - {model_name}', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_dir, 'graphs', 
                                 f'monte_carlo_{metric}_{model_name}_{get_timestamp()}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Distribution plot for {metric} - {model_name} saved")
        plt.close()
    
    def plot_pdf(self, model_name, metric='r2_scores'):
        """
        Plot Probability Density Function (PDF) for a metric.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        metric : str
            Metric to plot
        
        Returns:
        --------
        None
        """
        if model_name not in self.simulation_results:
            raise ValueError(f"No simulation results for {model_name}")
        
        results = self.simulation_results[model_name]
        values = results[metric]
        
        metric_label = metric.replace('_scores', '').upper()
        
        plt.figure(figsize=(10, 6))
        
        # Plot PDF using KDE
        sns.kdeplot(values, shade=True, color='darkgreen', alpha=0.7)
        plt.axvline(x=results[f'{metric_label.lower()}_mean'], color='r', linestyle='--', lw=2,
                   label=f'Mean: {results[f"{metric_label.lower()}_mean"]:.4f}')
        plt.axvline(x=np.median(values), color='orange', linestyle='--', lw=2,
                   label=f'Median: {np.median(values):.4f}')
        
        plt.xlabel(metric_label, fontsize=12)
        plt.ylabel('Probability Density', fontsize=12)
        plt.title(f'PDF of {metric_label} - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_dir, 'graphs',
                                 f'monte_carlo_pdf_{metric}_{model_name}_{get_timestamp()}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"PDF plot for {metric} - {model_name} saved")
        plt.close()
    
    def plot_cdf(self, model_name, metric='r2_scores'):
        """
        Plot Cumulative Distribution Function (CDF) for a metric.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        metric : str
            Metric to plot
        
        Returns:
        --------
        None
        """
        if model_name not in self.simulation_results:
            raise ValueError(f"No simulation results for {model_name}")
        
        results = self.simulation_results[model_name]
        values = results[metric]
        
        metric_label = metric.replace('_scores', '').upper()
        
        # Sort values for CDF
        sorted_values = np.sort(values)
        cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        
        plt.figure(figsize=(10, 6))
        
        # Plot CDF
        plt.plot(sorted_values, cdf, linewidth=2, color='purple')
        plt.axvline(x=results[f'{metric_label.lower()}_mean'], color='r', linestyle='--', lw=2,
                   label=f'Mean: {results[f"{metric_label.lower()}_mean"]:.4f}')
        plt.axvline(x=np.median(values), color='orange', linestyle='--', lw=2,
                   label=f'Median: {np.median(values):.4f}')
        
        plt.xlabel(metric_label, fontsize=12)
        plt.ylabel('Cumulative Probability', fontsize=12)
        plt.title(f'CDF of {metric_label} - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_dir, 'graphs',
                                 f'monte_carlo_cdf_{metric}_{model_name}_{get_timestamp()}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"CDF plot for {metric} - {model_name} saved")
        plt.close()
    
    def plot_all_distributions(self):
        """
        Plot all distributions (PDF and CDF) for all models.
        
        Returns:
        --------
        None
        """
        print_section("GENERATING MONTE CARLO DISTRIBUTION PLOTS")
        
        metrics = ['r2_scores', 'mae_scores', 'rmse_scores']
        
        for model_name in self.simulation_results.keys():
            for metric in metrics:
                self.plot_distribution(model_name, metric)
                self.plot_pdf(model_name, metric)
                self.plot_cdf(model_name, metric)
        
        print_section("ALL MONTE CARLO PLOTS GENERATED")
    
    def create_summary_table(self):
        """
        Create a summary table of Monte Carlo simulation results.
        
        Returns:
        --------
        summary_df : pandas.DataFrame
            Summary DataFrame
        """
        print_section("MONTE CARLO SIMULATION SUMMARY")
        
        summary_data = []
        for model_name, results in self.simulation_results.items():
            summary_data.append({
                'Model': model_name,
                'R² Mean': results['r2_mean'],
                'R² Std': results['r2_std'],
                'R² Min': results['r2_min'],
                'R² Max': results['r2_max'],
                'MAE Mean': results['mae_mean'],
                'MAE Std': results['mae_std'],
                'RMSE Mean': results['rmse_mean'],
                'RMSE Std': results['rmse_std']
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        return summary_df
    
    def generate_report(self):
        """
        Generate a comprehensive Monte Carlo simulation report.
        
        Returns:
        --------
        report : str
            Simulation report text
        """
        print_section("GENERATING MONTE CARLO REPORT")
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("MONTE CARLO SIMULATION REPORT".center(80))
        report_lines.append("="*80)
        report_lines.append(f"\nNumber of Simulations: {self.n_simulations}")
        report_lines.append(f"Test Size: {self.test_size}")
        report_lines.append(f"\nGenerated on: {get_timestamp()}")
        report_lines.append("\n" + "="*80)
        
        # Summary table
        summary_df = self.create_summary_table()
        report_lines.append("\nSIMULATION SUMMARY:")
        report_lines.append("-"*80)
        report_lines.append(summary_df.to_string(index=False))
        
        # Interpretation
        report_lines.append("\n" + "="*80)
        report_lines.append("\nINTERPRETATION:")
        report_lines.append("-"*80)
        report_lines.append("\n- Mean values represent the average performance across all simulations")
        report_lines.append("- Standard deviation (Std) indicates the stability of the model")
        report_lines.append("- Lower Std values indicate more stable and reliable predictions")
        report_lines.append("- Min and Max values show the range of possible performance")
        
        report_lines.append("\n" + "="*80)
        report_lines.append("END OF REPORT".center(80))
        report_lines.append("="*80)
        
        report = "\n".join(report_lines)
        
        # Save report
        output_path = os.path.join(self.output_dir, 'reports', f'monte_carlo_report_{get_timestamp()}.txt')
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Monte Carlo report saved to: {output_path}")
        
        return report


if __name__ == "__main__":
    # Example usage
    print("Monte Carlo Simulation Module")
    print("This module provides Monte Carlo simulation functionality.")
    print("Import and use the MonteCarloSimulation class in your main script.")
