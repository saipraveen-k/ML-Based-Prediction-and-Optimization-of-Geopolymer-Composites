"""
Exploratory Data Analysis Module for UHPGC Compressive Strength Prediction
==========================================================================
This module performs comprehensive exploratory data analysis and generates
visualizations for understanding the dataset.

Key Features:
- Automatic numeric column detection
- Robust handling of mixed data types
- Professional-quality visualizations
- Research-paper ready heatmaps
- Error-safe statistical computations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for professional visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import print_section, print_separator, create_directory, get_timestamp


class EDAAnalyzer:
    """
    A class for performing exploratory data analysis on UHPGC data.
    
    This class automatically detects and handles numeric columns,
    ensuring robust statistical analysis even with mixed data types.
    """
    
    def __init__(self, df, target_column='Compressive_Strength_MPa', output_dir='outputs/graphs'):
        """
        Initialize the EDAAnalyzer.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataset to analyze
        target_column : str
            Name of the target column
        output_dir : str
            Directory to save visualizations
        """
        self.df = df
        self.target_column = target_column
        self.output_dir = output_dir
        create_directory(output_dir)
        
        # Automatically detect numeric columns
        self.numeric_df = self._get_numeric_data()
        self.numeric_columns = self.numeric_df.columns.tolist()
        
    def _get_numeric_data(self):
        """
        Extract only numeric columns from the DataFrame.
        
        This helper function ensures all statistical operations
        are performed only on numeric data, avoiding errors
        from categorical/string columns.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing only numeric columns
        """
        numeric_df = self.df.select_dtypes(include=[np.number])
        print(f"\nDetected {len(numeric_df.columns)} numeric columns out of {len(self.df.columns)} total columns")
        return numeric_df
    
    def dataset_info(self):
        """
        Print comprehensive dataset information.
        
        Returns:
        --------
        None
        """
        print_section("DATASET INFORMATION")
        
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Number of Rows: {self.df.shape[0]}")
        print(f"Number of Columns: {self.df.shape[1]}")
        
        print("\n" + "="*80)
        print("Column Names:")
        for i, col in enumerate(self.df.columns, 1):
            print(f"{i}. {col}")
        
        print("\n" + "="*80)
        print("Data Types:")
        print(self.df.dtypes)
        
        print("\n" + "="*80)
        print("Missing Values:")
        missing = self.df.isnull().sum()
        missing_df = pd.DataFrame(missing, columns=['Missing Count'])
        missing_df['Missing %'] = (missing_df['Missing Count'] / len(self.df)) * 100
        print(missing_df)
        
        print("\n" + "="*80)
        print("Duplicate Rows:")
        print(f"Number of duplicate rows: {self.df.duplicated().sum()}")
        
        print("\n" + "="*80)
        print("Memory Usage:")
        print(self.df.memory_usage(deep=True))
        
    def statistical_summary(self):
        """
        Generate and display statistical summary for numeric columns only.
        
        This method computes descriptive statistics, skewness, and kurtosis
        exclusively on numeric data to avoid type conversion errors.
        
        Returns:
        --------
        summary : pandas.DataFrame
            Statistical summary
        """
        print_section("STATISTICAL SUMMARY")
        
        # Use only numeric columns for statistical summary
        summary = self.numeric_df.describe()
        print(summary)
        
        # Additional statistics (only for numeric columns)
        print("\n" + "="*80)
        print("Additional Statistics:")
        
        # Skewness - measures asymmetry of the distribution
        skewness = self.numeric_df.skew()
        print(f"\nSkewness (measures distribution asymmetry):")
        print(skewness)
        
        # Kurtosis - measures tailedness of the distribution
        kurtosis = self.numeric_df.kurtosis()
        print(f"\nKurtosis (measures distribution tailedness):")
        print(kurtosis)
        
        return summary
    
    def correlation_analysis(self):
        """
        Perform correlation analysis on numeric columns.
        
        Returns:
        --------
        corr_matrix : pandas.DataFrame
            Correlation matrix
        """
        print_section("CORRELATION ANALYSIS")
        
        # Calculate correlation matrix (only for numeric columns)
        corr_matrix = self.numeric_df.corr()
        
        # Print correlation with target
        if self.target_column in corr_matrix.columns:
            print(f"\nCorrelation with {self.target_column}:")
            target_corr = corr_matrix[self.target_column].sort_values(ascending=False)
            print(target_corr)
        
        return corr_matrix
    
    def correlation_heatmap(self):
        """
        Generate a professional correlation heatmap for numeric columns.
        
        This function creates a research-paper quality heatmap with:
        - Upper triangle masking for cleaner visualization
        - Annotated correlation coefficients
        - Professional color scheme
        - Proper label rotation
        - High resolution output
        
        Returns:
        --------
        None
        """
        print_section("CORRELATION HEATMAP")
        
        # Get correlation matrix for numeric columns only
        corr_matrix = self.numeric_df.corr()
        
        # Create figure with appropriate size based on number of columns
        n_cols = len(corr_matrix.columns)
        figsize = (max(14, n_cols * 0.8), max(12, n_cols * 0.8))
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create mask for upper triangle (shows only lower triangle for clarity)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        
        # Generate professional heatmap
        sns.heatmap(
            corr_matrix,
            annot=True,           # Show correlation values
            cmap='RdBu_r',        # Red-Blue colormap (reversed)
            center=0,             # Center colormap at 0
            vmin=-1,              # Minimum value
            vmax=1,               # Maximum value
            square=False,         # Don't force square cells
            linewidths=0.5,       # Line width between cells
            linecolor='white',    # Line color
            cbar_kws={
                "shrink": 0.8,
                "label": "Correlation Coefficient"
            },
            fmt='.2f',            # Format to 2 decimal places
            annot_kws={
                "size": min(10, max(6, 12 - n_cols * 0.2)),
                "weight": "bold"
            },
            mask=mask,            # Mask upper triangle
            ax=ax
        )
        
        # Enhance title and labels
        ax.set_title(
            'Feature Correlation Heatmap',
            fontsize=18,
            fontweight='bold',
            pad=20
        )
        
        # Rotate x-axis labels for better readability
        plt.xticks(
            rotation=45,
            ha='right',
            fontsize=min(10, max(7, 12 - n_cols * 0.15))
        )
        
        # Keep y-axis labels horizontal
        plt.yticks(
            rotation=0,
            fontsize=min(10, max(7, 12 - n_cols * 0.15))
        )
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot with high resolution
        output_path = os.path.join(self.output_dir, f'correlation_heatmap_{get_timestamp()}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Correlation heatmap saved to: {output_path}")
        plt.close()
        
        print("Correlation heatmap generated successfully!")
    
    def plot_distributions(self):
        """
        Generate distribution plots for all numerical features.
        
        Returns:
        --------
        None
        """
        print_section("FEATURE DISTRIBUTIONS")
        
        for col in self.numeric_columns:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Histogram
            sns.histplot(self.df[col], kde=True, ax=axes[0], color='skyblue')
            axes[0].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
            axes[0].set_xlabel(col, fontsize=10)
            axes[0].set_ylabel('Frequency', fontsize=10)
            
            # Box plot
            sns.boxplot(y=self.df[col], ax=axes[1], color='lightcoral')
            axes[1].set_title(f'Box Plot of {col}', fontsize=12, fontweight='bold')
            axes[1].set_ylabel(col, fontsize=10)
            
            plt.tight_layout()
            
            # Save plot
            output_path = os.path.join(self.output_dir, f'distribution_{col}_{get_timestamp()}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Distribution plot for {col} saved")
            plt.close()
        
        print("All distribution plots saved successfully")
    
    def plot_target_distribution(self):
        """
        Generate distribution plot for target variable.
        
        Returns:
        --------
        None
        """
        print_section("TARGET DISTRIBUTION")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        sns.histplot(self.df[self.target_column], kde=True, ax=axes[0], color='darkgreen')
        axes[0].set_title(f'Distribution of {self.target_column}', fontsize=12, fontweight='bold')
        axes[0].set_xlabel(self.target_column, fontsize=10)
        axes[0].set_ylabel('Frequency', fontsize=10)
        
        # Box plot
        sns.boxplot(y=self.df[self.target_column], ax=axes[1], color='orange')
        axes[1].set_title(f'Box Plot of {self.target_column}', fontsize=12, fontweight='bold')
        axes[1].set_ylabel(self.target_column, fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_dir, f'target_distribution_{get_timestamp()}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Target distribution plot saved to: {output_path}")
        plt.close()
    
    def plot_scatter_matrix(self):
        """
        Generate scatter plot matrix for features.
        
        Returns:
        --------
        None
        """
        print_section("SCATTER PLOT MATRIX")
        
        # Limit to first 8 columns to avoid overcrowding
        plot_columns = self.numeric_columns[:8] if len(self.numeric_columns) > 8 else self.numeric_columns
        
        if len(self.numeric_columns) > 8:
            print(f"Showing scatter matrix for first 8 features")
        
        # Create pairplot
        sns.pairplot(self.df[plot_columns], diag_kind='kde', 
                     plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'k'},
                     corner=True)
        plt.suptitle('Scatter Plot Matrix', y=1.02, fontsize=16, fontweight='bold')
        
        # Save plot
        output_path = os.path.join(self.output_dir, f'scatter_matrix_{get_timestamp()}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Scatter matrix saved to: {output_path}")
        plt.close()
    
    def plot_feature_vs_target(self):
        """
        Generate scatter plots of each feature vs target.
        
        Returns:
        --------
        None
        """
        print_section("FEATURE VS TARGET PLOTS")
        
        # Get numeric columns excluding target
        feature_columns = [col for col in self.numeric_columns if col != self.target_column]
        
        n_cols = 3
        n_rows = (len(feature_columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for idx, col in enumerate(feature_columns):
            if idx < len(axes):
                axes[idx].scatter(self.df[col], self.df[self.target_column], 
                                alpha=0.6, s=50, edgecolor='k', color='steelblue')
                axes[idx].set_xlabel(col, fontsize=10)
                axes[idx].set_ylabel(self.target_column, fontsize=10)
                axes[idx].set_title(f'{col} vs {self.target_column}', fontsize=11, fontweight='bold')
                axes[idx].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(feature_columns), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_dir, f'feature_vs_target_{get_timestamp()}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Feature vs target plots saved to: {output_path}")
        plt.close()
    
    def plot_boxplots(self):
        """
        Generate boxplots for all numerical features.
        
        Returns:
        --------
        None
        """
        print_section("BOXPLOTS")
        
        fig, axes = plt.subplots(1, figsize=(14, 8))
        self.numeric_df.boxplot(ax=axes, rot=45)
        axes.set_title('Boxplot of All Features', fontsize=14, fontweight='bold')
        axes.set_ylabel('Values', fontsize=12)
        axes.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_dir, f'boxplots_all_{get_timestamp()}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Boxplots saved to: {output_path}")
        plt.close()
    
    def generate_all_plots(self):
        """
        Generate all EDA visualizations.
        
        This comprehensive function generates all exploratory data analysis
        plots including statistical summaries, distributions, correlations,
        and professional heatmaps.
        
        Returns:
        --------
        None
        """
        print_section("GENERATING ALL EDA VISUALIZATIONS")
        
        # Dataset information
        self.dataset_info()
        
        # Statistical summary
        self.statistical_summary()
        
        # Correlation analysis (prints correlation values)
        self.correlation_analysis()
        
        # Professional correlation heatmap
        self.correlation_heatmap()
        
        # Feature distributions
        self.plot_distributions()
        
        # Target distribution
        self.plot_target_distribution()
        
        # Scatter matrix
        self.plot_scatter_matrix()
        
        # Feature vs target plots
        self.plot_feature_vs_target()
        
        # Boxplots
        self.plot_boxplots()
        
        print_section("EDA VISUALIZATIONS COMPLETED")
        print(f"All plots saved to: {self.output_dir}")


if __name__ == "__main__":
    # Example usage
    print("Exploratory Data Analysis Module")
    print("This module provides EDA functionality.")
    print("Import and use the EDAAnalyzer class in your main script.")