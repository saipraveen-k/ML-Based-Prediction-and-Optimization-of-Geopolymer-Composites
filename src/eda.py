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
from src.utils import print_section, print_separator, create_directory, get_timestamp, format_feature_name


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
        corr_matrix = self.numeric_df.corr().copy()
        
        # Clean up column and index names for printing
        corr_matrix.columns = [format_feature_name(col) for col in corr_matrix.columns]
        corr_matrix.index = [format_feature_name(idx) for idx in corr_matrix.index]
        
        # Create figure with large figure size
        fig, ax = plt.subplots(figsize=(18, 14))
        
        # Create mask for upper triangle (shows only lower triangle for clarity)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        
        # Generate professional heatmap
        sns.heatmap(
            corr_matrix,
            annot=True,           # Show correlation values
            cmap='coolwarm',      # coolwarm colormap
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
                "size": 9
            },
            mask=mask,            # Mask upper triangle
            ax=ax
        )
        
        # Enhance title and labels
        ax.set_title(
            'Pearson Correlation Matrix of Geopolymer Composite Variables',
            fontsize=20,
            fontweight='bold',
            pad=20
        )
        
        # Rotate x-axis labels for better readability
        plt.xticks(
            rotation=45,
            ha='right',
            fontsize=13
        )
        
        # Keep y-axis labels horizontal
        plt.yticks(
            rotation=0,
            fontsize=13
        )
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot with high resolution
        output_path = os.path.join(self.output_dir, f'correlation_heatmap_{get_timestamp()}.png')
        plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
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
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            
            cleaned_col = format_feature_name(col)
            mean_val = self.df[col].mean()
            median_val = self.df[col].median()
            
            # Histogram
            sns.histplot(self.df[col], kde=True, bins=30, edgecolor='black', linewidth=1, alpha=0.8, ax=axes[0], color='skyblue')
            axes[0].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            axes[0].axvline(median_val, color='green', linestyle='-', linewidth=2, label=f'Median: {median_val:.2f}')
            axes[0].set_title(f'Distribution of {cleaned_col}', fontsize=16, fontweight='bold', pad=10)
            axes[0].set_xlabel(cleaned_col, fontsize=14)
            axes[0].set_ylabel('Frequency', fontsize=14)
            axes[0].legend(fontsize=11)
            
            # Box plot
            sns.boxplot(y=self.df[col], ax=axes[1], color='lightcoral', width=0.4, linewidth=1.5)
            axes[1].set_title(f'Box Plot of {cleaned_col}', fontsize=16, fontweight='bold', pad=10)
            axes[1].set_ylabel(cleaned_col, fontsize=14)
            
            plt.tight_layout()
            
            # Save plot
            output_path = os.path.join(self.output_dir, f'distribution_{col}_{get_timestamp()}.png')
            plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
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
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        cleaned_target = format_feature_name(self.target_column)
        mean_val = self.df[self.target_column].mean()
        median_val = self.df[self.target_column].median()
        
        # Histogram
        sns.histplot(self.df[self.target_column], kde=True, bins=30, edgecolor='black', linewidth=1, alpha=0.8, ax=axes[0], color='darkgreen')
        axes[0].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        axes[0].axvline(median_val, color='green', linestyle='-', linewidth=2, label=f'Median: {median_val:.2f}')
        axes[0].set_title(f'Distribution of {cleaned_target}', fontsize=16, fontweight='bold', pad=10)
        axes[0].set_xlabel(cleaned_target, fontsize=14)
        axes[0].set_ylabel('Frequency', fontsize=14)
        axes[0].legend(fontsize=11)
        
        # Box plot
        sns.boxplot(y=self.df[self.target_column], ax=axes[1], color='orange', width=0.4, linewidth=1.5)
        axes[1].set_title(f'Box Plot of {cleaned_target}', fontsize=16, fontweight='bold', pad=10)
        axes[1].set_ylabel(cleaned_target, fontsize=14)
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_dir, f'target_distribution_{get_timestamp()}.png')
        plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
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
        
        # Clean column names for plotting
        plot_df = self.df[plot_columns].copy()
        plot_df.columns = [format_feature_name(col) for col in plot_df.columns]
        
        # Create pairplot
        sns.pairplot(plot_df, diag_kind='kde', 
                     plot_kws={'alpha': 0.7, 's': 50, 'edgecolor': 'k'},
                     corner=True)
        plt.suptitle('Scatter Plot Matrix', y=1.02, fontsize=20, fontweight='bold')
        
        # Save plot
        output_path = os.path.join(self.output_dir, f'scatter_matrix_{get_timestamp()}.png')
        plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
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
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        cleaned_target = format_feature_name(self.target_column)
        
        for idx, col in enumerate(feature_columns):
            if idx < len(axes):
                cleaned_feature = format_feature_name(col)
                # Plot with regression line
                sns.regplot(
                    x=self.df[col], 
                    y=self.df[self.target_column], 
                    ax=axes[idx],
                    scatter_kws={'alpha': 0.7, 's': 80, 'edgecolor': 'k', 'color': 'steelblue'},
                    line_kws={'color': 'red', 'linewidth': 2}
                )
                axes[idx].set_xlabel(cleaned_feature, fontsize=14)
                axes[idx].set_ylabel(cleaned_target, fontsize=14)
                axes[idx].set_title(f'{cleaned_feature} vs {cleaned_target}', fontsize=16, fontweight='bold', pad=10)
                axes[idx].grid(True, linestyle='--', alpha=0.5)
        
        # Hide unused subplots
        for idx in range(len(feature_columns), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_dir, f'feature_vs_target_{get_timestamp()}.png')
        plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
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
        
        # Clean columns of numeric_df in a copy
        plot_df = self.numeric_df.copy()
        plot_df.columns = [format_feature_name(col) for col in plot_df.columns]
        
        fig, axes = plt.subplots(1, figsize=(18, 10))
        
        # Plot using seaborn boxplot for style
        sns.boxplot(data=plot_df, ax=axes, palette='Set2')
        plt.xticks(rotation=45, ha='right', fontsize=12)
        axes.set_title('Boxplot of All Features', fontsize=20, fontweight='bold', pad=15)
        axes.set_ylabel('Values', fontsize=16)
        axes.grid(True, linestyle='--', alpha=0.5, axis='y')
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_dir, f'boxplots_all_{get_timestamp()}.png')
        plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
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
    import pandas as pd
    data_path = 'data/dataset.csv'
    if os.path.exists(data_path):
        print(f"Loading dataset from {data_path}...")
        df = pd.read_csv(data_path)
        analyzer = EDAAnalyzer(df)
        analyzer.generate_all_plots()
        print("EDA generation complete!")
    else:
        print(f"Error: Dataset not found at {data_path}")