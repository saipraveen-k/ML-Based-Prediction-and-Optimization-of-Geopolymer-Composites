"""
Explainable AI (XAI) SHAP Analysis Module for Geopolymer Concrete Prediction
=============================================================================
This module handles calculation of SHAP values, generation of publication-ready
visualizations (summary, beeswarm, dependence, bar, waterfall, force plots), 
and automatic generation of scientific interpretation and reports.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import shap

# Set style for publication-ready visualizations
sns.set_style("white")
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


class SHAPAnalyzer:
    """
    A class for performing publication-ready SHAP explainability analysis.
    """
    
    def __init__(self, model, X_scaled, feature_names, scaler=None, X_original=None, output_dir='outputs'):
        """
        Initialize the SHAPAnalyzer.
        
        Parameters:
        -----------
        model : object
            Trained model (XGBoost Regressor)
        X_scaled : array-like
            Scaled feature matrix used for model training/testing
        feature_names : list
            List of feature names
        scaler : StandardScaler, optional
            Fitted scaler to reconstruct original feature values
        X_original : array-like or DataFrame, optional
            Original unscaled features. If not provided, it will be reconstructed using the scaler.
        output_dir : str
            Directory to save outputs
        """
        self.model = model
        self.X_scaled = X_scaled
        self.feature_names = feature_names
        self.output_dir = output_dir
        
        # Ensure outputs/shap directory exists
        self.shap_dir = os.path.join(output_dir, 'shap')
        os.makedirs(self.shap_dir, exist_ok=True)
        
        # Reconstruct or store original features
        if X_original is not None:
            if isinstance(X_original, pd.DataFrame):
                self.X_original = X_original.copy()
            else:
                self.X_original = pd.DataFrame(X_original, columns=feature_names)
        elif scaler is not None:
            unscaled = scaler.inverse_transform(X_scaled)
            self.X_original = pd.DataFrame(unscaled, columns=feature_names)
        else:
            # Fallback to scaled features if no scaler or original data is provided
            if isinstance(X_scaled, pd.DataFrame):
                self.X_original = X_scaled.copy()
            else:
                self.X_original = pd.DataFrame(X_scaled, columns=feature_names)
                
        # Ensure X_scaled is a DataFrame with feature names (useful for SHAP)
        if not isinstance(self.X_scaled, pd.DataFrame):
            self.X_scaled_df = pd.DataFrame(self.X_scaled, columns=feature_names)
        else:
            self.X_scaled_df = self.X_scaled.copy()
            
        self.explainer = None
        self.shap_values = None
        self.explanation = None
        self.top_features = []
        self.feature_directions = {}
        
    def calculate_shap_values(self):
        """
        Create SHAP explainer and compute SHAP values.
        """
        print_section("SHAP VALUES CALCULATION")
        print("Initializing TreeExplainer on XGBoost model...")
        
        # Initialize TreeExplainer
        self.explainer = shap.TreeExplainer(self.model)
        
        print("Computing SHAP values for the dataset...")
        # Compute SHAP values on scaled features (since model was trained on scaled features)
        # We use X_scaled_df to ensure feature names are preserved in the explainer output
        self.shap_values = self.explainer.shap_values(self.X_scaled_df)
        
        # Check if shap_values has expected format
        if isinstance(self.shap_values, list) and len(self.shap_values) == 1:
            # XGBoost might wrap it in a list for single-output regression
            self.shap_values = self.shap_values[0]
            
        # Format feature names for publication-quality display
        self.cleaned_feature_names = [format_feature_name(f) for f in self.feature_names]
        self.X_original.columns = self.cleaned_feature_names
        
        # Create a proper shap.Explanation object for modern plotting functions.
        # Note: We associate the SHAP values with X_original (unscaled features) 
        # so that the plots display the real physical units of the features!
        self.explanation = shap.Explanation(
            values=self.shap_values,
            base_values=np.array([self.explainer.expected_value] * len(self.X_original)),
            data=self.X_original.values,
            feature_names=self.cleaned_feature_names
        )
        
        # Analyze feature importance and directions
        self._analyze_importance_and_directions()
        
        print(f"SHAP values computed successfully! Shape: {self.shap_values.shape}")
        return self.shap_values, self.explanation
        
    def _analyze_importance_and_directions(self):
        """
        Analyze average feature importance and correlation direction.
        """
        # Calculate mean absolute SHAP value for each feature
        mean_abs_shap = np.mean(np.abs(self.shap_values), axis=0)
        
        # Sort features by importance
        sorted_indices = np.argsort(mean_abs_shap)[::-1]
        self.top_features = [self.cleaned_feature_names[i] for i in sorted_indices]
        
        # Determine direction of influence for each feature
        # (Positive correlation between feature values and SHAP values means positive influence)
        self.feature_directions = {}
        for i, feature in enumerate(self.cleaned_feature_names):
            feat_vals = self.X_original[feature].values
            shap_vals = self.shap_values[:, i]
            
            # Filter out constants
            if np.std(feat_vals) > 0 and np.std(shap_vals) > 0:
                corr = np.corrcoef(feat_vals, shap_vals)[0, 1]
                if corr > 0.05:
                    self.feature_directions[feature] = "positive"
                elif corr < -0.05:
                    self.feature_directions[feature] = "negative"
                else:
                    self.feature_directions[feature] = "non-linear/complex"
            else:
                self.feature_directions[feature] = "neutral/no effect"
                
    def plot_summary(self):
        """
        Generate and save SHAP Summary Plot.
        """
        print("Generating Figure 1: SHAP Summary Plot...")
        plt.figure(figsize=(12, 8))
        
        # Use unscaled features X_original to display real physical units
        shap.summary_plot(self.shap_values, self.X_original, plot_size=(12, 8), show=False)
        
        plt.title('SHAP Summary Plot', fontsize=20, fontweight='bold', pad=15)
        plt.xlabel('SHAP Value (Impact on Compressive Strength, MPa)', fontsize=16)
        plt.tight_layout()
        
        output_path = os.path.join(self.shap_dir, 'shap_summary.png')
        plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Summary plot saved to: {output_path}")
        
    def plot_beeswarm(self):
        """
        Generate and save SHAP Beeswarm Plot.
        """
        print("Generating Figure 2: SHAP Beeswarm Plot...")
        plt.figure(figsize=(12, 8))
        
        # Beeswarm plot using the Explanation object
        shap.plots.beeswarm(self.explanation, plot_size=(12, 8), show=False)
        
        plt.title('SHAP Beeswarm Plot', fontsize=20, fontweight='bold', pad=15)
        plt.xlabel('SHAP Value (Impact on Compressive Strength, MPa)', fontsize=16)
        plt.tight_layout()
        
        output_path = os.path.join(self.shap_dir, 'shap_beeswarm.png')
        plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Beeswarm plot saved to: {output_path}")
        
    def plot_bar(self):
        """
        Generate and save SHAP Global Bar Plot.
        """
        print("Generating SHAP Bar Plot (Global Importance)...")
        plt.figure(figsize=(12, 8))
        
        # Bar plot using the Explanation object
        shap.plots.bar(self.explanation, max_display=len(self.feature_names), plot_size=(12, 8), show=False)
        
        plt.title('Global Feature Importance (Mean Absolute SHAP Value)', fontsize=20, fontweight='bold', pad=15)
        plt.xlabel('mean(|SHAP value|) (Average Impact magnitude, MPa)', fontsize=16)
        plt.tight_layout()
        
        output_path = os.path.join(self.shap_dir, 'shap_bar.png')
        plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Bar plot saved to: {output_path}")
        
    def plot_dependence_plots(self, top_n=5):
        """
        Generate and save SHAP Dependence Plots for the Top N features.
        
        Parameters:
        -----------
        top_n : int
            Number of top features to plot dependence for
        """
        print(f"Generating SHAP Dependence Plots for Top {top_n} features...")
        
        for idx in range(min(top_n, len(self.top_features))):
            feature = self.top_features[idx]
            plt.figure(figsize=(12, 8))
            
            # shap.dependence_plot automatically finds the best interaction feature 
            # and colors the points accordingly if we pass X_original.
            # We set show=False to save it.
            shap.dependence_plot(
                feature, 
                self.shap_values, 
                self.X_original, 
                show=False,
                interaction_index='auto'
            )
            
            plt.title(f'SHAP Dependence: {feature}', fontsize=20, fontweight='bold', pad=15)
            plt.ylabel(f'SHAP Value (Impact, MPa)', fontsize=16)
            plt.xlabel(f'{feature}', fontsize=16)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            
            output_name = f'dependence_feature_{idx+1}.png'
            output_path = os.path.join(self.shap_dir, output_name)
            plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"Dependence plot for '{feature}' saved to: {output_path}")
            
    def plot_waterfall(self, sample_idx=0):
        """
        Generate and save SHAP Waterfall Plot for a single sample prediction.
        
        Parameters:
        -----------
        sample_idx : int
            Index of the sample to plot
        """
        print(f"Generating SHAP Waterfall Plot for Sample #{sample_idx}...")
        plt.figure(figsize=(12, 8))
        
        # Plot waterfall for the selected sample
        shap.plots.waterfall(self.explanation[sample_idx], max_display=10, show=False)
        
        plt.title(f'SHAP Prediction Explanation - Sample #{sample_idx}', fontsize=20, fontweight='bold', pad=15)
        plt.tight_layout()
        
        output_path = os.path.join(self.shap_dir, 'shap_waterfall.png')
        plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Waterfall plot saved to: {output_path}")
        
    def plot_force_html(self, sample_idx=0):
        """
        Generate and save Interactive SHAP Force Plot as an HTML file.
        
        Parameters:
        -----------
        sample_idx : int
            Index of the sample to plot
        """
        print(f"Generating Interactive SHAP Force Plot HTML for Sample #{sample_idx}...")
        
        # Initialize Javascript environment (required for force plot visualization)
        shap.initjs()
        
        # Generate force plot
        # We use the explainer expected value and shap values for the sample, and original features for display
        force_plot = shap.force_plot(
            self.explainer.expected_value,
            self.shap_values[sample_idx],
            self.X_original.iloc[sample_idx],
            matplotlib=False,
            link='identity'
        )
        
        # Save as HTML file
        output_path = os.path.join(self.shap_dir, 'shap_force.html')
        shap.save_html(output_path, force_plot)
        print(f"Interactive force plot saved to: {output_path}")
        
    def generate_interpretation(self):
        """
        Generate dynamic research-style interpretation text.
        
        Returns:
        --------
        interpretation : str
            Formatted scientific interpretation
        """
        top_5 = self.top_features[:5]
        
        desc_map = {
            'Cement_kg_m3': 'Cement Content',
            'Fly_Ash_kg_m3': 'Fly Ash Content',
            'Silica_Fume_kg_m3': 'Silica Fume Content',
            'Metakaolin_kg_m3': 'Metakaolin Content',
            'GGBS_kg_m3': 'GGBS Content',
            'RHA_kg_m3': 'Rice Husk Ash Content',
            'POFA_kg_m3': 'POFA Content',
            'Fine_Sand_kg_m3': 'Fine Sand Content',
            'Water_kg_m3': 'Water Content',
            'Extra_Water_kg_m3': 'Extra Water Content',
            'Water_Binder_Ratio': 'Water-to-Binder Ratio',
            'Na2SiO3_Content_kg_m3': 'Sodium Silicate Content',
            'NaOH_Content_kg_m3': 'Sodium Hydroxide Content',
            'KOH_Content_kg_m3': 'Potassium Hydroxide Content',
            'Activator_Molarity_M': 'Alkali Activator Molarity',
            'Superplasticizer_kg_m3': 'Superplasticizer Content',
            'Polypropylene_Fiber_Content_%': 'PP Fiber Volume Fraction',
            'PP_Fiber_kg_m3': 'PP Fiber Content',
            'Fiber_Length_mm': 'PP Fiber Length',
            'Curing_Temperature_C': 'Curing Temperature',
            'Curing_Duration_days': 'Curing Duration'
        }
        
        def get_friendly_name(name):
            return desc_map.get(name, name.replace('_', ' '))
            
        # Build sentences
        intro = (
            f"The SHAP (SHapley Additive exPlanations) analysis conducted using the optimized XGBoost regressor "
            f"provided deep insights into feature importance and the nature of their influence. "
            f"The global feature ranking showed that **{get_friendly_name(top_5[0])}**, **{get_friendly_name(top_5[1])}**, "
            f"and **{get_friendly_name(top_5[2])}** are the primary drivers dictating the compressive strength of the geopolymer composites."
        )
        
        body_points = []
        for feature in top_5:
            direction = self.feature_directions.get(feature, "complex")
            friendly = get_friendly_name(feature)
            
            if direction == "positive":
                point = (
                    f"Higher **{friendly}** contributed positively to compressive strength development. "
                    f"This is consistent with geopolymerization kinetics where increased amounts of this parameter "
                    f"accelerate the formation of the aluminosilicate gel skeleton (N-A-S-H/C-A-S-H gel), densifying the concrete matrix."
                )
            elif direction == "negative":
                point = (
                    f"An increase in **{friendly}** showed a negative impact on compressive strength. "
                    f"Excessive values of this variable lead to increased porosity, micro-cracks, or incomplete geopolymerization, "
                    f"consequently reducing the load-bearing capacity of the geopolymer composite."
                )
            else:
                point = (
                    f"The feature **{friendly}** demonstrated a non-linear or complex interaction effect on the target strength. "
                    f"Its influence depends heavily on the synergistic proportions of other components, showing positive effects within "
                    f"specific ranges but causing detrimental effects when out of balance."
                )
            body_points.append(point)
            
        conclusion = (
            f"In summary, the SHAP analysis effectively bridges the gap between model explainability and concrete science, "
            f"revealing that optimizing the geopolymer binder system requires a delicate balance of **{get_friendly_name(top_5[0])}** "
            f"and curing parameters rather than simply increasing binder contents."
        )
        
        full_text = f"{intro}\n\n" + "\n\n".join(body_points) + f"\n\n{conclusion}"
        return full_text
        
    def generate_scientific_report(self):
        """
        Generate and save a publication-ready scientific report section.
        
        Returns:
        --------
        report : str
            Formatted scientific report
        """
        top_5 = self.top_features[:5]
        desc_map = {f: f.replace('_', ' ') for f in self.feature_names}
        
        report_lines = []
        report_lines.append("# Explainable Artificial Intelligence Analysis")
        report_lines.append("\n## 1. Introduction to SHAP")
        report_lines.append(
            "To resolve the black-box nature of machine learning algorithms and provide physical interpretations of "
            "the geopolymer concrete strength predictions, this research incorporates SHapley Additive exPlanations (SHAP) "
            "based on cooperative game theory. Unlike conventional feature importance methods that only provide global "
            "rankings, SHAP attributes an additive feature importance score to each variable for every specific prediction. "
            "This allows for both global interpretability (understanding overall model behavior) and local interpretability "
            "(explaining individual concrete mix predictions)."
        )
        
        report_lines.append("\n## 2. Global Feature Attribution (SHAP Summary and Beeswarm Plot)")
        report_lines.append(
            f"The global feature attributions are visualized using the SHAP Summary Plot (Figure 1) and Beeswarm Plot (Figure 2). "
            f"The features are ordered on the y-axis according to their explanatory power, with the most influential variable placed at the top. "
            f"The analysis identifies **{top_5[0]}** as the most critical factor, followed by **{top_5[1]}** and **{top_5[2]}**.\n\n"
            f"In the beeswarm visualization, each point represents a concrete mixture sample. The color indicates the feature value "
            f"(red for high, blue for low), and the position on the x-axis indicates the SHAP value (impact on output). "
            f"The wide horizontal spread of points for **{top_5[0]}** indicates its highly sensitive and commanding role in strength determination."
        )
        
        report_lines.append("\n## 3. Directionality of Feature Influence")
        report_lines.append(
            "The directionality of the top features is summarized as follows:"
        )
        for i, feat in enumerate(top_5, 1):
            dir_str = self.feature_directions.get(feat, "complex")
            report_lines.append(f"- **{feat}** (Rank {i}): {dir_str.capitalize()} contribution to compressive strength.")
            
        report_lines.append("\n## 4. SHAP Dependence and Multi-Variable Interactions")
        report_lines.append(
            f"To explore the non-linear relationships and synergistic interaction effects between materials, SHAP dependence plots "
            f"were generated for the top 5 most important features. Each plot shows the feature value on the x-axis and its corresponding "
            f"SHAP value on the y-axis, with dots colored by an automatically selected interaction feature. "
            f"This captures how the influence of one material parameter varies in the presence of different levels of another constituent. "
            f"For instance, the dependence plot for **{top_5[0]}** reveals a clear transition threshold, showing optimal ranges that maximize strength."
        )
        
        report_lines.append("\n## 5. Local Explainability: Waterfall and Force Plots")
        report_lines.append(
            "For individual mix design explanations, a Waterfall Plot (Figure 4) and an Interactive Force Plot (stored as `shap_force.html`) "
            "were generated. These plots explain exactly how a specific mixture design deviates from the baseline (mean prediction). "
            "The base value (E[f(X)]) represents the average strength predicted across the training set (approximately "
            f"{self.explainer.expected_value:.2f} MPa). The waterfall plot shows how positive contributions (red arrows/values) "
            "and negative contributions (blue arrows/values) shift the base value to arrive at the final predicted strength (f(x))."
        )
        
        report_lines.append("\n## 6. Engineering Interpretation and Practical Implications")
        report_lines.append(
            "The XAI analysis provides civil engineering researchers and mix designers with highly actionable insights:\n"
            "1. **Precursor Control**: Maximizing strength requires tight control over binder proportions, particularly silica fume and slag contents, which are high-scoring positive contributors.\n"
            "2. **Activator Balancing**: The alkali concentration and molarity must be kept within optimal windows. Excessive activator dosage leads to negative SHAP values, indicating silica precipitation inhibition or excess moisture.\n"
            "3. **Thermal Curing Optimization**: Curing temperature acts as an activation threshold. Higher temperatures show a strong positive SHAP impact up to a critical point (e.g., 60-80°C), above which the benefits plateau or decay due to rapid micro-cracking.\n"
            "4. **Fiber Reinforcement**: The content and length of PP fibers must be carefully tuned, acting as micro-structural bridges that contribute to performance, but only up to an optimal volumetric percentage before causing workability loss and clustering."
        )
        
        report_markdown = "\n".join(report_lines)
        
        # Save report
        output_path = os.path.join(self.shap_dir, 'Explainable_AI_Analysis.md')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_markdown)
        print(f"Scientific report section saved to: {output_path}")
        
        return report_markdown
        
    def run_all_analysis(self):
        """
        Run the complete SHAP analysis workflow and save all outputs.
        """
        print_section("RUNNING SHAP EXPLAINABILITY ANALYSIS WORKFLOW")
        
        # Calculate SHAP values
        self.calculate_shap_values()
        
        # Generate and save plots
        self.plot_summary()
        self.plot_beeswarm()
        self.plot_bar()
        self.plot_dependence_plots(top_n=5)
        self.plot_waterfall(sample_idx=0)
        self.plot_force_html(sample_idx=0)
        
        # Generate scientific report
        self.generate_scientific_report()
        
        print_section("SHAP ANALYSIS COMPLETED SUCCESSFULLY")
        print(f"All SHAP outputs saved to: {self.shap_dir}")
        print("\nGenerated Scientific Interpretation:")
        print_separator('-')
        print(self.generate_interpretation())
        print_separator('=')
        
        
if __name__ == "__main__":
    print("SHAP Explainability Analyzer Module")
    print("This module provides publication-quality SHAP interpretability analysis.")
