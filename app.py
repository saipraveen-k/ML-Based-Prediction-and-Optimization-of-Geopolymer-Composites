"""
UHPGC Compressive Strength Prediction - Streamlit Frontend
===========================================================
Professional web application for predicting compressive strength of
Ultra-High-Performance Geopolymer Concrete using trained ML models,
integrated with complete Explainable AI (SHAP) interpretability.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
import shap
import streamlit.components.v1 as components

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set page configuration
st.set_page_config(
    page_title="UHPGC Compressive Strength Predictor & XAI System",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set visualization style
sns.set_style("whitegrid")

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        padding: 0.5rem;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Load models and scaler
@st.cache_resource
def load_models():
    """Load trained models and scaler."""
    models_dir = 'models'
    scaler_path = 'models/scaler.pkl'
    
    models = {}
    scaler = None
    
    # Load scaler
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    
    # Load models
    model_files = {
        'Random Forest': 'random_forest.pkl',
        'SVR': 'svr.pkl',
        'XGBoost': 'xgboost.pkl'
    }
    
    for model_name, filename in model_files.items():
        model_path = os.path.join(models_dir, filename)
        if os.path.exists(model_path):
            models[model_name] = joblib.load(model_path)
    
    return models, scaler


# Feature names
FEATURE_NAMES = [
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


def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🏗️ UHPGC Compressive Strength Predictor & XAI</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.1rem; color: #555;">'
                'Machine Learning Explainability (SHAP) System for Geopolymer Concrete</p>', 
                unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar
    st.sidebar.markdown('<h2 class="sub-header">⚙️ Settings</h2>', unsafe_allow_html=True)
    
    # Load models
    models, scaler = load_models()
    
    if not models:
        st.markdown('<div class="warning-box">⚠️ <strong>Warning:</strong> No trained models found. '
                    'Please train the models first by running the notebook or scripts.</div>', 
                    unsafe_allow_html=True)
        st.info("To train models, run: `jupyter notebook notebooks/main.ipynb`")
        return
    
    # Model selection
    st.sidebar.markdown("### Model Selection")
    selected_model = st.sidebar.selectbox(
        "Choose ML Model for Predictions",
        options=list(models.keys()),
        index=2  # Default to XGBoost
    )
    
    st.sidebar.markdown("---")
    
    # Initialize tabs
    tab1, tab2 = st.tabs(["🔮 Predict Compressive Strength", "📊 Explainable AI (SHAP) Analysis"])
    
    # Collect inputs (needs to be available outside the tab so both tabs can access)
    input_values = {}
    
    with tab1:
        # Feature input section
        st.markdown('<h2 class="sub-header">📊 Input Material Composition</h2>', 
                    unsafe_allow_html=True)
        
        st.markdown('<div class="info-box">Enter the material composition values below. '
                    'All values should be in appropriate units (kg/m³ for materials, ratio for liquid/binder, °C for temperature).</div>', 
                    unsafe_allow_html=True)
        
        # Create input columns
        col1, col2, col3 = st.columns(3)
        
        # Column 1 inputs
        with col1:
            st.markdown("### Binders")
            input_values['Cement_kg_m3'] = st.number_input('Cement (kg/m³)', min_value=0.0, max_value=1000.0, value=780.0, step=10.0)
            input_values['Fly_Ash_kg_m3'] = st.number_input('Fly Ash (kg/m³)', min_value=0.0, max_value=500.0, value=120.0, step=10.0)
            input_values['Silica_Fume_kg_m3'] = st.number_input('Silica Fume (kg/m³)', min_value=0.0, max_value=200.0, value=50.0, step=5.0)
            input_values['Metakaolin_kg_m3'] = st.number_input('Metakaolin (kg/m³)', min_value=0.0, max_value=200.0, value=0.0, step=5.0)
            input_values['GGBS_kg_m3'] = st.number_input('GGBS (kg/m³)', min_value=0.0, max_value=500.0, value=0.0, step=10.0)
            input_values['RHA_kg_m3'] = st.number_input('Rice Husk Ash (kg/m³)', min_value=0.0, max_value=150.0, value=0.0, step=5.0)
            input_values['POFA_kg_m3'] = st.number_input('POFA (kg/m³)', min_value=0.0, max_value=150.0, value=0.0, step=5.0)
        
        # Column 2 inputs
        with col2:
            st.markdown("### Aggregates & Water")
            input_values['Fine_Sand_kg_m3'] = st.number_input('Fine Sand (kg/m³)', min_value=0.0, max_value=1000.0, value=0.0, step=10.0)
            input_values['Water_kg_m3'] = st.number_input('Water (kg/m³)', min_value=0.0, max_value=300.0, value=0.0, step=5.0)
            input_values['Extra_Water_kg_m3'] = st.number_input('Extra Water (kg/m³)', min_value=0.0, max_value=100.0, value=0.0, step=5.0)
            input_values['Water_Binder_Ratio'] = st.number_input('Water/Binder Ratio', min_value=0.0, max_value=1.0, value=0.0, step=0.01)
            input_values['Na2SiO3_Content_kg_m3'] = st.number_input('Na2SiO3 (kg/m³)', min_value=0.0, max_value=200.0, value=0.0, step=5.0)
            input_values['NaOH_Content_kg_m3'] = st.number_input('NaOH (kg/m³)', min_value=0.0, max_value=150.0, value=0.0, step=5.0)
            input_values['KOH_Content_kg_m3'] = st.number_input('KOH (kg/m³)', min_value=0.0, max_value=50.0, value=0.0, step=1.0)
        
        # Column 3 inputs
        with col3:
            st.markdown("### Additives & Curing")
            input_values['Activator_Molarity_M'] = st.number_input('Activator Molarity (M)', min_value=0.0, max_value=20.0, value=0.0, step=0.5)
            input_values['Superplasticizer_kg_m3'] = st.number_input('Superplasticizer (kg/m³)', min_value=0.0, max_value=50.0, value=0.0, step=0.5)
            input_values['Polypropylene_Fiber_Content_%'] = st.number_input('PP Fiber Content (%)', min_value=0.0, max_value=2.0, value=0.0, step=0.05)
            input_values['PP_Fiber_kg_m3'] = st.number_input('PP Fiber (kg/m³)', min_value=0.0, max_value=20.0, value=0.0, step=0.5)
            input_values['Fiber_Length_mm'] = st.number_input('Fiber Length (mm)', min_value=0.0, max_value=50.0, value=0.0, step=1.0)
            input_values['Curing_Temperature_C'] = st.number_input('Curing Temperature (°C)', min_value=20.0, max_value=100.0, value=25.0, step=1.0)
            input_values['Curing_Duration_days'] = st.number_input('Curing Duration (days)', min_value=1.0, max_value=90.0, value=28.0, step=1.0)
        
        st.markdown("---")
        
        # Predict button
        predict_button = st.button('🔮 Predict Compressive Strength', type='primary', use_container_width=True)
        
        if predict_button:
            # Make prediction
            with st.spinner('Processing prediction...'):
                # Convert to DataFrame
                input_df = pd.DataFrame([input_values])
                
                # Ensure correct column order
                input_df = input_df[FEATURE_NAMES]
                
                # Scale if scaler is available
                if scaler:
                    input_scaled = scaler.transform(input_df)
                else:
                    input_scaled = input_df.values
                
                # Get model
                model = models[selected_model]
                
                # Make prediction
                prediction = model.predict(input_scaled)[0]
            
            # Display results
            st.markdown('<h2 class="sub-header">🎯 Prediction Results</h2>', unsafe_allow_html=True)
            
            # Main prediction
            col_result1, col_result2 = st.columns([2, 1])
            
            with col_result1:
                st.markdown(f'<div class="success-box" style="text-align: center; font-size: 1.5rem;">'
                           f'<strong>Predicted Compressive Strength:</strong> {prediction:.2f} MPa</div>', 
                           unsafe_allow_html=True)
            
            with col_result2:
                st.metric(label="Model Used", value=selected_model)
            
            st.markdown("---")
            
            # Input summary
            st.markdown('<h2 class="sub-header">📋 Input Summary</h2>', unsafe_allow_html=True)
            
            summary_df = pd.DataFrame([input_values])
            summary_df = summary_df.T
            summary_df.columns = ['Value']
            st.dataframe(summary_df, use_container_width=True)
            
            st.markdown("---")
            
            # Compare with all models
            st.markdown('<h2 class="sub-header">🔄 Comparison with All Models</h2>', 
                        unsafe_allow_html=True)
            
            comparison_data = []
            for model_name, model in models.items():
                pred = model.predict(input_scaled)[0]
                comparison_data.append({
                    'Model': model_name,
                    'Predicted CS (MPa)': pred
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(comparison_df['Model'], comparison_df['Predicted CS (MPa)'], 
                         color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7, edgecolor='k')
            ax.set_ylabel('Compressive Strength (MPa)', fontsize=12)
            ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Feature importance section (if available)
        st.markdown('<h2 class="sub-header">📈 Feature Information</h2>', unsafe_allow_html=True)
        
        with st.expander("View Feature Descriptions"):
            st.markdown("""
            **Binders:**
            - **GGBS**: Ground Granulated Blast Furnace Slag - A supplementary cementitious material
            - **Silica Fume**: Ultra-fine pozzolanic material that improves strength and durability
            - **Fly Ash**: Coal combustion byproduct used as partial cement replacement
            - **Rice Husk Ash**: Agricultural waste ash with pozzolanic properties
            
            **Alkali Activators:**
            - **NaOH**: Sodium Hydroxide - Primary alkali activator
            - **Na2SiO3**: Sodium Silicate - Secondary alkali activator
            - **KOH**: Potassium Hydroxide - Alternative alkali activator
            - **Extra Water**: Additional water for workability adjustment
            
            **Fibers & Parameters:**
            - **Steel Fiber**: Steel fibers for reinforcement and ductility
            - **PP Fiber**: Polypropylene fibers for crack control
            - **Liquid/Binder Ratio**: Ratio of liquid to solid binder content
            - **Curing Temperature**: Temperature during curing process
            """)
            
    with tab2:
        st.markdown('<h2 class="sub-header">📊 Explainable AI (SHAP) Analysis</h2>', unsafe_allow_html=True)
        st.markdown(
            '<div class="info-box"><strong>What is SHAP?</strong> SHAP (SHapley Additive exPlanations) is a '
            'state-of-the-art cooperative game theory approach to explain the outputs of machine learning models. '
            'It allocates an attribution score (SHAP value) to each material component, explaining its positive '
            'or negative contribution to the concrete compressive strength. This satisfies Q1/Q2 journal requirements for XAI.</div>',
            unsafe_allow_html=True
        )
        
        # Inner tabs for Global vs Local Explainability
        shap_tab1, shap_tab2 = st.tabs(["🌍 Global Model Behavior", "🎯 Local Mix Explanation"])
        
        with shap_tab1:
            st.markdown("### Global Feature Attributions")
            st.markdown("These plots show the overall behavior of the XGBoost model across the entire dataset of 812 samples.")
            
            # Selector for global plots
            plot_choice = st.selectbox(
                "Select Global SHAP Plot to View",
                ["Beeswarm Distribution Plot", "Summary Dot Plot", "Feature Importance Bar Plot"]
            )
            
            # Paths to saved plots
            summary_path = 'outputs/shap/shap_summary.png'
            beeswarm_path = 'outputs/shap/shap_beeswarm.png'
            bar_path = 'outputs/shap/shap_bar.png'
            
            if plot_choice == "Beeswarm Distribution Plot" and os.path.exists(beeswarm_path):
                st.image(beeswarm_path, caption="Figure 1: SHAP Beeswarm Plot. Shows how feature values (red = high, blue = low) correspond to SHAP impact values.", use_column_width=True)
            elif plot_choice == "Summary Dot Plot" and os.path.exists(summary_path):
                st.image(summary_path, caption="Figure 2: SHAP Summary Plot. Displays feature importance alongside direction of influence.", use_column_width=True)
            elif plot_choice == "Feature Importance Bar Plot" and os.path.exists(bar_path):
                st.image(bar_path, caption="Figure 3: Global Feature Importance. Ranks features by their mean absolute SHAP value (impact magnitude).", use_column_width=True)
            else:
                st.warning("Pre-computed global SHAP plots not found. Please run the SHAP analyzer script `python src/run_shap.py` to generate these figures.")
            
            # Display pre-computed scientific report
            st.markdown("---")
            st.markdown("### 📝 Scientific Interpretation (Paper-Ready)")
            report_path = 'outputs/shap/Explainable_AI_Analysis.md'
            if os.path.exists(report_path):
                with open(report_path, 'r', encoding='utf-8') as f:
                    report_md = f.read()
                st.markdown(report_md)
            else:
                st.info("The detailed scientific report is available once the SHAP analysis pipeline is executed.")
                
        with shap_tab2:
            st.markdown("### 🔍 Mix-Specific Local Explanation")
            st.markdown(
                "This section explains how each material component in your current input composition (entered in the first tab) "
                "contributes to the predicted strength compared to the base value (average strength predicted across the dataset)."
            )
            
            # Reconstruct the current input vector as a scaled DataFrame
            input_df = pd.DataFrame([input_values])
            input_df = input_df[FEATURE_NAMES]
            
            # Get scaler and XGBoost model
            xgb_model = models.get('XGBoost')
            if xgb_model and scaler:
                try:
                    # Scale input
                    input_scaled = scaler.transform(input_df)
                    
                    # Compute SHAP explanation on the fly
                    with st.spinner("Computing local SHAP attributions..."):
                        explainer = shap.TreeExplainer(xgb_model)
                        local_shap = explainer.shap_values(input_scaled)
                        if isinstance(local_shap, list) and len(local_shap) == 1:
                            local_shap = local_shap[0]
                            
                        # Explanation object using original units
                        local_exp = shap.Explanation(
                            values=local_shap[0],
                            base_values=explainer.expected_value,
                            data=input_df.iloc[0].values,
                            feature_names=FEATURE_NAMES
                        )
                    
                    # Display Waterfall plot
                    st.markdown("#### 1. SHAP Waterfall Plot")
                    st.markdown(
                        "The waterfall plot shows how the base value (average prediction across the dataset, represented at the bottom) "
                        "is adjusted by each material to arrive at the final predicted strength (represented at the top). "
                        "Red blocks indicate positive contributions (increasing strength), and blue blocks indicate negative contributions."
                    )
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.plots.waterfall(local_exp, show=False)
                    plt.title("SHAP Waterfall: Material Contributions to Predicted Strength", fontsize=12, fontweight='bold', pad=10)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # Display Interactive Force Plot
                    st.markdown("---")
                    st.markdown("#### 2. SHAP Interactive Force Plot")
                    st.markdown(
                        "The interactive force plot visualizes how the positive (pushing higher) and negative (pushing lower) "
                        "forces of the materials balance to determine the predicted compressive strength."
                    )
                    
                    force_plot = shap.force_plot(
                        explainer.expected_value,
                        local_shap[0],
                        input_df.iloc[0],
                        matplotlib=False,
                        link='identity'
                    )
                    
                    # Convert to HTML string and render
                    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
                    components.html(shap_html, height=130)
                    
                except Exception as e:
                    st.error(f"Error generating local SHAP explanation: {e}")
                    st.info("Ensure that models are trained and compatible.")
            else:
                st.warning("XGBoost model and scaler must be loaded for local SHAP explainability.")

    st.markdown("---")
    
    # Footer
    st.markdown('<div style="text-align: center; color: #777; padding: 2rem;">'
                '<p><strong>UHPGC Compressive Strength Prediction System</strong></p>'
                '<p>Based on Research: "Investigation of machine learning models in predicting '
                'compressive strength for ultra-high-performance geopolymer concrete"</p>'
                '<p>Models: Random Forest | SVR | XGBoost | Explainable AI (SHAP)</p>'
                '</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
