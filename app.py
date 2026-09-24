"""
Sustainable Geopolymer AI Intelligence Platform - Main Application Entrypoint
=============================================================================
Professional AI Platform for predicting compressive strength of 
sustainable geopolymer concrete/composites, integrated with SHAP explainability,
Monte Carlo & 10-Fold CV validation, and material design insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ui.cards import inject_custom_css
from src.ui.components import render_sidebar_navigation, load_dataset_sample, render_footer
from src.ui.pages import (
    render_home_page,
    render_prediction_studio_page,
    render_data_explorer_page,
    render_model_intelligence_page,
    render_validation_page,
    render_explainable_ai_page,
    render_material_insights_page,
    render_sustainability_page,
    render_reports_page,
    render_about_page,
    load_evaluation_metrics
)

# Page configuration
st.set_page_config(
    page_title="Sustainable Geopolymer AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_cached_models():
    """Centralized loader for trained ML models and StandardScaler without retraining."""
    models_dir = 'models'
    scaler_path = 'models/scaler.pkl'
    
    models = {}
    scaler = None
    
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            st.error(f"Error loading scaler: {e}")
            
    model_files = {
        'XGBoost': 'xgboost.pkl',
        'Random Forest': 'random_forest.pkl',
        'SVR': 'svr.pkl'
    }
    
    for model_name, filename in model_files.items():
        model_path = os.path.join(models_dir, filename)
        if os.path.exists(model_path):
            try:
                models[model_name] = joblib.load(model_path)
            except Exception as e:
                st.warning(f"Could not load {model_name}: {e}")
                
    return models, scaler


def main():
    # Inject CSS custom theme styling
    inject_custom_css()
    
    # Load cached dataset, models, scaler, and metrics
    df = load_dataset_sample()
    models, scaler = load_cached_models()
    metrics_df = load_evaluation_metrics()
    
    # Render Sidebar Navigation with health validation
    selected_page, demo_mode = render_sidebar_navigation(df, models, scaler, metrics_df)
    
    if not models:
        st.warning("⚠️ **No trained models found in `models/` directory.**")
        return
        
    # Route pages
    if selected_page == "Overview":
        render_home_page(df, models)
    elif selected_page == "Prediction Studio":
        render_prediction_studio_page(df, models, scaler, demo_mode=demo_mode)
    elif selected_page == "Data Explorer":
        render_data_explorer_page(df)
    elif selected_page == "Model Intelligence":
        render_model_intelligence_page(df, models, scaler)
    elif selected_page == "Validation":
        render_validation_page()
    elif selected_page == "Explainable AI":
        render_explainable_ai_page(df, models, scaler)
    elif selected_page == "Material Insights":
        render_material_insights_page(df)
    elif selected_page == "Sustainability":
        render_sustainability_page(df)
    elif selected_page == "Reports":
        render_reports_page()
    elif selected_page == "About":
        render_about_page()
        
    # Render footer
    render_footer()


if __name__ == "__main__":
    main()
