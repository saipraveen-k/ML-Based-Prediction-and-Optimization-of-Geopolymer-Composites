"""
Sidebar Navigation, Demo Mix Controls, and System Health Components
=====================================================================
Renders visually quiet sidebar navigation, dataset demo presets, 
dynamic system status, and platform footers.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ui.data_validation import get_system_health_status

NAV_PAGES = [
    "Overview",
    "Prediction Studio",
    "Data Explorer",
    "Model Intelligence",
    "Validation",
    "Explainable AI",
    "Material Insights",
    "Sustainability",
    "Reports",
    "About"
]


def render_sidebar_navigation(df=None, models=None, scaler=None, metrics_df=None):
    """Render the refined sidebar navigation experience with dynamic system status."""
    st.sidebar.markdown("""
    <div style="padding: 0.5rem 0 1.25rem 0; border-bottom: 1px solid #f1f5f9; margin-bottom: 1rem;">
        <div style="font-size: 1.15rem; font-weight: 800; color: #064e3b; letter-spacing: -0.01em; display: flex; align-items: center; gap: 0.4rem;">
            <span>🌿</span> SUSTAINABLE
        </div>
        <div style="font-size: 1.15rem; font-weight: 800; color: #059669; letter-spacing: -0.01em; margin-bottom: 0.25rem;">
            GEOPOLYMER AI
        </div>
        <div style="font-size: 0.75rem; color: #64748b; font-weight: 500; line-height: 1.3;">
            AI-assisted sustainable material intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    selected_page = st.sidebar.radio(
        "Navigation",
        options=NAV_PAGES,
        index=0,
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    
    # Smart Demo Preset Quick Toggle
    demo_mode = st.sidebar.toggle("Use Demo Preset Mix", value=False, help="Load valid dataset sample mix parameters automatically into Prediction Studio.")
    
    # Dynamic System Health Validation Check
    health_status = get_system_health_status(df, models, scaler, metrics_df)
    status_label = health_status["overall_status"]
    dot_color = "#10b981" if status_label == "System Operational" else "#f59e0b"
    text_color = "#047857" if status_label == "System Operational" else "#b45309"
    
    st.sidebar.markdown(f"""
    <div style="margin-top: auto; padding-top: 2rem; border-top: 1px solid #f1f5f9;">
        <div style="display: flex; align-items: center; gap: 0.5rem; font-size: 0.8rem; font-weight: 600; color: {text_color};">
            <span style="width: 8px; height: 8px; background-color: {dot_color}; border-radius: 50%; display: inline-block;"></span>
            {status_label}
        </div>
        <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 0.4rem;">
            Platform v2.4 • XGBoost ML Engine
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    return selected_page, demo_mode


@st.cache_data
def load_dataset_sample():
    """Load dataset safely for data exploration and demo mode."""
    dataset_path = 'data/dataset.csv'
    if os.path.exists(dataset_path):
        try:
            df = pd.read_csv(dataset_path)
            return df
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return None
    return None


def get_demo_preset_mixes(df):
    """Extract representative valid dataset rows for preset selection."""
    if df is None:
        return {}
        
    presets = {}
    
    if 'Compressive_Strength_MPa' in df.columns:
        # High Strength Mix
        high_str_idx = df['Compressive_Strength_MPa'].idxmax()
        presets["High-Strength Mix"] = df.loc[high_str_idx].to_dict()
        
        # High Waste Eco Mix
        waste_score = (df.get('Fly_Ash_kg_m3', 0) + df.get('GGBS_kg_m3', 0) + df.get('RHA_kg_m3', 0) + df.get('POFA_kg_m3', 0))
        high_waste_idx = waste_score.idxmax() if isinstance(waste_score, pd.Series) else 0
        presets["High-Waste Mix"] = df.loc[high_waste_idx].to_dict()
        
        # Balanced Mix (Median Strength)
        median_idx = (df['Compressive_Strength_MPa'] - df['Compressive_Strength_MPa'].median()).abs().idxmin()
        presets["Balanced Mix"] = df.loc[median_idx].to_dict()
        
    return presets


def render_footer():
    """Render subtle application footer."""
    st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #94a3b8; font-size: 0.8rem; padding: 1.5rem 0; border-top: 1px solid #f1f5f9;">
        <span style="font-weight: 700; color: #475569;">Sustainable Geopolymer AI Intelligence Platform</span> &nbsp;•&nbsp; 
        Civil & Materials Engineering Research &nbsp;•&nbsp; 
        <a href="https://github.com/saipraveen-k/ML-Based-Prediction-and-Optimization-of-Geopolymer-Composites" target="_blank" style="color: #059669; text-decoration: none; font-weight: 600;">GitHub Repository</a>
    </div>
    """, unsafe_allow_html=True)
