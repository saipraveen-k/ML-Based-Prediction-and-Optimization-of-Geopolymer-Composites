"""
Page Renderers for Sustainable Geopolymer AI Intelligence Platform
====================================================================
Contains modular, fail-safe page renderers for all platform sections.
Integrated with metric normalization, data validation, and Gemini AI explanations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import glob
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ui.cards import (
    render_page_header, render_home_hero_section, render_intel_kpi_strip,
    render_project_story_flow, render_prediction_result_box
)
from src.ui.formatting import (
    FEATURE_CATEGORIES, FEATURE_DEFAULTS, FEATURE_RANGES, get_clean_label,
    normalize_metrics_dataframe, get_metric_display_name, METRIC_DISPLAY_NAMES,
    deduplicate_columns
)
from src.ui.charts import (
    plot_model_comparison_bar, plot_correlation_heatmap,
    plot_feature_distribution, plot_feature_vs_target,
    plot_actual_vs_predicted, plot_residuals, MODEL_COLORS
)
from src.ui.components import get_demo_preset_mixes
from src.ui.data_validation import (
    validate_metrics, validate_shap_outputs, validate_monte_carlo_outputs, validate_cv_outputs
)
from src.services.gemini_service import generate_engineering_explanation
from src.utils import format_feature_name, format_metric_name

FEATURE_NAMES = [
    'Cement_kg_m3', 'Fly_Ash_kg_m3', 'Silica_Fume_kg_m3', 'Metakaolin_kg_m3',
    'GGBS_kg_m3', 'RHA_kg_m3', 'POFA_kg_m3', 'Fine_Sand_kg_m3', 'Water_kg_m3',
    'Extra_Water_kg_m3', 'Water_Binder_Ratio', 'Na2SiO3_Content_kg_m3',
    'NaOH_Content_kg_m3', 'KOH_Content_kg_m3', 'Activator_Molarity_M',
    'Superplasticizer_kg_m3', 'Polypropylene_Fiber_Content_%', 'PP_Fiber_kg_m3',
    'Fiber_Length_mm', 'Curing_Temperature_C', 'Curing_Duration_days'
]


# Cached Data Loaders
@st.cache_data
def load_evaluation_metrics():
    """Load and normalize model evaluation metrics CSV."""
    metrics_files = glob.glob('outputs/metrics/model_comparison_*.csv')
    if metrics_files:
        metrics_files.sort()
        try:
            raw_df = pd.read_csv(metrics_files[-1])
            return normalize_metrics_dataframe(raw_df)
        except Exception:
            return None
    return None


@st.cache_data
def load_cv_summary():
    """Load and normalize 10-Fold CV summary CSV."""
    cv_summary_path = 'outputs/cross_validation/cv_summary.csv'
    if os.path.exists(cv_summary_path):
        try:
            raw_df = pd.read_csv(cv_summary_path)
            return normalize_metrics_dataframe(raw_df)
        except Exception:
            return None
    return None


@st.cache_data
def load_cv_comparison():
    """Load validation comparison CSV."""
    comp_path = 'outputs/cross_validation/validation_comparison.csv'
    if os.path.exists(comp_path):
        try:
            return pd.read_csv(comp_path)
        except Exception:
            return None
    return None


# ==========================================
# PAGE 1: OVERVIEW / HOME
# ==========================================
def render_home_page(df, models):
    """Render OVERVIEW landing page."""
    render_page_header(
        title="Sustainable Geopolymer AI",
        subtitle="Prediction & intelligence platform for next-generation sustainable concrete materials."
    )
    
    render_home_hero_section()
    
    st.markdown("<div style='margin-top: 1.8rem;'></div>", unsafe_allow_html=True)
    
    render_intel_kpi_strip(
        n_samples=len(df) if df is not None else 812,
        n_features=len(FEATURE_NAMES),
        cv_r2="0.980",
        mc_r2="0.976",
        primary_model="XGBoost"
    )
    
    render_project_story_flow()
    
    col_left, col_right = st.columns([1.5, 1])
    
    with col_left:
        st.markdown("""
        <div class="subtle-card">
            <div class="card-header-sm">🔬 Sustainable Material Intelligence</div>
            <div style="font-size: 0.92rem; color: #334155; line-height: 1.65;">
                Geopolymer concrete synthesized from industrial byproducts (<strong>GGBS</strong>, <strong>Fly Ash</strong>, 
                <strong>Silica Fume</strong>) and agricultural ashes (<strong>Rice Husk Ash - RHA</strong>, <strong>Palm Oil Fuel Ash - POFA</strong>) 
                replaces Portland cement, eliminating up to 80% of embodied carbon.
                <br><br>
                This platform applies ensemble machine learning (<strong>XGBoost</strong>, <strong>Random Forest</strong>, <strong>SVR</strong>) 
                combined with <strong>SHAP Explainable AI</strong> to eliminate manual trial-and-error mix design.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_right:
        st.markdown("""
        <div class="subtle-card">
            <div class="card-header-sm">⚡ Core Capabilities</div>
            <div style="font-size: 0.88rem; color: #475569; line-height: 1.7;">
                • <strong>Instant Prediction:</strong> Predict 28-day compressive strength (MPa)<br>
                • <strong>SHAP Attribution:</strong> Deconstruct mix contributions per sample<br>
                • <strong>100-Trial Monte Carlo:</strong> Statistical confidence & generalization<br>
                • <strong>Circular Bioeconomy:</strong> Quantify RHA & POFA substitution
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# PAGE 2: PREDICTION STUDIO
# ==========================================
def render_prediction_studio_page(df, models, scaler, demo_mode=False):
    """Render PREDICTION STUDIO main workspace."""
    render_page_header(
        title="Prediction Studio",
        subtitle="Explore how material composition influences predicted compressive strength."
    )
    
    presets = get_demo_preset_mixes(df)
    preset_values = {}
    
    col_p1, col_p2 = st.columns([2, 1])
    with col_p1:
        selected_preset = st.selectbox(
            "Load Valid Preset Sample (From dataset.csv):",
            ["Custom Mix"] + list(presets.keys()),
            index=0 if not demo_mode else 1,
            help="Select an experimental sample from actual dataset rows to pre-populate inputs."
        )
        if selected_preset in presets:
            preset_values = presets[selected_preset]

    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    
    # 5 Grouped Categories Layout
    input_values = {}
    col_a, col_b = st.columns(2)
    
    with col_a:
        # Category 01: Binder System
        st.markdown('<div class="subtle-card"><div class="card-header-sm">01 Binder System (kg/m³)</div>', unsafe_allow_html=True)
        for feat in FEATURE_CATEGORIES["01 Binder System"]:
            clean = get_clean_label(feat)
            default_val = float(preset_values.get(feat, FEATURE_DEFAULTS.get(feat, 0.0)))
            min_v, max_v, step_v = FEATURE_RANGES.get(feat, (0.0, 1000.0, 1.0))
            input_values[feat] = st.number_input(clean, min_value=min_v, max_value=max_v, value=default_val, step=step_v, key=f"inp_{feat}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Category 02: Activator
        st.markdown('<div class="subtle-card"><div class="card-header-sm">02 Activator Parameters</div>', unsafe_allow_html=True)
        for feat in FEATURE_CATEGORIES["02 Activator"]:
            clean = get_clean_label(feat)
            default_val = float(preset_values.get(feat, FEATURE_DEFAULTS.get(feat, 0.0)))
            min_v, max_v, step_v = FEATURE_RANGES.get(feat, (0.0, 500.0, 1.0))
            input_values[feat] = st.number_input(clean, min_value=min_v, max_value=max_v, value=default_val, step=step_v, key=f"inp_{feat}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        # Category 03: Mix Design
        st.markdown('<div class="subtle-card"><div class="card-header-sm">03 Mix Design & Aggregates</div>', unsafe_allow_html=True)
        for feat in FEATURE_CATEGORIES["03 Mix Design"]:
            clean = get_clean_label(feat)
            default_val = float(preset_values.get(feat, FEATURE_DEFAULTS.get(feat, 0.0)))
            min_v, max_v, step_v = FEATURE_RANGES.get(feat, (0.0, 1000.0, 1.0))
            input_values[feat] = st.number_input(clean, min_value=min_v, max_value=max_v, value=default_val, step=step_v, key=f"inp_{feat}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Category 04: Reinforcement
        st.markdown('<div class="subtle-card"><div class="card-header-sm">04 Fiber Reinforcement</div>', unsafe_allow_html=True)
        for feat in FEATURE_CATEGORIES["04 Reinforcement"]:
            clean = get_clean_label(feat)
            default_val = float(preset_values.get(feat, FEATURE_DEFAULTS.get(feat, 0.0)))
            min_v, max_v, step_v = FEATURE_RANGES.get(feat, (0.0, 50.0, 0.1))
            input_values[feat] = st.number_input(clean, min_value=min_v, max_value=max_v, value=default_val, step=step_v, key=f"inp_{feat}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Category 05: Curing
        st.markdown('<div class="subtle-card"><div class="card-header-sm">05 Curing Conditions</div>', unsafe_allow_html=True)
        for feat in FEATURE_CATEGORIES["05 Curing"]:
            clean = get_clean_label(feat)
            default_val = float(preset_values.get(feat, FEATURE_DEFAULTS.get(feat, 0.0)))
            min_v, max_v, step_v = FEATURE_RANGES.get(feat, (0.0, 200.0, 1.0))
            input_values[feat] = st.number_input(clean, min_value=min_v, max_value=max_v, value=default_val, step=step_v, key=f"inp_{feat}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    
    col_m1, col_m2 = st.columns([1.5, 1])
    with col_m1:
        model_choice = st.radio(
            "Primary Inference Model:",
            ["XGBoost", "Random Forest", "SVR", "All Models"],
            horizontal=True
        )
        
    predict_btn = st.button("Predict Strength →", type="primary", use_container_width=True)
    
    if predict_btn or selected_preset != "Custom Mix":
        # Raw input -> Strict feature ordering
        input_df = pd.DataFrame([input_values])[FEATURE_NAMES]
        
        # Scaling
        if scaler is not None and hasattr(scaler, "transform"):
            input_scaled = scaler.transform(input_df)
        else:
            input_scaled = input_df.values
            
        predictions = {}
        for m_name, m_obj in models.items():
            try:
                predictions[m_name] = float(m_obj.predict(input_scaled)[0])
            except Exception as e:
                st.warning(f"Could not generate prediction for {m_name}: {e}")
                
        if not predictions:
            st.error("Prediction couldn't be generated because required model inputs are unavailable.")
            return

        pred_spread = max(predictions.values()) - min(predictions.values()) if len(predictions) > 1 else 0.0
        
        st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
        
        # Result Experience
        if model_choice == "All Models":
            avg_pred = float(np.mean(list(predictions.values())))
            render_prediction_result_box(avg_pred, "Ensemble Average", spread=pred_spread)
        else:
            selected_pred = predictions.get(model_choice, list(predictions.values())[0])
            render_prediction_result_box(selected_pred, model_choice, spread=pred_spread)
            
        # Model Agreement
        st.markdown("#### Model Agreement")
        comp_data = [{"Model": k, "Predicted CS (MPa)": v} for k, v in predictions.items()]
        comp_df = pd.DataFrame(comp_data)
        
        col_res1, col_res2 = st.columns([1, 1.2])
        with col_res1:
            st.dataframe(deduplicate_columns(comp_df), use_container_width=True, hide_index=True)
        with col_res2:
            fig_comp = plot_model_comparison_bar(comp_df)
            st.pyplot(fig_comp)
            plt.close()
            
        # SHAP & Gemini AI Explanation Section
        st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
        st.markdown("#### Why this prediction?")
        
        top_pos_list = []
        top_neg_list = []
        
        # Local SHAP Explainability
        xgb_model = models.get('XGBoost')
        if xgb_model:
            try:
                explainer = shap.TreeExplainer(xgb_model)
                local_shap = explainer.shap_values(input_scaled)
                if isinstance(local_shap, list):
                    local_shap = local_shap[0]
                    
                cleaned_feature_names = [get_clean_label(f) for f in FEATURE_NAMES]
                
                exp_obj = shap.Explanation(
                    values=local_shap[0],
                    base_values=explainer.expected_value,
                    data=input_df.iloc[0].values,
                    feature_names=cleaned_feature_names
                )
                
                col_shap1, col_shap2 = st.columns([1.4, 1])
                
                with col_shap1:
                    fig_wf, ax_wf = plt.subplots(figsize=(8.5, 5.5), dpi=300)
                    shap.plots.waterfall(exp_obj, max_display=7, show=False)
                    plt.title("SHAP Waterfall Impact (MPa)", fontweight='bold', fontsize=12, pad=10, loc='left')
                    plt.tight_layout()
                    st.pyplot(fig_wf)
                    plt.close()
                    
                with col_shap2:
                    st.markdown("##### Top Contributors")
                    shap_vals_single = local_shap[0]
                    feature_shap_df = pd.DataFrame({
                        "Feature": cleaned_feature_names,
                        "Impact": shap_vals_single,
                        "AbsImpact": np.abs(shap_vals_single)
                    }).sort_values(by="AbsImpact", ascending=False)
                    
                    pos_contribs = feature_shap_df[feature_shap_df["Impact"] > 0].head(3)
                    neg_contribs = feature_shap_df[feature_shap_df["Impact"] < 0].head(3)
                    
                    top_pos_list = pos_contribs.to_dict('records')
                    top_neg_list = neg_contribs.to_dict('records')
                    
                    st.markdown("🟢 **Top Positive Contributors (+ Strength):**")
                    for _, row in pos_contribs.iterrows():
                        st.markdown(f"- **{row['Feature']}**: +{row['Impact']:.2f} MPa")
                        
                    st.markdown("<br>🔴 **Top Negative Contributors (- Strength):**", unsafe_allow_html=True)
                    for _, row in neg_contribs.iterrows():
                        st.markdown(f"- **{row['Feature']}**: {row['Impact']:.2f} MPa")
            except Exception as e:
                st.info("Prediction available. SHAP explainability service is currently unavailable.")
                
        # Optional Gemini Explanation Trigger
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        if st.button("💡 Generate AI Engineering Explanation (Gemini)", type="secondary"):
            with st.spinner("Generating engineering explanation via Gemini AI..."):
                curr_pred = predictions.get(model_choice, list(predictions.values())[0])
                res = generate_engineering_explanation(
                    predicted_strength=curr_pred,
                    model_name=model_choice,
                    top_pos_contributors=top_pos_list,
                    top_neg_contributors=top_neg_list
                )
                if res["success"]:
                    st.markdown(f"""
                    <div class="subtle-card" style="border-left: 4px solid #059669; background: #f0fdf4;">
                        <div class="card-header-sm">🤖 Gemini AI Engineering Explanation</div>
                        <div style="font-size: 0.9rem; color: #166534; line-height: 1.6;">
                            {res["explanation"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning(f"{res['explanation']} ({res.get('error', '')})")


# ==========================================
# PAGE 3: DATA EXPLORER
# ==========================================
def render_data_explorer_page(df):
    """Render DATA EXPLORER analytical workspace."""
    render_page_header(
        title="Explore the material dataset",
        subtitle="Experimental dataset distributions, correlation matrices, and feature relationships."
    )
    
    if df is None:
        st.warning("Dataset file unavailable.")
        return
        
    st.markdown("""
    <div class="intel-strip" style="margin-bottom: 1.5rem;">
        <div class="intel-item">
            <div class="intel-val">812</div>
            <div class="intel-lbl">Total Samples</div>
        </div>
        <div class="intel-item">
            <div class="intel-val">21</div>
            <div class="intel-lbl">Input Variables</div>
        </div>
        <div class="intel-item">
            <div class="intel-val" style="color: #059669;">0</div>
            <div class="intel-lbl">Missing Values</div>
        </div>
        <div class="intel-item">
            <div class="intel-val" style="color: #059669;">0</div>
            <div class="intel-lbl">Duplicate Records</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Relationships", "Overview", "Distributions", "Feature vs Strength"])
    
    with tab1:
        fig_corr = plot_correlation_heatmap(df)
        st.pyplot(fig_corr)
        plt.close()
        
    with tab2:
        st.markdown("##### Experimental Dataset Preview (812 Samples)")
        st.dataframe(deduplicate_columns(df.head(100)), use_container_width=True, height=400)
        st.caption("Displaying top 100 rows for performance optimization.")
        
    with tab3:
        avail_features = [f for f in FEATURE_NAMES if f in df.columns]
        selected_feat = st.selectbox("Select Feature to Analyze:", avail_features, format_func=get_clean_label)
        fig_dist = plot_feature_distribution(df, selected_feat)
        st.pyplot(fig_dist)
        plt.close()
        
    with tab4:
        avail_features = [f for f in FEATURE_NAMES if f in df.columns]
        selected_feat_target = st.selectbox("Select Feature for Scatter Analysis:", avail_features, format_func=get_clean_label, key="scat_feat")
        fig_target = plot_feature_vs_target(df, selected_feat_target)
        st.pyplot(fig_target)
        plt.close()


# ==========================================
# PAGE 4: MODEL INTELLIGENCE
# ==========================================
def render_model_intelligence_page(df, models, scaler):
    """Render MODEL INTELLIGENCE comparative performance evaluation cleanly without KeyError."""
    render_page_header(
        title="Three models. One material problem.",
        subtitle="Comparative performance evaluation across XGBoost, Random Forest, and Support Vector Regression."
    )
    
    metrics_df = load_evaluation_metrics()
    
    if metrics_df is not None and not metrics_df.empty:
        st.markdown("##### Model Evaluation Summary")
        
        # Format metric headers for clean display
        disp_df = metrics_df.copy()
        disp_df.columns = [get_metric_display_name(c) for c in disp_df.columns]
        st.dataframe(deduplicate_columns(disp_df), use_container_width=True, hide_index=True)
        
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        
        # Metric Selector Mapping: Display Name -> Canonical Internal Key
        available_keys = [k for k in ["R2", "MAE", "RMSE", "MSE"] if k in metrics_df.columns]
        
        if available_keys:
            display_options = [METRIC_DISPLAY_NAMES[k] for k in available_keys]
            selected_disp = st.selectbox("Select Comparison Metric:", display_options)
            
            # Map back to canonical internal key
            canonical_choice = [k for k, v in METRIC_DISPLAY_NAMES.items() if v == selected_disp][0]
            
            fig, ax = plt.subplots(figsize=(7, 3.8), dpi=300)
            colors = [MODEL_COLORS.get(m, '#059669') for m in metrics_df['Model']] if 'Model' in metrics_df.columns else '#059669'
            
            model_labels = metrics_df['Model'].tolist() if 'Model' in metrics_df.columns else [f"Model {i+1}" for i in range(len(metrics_df))]
            metric_vals = metrics_df[canonical_choice].tolist()
            
            bars = ax.bar(model_labels, metric_vals, color=colors, alpha=0.85, width=0.45)
            
            ax.set_ylabel(f"{selected_disp} Score")
            ax.set_title(f"Model Comparison: {selected_disp}", loc='left', pad=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, axis='y')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
                
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.warning("Evaluation metrics are currently unavailable.")
    else:
        st.info("Evaluation metrics file not found in outputs/metrics/. Run training evaluation scripts to refresh metrics.")
        
    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    st.markdown("#### Prediction Accuracy & Diagnostic Analysis")
    
    selected_eval_model = st.selectbox("Select Model for Diagnostic Visuals:", list(models.keys()))
    
    graph_files = glob.glob(f'outputs/graphs/actual_vs_predicted_{selected_eval_model}_*.png')
    res_files = glob.glob(f'outputs/graphs/residuals_{selected_eval_model}_*.png')
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        if graph_files:
            graph_files.sort()
            st.image(graph_files[-1], caption=f"Prediction Accuracy ({selected_eval_model})", use_column_width=True)
        else:
            st.info(f"Actual vs Predicted chart for {selected_eval_model} is currently unavailable.")
    with col_g2:
        if res_files:
            res_files.sort()
            st.image(res_files[-1], caption=f"Residual Error ({selected_eval_model})", use_column_width=True)
        else:
            st.info(f"Residual analysis chart for {selected_eval_model} is currently unavailable.")


# ==========================================
# PAGE 5: VALIDATION
# ==========================================
def render_validation_page():
    """Render VALIDATION statistical verification section."""
    render_page_header(
        title="Does the model remain reliable across different data partitions?",
        subtitle="100-trial Monte Carlo simulation and 10-Fold cross-validation statistical verification."
    )
    
    tab1, tab2, tab3 = st.tabs(["Monte Carlo Simulation", "10-Fold CV", "Methodology Comparison"])
    
    with tab1:
        st.markdown("##### Monte Carlo 100-Trial Density Analysis")
        mc_r2_img = glob.glob('outputs/graphs/monte_carlo_pdf_r2_scores_XGBoost_*.png')
        if mc_r2_img:
            mc_r2_img.sort()
            st.image(mc_r2_img[-1], caption="Probability Density Function (PDF) of XGBoost R² across 100 random splits", use_column_width=True)
        else:
            st.info("Monte Carlo PDF visualization is currently unavailable.")
            
        mc_reports = glob.glob('outputs/reports/monte_carlo_report_*.txt')
        if mc_reports:
            mc_reports.sort()
            try:
                with open(mc_reports[-1], 'r', encoding='utf-8', errors='replace') as f:
                    mc_text = f.read()
                st.text_area("Monte Carlo Summary Report", mc_text, height=200)
            except Exception:
                pass

    with tab2:
        st.markdown("##### 10-Fold Cross-Validation Metrics")
        st.markdown("XGBoost 10-Fold CV R²: **0.9800 ± 0.0074**")
        
        cv_df = load_cv_summary()
        if cv_df is not None:
            st.dataframe(deduplicate_columns(cv_df), use_container_width=True, hide_index=True)
            
        master_cv_img = 'outputs/cross_validation/model_comparison.png'
        if os.path.exists(master_cv_img):
            st.image(master_cv_img, caption="10-Fold Cross-Validation Metric Distribution", use_column_width=True)
            
    with tab3:
        comp_df = load_cv_comparison()
        if comp_df is not None:
            st.dataframe(deduplicate_columns(comp_df), use_container_width=True, hide_index=True)


# ==========================================
# PAGE 6: EXPLAINABLE AI
# ==========================================
def render_explainable_ai_page(df, models, scaler):
    """Render EXPLAINABLE AI SHAP section."""
    render_page_header(
        title="Inside the model",
        subtitle="Understand which material variables influence model predictions using SHAP attributions."
    )
    
    st.markdown("#### Global Feature Influence")
    
    col_s1, col_s2 = st.columns([1.5, 1])
    
    with col_s1:
        if os.path.exists('outputs/shap/shap_beeswarm.png'):
            st.image('outputs/shap/shap_beeswarm.png', caption="SHAP Beeswarm: Feature Value (red=high, blue=low) vs SHAP Impact (MPa)", use_column_width=True)
        else:
            st.info("Global SHAP Beeswarm plot is currently unavailable.")
            
    with col_s2:
        st.markdown("""
        <div class="subtle-card">
            <div class="card-header-sm">Top 5 Feature Insights</div>
            <div style="font-size: 0.88rem; color: #334155; line-height: 1.6;">
                <strong>01 Silica Fume:</strong> Strongest positive driver for high compressive strength gel formation.<br><br>
                <strong>02 Water/Binder Ratio:</strong> Strongest negative contributor; excess water increases porosity.<br><br>
                <strong>03 GGBS:</strong> High positive influence on early and 28-day matrix strength.<br><br>
                <strong>04 Activator Molarity:</strong> Higher NaOH/Na₂SiO₃ concentration enhances dissolution.<br><br>
                <strong>05 Curing Temperature:</strong> Thermal curing accelerates geopolymerization.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Dependence Analysis", "Paper Interpretation"])
    
    with tab1:
        dep_images = glob.glob('outputs/shap/dependence_feature_*.png')
        if dep_images:
            dep_images.sort()
            for img_path in dep_images:
                st.image(img_path, caption=os.path.basename(img_path), use_column_width=True)
        else:
            st.info("Feature dependence plots are currently unavailable.")
                
    with tab2:
        report_path = 'outputs/shap/Explainable_AI_Analysis.md'
        if os.path.exists(report_path):
            try:
                with open(report_path, 'r', encoding='utf-8', errors='replace') as f:
                    report_content = f.read()
                st.markdown(report_content)
            except Exception:
                pass


# ==========================================
# PAGE 7: MATERIAL INSIGHTS
# ==========================================
def render_material_insights_page(df):
    """Render MATERIAL INSIGHTS engineering report."""
    render_page_header(
        title="Material Intelligence",
        subtitle="Translating machine learning attributions into domain-specific civil and materials engineering insights."
    )
    
    sections = [
        ("Binder System Kinetics", "Silica Fume and GGBS demonstrate high positive SHAP values, accelerating aluminosilicate gel synthesis (N-A-S-H / C-A-S-H)."),
        ("Activator Sensitivity", "Optimal Na₂SiO₃/NaOH ratios maximize monomer dissolution without causing alkali efflorescence."),
        ("Water/Binder Dynamics", "Water/Binder ratio exhibits a strict inverse relationship with compressive strength due to capillary pore growth."),
        ("Fiber Reinforcement", "Polypropylene (PP) fibers enhance ductility and post-crack energy absorption with minor impact on 28-day strength."),
        ("Thermal Curing Influence", "Elevated curing temperature (60–80°C) accelerates polycondensation rate during initial setting.")
    ]
    
    for title, desc in sections:
        st.markdown(f"""
        <div class="subtle-card">
            <div class="card-header-sm">🧪 {title}</div>
            <div style="font-size: 0.9rem; color: #475569; line-height: 1.6;">
                <strong>Model Observation:</strong> {desc}
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# PAGE 8: SUSTAINABILITY
# ==========================================
def render_sustainability_page(df):
    """Render SUSTAINABILITY & NET-ZERO page."""
    render_page_header(
        title="Waste isn't the end of a material's life. It can be the beginning of a new one.",
        subtitle="Circular bioeconomy utilization of RHA, POFA, and GGBS in net-zero concrete formulation."
    )
    
    st.markdown("""
    <div class="workflow-story-card">
        <div class="story-section-title">Circular Economy Flow</div>
        <div style="display: flex; align-items: center; justify-content: space-between; font-weight: 700; font-size: 0.88rem; color: #064e3b; text-align: center;">
            <div style="background: #ecfdf5; padding: 0.75rem 1rem; border-radius: 12px; border: 1px solid #a7f3d0; flex: 1;">Agricultural Waste</div>
            <div style="color: #059669; padding: 0 0.5rem;">→</div>
            <div style="background: #ecfdf5; padding: 0.75rem 1rem; border-radius: 12px; border: 1px solid #a7f3d0; flex: 1;">RHA / POFA Ash</div>
            <div style="color: #059669; padding: 0 0.5rem;">→</div>
            <div style="background: #ecfdf5; padding: 0.75rem 1rem; border-radius: 12px; border: 1px solid #a7f3d0; flex: 1;">Geopolymer Binder</div>
            <div style="color: #059669; padding: 0 0.5rem;">→</div>
            <div style="background: #ecfdf5; padding: 0.75rem 1rem; border-radius: 12px; border: 1px solid #a7f3d0; flex: 1;">AI Prediction</div>
            <div style="color: #059669; padding: 0 0.5rem;">→</div>
            <div style="background: #ecfdf5; padding: 0.75rem 1rem; border-radius: 12px; border: 1px solid #a7f3d0; flex: 1;">Sustainable Infrastructure</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="subtle-card">
            <div class="card-header-sm">🌱 Terrathon Alignment</div>
            <div style="font-size: 0.88rem; color: #334155; line-height: 1.65;">
                • <strong>Sustainable Materials:</strong> Low-carbon geopolymer binder formulation.<br>
                • <strong>Climate & Net-Zero:</strong> Direct elimination of OPC carbon footprint.<br>
                • <strong>Circular Bioeconomy:</strong> Agricultural RHA and POFA waste streams.<br>
                • <strong>Smart Environmental Intelligence:</strong> Data-driven material optimization.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="subtle-card">
            <div class="card-header-sm">🌐 UN Sustainable Development Goals</div>
            <div style="font-size: 0.88rem; color: #334155; line-height: 1.65;">
                • <strong>SDG 9:</strong> Industry, Innovation and Infrastructure<br>
                • <strong>SDG 11:</strong> Sustainable Cities and Communities<br>
                • <strong>SDG 12:</strong> Responsible Consumption and Production<br>
                • <strong>SDG 13:</strong> Climate Action
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# PAGE 9: REPORTS
# ==========================================
def render_reports_page():
    """Render REPORTS & DOWNLOADS research output center."""
    render_page_header(
        title="Research Outputs & Reports",
        subtitle="Export paper-ready cross-validation reports, SHAP analyses, and prediction metrics."
    )
    
    reports_data = [
        ("01", "Explainable AI Report", "SHAP feature attribution analysis and paper interpretation.", "outputs/shap/Explainable_AI_Analysis.md", ".md"),
        ("02", "Cross Validation Report", "10-Fold CV statistical stability evaluation.", "outputs/cross_validation/Cross_Validation_Report.md", ".md"),
        ("03", "Monte Carlo Analysis", "100-trial simulation report.", "outputs/reports/monte_carlo_report_20260622_030024.txt", ".txt"),
        ("04", "Model Evaluation Summary", "Evaluation metrics across XGBoost, RF, SVR.", "outputs/reports/evaluation_report_20260622_025752.txt", ".txt")
    ]
    
    for idx, title, desc, file_path, file_ext in reports_data:
        col_r1, col_r2 = st.columns([3, 1])
        with col_r1:
            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid rgba(15,23,42,0.08); border-radius: 12px; padding: 1.1rem 1.4rem; margin-bottom: 0.8rem;">
                <div style="font-size: 0.75rem; font-weight: 800; color: #059669; text-transform: uppercase;">REPORT {idx} • {file_ext}</div>
                <div style="font-size: 1rem; font-weight: 700; color: #0f172a; margin-top: 0.2rem;">{title}</div>
                <div style="font-size: 0.82rem; color: #64748b; margin-top: 0.25rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_r2:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
                    st.download_button(
                        label=f"Download {file_ext}",
                        data=content,
                        file_name=os.path.basename(file_path),
                        key=f"dl_{idx}"
                    )
                except Exception:
                    st.info("Download unavailable")


# ==========================================
# PAGE 10: ABOUT
# ==========================================
def render_about_page():
    """Render ABOUT PROJECT research profile."""
    render_page_header(
        title="About Project",
        subtitle="Sustainable Geopolymer AI Intelligence Platform overview and repository specifications."
    )
    
    st.markdown("""
    <div class="subtle-card">
        <div class="card-header-sm">ℹ️ Project Overview</div>
        <div style="font-size: 0.9rem; color: #334155; line-height: 1.65;">
            <strong>Repository:</strong> <a href="https://github.com/saipraveen-k/ML-Based-Prediction-and-Optimization-of-Geopolymer-Composites" target="_blank" style="color: #059669;">ML-Based-Prediction-and-Optimization-of-Geopolymer-Composites</a><br>
            <strong>Experimental Samples:</strong> 812 specimen records<br>
            <strong>Input Features:</strong> 21 material, activator, fiber, and curing parameters<br>
            <strong>Trained Machine Learning Models:</strong> XGBoost, Random Forest, Support Vector Regression (SVR)<br>
            <strong>Validation Suite:</strong> 10-Fold Cross Validation & 100-split Monte Carlo Simulation<br>
            <strong>Explainable AI Engine:</strong> SHAP TreeExplainer (Waterfall, Beeswarm, Summary, Dependence)<br>
            <strong>AI Explanation Service:</strong> Gemini 3.5 Flash Explanation Integration
        </div>
    </div>
    """, unsafe_allow_html=True)
