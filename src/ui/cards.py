"""
UI Cards, Custom Theme Injection, and Product Layout Components
===============================================================
Injects modern AI SaaS styling, top headers, hero sections, horizontal 
intelligence strips, visual workflow pipelines, and result cards.
"""

import streamlit as st


def inject_custom_css():
    """Inject modern, clean, sustainable engineering product styled CSS."""
    st.markdown("""
    <style>
        /* Modern Font Imports & Custom Color Tokens */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

        :root {
            --primary-dark: #064e3b;
            --primary-emerald: #059669;
            --accent-teal: #0d9488;
            --accent-mint: #10b981;
            --supporting-blue: #2563eb;
            --bg-page: #f8fafc;
            --bg-card: #ffffff;
            --border-subtle: rgba(15, 23, 42, 0.08);
            --border-hover: rgba(16, 185, 129, 0.3);
            --text-main: #0f172a;
            --text-muted: #64748b;
            --text-light: #94a3b8;
        }

        /* Global Font & Resets */
        html, body, [class*="css"], .stApp {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
            background-color: var(--bg-page) !important;
            color: var(--text-main) !important;
        }

        /* Container Padding */
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 3rem !important;
            max-width: 1250px !important;
        }

        /* Hide Streamlit Header & Footer Branding */
        header[data-testid="stHeader"] {
            background: transparent !important;
        }
        footer {
            visibility: hidden;
        }

        /* Top Bar System Status Header */
        .top-app-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.85rem 1.5rem;
            background: #ffffff;
            border: 1px solid var(--border-subtle);
            border-radius: 14px;
            margin-bottom: 1.8rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
        }
        .top-app-brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        .top-app-title {
            font-size: 0.95rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            color: var(--primary-dark);
            text-transform: uppercase;
        }
        .top-app-subtitle {
            font-size: 0.8rem;
            color: var(--text-muted);
            font-weight: 500;
        }
        .status-pill-online {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: #ecfdf5;
            color: #047857;
            font-weight: 700;
            font-size: 0.75rem;
            padding: 0.3rem 0.75rem;
            border-radius: 9999px;
            border: 1px solid #a7f3d0;
            letter-spacing: 0.04em;
        }
        .status-pulse-dot {
            width: 7px;
            height: 7px;
            background-color: #10b981;
            border-radius: 50%;
            display: inline-block;
            box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2);
        }

        /* Hero Container Styling */
        .product-hero-container {
            background: linear-gradient(135deg, #04392b 0%, #064e3b 60%, #0d9488 100%);
            border-radius: 20px;
            padding: 2.8rem 2.5rem;
            color: #ffffff;
            box-shadow: 0 20px 25px -5px rgba(6, 78, 59, 0.15), 0 10px 10px -5px rgba(6, 78, 59, 0.04);
            margin-bottom: 2rem;
            position: relative;
            overflow: hidden;
        }
        .hero-eyebrow {
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #6ee7b7;
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .hero-headline {
            font-size: 2.5rem;
            font-weight: 800;
            line-height: 1.15;
            letter-spacing: -0.03em;
            margin-bottom: 1rem;
            color: #ffffff;
        }
        .hero-description {
            font-size: 1.05rem;
            line-height: 1.6;
            color: #d1fae5;
            max-width: 620px;
            margin-bottom: 1.8rem;
            font-weight: 400;
        }

        /* Horizontal Intelligence Metric Strip */
        .intel-strip {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: #ffffff;
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            padding: 1.25rem 2rem;
            margin-bottom: 2.2rem;
            box-shadow: 0 2px 8px -2px rgba(0, 0, 0, 0.04);
        }
        .intel-item {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            flex: 1;
            padding: 0 1rem;
        }
        .intel-item:not(:last-child) {
            border-right: 1px solid #f1f5f9;
        }
        .intel-item:first-child {
            padding-left: 0;
        }
        .intel-item:last-child {
            padding-right: 0;
        }
        .intel-val {
            font-size: 1.6rem;
            font-weight: 800;
            color: var(--text-main);
            letter-spacing: -0.02em;
            line-height: 1.2;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }
        .intel-lbl {
            font-size: 0.76rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 0.2rem;
        }

        /* Product Story Horizontal Visual Flow */
        .workflow-story-card {
            background: #ffffff;
            border: 1px solid var(--border-subtle);
            border-radius: 18px;
            padding: 1.75rem 2rem;
            margin-bottom: 2.2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        }
        .story-section-title {
            font-size: 0.82rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--primary-emerald);
            margin-bottom: 1.25rem;
        }
        .story-flow-grid {
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 0.75rem;
            align-items: stretch;
        }
        .story-step-box {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1rem 0.85rem;
            transition: all 0.2s ease;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .story-step-box:hover {
            border-color: var(--primary-emerald);
            background: #ffffff;
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.08);
            transform: translateY(-2px);
        }
        .step-num {
            font-size: 0.7rem;
            font-weight: 800;
            color: var(--primary-emerald);
            background: #ecfdf5;
            padding: 0.15rem 0.5rem;
            border-radius: 6px;
            display: inline-block;
            width: fit-content;
            margin-bottom: 0.5rem;
        }
        .step-title {
            font-size: 0.85rem;
            font-weight: 700;
            color: var(--text-main);
            margin-bottom: 0.35rem;
        }
        .step-desc {
            font-size: 0.75rem;
            color: var(--text-muted);
            line-height: 1.4;
        }

        /* Generic Section Card Container */
        .subtle-card {
            background: #ffffff;
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            padding: 1.6rem 1.8rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.02);
            transition: border-color 0.2s ease;
        }
        .subtle-card:hover {
            border-color: rgba(15, 23, 42, 0.15);
        }
        .card-header-sm {
            font-size: 0.95rem;
            font-weight: 700;
            color: var(--text-main);
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Prediction Result Box */
        .result-hero-box {
            background: linear-gradient(135deg, #064e3b 0%, #04392b 100%);
            border-radius: 18px;
            padding: 2.2rem 2rem;
            color: #ffffff;
            text-align: center;
            box-shadow: 0 12px 24px -4px rgba(6, 78, 59, 0.2);
            margin: 1.5rem 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .result-hero-label {
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #a7f3d0;
            margin-bottom: 0.6rem;
        }
        .result-hero-value {
            font-size: 3.8rem;
            font-weight: 900;
            line-height: 1.05;
            color: #ffffff;
            letter-spacing: -0.03em;
            margin-bottom: 0.5rem;
        }
        .result-hero-unit {
            font-size: 1.6rem;
            font-weight: 600;
            color: #6ee7b7;
            margin-left: 0.2rem;
        }
        .result-hero-badge {
            display: inline-block;
            background: rgba(255, 255, 255, 0.14);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.25);
            color: #ffffff;
            padding: 0.35rem 1.2rem;
            border-radius: 9999px;
            font-size: 0.88rem;
            font-weight: 600;
            margin-top: 0.4rem;
        }
        .result-hero-subtext {
            font-size: 0.85rem;
            color: #d1fae5;
            margin-top: 0.8rem;
        }

        /* Streamlit Button Overrides */
        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.7rem 1.8rem !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
            letter-spacing: 0.02em !important;
            box-shadow: 0 4px 12px rgba(5, 150, 105, 0.25) !important;
            transition: all 0.2s ease !important;
        }
        div.stButton > button[kind="primary"]:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 16px rgba(5, 150, 105, 0.35) !important;
            background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        }

        div.stButton > button[kind="secondary"] {
            background: #ffffff !important;
            color: var(--text-main) !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 12px !important;
            padding: 0.65rem 1.4rem !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            transition: all 0.2s ease !important;
        }
        div.stButton > button[kind="secondary"]:hover {
            border-color: var(--primary-emerald) !important;
            color: var(--primary-emerald) !important;
            background: #f8fafc !important;
        }

        /* Streamlit Tabs Customization */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem !important;
            border-bottom: 1px solid #e2e8f0 !important;
            padding-bottom: 0.2rem !important;
        }
        .stTabs [data-baseweb="tab"] {
            font-weight: 600 !important;
            font-size: 0.88rem !important;
            color: var(--text-muted) !important;
            border-radius: 8px 8px 0 0 !important;
            padding: 0.5rem 1.1rem !important;
            background: transparent !important;
        }
        .stTabs [aria-selected="true"] {
            color: var(--primary-dark) !important;
            border-bottom: 3px solid var(--primary-emerald) !important;
            background: transparent !important;
        }

        /* Input Controls Clean Styling */
        div[data-baseweb="input"] {
            border-radius: 10px !important;
            border-color: #cbd5e1 !important;
            background-color: #ffffff !important;
        }
        div[data-baseweb="input"]:focus-within {
            border-color: var(--primary-emerald) !important;
            box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2) !important;
        }

        /* Streamlit Sidebar Clean Quiet Styling */
        section[data-testid="stSidebar"] {
            background-color: #ffffff !important;
            border-right: 1px solid var(--border-subtle) !important;
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 2rem !important;
        }

        /* Custom Radio Buttons Pill Styling */
        div[data-testid="stRadio"] > label {
            font-weight: 700 !important;
            font-size: 0.85rem !important;
            color: var(--text-muted) !important;
            text-transform: uppercase !alignment;
            letter-spacing: 0.05em !important;
        }
    </style>
    """, unsafe_allow_html=True)


def render_page_header(title, subtitle=None):
    """Render standardized top header bar across all subpages."""
    st.markdown(f"""
    <div class="top-app-bar">
        <div class="top-app-brand">
            <div>
                <div class="top-app-title">🌿 SUSTAINABLE GEOPOLYMER AI</div>
                <div class="top-app-subtitle">{subtitle if subtitle else 'Prediction & intelligence platform for next-generation sustainable concrete materials.'}</div>
            </div>
        </div>
        <div>
            <span class="status-pill-online">
                <span class="status-pulse-dot"></span> AI ENGINE ONLINE
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_home_hero_section():
    """Render the main hero section on the Home / Overview page."""
    col_left, col_right = st.columns([1.35, 1.0])
    
    with col_left:
        st.markdown("""
        <div class="product-hero-container">
            <div class="hero-eyebrow">🌱 AI FOR SUSTAINABLE MATERIALS</div>
            <div class="hero-headline">Design stronger.<br>Build greener.</div>
            <div class="hero-description">
                Machine learning predicts the compressive strength of geopolymer composites 
                synthesized from agricultural and industrial waste materials, optimizing binder 
                performance for low-carbon engineering.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_right:
        st.markdown("""
        <div style="background: #ffffff; border: 1px solid rgba(15, 23, 42, 0.08); border-radius: 20px; padding: 1.8rem; height: 100%; display: flex; flex-direction: column; justify-content: center; box-shadow: 0 4px 16px -2px rgba(0,0,0,0.03);">
            <div style="font-size: 0.78rem; font-weight: 800; letter-spacing: 0.08em; color: #059669; text-transform: uppercase; margin-bottom: 1rem;">
                MATERIAL × NEURAL ARCHITECTURE
            </div>
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 14px; padding: 1.25rem; margin-bottom: 1rem;">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.8rem;">
                    <span style="font-weight: 700; font-size: 0.88rem; color: #0f172a;">🌾 Agricultural Waste</span>
                    <span style="font-size: 0.78rem; color: #059669; font-weight: 600; background: #ecfdf5; padding: 0.2rem 0.5rem; border-radius: 6px;">RHA + POFA</span>
                </div>
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.8rem;">
                    <span style="font-weight: 700; font-size: 0.88rem; color: #0f172a;">🏭 Industrial Byproducts</span>
                    <span style="font-size: 0.78rem; color: #0284c7; font-weight: 600; background: #f0f9ff; padding: 0.2rem 0.5rem; border-radius: 6px;">GGBS + Silica Fume</span>
                </div>
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <span style="font-weight: 700; font-size: 0.88rem; color: #0f172a;">⚡ Machine Learning Engine</span>
                    <span style="font-size: 0.78rem; color: #7c3aed; font-weight: 600; background: #f5f3ff; padding: 0.2rem 0.5rem; border-radius: 6px;">XGBoost + SHAP</span>
                </div>
            </div>
            <div style="font-size: 0.8rem; color: #64748b; line-height: 1.45;">
                Instant multi-model predictions with cooperative game-theoretic feature attributions.
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_intel_kpi_strip(n_samples=812, n_features=21, cv_r2="0.980", mc_r2="0.976", primary_model="XGBoost"):
    """Render clean horizontal intelligence metric strip."""
    st.markdown(f"""
    <div class="intel-strip">
        <div class="intel-item">
            <div class="intel-val">{n_samples}</div>
            <div class="intel-lbl">Experimental Samples</div>
        </div>
        <div class="intel-item">
            <div class="intel-val">{n_features}</div>
            <div class="intel-lbl">Input Variables</div>
        </div>
        <div class="intel-item">
            <div class="intel-val">{cv_r2}</div>
            <div class="intel-lbl">10-Fold CV R²</div>
        </div>
        <div class="intel-item">
            <div class="intel-val">{mc_r2}</div>
            <div class="intel-lbl">Monte Carlo R²</div>
        </div>
        <div class="intel-item">
            <div class="intel-val" style="color: #059669;">{primary_model}</div>
            <div class="intel-lbl">Primary Model</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_project_story_flow():
    """Render horizontal visual story flow section."""
    st.markdown("""
    <div class="workflow-story-card">
        <div class="story-section-title">From waste to engineering insight</div>
        <div class="story-flow-grid">
            <div class="story-step-box">
                <div>
                    <span class="step-num">01</span>
                    <div class="step-title">Waste Materials</div>
                </div>
                <div class="step-desc">RHA, POFA, GGBS & agricultural fly ash.</div>
            </div>
            <div class="story-step-box">
                <div>
                    <span class="step-num">02</span>
                    <div class="step-title">Experimental Data</div>
                </div>
                <div class="step-desc">812 validated lab composite specimens.</div>
            </div>
            <div class="story-step-box">
                <div>
                    <span class="step-num">03</span>
                    <div class="step-title">Machine Learning</div>
                </div>
                <div class="step-desc">XGBoost, Random Forest & SVR algorithms.</div>
            </div>
            <div class="story-step-box">
                <div>
                    <span class="step-num">04</span>
                    <div class="step-title">Validation</div>
                </div>
                <div class="step-desc">10-Fold CV & 100-split Monte Carlo.</div>
            </div>
            <div class="story-step-box">
                <div>
                    <span class="step-num">05</span>
                    <div class="step-title">Explainability</div>
                </div>
                <div class="step-desc">SHAP game-theoretic feature impact.</div>
            </div>
            <div class="story-step-box">
                <div>
                    <span class="step-num">06</span>
                    <div class="step-title">Material Insight</div>
                </div>
                <div class="step-desc">Optimized eco-friendly concrete design.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_prediction_result_box(strength, model_name="XGBoost", spread=None):
    """Render polished result box for predicted compressive strength."""
    spread_sub = f"Across XGBoost, RF, SVR: ±{spread:.2f} MPa" if spread is not None else "Model estimate based on selected composition"
    st.markdown(f"""
    <div class="result-hero-box">
        <div class="result-hero-label">Predicted Compressive Strength</div>
        <div class="result-hero-value">{strength:.2f} <span class="result-hero-unit">MPa</span></div>
        <div>
            <span class="result-hero-badge">Selected Model: {model_name}</span>
        </div>
        <div class="result-hero-subtext">{spread_sub}</div>
    </div>
    """, unsafe_allow_html=True)
