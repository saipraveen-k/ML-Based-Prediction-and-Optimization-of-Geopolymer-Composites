# Sustainable Geopolymer AI Intelligence Platform

## 🏗️ AI-Assisted Prediction & Optimization of Sustainable Geopolymer Composites

A state-of-the-art, end-to-end Machine Learning intelligence platform and research dashboard for predicting the compressive strength of sustainable geopolymer concrete/composites. Integrated with **SHAP Explainable AI (XAI)**, **10-Fold Cross-Validation**, and **100-Trial Monte Carlo Simulations**.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Research & Sustainability Motivation](#-research--sustainability-motivation)
- [Terrathon 2026 & UN SDG Alignment](#-terrathon-2026--un-sdg-alignment)
- [Application Architecture (10 Sections)](#-application-architecture-10-sections)
- [Dataset Specifications](#-dataset-specifications)
- [Machine Learning Pipeline & Models](#-machine-learning-pipeline--models)
- [Validation Framework (Monte Carlo & 10-Fold CV)](#-validation-framework-monte-carlo--10-fold-cv)
- [Explainable AI (SHAP)](#-explainable-ai-shap)
- [Installation Guide](#-installation-guide)
- [Running Instructions](#-running-instructions)
- [Project Structure](#-project-structure)
- [License & Reference](#-license--reference)

---

## 🎯 Project Overview

This project delivers a scientific Machine Learning framework to predict the **Compressive Strength (MPa)** of sustainable geopolymer composites utilizing industrial and agricultural waste materials such as **Rice Husk Ash (RHA)**, **Palm Oil Fuel Ash (POFA)**, **Ground Granulated Blast-Furnace Slag (GGBS)**, **Fly Ash**, and **Silica Fume**.

### Research Objective
To provide transparent, reproducible, and interpretable AI support for civil engineers and materials scientists designing low-carbon concrete alternatives to conventional Ordinary Portland Cement (OPC).

---

## 🌱 Research & Sustainability Motivation

Ordinary Portland Cement (OPC) production accounts for approximately **8% of global CO₂ emissions**. Geopolymer composites utilize alkali-activated aluminosilicate precursors derived from industrial byproducts and agricultural wastes:
- **GGBS & Fly Ash:** Reduce landfill burden and industrial waste.
- **RHA & POFA:** Support circular bioeconomy by repurposing agricultural biomass ash.
- **Cement Reduction:** Potential reduction of carbon footprint by up to **70–80%**.

---

## 🌐 Terrathon 2026 & UN SDG Alignment

### Terrathon 2026 Themes Supported
1. **Sustainable Materials & Clean Energy:** Low-carbon geopolymer binder formulations.
2. **Climate & Net-Zero Solutions:** Decarbonization pathway for construction materials.
3. **Circular Bioeconomy & Resource Recovery:** Biomass ash (RHA & POFA) recycling.
4. **Smart Environmental Intelligence:** AI-assisted experimental mix design.

### UN Sustainable Development Goals
- 🏗️ **SDG 9:** Industry, Innovation and Infrastructure
- 🏙️ **SDG 11:** Sustainable Cities and Communities
- 🔄 **SDG 12:** Responsible Consumption and Production
- 🌍 **SDG 13:** Climate Action

---

## 💻 Application Architecture (10 Sections)

The Streamlit web application (`app.py`) is structured into 10 dedicated sections:

1. **🏠 HOME / OVERVIEW:** Executive dashboard with KPI cards (812 Samples, 21 Features, Best Model XGBoost, 10-Fold R² 0.9800, Monte Carlo R² 0.9764) and visual research story pipeline banner.
2. **🔮 PREDICTION STUDIO:** Main feature allowing real-time single and ensemble mix predictions across XGBoost, Random Forest, SVR, or "All Models", complete with local SHAP waterfall plots and positive/negative driver attributions.
3. **📊 DATA EXPLORER:** Interactive dataset overview, Pearson correlation heatmap with upper triangle mask, univariate feature distribution plots (KDE + Boxplot), and bivariate scatter plots vs. target.
4. **🧪 MODEL LAB:** Side-by-side performance evaluation tables and comparative charts (R², MAE, RMSE, MSE) across models.
5. **🎯 VALIDATION CENTER:** Dedicated tabs for Monte Carlo (100 random splits), 10-Fold Cross-Validation, methodology comparison, and stability reports.
6. **💡 EXPLAINABLE AI (SHAP):** Global beeswarm, summary dot, feature importance bar, and dependence plots for journal publication requirements.
7. **🏗️ MATERIAL INSIGHTS:** Material kinetics, pozzolanic activity interpretation, alkali activator molarity sensitivity, and fiber reinforcement mechanics.
8. **🌱 SUSTAINABILITY DASHBOARD:** Circular bioeconomy impact metrics, waste material replacement pathways, and SDG alignments.
9. **📑 REPORTS & DOWNLOADS:** Centralized export center for scientific markdown reports, validation CSV summaries, and feature importance rankings.
10. **ℹ️ ABOUT PROJECT:** Research context, repository sitemap, architectural documentation, and metadata.

---

## 📊 Dataset Specifications

- **Total Samples:** 812 experimental mix formulations
- **Target Variable:** `Compressive_Strength_MPa` (Compressive Strength in MPa)
- **Input Variables (21 Features):**
  - **Binders:** Cement, Fly Ash, Silica Fume, Metakaolin, GGBS, Rice Husk Ash (RHA), POFA (all in kg/m³)
  - **Alkali Activators:** Na₂SiO₃, NaOH, KOH (kg/m³), Activator Molarity (M), Extra Water (kg/m³)
  - **Fibers:** Polypropylene Fiber Content (%), PP Fiber (kg/m³), Fiber Length (mm)
  - **Mix Parameters:** Fine Sand (kg/m³), Water (kg/m³), Water/Binder Ratio, Superplasticizer (kg/m³)
  - **Curing Conditions:** Curing Temperature (°C), Curing Duration (days)

---

## 🤖 Machine Learning Pipeline & Models

Three machine learning models were trained and tuned using scikit-learn and XGBoost:
1. **XGBoost Regressor (Best Performing):** R² = 0.9800 (10-Fold CV)
2. **Random Forest Regressor:** R² = 0.9650
3. **Support Vector Regression (SVR):** R² = 0.9120

### Preprocessing & Scaling Pipeline
- Standardized feature scaling (`StandardScaler`) fitted strictly on training data.
- Identical feature order and scaling applied during Streamlit inference.

---

## 🎯 Validation Framework (Monte Carlo & 10-Fold CV)

- **Monte Carlo Simulation:** 100 random train/test splits to generate Probability Density Functions (PDF) and Cumulative Density Functions (CDF) for model stability verification.
- **10-Fold Cross-Validation:** Guarantees every experimental sample is evaluated in testing subsets.

---

## 💡 Explainable AI (SHAP)

Uses cooperative game theory (`shap.TreeExplainer`) to break black-box ML predictions into interpretable additive feature contributions:
- **Global Beeswarm & Summary Plots:** Rank features by mean |SHAP value|.
- **Top Drivers:** Silica Fume (kg/m³), GGBS (kg/m³), and Fiber Content demonstrate positive attributions, while excessive Water/Binder Ratio shows negative attributions.
- **Local Waterfall & Force Plots:** Instant explainability for user-entered mix compositions.

---

## 🚀 Installation Guide

### Prerequisites
- Python 3.8 to 3.10
- Git

### Step 1: Clone Repository
```bash
git clone https://github.com/saipraveen-k/ML-Based-Prediction-and-Optimization-of-Geopolymer-Composites.git
cd "ML-Based-Prediction-and-Optimization-of-Geopolymer-Composites"
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📖 Running Instructions

### Launch Web Application
```bash
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

### Execute Pipeline Scripts (Optional / Maintenance)
```bash
# Data Preprocessing
python src/data_preprocessing.py

# Model Training
python src/model_training.py

# Model Evaluation
python src/evaluation.py

# Monte Carlo Simulation
python src/monte_carlo.py

# 10-Fold Cross Validation
python src/cross_validation.py

# SHAP Explainability Analysis
python src/run_shap.py
```

---

## 📁 Project Structure

```
ML-Based Prediction and Optimization of Geopolymer Composites/
│
├── app.py                         # Main Streamlit Application
├── requirements.txt               # Dependencies
├── README.md                      # Documentation
│
├── data/
│   ├── dataset.csv                # 812 Samples Dataset
│   └── Sample Data.xlsx           # Excel format dataset
│
├── models/
│   ├── xgboost.pkl                # Trained XGBoost Model
│   ├── random_forest.pkl          # Trained Random Forest Model
│   ├── svr.pkl                    # Trained SVR Model
│   └── scaler.pkl                 # Fitted StandardScaler
│
├── src/
│   ├── data_preprocessing.py      # Preprocessing Pipeline
│   ├── model_training.py          # Model Training & Tuning
│   ├── evaluation.py              # Performance Metrics
│   ├── monte_carlo.py             # Monte Carlo Simulation
│   ├── cross_validation.py        # 10-Fold CV Analysis
│   ├── shap_analyzer.py           # SHAP Analysis Engine
│   ├── run_shap.py                # SHAP Script
│   ├── utils.py                   # Formatting & Helper Utilities
│   └── ui/                        # UI Components & Page Renderers
│       ├── __init__.py
│       ├── cards.py               # Custom CSS & KPI Cards
│       ├── charts.py              # High-Res Matplotlib & Seaborn Plots
│       ├── components.py          # Sidebar Navigation & Controls
│       ├── formatting.py          # Feature Labels & Units
│       └── pages.py               # 10 Page Renderers
│
└── outputs/                       # Generated Visualizations & Reports
    ├── cross_validation/
    ├── graphs/
    ├── metrics/
    ├── reports/
    └── shap/
```

---

## 📄 License & Reference

This repository is maintained for research and open-source scientific advancement in sustainable construction materials.
