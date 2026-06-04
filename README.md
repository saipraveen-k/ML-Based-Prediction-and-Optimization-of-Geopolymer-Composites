# UHPGC Compressive Strength Prediction System

## 🏗️ Machine Learning Project for Ultra-High-Performance Geopolymer Concrete

A comprehensive end-to-end Machine Learning system for predicting the compressive strength of Ultra-High-Performance Geopolymer Concrete (UHPGC) using advanced ML algorithms.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Technologies Used](#technologies-used)
- [Dataset Explanation](#dataset-explanation)
- [ML Models Explanation](#ml-models-explanation)
- [Installation Guide](#installation-guide)
- [Usage Guide](#usage-guide)
- [Folder Structure](#folder-structure)
- [Outputs Explanation](#outputs-explanation)
- [Future Scope](#future-scope)
- [Conclusion](#conclusion)

---

## 🎯 Project Overview

This project implements a complete Machine Learning pipeline for predicting the compressive strength of Ultra-High-Performance Geopolymer Concrete (UHPGC). The system uses three advanced ML algorithms - Random Forest, Support Vector Regression (SVR), and XGBoost - to accurately predict compressive strength based on material composition and curing parameters.

**Research Paper Reference:**
"Investigation of machine learning models in predicting compressive strength for ultra-high-performance geopolymer concrete: A comparative study"

### Key Features

- ✅ Automated data preprocessing and cleaning
- ✅ Comprehensive Exploratory Data Analysis (EDA)
- ✅ Hyperparameter tuning for optimal model performance
- ✅ Multiple ML model implementation (RF, SVR, XGBoost)
- ✅ Detailed model evaluation and comparison
- ✅ Feature importance analysis
- ✅ 100-run Monte Carlo simulation for stability analysis
- ✅ Explainable AI (XAI) with publication-ready SHAP analysis (beeswarm, summary, bar, waterfall, and interactive force plots)
- ✅ Professional Streamlit web application
- ✅ Complete documentation and reports

---

## 🎯 Objectives

1. **Develop** a robust ML system for UHPGC compressive strength prediction
2. **Compare** performance of Random Forest, SVR, and XGBoost models
3. **Identify** the most influential material composition features
4. **Analyze** model stability using Monte Carlo simulation
5. **Provide** a user-friendly prediction interface
6. **Generate** comprehensive reports and visualizations

---

## 💻 Technologies Used

### Programming Language
- **Python 3.8+**

### Core Libraries
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **matplotlib** - Data visualization
- **seaborn** - Statistical visualization
- **scikit-learn** - Machine learning algorithms
- **xgboost** - Gradient boosting framework
- **openpyxl** - Excel file handling
- **joblib** - Model serialization
- **streamlit** - Web application framework
- **scipy** - Scientific computing

### Development Environment
- **Jupyter Notebook** - Interactive development
- **VS Code** - Code editing
- **Git** - Version control

---

## 📊 Dataset Explanation

### Input Features

The system uses the following material composition and curing parameters as input features:

**Binders:**
- Cement_kg_m3
- Fly_Ash_kg_m3
- Silica_Fume_kg_m3
- Metakaolin_kg_m3
- GGBS_kg_m3
- RHA_kg_m3 (Rice Husk Ash)
- POFA_kg_m3

**Aggregates & Water:**
- Fine_Sand_kg_m3
- Water_kg_m3
- Extra_Water_kg_m3
- Water_Binder_Ratio

**Alkali Activators:**
- Na2SiO3_Content_kg_m3
- NaOH_Content_kg_m3
- KOH_Content_kg_m3
- Activator_Molarity_M

**Additives & Fibers:**
- Superplasticizer_kg_m3
- Polypropylene_Fiber_Content_%
- PP_Fiber_kg_m3
- Fiber_Length_mm

**Curing Parameters:**
- Curing_Temperature_C
- Curing_Duration_days

### Target Variable
- **Compressive_Strength_MPa** - Measured in MPa

### Dataset Requirements
- Format: CSV (.csv)
- No missing values (handled automatically if present)
- Numerical values for all features
- Minimum 50 samples for reliable training
- Target column: Compressive_Strength_MPa

---

## 🤖 ML Models Explanation

### 1. Random Forest Regressor
- **Type:** Ensemble learning method
- **Mechanism:** Builds multiple decision trees and merges their predictions
- **Advantages:** Handles non-linear relationships, robust to outliers, provides feature importance
- **Hyperparameters Tuned:** n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features

### 2. Support Vector Regression (SVR)
- **Type:** Kernel-based regression
- **Mechanism:** Finds optimal hyperplane in high-dimensional space
- **Advantages:** Effective in high-dimensional spaces, versatile with different kernels
- **Hyperparameters Tuned:** C, epsilon, kernel, gamma

### 3. XGBoost Regressor
- **Type:** Gradient boosting framework
- **Mechanism:** Sequentially builds trees, correcting errors from previous trees
- **Advantages:** High accuracy, handles missing values, regularization, parallel processing
- **Hyperparameters Tuned:** n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, gamma, reg_alpha, reg_lambda

### Evaluation Metrics
- **R² Score:** Coefficient of determination (higher is better)
- **MAE:** Mean Absolute Error (lower is better)
- **RMSE:** Root Mean Squared Error (lower is better)
- **MSE:** Mean Squared Error (lower is better)

---

## 🚀 Installation Guide

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)

### Step 1: Clone/Download the Project
```bash
cd "ML-Based Prediction and Optimization of Geopolymer Composites"
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Place Dataset
- Place your CSV dataset file named `dataset.csv` in the `data/` folder
- Ensure the dataset contains the required features and target variable (Compressive_Strength_MPa)

---

## 📖 Usage Guide

### Option 1: Using Jupyter Notebook (Recommended)

1. **Open Jupyter Notebook**
```bash
jupyter notebook notebooks/main.ipynb
```

2. **Run All Cells**
- Execute cells sequentially from top to bottom
- The notebook will automatically:
  - Load CSV dataset
  - Preprocess data
  - Perform EDA
  - Train models
  - Evaluate performance
  - Generate visualizations
  - Save models and outputs

### Option 2: Using Streamlit Web App

1. **Train Models First**
- Run the notebook to train and save models
- Models will be saved in the `models/` folder

2. **Launch Streamlit App**
```bash
streamlit run app.py
```

3. **Access the App**
- Open your browser and navigate to `http://localhost:8501`
- Enter material composition values
- Click "Predict Compressive Strength"
- View predictions and model comparisons

### Option 3: Using Python Scripts

```python
# Import modules
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from src.prediction import PredictionSystem

# Preprocess data
preprocessor = DataPreprocessor('data/dataset.xlsx')
X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline()

# Train models
trainer = ModelTrainer()
models = trainer.train_all_models(X_train, y_train)

# Make predictions
predictor = PredictionSystem()
predictor.load_models()
prediction = predictor.predict_single(input_dict, 'XGBoost')
```

---

## 📁 Folder Structure

```
project/
│
├── data/
│   └── dataset.csv               # Your dataset file (CSV format)
│
├── notebooks/
│   └── main.ipynb                # Complete Jupyter notebook
│
├── src/
│   ├── data_preprocessing.py     # Data cleaning and preprocessing
│   ├── eda.py                    # Exploratory data analysis
│   ├── model_training.py         # ML model training
│   ├── evaluation.py             # Model evaluation
│   ├── prediction.py             # Prediction system
│   ├── monte_carlo.py            # Monte Carlo simulation
│   ├── feature_importance.py     # Feature importance analysis
│   └── utils.py                  # Utility functions
│
├── models/
│   ├── random_forest.pkl         # Trained Random Forest model
│   ├── svr.pkl                   # Trained SVR model
│   ├── xgboost.pkl               # Trained XGBoost model
│   └── scaler.pkl                # Fitted scaler
│
├── outputs/
│   ├── graphs/                   # All visualizations
│   │   ├── correlation_heatmap.png
│   │   ├── distribution_*.png
│   │   ├── actual_vs_predicted_*.png
│   │   ├── residuals_*.png
│   │   ├── feature_importance_*.png
│   │   └── monte_carlo_*.png
│   ├── reports/                  # Text reports
│   │   ├── evaluation_report_*.txt
│   │   ├── feature_importance_report_*.txt
│   │   └── monte_carlo_report_*.txt
│   ├── metrics/                  # Performance metrics
│   │   └── model_comparison_*.csv
│   └── predictions/              # Prediction outputs
│
├── app.py                        # Streamlit web application
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── README_EXECUTION.md          # Detailed execution guide
└── PROJECT_EXPLANATION.md       # Technical documentation
```

---

## 📈 Outputs Explanation

### Graphs (outputs/graphs/)
- **Correlation Heatmap:** Shows relationships between all features
- **Distribution Plots:** Histograms and boxplots for each feature
- **Scatter Plots:** Feature vs target relationships
- **Actual vs Predicted:** Model prediction accuracy visualization
- **Residual Plots:** Error analysis plots
- **Feature Importance:** Most influential features for each model
- **Monte Carlo Plots:** PDF, CDF, and distribution plots from 100 simulations

### Reports (outputs/reports/)
- **Evaluation Report:** Comprehensive model performance summary
- **Feature Importance Report:** Detailed feature analysis
- **Monte Carlo Report:** Stability analysis results

### Metrics (outputs/metrics/)
- **Model Comparison Table:** Side-by-side model performance comparison

---

## 🔮 Future Scope

1. **Additional Models:** Implement neural networks, ensemble methods
2. **Real-time Prediction:** Deploy as REST API
3. **Mobile App:** Develop mobile application for field use
4. **Optimization:** Add genetic algorithm for mix design optimization
5. **Cost Analysis:** Include cost prediction and optimization
6. **Database Integration:** Store predictions and track performance over time
7. **Multi-output Prediction:** Predict additional properties (durability, workability)

---

## 🎓 Conclusion

This project provides a complete, production-ready Machine Learning system for predicting UHPGC compressive strength. The modular architecture, comprehensive documentation, and professional web application make it suitable for:

- **Research Publication:** Complete methodology and results
- **Client Delivery:** Professional deliverable with reports
- **Final Year Project:** Well-documented academic project
- **Portfolio Project:** Demonstrates full ML pipeline skills
- **Industrial Use:** Practical prediction tool for concrete industry

The system achieves high accuracy (R² ≈ 0.84+) and provides valuable insights into the factors affecting compressive strength, making it a valuable tool for researchers and engineers working with geopolymer concrete.

---

## 📧 Contact & Support

For questions, issues, or contributions, please refer to the project documentation or contact the development team.

---

## 📄 License

This project is for educational and research purposes. Please cite the research paper if used in academic work.

---

**Built with ❤️ for the Concrete and ML Research Community**
