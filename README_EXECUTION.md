# Execution Guide - UHPGC Compressive Strength Prediction System

## 📚 Step-by-Step Instructions for Running the Project

This guide provides detailed instructions for setting up and running the UHPGC Compressive Strength Prediction System on your machine.

---

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Python Installation](#python-installation)
3. [Virtual Environment Setup](#virtual-environment-setup)
4. [Dependency Installation](#dependency-installation)
5. [Dataset Preparation](#dataset-preparation)
6. [Running the Jupyter Notebook](#running-the-jupyter-notebook)
7. [Running the Streamlit App](#running-the-streamlit-app)
8. [Understanding Outputs](#understanding-outputs)
9. [Troubleshooting](#troubleshooting)

---

## 💻 System Requirements

### Minimum Requirements
- **Operating System:** Windows 10/11, Linux, or macOS
- **RAM:** 8 GB (16 GB recommended)
- **Storage:** 2 GB free space
- **Processor:** Intel Core i5 or equivalent

### Software Requirements
- **Python:** Version 3.8 or higher
- **pip:** Python package installer (comes with Python)
- **Git:** Optional, for version control

---

## 🐍 Python Installation

### Windows

1. **Download Python**
   - Visit: https://www.python.org/downloads/
   - Download the latest Python 3.x installer (3.8 or higher)

2. **Install Python**
   - Run the installer
   - **IMPORTANT:** Check "Add Python to PATH" during installation
   - Click "Install Now"

3. **Verify Installation**
   ```bash
   python --version
   pip --version
   ```

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

### macOS

```bash
# Using Homebrew
brew install python3
```

---

## 🌳 Virtual Environment Setup

Using a virtual environment is **highly recommended** to avoid conflicts with system packages.

### Windows

```bash
# Navigate to project directory
cd "ML-Based Prediction and Optimization of Geopolymer Composites"

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Verify activation (you should see (venv) in your terminal)
```

### Linux/macOS

```bash
# Navigate to project directory
cd "ML-Based Prediction and Optimization of Geopolymer Composites"

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify activation (you should see (venv) in your terminal)
```

### Deactivate Virtual Environment

```bash
deactivate
```

---

## 📦 Dependency Installation

### Step 1: Navigate to Project Directory

```bash
cd "ML-Based Prediction and Optimization of Geopolymer Composites"
```

### Step 2: Ensure Virtual Environment is Active

```bash
# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### Step 3: Install Requirements

```bash
pip install -r requirements.txt
```

This will install all required packages:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- openpyxl
- joblib
- streamlit
- scipy

### Step 4: Verify Installation

```bash
python -c "import pandas, numpy, sklearn, xgboost, streamlit; print('All packages installed successfully!')"
```

---

## 📊 Dataset Preparation

### Step 1: Obtain Your Dataset

- Your dataset should be in Excel (.xlsx) or CSV format
- It should contain material composition features and compressive strength values

### Step 2: Place Dataset in Correct Location

```bash
# Place your dataset file here:
data/dataset.csv
```

### Step 3: Verify Dataset Format

Your dataset should have columns similar to:
- Cement_kg_m3, Fly_Ash_kg_m3, Silica_Fume_kg_m3, Metakaolin_kg_m3
- GGBS_kg_m3, RHA_kg_m3, POFA_kg_m3
- Fine_Sand_kg_m3, Water_kg_m3, Extra_Water_kg_m3
- Water_Binder_Ratio
- Na2SiO3_Content_kg_m3, NaOH_Content_kg_m3, KOH_Content_kg_m3
- Activator_Molarity_M
- Superplasticizer_kg_m3
- Polypropylene_Fiber_Content_%, PP_Fiber_kg_m3, Fiber_Length_mm
- Curing_Temperature_C, Curing_Duration_days
- Compressive_Strength_MPa - **This is the target variable**

### Step 4: Check Dataset

```python
import pandas as pd
df = pd.read_csv('data/dataset.csv')
print(df.head())
print(df.shape)
print(df.columns)
```

---

## 📓 Running the Jupyter Notebook

### Option 1: Using VS Code (Recommended)

1. **Open Project in VS Code**
   ```bash
   code .
   ```

2. **Open Notebook**
   - Navigate to `notebooks/main.ipynb`
   - Open the file

3. **Select Kernel**
   - Click the kernel selector in the top right
   - Select your virtual environment (venv)

4. **Run Cells**
   - Click "Run All" or run cells sequentially
   - Wait for execution (may take 10-30 minutes depending on dataset size)

### Option 2: Using Jupyter Notebook Command

```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/macOS

# Install Jupyter (if not already installed)
pip install jupyter

# Start Jupyter Notebook
jupyter notebook

# Navigate to notebooks/main.ipynb
# Run all cells
```

### Option 3: Using Jupyter Lab

```bash
# Install Jupyter Lab
pip install jupyterlab

# Start Jupyter Lab
jupyter lab

# Navigate to notebooks/main.ipynb
# Run all cells
```

### What Happens During Execution

The notebook will automatically:

1. **Load and Explore Data**
   - Read dataset
   - Display dataset information
   - Show statistical summary

2. **Preprocess Data**
   - Handle missing values
   - Remove duplicates
   - Detect outliers
   - Scale features
   - Split into train-test sets

3. **Perform EDA**
   - Generate correlation heatmap
   - Create distribution plots
   - Generate scatter plots
   - Save all visualizations to `outputs/graphs/`

4. **Train Models**
   - Train Random Forest with hyperparameter tuning
   - Train SVR with hyperparameter tuning
   - Train XGBoost with hyperparameter tuning
   - Save models to `models/`

5. **Evaluate Models**
   - Calculate R², MAE, RMSE, MSE
   - Generate actual vs predicted plots
   - Create residual plots
   - Generate comparison table

6. **Feature Importance**
   - Analyze feature importance for each model
   - Generate feature importance plots
   - Create comparison visualization

7. **Monte Carlo Simulation**
   - Run 100 simulations per model
   - Generate PDF and CDF plots
   - Analyze model stability

8. **Prediction Demo**
   - Demonstrate prediction system
   - Show sample predictions

---

## 🌐 Running the Streamlit App

### Prerequisites

- Complete the Jupyter notebook execution first to train and save models
- Models should be in the `models/` folder

### Step 1: Activate Virtual Environment

```bash
# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### Step 2: Run Streamlit App

```bash
streamlit run app.py
```

### Step 3: Access the App

- Open your web browser
- Navigate to: `http://localhost:8501`
- The app will open automatically

### Step 4: Use the App

1. **Select Model**
   - Choose from Random Forest, SVR, or XGBoost in the sidebar

2. **Enter Input Values**
   - Fill in material composition values
   - Adjust sliders or enter numbers directly

3. **Click Predict**
   - Click the "Predict Compressive Strength" button
   - View prediction results

4. **Compare Models**
   - See predictions from all models
   - View comparison chart

### Step 5: Stop the App

- Press `Ctrl+C` in the terminal to stop the Streamlit server

---

## 📁 Understanding Outputs

### Generated Files and Folders

After running the notebook, you will find:

#### 1. Models Folder (`models/`)
- `random_forest.pkl` - Trained Random Forest model
- `svr.pkl` - Trained SVR model
- `xgboost.pkl` - Trained XGBoost model
- `scaler.pkl` - Fitted feature scaler

#### 2. Graphs Folder (`outputs/graphs/`)
- `correlation_heatmap_*.png` - Correlation matrix
- `distribution_*.png` - Feature distributions
- `target_distribution_*.png` - Target variable distribution
- `scatter_matrix_*.png` - Scatter plot matrix
- `feature_vs_target_*.png` - Feature vs target plots
- `boxplots_all_*.png` - Boxplots of all features
- `actual_vs_predicted_*.png` - Prediction accuracy plots
- `residuals_*.png` - Residual analysis plots
- `error_distribution_*.png` - Error distribution plots
- `model_comparison_*.png` - Model comparison charts
- `feature_importance_*.png` - Feature importance plots
- `monte_carlo_*.png` - Monte Carlo simulation plots

#### 3. Reports Folder (`outputs/reports/`)
- `evaluation_report_*.txt` - Model evaluation summary
- `feature_importance_report_*.txt` - Feature importance analysis
- `monte_carlo_report_*.txt` - Monte Carlo simulation results

#### 4. Metrics Folder (`outputs/metrics/`)
- `model_comparison_*.csv` - Performance comparison table

---

## 🔧 Troubleshooting

### Issue 1: Python Not Found

**Error:** `'python' is not recognized as an internal or external command`

**Solution:**
- Ensure Python is installed
- Add Python to PATH during installation
- Or use full path: `C:\Python39\python.exe`

### Issue 2: Virtual Environment Activation Fails

**Error:** Activation script not found

**Solution:**
```bash
# Recreate virtual environment
python -m venv venv --clear
```

### Issue 3: Package Installation Fails

**Error:** Could not find a version that satisfies the requirement

**Solution:**
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install packages individually
pip install pandas numpy matplotlib seaborn scikit-learn xgboost openpyxl joblib streamlit scipy
```

### Issue 4: XGBoost Installation Fails

**Error:** Microsoft Visual C++ 14.0 is required

**Solution:**
```bash
# Install pre-compiled wheel
pip install xgboost --no-cache-dir
```

### Issue 5: Dataset Not Found

**Error:** FileNotFoundError: dataset.csv not found

**Solution:**
- Ensure dataset is placed in `data/` folder
- Check filename spelling (case-sensitive on Linux/macOS)
- Verify file format (.csv)

### Issue 6: Memory Error During Training

**Error:** MemoryError or Out of Memory

**Solution:**
- Reduce dataset size
- Close other applications
- Use smaller hyperparameter grid
- Set `tune_hyperparameters=False` in notebook

### Issue 7: Streamlit App Not Loading Models

**Error:** No trained models found

**Solution:**
- Run the Jupyter notebook first to train models
- Check `models/` folder for .pkl files
- Verify models were saved successfully

### Issue 8: Port Already in Use

**Error:** Port 8501 is already in use

**Solution:**
```bash
# Use different port
streamlit run app.py --server.port 8502
```

### Issue 9: Import Errors

**Error:** ModuleNotFoundError: No module named 'src'

**Solution:**
- Ensure you're running from the project root directory
- Check that `src/` folder exists
- Verify virtual environment is active

### Issue 10: Slow Performance

**Issue:** Training or prediction is very slow

**Solution:**
- Reduce `n_simulations` in Monte Carlo (default: 100)
- Set `tune_hyperparameters=False` for faster training
- Use smaller dataset for testing
- Close other applications

---

## 📞 Additional Help

For additional support:
1. Check the `PROJECT_EXPLANATION.md` for technical details
2. Review the Jupyter notebook comments for code explanations
3. Verify all dependencies are correctly installed
4. Ensure dataset format matches requirements

---

## ✅ Quick Start Checklist

Before running the project, ensure:

- [ ] Python 3.8+ is installed
- [ ] Virtual environment is created and activated
- [ ] All dependencies are installed (`pip install -r requirements.txt`)
- [ ] Dataset is placed in `data/dataset.csv`
- [ ] Dataset has correct format and columns (Compressive_Strength_MPa as target)
- [ ] Sufficient disk space (2 GB+)
- [ ] Sufficient RAM (8 GB+ recommended)

---

**Happy Predicting! 🎯**
