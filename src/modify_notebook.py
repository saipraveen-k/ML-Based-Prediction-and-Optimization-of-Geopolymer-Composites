"""
Script to Programmatically Update notebooks/main.ipynb
======================================================
This script modifies the main.ipynb Jupyter Notebook to add the SHAP analysis section
and update the Table of Contents and Conclusion.
"""

import json
import os

def main():
    notebook_path = 'notebooks/main.ipynb'
    if not os.path.exists(notebook_path):
        print(f"Error: Notebook not found at {notebook_path}")
        return

    print("Loading notebooks/main.ipynb...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    cells = notebook['cells']
    
    # 1. Update Table of Contents (cell 1 or search by TOC title)
    toc_idx = None
    for idx, cell in enumerate(cells):
        if cell['cell_type'] == 'markdown' and any('## Table of Contents' in line for line in cell['source']):
            toc_idx = idx
            break
            
    if toc_idx is not None:
        print(f"Updating Table of Contents in cell #{toc_idx}...")
        cells[toc_idx]['source'] = [
            "## Table of Contents\n",
            "1. [Imports and Setup](#imports)\n",
            "2. [Data Loading](#data-loading)\n",
            "3. [Data Preprocessing](#preprocessing)\n",
            "4. [Exploratory Data Analysis](#eda)\n",
            "5. [Model Training](#model-training)\n",
            "6. [Model Evaluation](#evaluation)\n",
            "7. [Feature Importance Analysis](#feature-importance)\n",
            "8. [Monte Carlo Simulation](#monte-carlo)\n",
            "9. [Prediction System](#prediction)\n",
            "10. [Explainable AI (SHAP) Analysis](#shap-analysis)\n",
            "11. [Conclusion](#conclusion)"
        ]

    # 2. Find Conclusion cell to insert SHAP section right before it
    conclusion_idx = None
    for idx, cell in enumerate(cells):
        if cell['cell_type'] == 'markdown' and any("<a id='conclusion'>" in line for line in cell['source']):
            conclusion_idx = idx
            break
            
    if conclusion_idx is not None:
        print(f"Found Conclusion at cell #{conclusion_idx}. Renumbering to Section 11...")
        # Renumber Conclusion
        source = cells[conclusion_idx]['source']
        for i, line in enumerate(source):
            if '## 10. Conclusion' in line:
                source[i] = line.replace('## 10. Conclusion', '## 11. Conclusion')
            if '7. Prediction' in line:
                # Add SHAP point
                source[i] = "7. **Prediction**: Created a functional prediction system\n"
                # Insert point 8
                source.insert(i + 1, "8. **Explainable AI (SHAP)**: Added publication-ready SHAP explainability analysis\n")
        
        cells[conclusion_idx]['source'] = source
        
        # 3. Create the new SHAP analysis cells
        shap_markdown_cell = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "<a id='shap-analysis'></a>\n",
                "## 10. Explainable AI (SHAP) Analysis\n",
                "\n",
                "Apply SHAP (SHapley Additive exPlanations) to explain the optimized XGBoost regressor globally and locally."
            ]
        }
        
        shap_code_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Import SHAP analyzer\n",
                "from src.shap_analyzer import SHAPAnalyzer\n",
                "import pandas as pd\n",
                "\n",
                "# Use the trained XGBoost model as the primary explainability model\n",
                "xgb_model = models['XGBoost']\n",
                "scaler = preprocessor.scaler\n",
                "\n",
                "# Reconstruct original (unscaled) features\n",
                "X_original = pd.DataFrame(\n",
                "    scaler.inverse_transform(preprocessor.X),\n",
                "    columns=feature_names\n",
                ")\n",
                "\n",
                "# Initialize the SHAPAnalyzer\n",
                "shap_analyzer = SHAPAnalyzer(\n",
                "    model=xgb_model,\n",
                "    X_scaled=preprocessor.X,\n",
                "    feature_names=feature_names,\n",
                "    scaler=scaler,\n",
                "    X_original=X_original,\n",
                "    output_dir='../outputs'\n",
                ")\n",
                "\n",
                "# Execute the complete SHAP analysis pipeline\n",
                "shap_analyzer.run_all_analysis()"
            ]
        }
        
        # Insert cells before the conclusion cell
        print("Inserting SHAP markdown and code cells...")
        cells.insert(conclusion_idx, shap_markdown_cell)
        cells.insert(conclusion_idx + 1, shap_code_cell)
        
    notebook['cells'] = cells
    
    print("Saving modified main.ipynb...")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
        
    print("Notebook modified successfully!")

if __name__ == '__main__':
    main()
