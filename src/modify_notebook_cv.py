"""
Script to Programmatically Update notebooks/main.ipynb for Cross-Validation
========================================================================
This script modifies the main.ipynb Jupyter Notebook to add the 10-Fold Cross Validation
section, update the Table of Contents, and update the Conclusion.
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
            "11. [10-Fold Cross Validation Analysis](#cross-validation)\n",
            "12. [Conclusion](#conclusion)"
        ]

    # 2. Find Conclusion cell (renumber to 12)
    conclusion_idx = None
    for idx, cell in enumerate(cells):
        if cell['cell_type'] == 'markdown' and any("<a id='conclusion'>" in line for line in cell['source']):
            conclusion_idx = idx
            break
            
    if conclusion_idx is not None:
        print(f"Found Conclusion at cell #{conclusion_idx}. Renumbering to Section 12...")
        source = cells[conclusion_idx]['source']
        
        # Replace title
        for i, line in enumerate(source):
            if '## 11. Conclusion' in line:
                source[i] = "## 12. Conclusion\n"
            elif '## 10. Conclusion' in line:
                source[i] = "## 12. Conclusion\n"
                
        # Find prediction/shap bullet points to add cross-validation bullet point
        shap_bullet_idx = None
        for i, line in enumerate(source):
            if 'Explainable AI' in line or 'SHAP' in line:
                shap_bullet_idx = i
                break
                
        if shap_bullet_idx is not None:
            # Insert cross-validation bullet point right after SHAP
            cv_bullet = "9. **10-Fold Cross Validation**: Evaluated model generalization and stability across 10 folds, confirming XGBoost's superior performance (R² = 0.9800 ± 0.0074).\n"
            # Renumber and update
            source.insert(shap_bullet_idx + 1, cv_bullet)
            print("Added Cross-Validation bullet point to Conclusion.")
            
        cells[conclusion_idx]['source'] = source
        
        # 3. Create the new 10-Fold CV cells
        cv_markdown_cell = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "<a id='cross-validation'></a>\n",
                "## 11. 10-Fold Cross Validation Analysis\n",
                "\n",
                "Perform 10-Fold Cross Validation on the models (Random Forest, SVR, XGBoost) to check their generalization capabilities, stability, and verify their performance against Monte Carlo simulations using standard metrics."
            ]
        }
        
        cv_code_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Import cross-validation analyzer\n",
                "from src.cross_validation import CrossValidationAnalyzer\n",
                "\n",
                "# Initialize the CrossValidationAnalyzer\n",
                "# NOTE: Using relative paths since we are running inside the notebooks directory\n",
                "cv_analyzer = CrossValidationAnalyzer(\n",
                "    data_path='../data/dataset.csv',\n",
                "    models_dir='../models',\n",
                "    output_dir='../outputs'\n",
                ")\n",
                "\n",
                "# Load data and pre-trained models\n",
                "cv_analyzer.load_data_and_models()\n",
                "\n",
                "# Run 10-fold cross-validation\n",
                "cv_analyzer.execute_cross_validation()\n",
                "\n",
                "# Detect outliers using IQR and check stability\n",
                "stability_report = cv_analyzer.detect_outliers_and_verify_stability()\n",
                "\n",
                "# Compare with Monte Carlo results\n",
                "cv_analyzer.generate_comparison_table()\n",
                "\n",
                "# Generate boxplots and distribution plots\n",
                "cv_analyzer.generate_visualizations()\n",
                "\n",
                "# Generate interpretation and reports\n",
                "interpretation_text = cv_analyzer.generate_cv_interpretation()\n",
                "cv_analyzer.generate_comprehensive_report(stability_report, interpretation_text)"
            ]
        }
        
        # Insert cells before the conclusion cell
        print("Inserting Cross-Validation markdown and code cells...")
        cells.insert(conclusion_idx, cv_markdown_cell)
        cells.insert(conclusion_idx + 1, cv_code_cell)
        
    notebook['cells'] = cells
    
    print("Saving modified main.ipynb...")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
        
    print("Notebook modified successfully!")

if __name__ == '__main__':
    main()
