"""
UI Module Package Initialization
================================
Exposes main UI pages, styling, and components.
"""

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
    render_about_page
)
