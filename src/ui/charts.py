"""
High-Resolution Publication-Quality Chart Generator
=====================================================
Unified design system for interactive and exported Matplotlib / Seaborn charts.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import format_feature_name, format_metric_name

# Unified Scientific Color Palette
MODEL_COLORS = {
    'XGBoost': '#059669',       # Deep Emerald
    'Random Forest': '#0284c7', # Soft Blue
    'SVR': '#f97316',           # Warm Orange
    'Average': '#6366f1'        # Indigo
}

# Apply clean publication styling to Matplotlib global config
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'axes.labelcolor': '#0f172a',
    'axes.edgecolor': '#cbd5e1',
    'axes.linewidth': 0.8,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#ffffff',
    'grid.color': '#f1f5f9',
    'grid.linestyle': '--',
    'grid.alpha': 0.7
})


def plot_model_comparison_bar(comparison_df):
    """Plot bar chart comparing model predictions."""
    fig, ax = plt.subplots(figsize=(7, 3.8), dpi=300)
    
    models = comparison_df['Model'].tolist()
    preds = comparison_df['Predicted CS (MPa)'].tolist()
    colors = [MODEL_COLORS.get(m, '#059669') for m in models]
    
    bars = ax.bar(models, preds, color=colors, alpha=0.9, edgecolor='none', width=0.45)
    
    ax.set_ylabel('Predicted Compressive Strength (MPa)')
    ax.set_title('Model Prediction Comparison', pad=12)
    ax.grid(True, axis='y')
    
    # Hide top and right spines for clean look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    max_val = max(preds) if preds else 100
    ax.set_ylim(0, max_val * 1.18)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (max_val * 0.02),
                f'{height:.2f} MPa', ha='center', va='bottom', fontweight='bold', fontsize=10, color='#0f172a')
                
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df, columns=None, annot_size=8):
    """Plot publication-quality correlation heatmap with upper triangle mask and clean labels."""
    if columns:
        corr_df = df[columns].corr()
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        corr_df = df[num_cols].corr()
        
    clean_labels = [format_feature_name(c) for c in corr_df.columns]
    
    # Upper triangle mask
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    # Custom divergent color palette (Teal-Emerald to Warm Orange/Red)
    cmap = sns.diverging_palette(150, 20, s=80, l=55, as_cmap=True)
    
    sns.heatmap(
        corr_df,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor='#ffffff',
        cbar_kws={"shrink": 0.75, "label": "Pearson Correlation Coefficient (r)"},
        xticklabels=clean_labels,
        yticklabels=clean_labels,
        annot_kws={"size": annot_size, "weight": "bold"},
        ax=ax
    )
    
    ax.set_title("Material Relationship Map", pad=15, fontsize=15, loc='left')
    plt.suptitle("Pearson correlation across numerical variables", fontsize=10, color='#64748b', x=0.125, y=0.92, ha='left')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig


def plot_feature_distribution(df, feature_name):
    """Plot feature histogram with KDE and boxplot."""
    clean_title = format_feature_name(feature_name)
    data = df[feature_name].dropna()
    
    fig, (ax_box, ax_hist) = plt.subplots(
        2, 1, figsize=(7, 4.5), dpi=300, sharex=True,
        gridspec_kw={'height_ratios': [0.22, 0.78]}
    )
    
    sns.boxplot(x=data, ax=ax_box, color='#10b981', fliersize=3, linewidth=1)
    ax_box.set(xlabel='')
    ax_box.set_title(f'Distribution: {clean_title}', pad=8, loc='left')
    ax_box.spines['top'].set_visible(False)
    ax_box.spines['right'].set_visible(False)
    ax_box.spines['left'].set_visible(False)
    ax_box.tick_params(left=False, labelleft=False)
    
    sns.histplot(data, kde=True, ax=ax_hist, color='#059669', edgecolor='#047857', alpha=0.5, linewidth=0.8)
    ax_hist.set_xlabel(clean_title)
    ax_hist.set_ylabel('Sample Frequency')
    ax_hist.spines['top'].set_visible(False)
    ax_hist.spines['right'].set_visible(False)
    ax_hist.grid(True, axis='y')
    
    mean_val = data.mean()
    median_val = data.median()
    
    ax_hist.axvline(mean_val, color='#ef4444', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
    ax_hist.axvline(median_val, color='#0284c7', linestyle='-', linewidth=1.5, label=f'Median: {median_val:.2f}')
    ax_hist.legend(loc='upper right')
    
    plt.tight_layout()
    return fig


def plot_feature_vs_target(df, feature_name, target_name='Compressive_Strength_MPa'):
    """Plot feature vs target scatter plot with regression trendline."""
    clean_feature = format_feature_name(feature_name)
    clean_target = format_feature_name(target_name)
    
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=300)
    
    sns.regplot(
        x=df[feature_name],
        y=df[target_name],
        ax=ax,
        color='#059669',
        scatter_kws={'alpha': 0.55, 's': 28, 'color': '#059669', 'edgecolor': 'none'},
        line_kws={'color': '#ef4444', 'linewidth': 1.8, 'label': 'Linear Fit Trend'}
    )
    
    corr_val = df[feature_name].corr(df[target_name])
    ax.set_xlabel(clean_feature)
    ax.set_ylabel(clean_target)
    ax.set_title(f'{clean_feature} vs. {clean_target}', pad=12, loc='left')
    plt.suptitle(f"Pearson Correlation r = {corr_val:.4f}", fontsize=10, color='#64748b', x=0.125, y=0.92, ha='left')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    return fig


def plot_actual_vs_predicted(y_true, y_pred, model_name="Model", r2=None, mae=None, rmse=None):
    """Plot high-resolution Actual vs Predicted scatter plot with identity line."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    
    color = MODEL_COLORS.get(model_name, '#059669')
    ax.scatter(y_true, y_pred, alpha=0.6, color=color, edgecolors='none', s=35, label='Experimental Samples')
    
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], '--', color='#ef4444', linewidth=1.8, label='Ideal Identity (y = x)')
    
    ax.set_xlabel('Observed Compressive Strength (MPa)')
    ax.set_ylabel('Model Predicted Strength (MPa)')
    ax.set_title('Prediction Accuracy', pad=15, loc='left')
    plt.suptitle(f"Observed vs model-predicted compressive strength ({model_name})", fontsize=10, color='#64748b', x=0.125, y=0.92, ha='left')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True)
    
    stats_text = ""
    if r2 is not None:
        stats_text += f"R² = {r2:.4f}\n"
    if mae is not None:
        stats_text += f"MAE = {mae:.2f} MPa\n"
    if rmse is not None:
        stats_text += f"RMSE = {rmse:.2f} MPa"
        
    if stats_text:
        ax.text(0.05, 0.82, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8fafc', edgecolor='#cbd5e1', alpha=0.95))
                
    ax.legend(loc='lower right')
    plt.tight_layout()
    return fig


def plot_residuals(y_true, y_pred, model_name="Model"):
    """Plot residual analysis (Residual vs Predicted and Residual Distribution)."""
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=300)
    
    color = MODEL_COLORS.get(model_name, '#059669')
    ax1.scatter(y_pred, residuals, alpha=0.55, color=color, edgecolors='none', s=30)
    ax1.axhline(0, color='#ef4444', linestyle='--', linewidth=1.5)
    ax1.set_xlabel('Predicted Strength (MPa)')
    ax1.set_ylabel('Residual Error (Actual - Predicted) [MPa]')
    ax1.set_title(f'Residuals vs. Predicted ({model_name})', pad=10, loc='left')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True)
    
    sns.histplot(residuals, kde=True, ax=ax2, color='#059669', edgecolor='#047857', alpha=0.5)
    ax2.axvline(0, color='#ef4444', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('Residual Error (MPa)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Residual Error Distribution', pad=10, loc='left')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, axis='y')
    
    plt.tight_layout()
    return fig
