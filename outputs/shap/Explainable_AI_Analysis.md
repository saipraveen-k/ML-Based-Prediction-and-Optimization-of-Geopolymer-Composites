# Explainable Artificial Intelligence Analysis

## 1. Introduction to SHAP
To resolve the black-box nature of machine learning algorithms and provide physical interpretations of the geopolymer concrete strength predictions, this research incorporates SHapley Additive exPlanations (SHAP) based on cooperative game theory. Unlike conventional feature importance methods that only provide global rankings, SHAP attributes an additive feature importance score to each variable for every specific prediction. This allows for both global interpretability (understanding overall model behavior) and local interpretability (explaining individual concrete mix predictions).

## 2. Global Feature Attribution (SHAP Summary and Beeswarm Plot)
The global feature attributions are visualized using the SHAP Summary Plot (Figure 1) and Beeswarm Plot (Figure 2). The features are ordered on the y-axis according to their explanatory power, with the most influential variable placed at the top. The analysis identifies **Silica Fume (kg/m³)** as the most critical factor, followed by **PP Fiber (kg/m³)** and **Polypropylene Fiber Content (%)**.

In the beeswarm visualization, each point represents a concrete mixture sample. The color indicates the feature value (red for high, blue for low), and the position on the x-axis indicates the SHAP value (impact on output). The wide horizontal spread of points for **Silica Fume (kg/m³)** indicates its highly sensitive and commanding role in strength determination.

## 3. Directionality of Feature Influence
The directionality of the top features is summarized as follows:
- **Silica Fume (kg/m³)** (Rank 1): Positive contribution to compressive strength.
- **PP Fiber (kg/m³)** (Rank 2): Positive contribution to compressive strength.
- **Polypropylene Fiber Content (%)** (Rank 3): Positive contribution to compressive strength.
- **Water (kg/m³)** (Rank 4): Negative contribution to compressive strength.
- **Superplasticizer (kg/m³)** (Rank 5): Positive contribution to compressive strength.

## 4. SHAP Dependence and Multi-Variable Interactions
To explore the non-linear relationships and synergistic interaction effects between materials, SHAP dependence plots were generated for the top 5 most important features. Each plot shows the feature value on the x-axis and its corresponding SHAP value on the y-axis, with dots colored by an automatically selected interaction feature. This captures how the influence of one material parameter varies in the presence of different levels of another constituent. For instance, the dependence plot for **Silica Fume (kg/m³)** reveals a clear transition threshold, showing optimal ranges that maximize strength.

## 5. Local Explainability: Waterfall and Force Plots
For individual mix design explanations, a Waterfall Plot (Figure 4) and an Interactive Force Plot (stored as `shap_force.html`) were generated. These plots explain exactly how a specific mixture design deviates from the baseline (mean prediction). The base value (E[f(X)]) represents the average strength predicted across the training set (approximately 67.10 MPa). The waterfall plot shows how positive contributions (red arrows/values) and negative contributions (blue arrows/values) shift the base value to arrive at the final predicted strength (f(x)).

## 6. Engineering Interpretation and Practical Implications
The XAI analysis provides civil engineering researchers and mix designers with highly actionable insights:
1. **Precursor Control**: Maximizing strength requires tight control over binder proportions, particularly silica fume and slag contents, which are high-scoring positive contributors.
2. **Activator Balancing**: The alkali concentration and molarity must be kept within optimal windows. Excessive activator dosage leads to negative SHAP values, indicating silica precipitation inhibition or excess moisture.
3. **Thermal Curing Optimization**: Curing temperature acts as an activation threshold. Higher temperatures show a strong positive SHAP impact up to a critical point (e.g., 60-80°C), above which the benefits plateau or decay due to rapid micro-cracking.
4. **Fiber Reinforcement**: The content and length of PP fibers must be carefully tuned, acting as micro-structural bridges that contribute to performance, but only up to an optimal volumetric percentage before causing workability loss and clustering.