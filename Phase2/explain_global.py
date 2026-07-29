import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Add Phase 2 to path and import configs
import config
from explainers import SepsisExplainer

def generate_global_explanations():
    print("--- Phase 2: Starting Global Explainability Pipeline ---")
    
    # 1. Initialize SepsisExplainer
    # We will initialize with background data for exact expected values
    explainer = SepsisExplainer(use_background_data=True)
    
    # 2. Load test set
    print("Loading preprocessed test data...")
    X_test = joblib.load(config.X_TEST_PATH)
    y_test = joblib.load(config.Y_TEST_PATH)
    print(f"Loaded test data. Shape: {X_test.shape}")
    
    # 3. Subsample test set for SHAP computation
    # Using a subset of X_test makes explanation generation fast and representative
    np.random.seed(config.RANDOM_SEED)
    sample_size = min(config.SHAP_TEST_SAMPLES, X_test.shape[0])
    print(f"Subsampling {sample_size} test observations for global SHAP calculation...")
    sample_indices = np.random.choice(X_test.shape[0], sample_size, replace=False)
    X_sample = X_test[sample_indices]
    
    # Convert sample back to DataFrame to preserve feature names in SHAP plots
    X_sample_df = pd.DataFrame(X_sample, columns=explainer.feature_names)
    
    # 4. Compute SHAP values
    print("Computing SHAP values for the sample...")
    # TreeExplainer is highly optimized, this should take only a few seconds
    shap_results = explainer.shap_explainer.shap_values(X_sample_df)
    
    # Handle list outputs for binary classification
    if isinstance(shap_results, list):
        # Index 1 corresponds to sepsis class
        shap_values_matrix = shap_results[1] if len(shap_results) == 2 else shap_results[0]
    else:
        shap_values_matrix = shap_results
        
    print(f"SHAP values computed. Shape: {shap_values_matrix.shape}")
    
    # 5. Generate and Save SHAP Beeswarm Plot (Summary Plot)
    print("Generating SHAP Summary Plot (Beeswarm)...")
    plt.figure(figsize=(10, 8))
    # We use shap.summary_plot
    shap.summary_plot(shap_values_matrix, X_sample_df, show=False)
    plt.title("SHAP Beeswarm Plot: Feature Impact on Sepsis Risk", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    summary_plot_path = os.path.join(config.VISUALIZATIONS_DIR, "shap_summary_beeswarm.png")
    plt.savefig(summary_plot_path, dpi=300)
    plt.close()
    print(f"Saved SHAP Beeswarm Plot to {summary_plot_path}")
    
    # 6. Generate and Save SHAP Feature Importance Bar Plot
    print("Generating SHAP Feature Importance Plot (Bar)...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_matrix, X_sample_df, plot_type="bar", show=False)
    plt.title("SHAP Global Feature Importance (Mean Absolute SHAP Value)", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    bar_plot_path = os.path.join(config.VISUALIZATIONS_DIR, "shap_feature_importance_bar.png")
    plt.savefig(bar_plot_path, dpi=300)
    plt.close()
    print(f"Saved SHAP Bar Plot to {bar_plot_path}")
    
    # 7. Compare SHAP Importance with XGBoost Built-In Importance
    print("Computing and comparing XGBoost built-in feature importances...")
    # XGBoost default is 'gain'
    xgb_importances = explainer.model.feature_importances_
    
    # Calculate mean absolute SHAP value for each feature
    mean_abs_shap = np.mean(np.abs(shap_values_matrix), axis=0)
    
    # Construct comparison DataFrame
    importance_df = pd.DataFrame({
        'Feature': explainer.feature_names,
        'Mean_Abs_SHAP': mean_abs_shap,
        'XGB_BuiltIn_Weight': xgb_importances
    })
    
    # Normalize importances so they sum to 1 or are comparable in ranks
    importance_df['SHAP_Rank'] = importance_df['Mean_Abs_SHAP'].rank(ascending=False)
    importance_df['XGB_Rank'] = importance_df['XGB_BuiltIn_Weight'].rank(ascending=False)
    
    # Sort by SHAP importance
    importance_df = importance_df.sort_values(by='Mean_Abs_SHAP', ascending=False).reset_index(drop=True)
    
    # Print Top 15 features
    print("\nTop 15 Features Comparison:")
    print(importance_df.head(15).to_string(index=False))
    
    # Save comparison dataframe to a CSV log in visualizations
    csv_path = os.path.join(config.BASE_DIR, "global_feature_importance_comparison.csv")
    importance_df.to_csv(csv_path, index=False)
    print(f"\nSaved feature importance comparison data to {csv_path}")
    
    # 8. Plot Top 15 Comparison Chart
    plt.figure(figsize=(12, 6))
    top_15 = importance_df.head(15)
    
    # We will plot normalized importances side-by-side
    top_15_melted = pd.melt(
        top_15, 
        id_vars=['Feature'], 
        value_vars=['Mean_Abs_SHAP', 'XGB_BuiltIn_Weight'],
        var_name='Importance_Type', 
        value_name='Value'
    )
    
    # Normalize values for side-by-side plotting
    # Simple Min-Max scaling for display purposes
    for val_type in ['Mean_Abs_SHAP', 'XGB_BuiltIn_Weight']:
        mask = top_15_melted['Importance_Type'] == val_type
        max_val = top_15_melted.loc[mask, 'Value'].max()
        if max_val > 0:
            top_15_melted.loc[mask, 'Value'] = top_15_melted.loc[mask, 'Value'] / max_val
            
    sns.barplot(data=top_15_melted, x='Feature', y='Value', hue='Importance_Type', palette='viridis')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Normalized Importance (Relative to Top Feature)')
    plt.title('Comparison of Top 15 Features: SHAP vs XGBoost Built-In Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    comparison_plot_path = os.path.join(config.VISUALIZATIONS_DIR, "importance_comparison_top15.png")
    plt.savefig(comparison_plot_path, dpi=300)
    plt.close()
    print(f"Saved Feature Importance Comparison Plot to {comparison_plot_path}")
    print("\n--- Global Explainability Pipeline Completed Successfully ---")

if __name__ == "__main__":
    generate_global_explanations()
