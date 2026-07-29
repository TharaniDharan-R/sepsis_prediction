import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# Add Phase 2 config and import explainer
import config
from explainers import SepsisExplainer

def run_local_explanations():
    print("--- Phase 2: Starting Local Patient Explainability ---")
    
    # 1. Load data
    print("Loading test data split...")
    X_test = joblib.load(config.X_TEST_PATH)
    y_test = joblib.load(config.Y_TEST_PATH)
    
    # 2. Initialize explainer
    explainer = SepsisExplainer(use_background_data=True)
    tuned_threshold = explainer.tuned_threshold
    
    # 3. Compute probabilities for the whole test set to categorize predictions
    print("Evaluating XGBoost model probabilities on test set...")
    probs = explainer.model.predict_proba(X_test)
    y_prob = probs[:, 1] if len(probs.shape) > 1 and probs.shape[1] > 1 else probs
    y_pred = (y_prob >= tuned_threshold).astype(int)
    
    # Identify patient indices for each category
    tp_indices = np.where((y_test == 1) & (y_pred == 1))[0]
    tn_indices = np.where((y_test == 0) & (y_pred == 0))[0]
    fp_indices = np.where((y_test == 0) & (y_pred == 1))[0]
    fn_indices = np.where((y_test == 1) & (y_pred == 0))[0]
    
    print(f"Identified cases in test set:")
    print(f" - True Positives: {len(tp_indices)}")
    print(f" - True Negatives: {len(tn_indices)}")
    print(f" - False Positives: {len(fp_indices)}")
    print(f" - False Negatives: {len(fn_indices)}")
    
    # Select one representative index for each category
    cases = {}
    if len(tp_indices) > 0:
        # Choose one with a high confidence sepsis prediction
        cases['True Positive'] = tp_indices[np.argmax(y_prob[tp_indices])]
    if len(tn_indices) > 0:
        # Choose one with a low sepsis prediction probability
        cases['True Negative'] = tn_indices[np.argmin(y_prob[tn_indices])]
    if len(fp_indices) > 0:
        # Choose one false alarm
        cases['False Positive'] = fp_indices[np.argmax(y_prob[fp_indices])]
    if len(fn_indices) > 0:
        # Choose one missed sepsis case
        cases['False Negative'] = fn_indices[np.argmax(y_prob[fn_indices])]
        
    print("\nSelected Patient Instances for Local Explanation:")
    for name, idx in cases.items():
        print(f" - {name} at Index {idx} (True Label: {y_test[idx]}, Predicted Prob: {y_prob[idx]*100:.2f}%)")
        
    # Set plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 4. Generate local explanation plots and JSON outputs for each patient
    for case_name, idx in cases.items():
        print(f"\n==================================================")
        print(f"Analyzing Local Patient Case: {case_name} (Index {idx})")
        print(f"==================================================")
        
        row_preprocessed = X_test[idx]
        
        # A. Compute and Print JSON Explanation Payload (representing API integration)
        explanation_dict = explainer.explain_patient_hour(row_preprocessed)
        
        # Trim contributions to top 5 for cleaner console display
        print("\nAPI Structured Explanation Payload (Top 5 contributions):")
        display_dict = {
            'prediction': explanation_dict['prediction'],
            'shap': {
                'base_value': explanation_dict['shap']['base_value'],
                'prediction_value': explanation_dict['shap']['prediction_value'],
                'top_contributions': explanation_dict['shap']['contributions'][:5]
            },
            'lime': {
                'intercept': explanation_dict['lime']['intercept'],
                'top_contributions': explanation_dict['lime']['contributions'][:5]
            }
        }
        print(json.dumps(display_dict, indent=2))
        
        # Save complete JSON payload to visualizations folder
        json_filename = f"local_explanation_{case_name.lower().replace(' ', '_')}.json"
        json_path = os.path.join(config.VISUALIZATIONS_DIR, json_filename)
        with open(json_path, 'w') as f:
            json.dump(explanation_dict, f, indent=4)
        print(f"Saved complete JSON payload to {json_path}")
        
        # B. Plot SHAP Waterfall Plot
        print(f"Generating SHAP Waterfall Plot for {case_name}...")
        exp_obj = explainer.explain_instance_shap(row_preprocessed)
        
        plt.figure(figsize=(10, 6))
        try:
            # SHAP waterfall plotting function
            shap.plots.waterfall(exp_obj, max_display=10, show=False)
            plt.title(f"SHAP Waterfall: {case_name} Explanation\n(Predicted Prob: {y_prob[idx]*100:.1f}% vs Thr: {tuned_threshold*100:.1f}%)", 
                      fontsize=12, fontweight='bold', pad=15)
            plt.tight_layout()
            waterfall_plot_path = os.path.join(config.VISUALIZATIONS_DIR, f"shap_waterfall_{case_name.lower().replace(' ', '_')}.png")
            plt.savefig(waterfall_plot_path, dpi=300)
            plt.close()
            print(f"Saved SHAP waterfall plot to {waterfall_plot_path}")
        except Exception as e:
            print(f"SHAP Waterfall plot failed: {e}. Drawing fallback bar chart.")
            # Custom horizontal bar plot as fallback
            contributions = explanation_dict['shap']['contributions'][:10]
            features = [c['feature'] for c in contributions]
            impacts = [c['impact'] for c in contributions]
            
            plt.figure(figsize=(10, 6))
            colors = ['crimson' if x > 0 else 'dodgerblue' for x in impacts]
            plt.barh(features, impacts, color=colors)
            plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
            plt.xlabel("SHAP Value (Impact on Log-Odds Sepsis Risk)")
            plt.title(f"SHAP Local Feature Contributions: {case_name}\n(Predicted Prob: {y_prob[idx]*100:.1f}%)", 
                      fontsize=12, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            waterfall_plot_path = os.path.join(config.VISUALIZATIONS_DIR, f"shap_waterfall_{case_name.lower().replace(' ', '_')}.png")
            plt.savefig(waterfall_plot_path, dpi=300)
            plt.close()
            print(f"Saved fallback SHAP plot to {waterfall_plot_path}")
            
        # C. Plot LIME Explanation Plot
        print(f"Generating LIME Plot for {case_name}...")
        lime_exp = explainer.explain_instance_lime_obj(row_preprocessed)
        
        try:
            fig = lime_exp.as_pyplot_figure(label=1)
            plt.title(f"LIME Local Explanation: {case_name}\n(Predicted Prob: {y_prob[idx]*100:.1f}%)", 
                      fontsize=12, fontweight='bold')
            plt.tight_layout()
            lime_plot_path = os.path.join(config.VISUALIZATIONS_DIR, f"lime_local_{case_name.lower().replace(' ', '_')}.png")
            plt.savefig(lime_plot_path, dpi=300)
            plt.close()
            print(f"Saved LIME plot to {lime_plot_path}")
        except Exception as e:
            print(f"LIME pyplot failed: {e}. Drawing fallback custom bar chart.")
            # Custom horizontal bar plot for LIME contributions
            contributions = explanation_dict['lime']['contributions'][:10]
            features = [c['feature'] for c in contributions]
            impacts = [c['impact'] for c in contributions]
            
            plt.figure(figsize=(10, 6))
            colors = ['crimson' if x > 0 else 'dodgerblue' for x in impacts]
            plt.barh(features, impacts, color=colors)
            plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
            plt.xlabel("LIME Contribution Score (Impact on Sepsis Class)")
            plt.title(f"LIME Local Feature Contributions: {case_name}\n(Predicted Prob: {y_prob[idx]*100:.1f}%)", 
                      fontsize=12, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            lime_plot_path = os.path.join(config.VISUALIZATIONS_DIR, f"lime_local_{case_name.lower().replace(' ', '_')}.png")
            plt.savefig(lime_plot_path, dpi=300)
            plt.close()
            print(f"Saved fallback LIME plot to {lime_plot_path}")
            
    print("\n--- Local Explainability Pipeline Completed Successfully ---")

if __name__ == "__main__":
    run_local_explanations()
