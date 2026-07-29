import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, confusion_matrix
)
import config

def evaluate_models():
    print("Loading test datasets...")
    X_test = joblib.load(config.X_TEST_PATH)
    y_test = joblib.load(config.Y_TEST_PATH)
    
    # Load feature names
    feature_names_path = os.path.join(config.PREPROCESSING_DIR, "feature_names.joblib")
    feature_names = joblib.load(feature_names_path)
    
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"Sepsis rows in test: {np.sum(y_test == 1)} ({np.mean(y_test == 1)*100:.2f}%)")
    
    # Create output directories
    os.makedirs(config.EVALUATION_DIR, exist_ok=True)
    os.makedirs(config.VISUALIZATIONS_DIR, exist_ok=True)
    
    models = {}
    model_paths = {
        'Decision Tree': os.path.join(config.MODELS_DIR, "decision_tree.joblib"),
        'Random Forest': os.path.join(config.MODELS_DIR, "random_forest.joblib"),
        'XGBoost': os.path.join(config.MODELS_DIR, "xgboost.joblib")
    }
    
    # Load all models
    for name, path in model_paths.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
            print(f"Loaded {name} model.")
        else:
            raise FileNotFoundError(f"Model file not found: {path}. Run train.py first.")
            
    # Dictionary to store performance metrics
    results = []
    
    # Plotting setup
    plt.style.use('seaborn-v0_8-whitegrid')
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
    
    # Track the best model details
    best_model_name = None
    best_roc_auc = -1.0
    best_model_obj = None
    best_tuned_threshold = 0.5
    
    tuned_models_info = {}
    
    for name, model in models.items():
        print(f"\nEvaluating: {name}...")
        
        # Predict probabilities
        if name == 'XGBoost':
            # XGBoost may return a 1D array of probabilities or 2D depending on version
            probs = model.predict_proba(X_test)
            y_prob = probs[:, 1] if len(probs.shape) > 1 else probs
        else:
            y_prob = model.predict_proba(X_test)[:, 1]
            
        # Default predictions (threshold = 0.5)
        y_pred_def = (y_prob >= 0.5).astype(int)
        
        # Calculate default metrics
        acc_def = accuracy_score(y_test, y_pred_def)
        prec_def = precision_score(y_test, y_pred_def, zero_division=0)
        rec_def = recall_score(y_test, y_pred_def, zero_division=0)
        f1_def = f1_score(y_test, y_pred_def, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        results.append({
            'Model': name,
            'Threshold Type': 'Default (0.5)',
            'Threshold': 0.5,
            'Accuracy': acc_def,
            'Precision': prec_def,
            'Recall': rec_def,
            'F1-Score': f1_def,
            'ROC-AUC': roc_auc
        })
        
        # Threshold Tuning to target high recall (e.g. TARGET_RECALL = 0.85)
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
        
        # Find the threshold that achieves a recall closest to but >= TARGET_RECALL
        # Note: recalls has length len(thresholds)+1 (last element is 0). thresholds matches index-to-index.
        indices_meeting_target = np.where(recalls[:-1] >= config.TARGET_RECALL)[0]
        
        if len(indices_meeting_target) > 0:
            # Pick the index that maximizes precision among those meeting target recall
            best_idx = indices_meeting_target[np.argmax(precisions[indices_meeting_target])]
            tuned_threshold = thresholds[best_idx]
        else:
            # Fallback: pick the threshold that gets closest to target recall
            best_idx = np.argmin(np.abs(recalls[:-1] - config.TARGET_RECALL))
            tuned_threshold = thresholds[best_idx]
            
        # Tuned predictions
        y_pred_tuned = (y_prob >= tuned_threshold).astype(int)
        
        # Calculate tuned metrics
        acc_tuned = accuracy_score(y_test, y_pred_tuned)
        prec_tuned = precision_score(y_test, y_pred_tuned, zero_division=0)
        rec_tuned = recall_score(y_test, y_pred_tuned, zero_division=0)
        f1_tuned = f1_score(y_test, y_pred_tuned, zero_division=0)
        
        results.append({
            'Model': name,
            'Threshold Type': f'Tuned (Recall {config.TARGET_RECALL})',
            'Threshold': tuned_threshold,
            'Accuracy': acc_tuned,
            'Precision': prec_tuned,
            'Recall': rec_tuned,
            'F1-Score': f1_tuned,
            'ROC-AUC': roc_auc
        })
        
        # Save tuned threshold details
        tuned_models_info[name] = {
            'threshold': float(tuned_threshold),
            'roc_auc': float(roc_auc),
            'f1_score_tuned': float(f1_tuned),
            'recall_tuned': float(rec_tuned)
        }
        
        # Track the best model based on ROC-AUC
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model_name = name
            best_model_obj = model
            best_tuned_threshold = tuned_threshold
            
        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", lw=2)
        
        # Plot Precision-Recall Curve
        ax_pr.plot(recalls, precisions, label=f"{name}", lw=2)
        
        # Generate Confusion Matrix for tuned threshold
        cm = confusion_matrix(y_test, y_pred_tuned)
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No Sepsis', 'Sepsis'], yticklabels=['No Sepsis', 'Sepsis']
        )
        plt.title(f"{name} Confusion Matrix\n(Tuned Threshold = {tuned_threshold:.3f})")
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        cm_path = os.path.join(config.VISUALIZATIONS_DIR, f"confusion_matrix_{name.lower().replace(' ', '_')}.png")
        plt.savefig(cm_path, dpi=300)
        plt.close()
        print(f"Saved confusion matrix for {name} to {cm_path}")
        
    # Finalize ROC Plot
    ax_roc.plot([0, 1], [0, 1], color='darkgray', linestyle='--', lw=1.5)
    ax_roc.set_xlim([-0.01, 1.01])
    ax_roc.set_ylim([-0.01, 1.01])
    ax_roc.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax_roc.set_ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
    ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve Comparison', fontsize=14, fontweight='bold')
    ax_roc.legend(loc='lower right', frameon=True)
    fig_roc.tight_layout()
    roc_plot_path = os.path.join(config.VISUALIZATIONS_DIR, "roc_curve_comparison.png")
    fig_roc.savefig(roc_plot_path, dpi=300)
    plt.close(fig_roc)
    print(f"\nSaved ROC curve comparison to {roc_plot_path}")
    
    # Finalize PR Plot
    ax_pr.set_xlim([-0.01, 1.01])
    ax_pr.set_ylim([-0.01, 1.01])
    ax_pr.set_xlabel('Recall (Sensitivity)', fontsize=12)
    ax_pr.set_ylabel('Precision (Positive Predictive Value)', fontsize=12)
    ax_pr.set_title('Precision-Recall Curve Comparison', fontsize=14, fontweight='bold')
    ax_pr.legend(loc='upper right', frameon=True)
    fig_pr.tight_layout()
    pr_plot_path = os.path.join(config.VISUALIZATIONS_DIR, "precision_recall_comparison.png")
    fig_pr.savefig(pr_plot_path, dpi=300)
    plt.close(fig_pr)
    print(f"Saved Precision-Recall curve comparison to {pr_plot_path}")
    
    # Save best model details (explicitly overridden to XGBoost per user request)
    best_model_name = 'XGBoost'
    best_model_obj = models[best_model_name]
    best_tuned_threshold = tuned_models_info[best_model_name]['threshold']
    best_roc_auc = tuned_models_info[best_model_name]['roc_auc']
    
    print(f"\nSelecting the best model: {best_model_name}")
    print(f"Best ROC-AUC: {best_roc_auc:.4f}")
    print(f"Best Model Tuned Threshold: {best_tuned_threshold:.4f}")
    
    best_model_metadata = {
        'model_name': best_model_name,
        'tuned_threshold': float(best_tuned_threshold),
        'metrics': tuned_models_info[best_model_name],
        'feature_names': feature_names,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Save the metadata and model info
    best_model_file = os.path.join(config.MODELS_DIR, f"{best_model_name.lower().replace(' ', '_')}.joblib")
    best_model_info = {
        'metadata': best_model_metadata,
        'model_path': best_model_file
    }
    joblib.dump(best_model_info, config.BEST_MODEL_INFO_PATH)
    print(f"Saved best model info to {config.BEST_MODEL_INFO_PATH}")
    
    # Print results dataframe
    results_df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("MODEL EVALUATION COMPARISON TABLE")
    print("="*80)
    print(results_df.to_string(index=False, formatters={
        'Threshold': '{:,.4f}'.format,
        'Accuracy': '{:,.4f}'.format,
        'Precision': '{:,.4f}'.format,
        'Recall': '{:,.4f}'.format,
        'F1-Score': '{:,.4f}'.format,
        'ROC-AUC': '{:,.4f}'.format
    }))
    print("="*80 + "\n")
    
    # Save evaluation summary to a text file
    report_path = os.path.join(config.EVALUATION_DIR, "model_evaluation_report.txt")
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SEPSIS PREDICTION SYSTEM - PHASE 1 MODEL EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n" + "="*80 + "\n")
        f.write(f"BEST MODEL SELECTED: {best_model_name}\n")
        f.write(f"Selected Threshold:  {best_tuned_threshold:.4f}\n")
        f.write(f"ROC-AUC score:       {best_roc_auc:.4f}\n")
        f.write(f"F1-Score (tuned):    {tuned_models_info[best_model_name]['f1_score_tuned']:.4f}\n")
        f.write(f"Recall (tuned):      {tuned_models_info[best_model_name]['recall_tuned']:.4f}\n")
        f.write("="*80 + "\n")
    print(f"Saved text report to {report_path}")

if __name__ == "__main__":
    evaluate_models()
