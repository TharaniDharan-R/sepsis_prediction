import os
import sys
import joblib
import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular

# Import Phase 2 config
import config

class SepsisExplainer:
    """
    Explainable AI (XAI) manager for the Sepsis Prediction system.
    Provides local and global predictions explanations using SHAP and LIME.
    """
    def __init__(self, use_background_data=True):
        # 1. Load preprocessor and model assets
        print(f"Loading preprocessor from {config.PREPROCESSOR_PATH}...")
        if not os.path.exists(config.PREPROCESSOR_PATH):
            raise FileNotFoundError(f"Preprocessor not found at {config.PREPROCESSOR_PATH}. Ensure Phase 1 ran successfully.")
        self.preprocessor = joblib.load(config.PREPROCESSOR_PATH)
        
        print(f"Loading best model info from {config.BEST_MODEL_INFO_PATH}...")
        if not os.path.exists(config.BEST_MODEL_INFO_PATH):
            raise FileNotFoundError(f"Best model info not found at {config.BEST_MODEL_INFO_PATH}.")
        best_info = joblib.load(config.BEST_MODEL_INFO_PATH)
        
        self.model_name = best_info['metadata']['model_name']
        self.tuned_threshold = best_info['metadata']['tuned_threshold']
        
        # We explicitly load the xgboost model
        model_filename = "xgboost.joblib"
        model_path = os.path.join(config.MODELS_DIR, model_filename)
        print(f"Loading XGBoost model from {model_path}...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"XGBoost model not found at {model_path}.")
        self.model = joblib.load(model_path)
        
        # Load preprocessed feature names
        feature_names_path = os.path.join(config.PREPROCESSING_DIR, "feature_names.joblib")
        if not os.path.exists(feature_names_path):
            raise FileNotFoundError(f"Feature names file not found at {feature_names_path}.")
        self.feature_names = joblib.load(feature_names_path)
        
        # 2. Initialize SHAP explainer (TreeExplainer optimized for XGBoost)
        self.background_data = None
        if use_background_data and os.path.exists(config.X_TRAIN_PATH):
            print("Loading and subsampling training data for SHAP background baseline...")
            X_train = joblib.load(config.X_TRAIN_PATH)
            np.random.seed(config.RANDOM_SEED)
            if X_train.shape[0] > config.SHAP_BACKGROUND_SAMPLES:
                idx = np.random.choice(X_train.shape[0], config.SHAP_BACKGROUND_SAMPLES, replace=False)
                self.background_data = X_train[idx]
            else:
                self.background_data = X_train
            
            # TreeExplainer with background dataset computes baseline from training statistics
            self.shap_explainer = shap.TreeExplainer(self.model, data=self.background_data)
        else:
            # TreeExplainer without data uses path probabilities based on trees
            self.shap_explainer = shap.TreeExplainer(self.model)
            
        # 3. Initialize LIME Tabular Explainer
        if os.path.exists(config.X_TRAIN_PATH):
            print("Loading training data for LIME explainer baseline...")
            X_train = joblib.load(config.X_TRAIN_PATH)
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_train,
                feature_names=self.feature_names,
                class_names=['No Sepsis', 'Sepsis'],
                mode='classification',
                random_state=config.RANDOM_SEED
            )
        else:
            self.lime_explainer = None
            print("Warning: Training data not found. LIME explainer will not be available.")

        print("SepsisExplainer successfully loaded and initialized!")

    def explain_patient_hour(self, preprocessed_row):
        """
        Computes both SHAP and LIME contributions for a single preprocessed observation.
        
        Parameters:
        -----------
        preprocessed_row : numpy.ndarray
            A 1D array or 2D array with 1 row, representing scaled and imputed features for 1 hour.
            
        Returns:
        --------
        dict:
            Structured explanations containing model output, SHAP values, and LIME values.
        """
        # Format input row
        row_1d = np.squeeze(preprocessed_row)
        if row_1d.ndim != 1:
            raise ValueError("preprocessed_row must represent a single row (1D observation).")
        row_2d = row_1d.reshape(1, -1)
        
        # Predict probability
        probs = self.model.predict_proba(row_2d)
        prob = float(probs[0, 1] if len(probs.shape) > 1 and probs.shape[1] > 1 else probs[0])
        pred_class = int(prob >= self.tuned_threshold)
        
        # --- SHAP Local explanation ---
        # Note: tree model SHAP outputs are typically in raw log-odds margins
        shap_vals = self.shap_explainer.shap_values(row_2d)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1] if len(shap_vals) == 2 else shap_vals[0]
        shap_vals = np.squeeze(shap_vals)
        
        base_val = self.shap_explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = base_val[1] if len(base_val) == 2 else base_val[0]
        base_val = float(base_val)
        
        shap_contributions = []
        for name, val, shap_val in zip(self.feature_names, row_1d, shap_vals):
            shap_contributions.append({
                'feature': name,
                'value': float(val),
                'impact': float(shap_val),
                'importance': float(abs(shap_val))
            })
        # Sort by absolute contribution importance
        shap_contributions = sorted(shap_contributions, key=lambda x: x['importance'], reverse=True)
        
        # --- LIME Local explanation ---
        lime_contributions = []
        lime_intercept = 0.0
        lime_class_prob = 0.0
        
        if self.lime_explainer is not None:
            # LIME needs a standardized prediction function that returns (N, 2)
            def predict_fn(x):
                probs_in = self.model.predict_proba(x)
                if len(probs_in.shape) == 1 or probs_in.shape[1] == 1:
                    p = np.squeeze(probs_in)
                    return np.vstack([1 - p, p]).T
                return probs_in
                
            exp = self.lime_explainer.explain_instance(
                data_row=row_1d,
                predict_fn=predict_fn,
                num_features=len(self.feature_names)
            )
            
            # Try to fetch class 1 intercept
            try:
                lime_intercept = float(exp.intercept[1])
            except (AttributeError, KeyError, IndexError):
                lime_intercept = 0.0
                
            local_exp_list = exp.local_exp.get(1, exp.local_exp.get(0, []))
            
            # Map index back to name
            idx_to_name = {i: name for i, name in enumerate(self.feature_names)}
            
            for feat_idx, weight in local_exp_list:
                name = idx_to_name[feat_idx]
                val = float(row_1d[feat_idx])
                lime_contributions.append({
                    'feature': name,
                    'value': val,
                    'impact': float(weight),
                    'importance': float(abs(weight))
                })
            # Sort by LIME weight absolute value
            lime_contributions = sorted(lime_contributions, key=lambda x: x['importance'], reverse=True)
            
        return {
            'prediction': {
                'probability': prob,
                'prediction_class': pred_class,
                'tuned_threshold': self.tuned_threshold
            },
            'shap': {
                'base_value': base_val,
                'prediction_value': float(base_val + np.sum(shap_vals)),
                'contributions': shap_contributions
            },
            'lime': {
                'intercept': lime_intercept,
                'contributions': lime_contributions
            }
        }
        
    def explain_instance_shap(self, preprocessed_row):
        """
        Helper method to get raw SHAP values object for a single patient record (useful for waterfall plots).
        """
        row_2d = preprocessed_row.reshape(1, -1)
        # We need an Explanation object for shap.plots.waterfall
        explanation = self.shap_explainer(row_2d)
        
        # If output is multi-class or list, extract class 1
        if len(explanation.shape) == 3:  # (samples, features, classes)
            # Newer SHAP Explanation objects might be 3D
            # We slice class 1
            explanation = explanation[:, :, 1]
            
        return explanation[0]  # Return the Explanation object for the single row
        
    def explain_instance_lime_obj(self, preprocessed_row):
        """
        Helper method to get the raw LIME explanation object (useful for plotting).
        """
        row_1d = np.squeeze(preprocessed_row)
        def predict_fn(x):
            probs_in = self.model.predict_proba(x)
            if len(probs_in.shape) == 1 or probs_in.shape[1] == 1:
                p = np.squeeze(probs_in)
                return np.vstack([1 - p, p]).T
            return probs_in
            
        return self.lime_explainer.explain_instance(
            data_row=row_1d,
            predict_fn=predict_fn,
            num_features=10
        )
