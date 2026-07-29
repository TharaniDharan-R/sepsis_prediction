import os
import joblib
import pandas as pd
import numpy as np
import config
from preprocessing.transformers import SepsisFeatureExtractor

class SepsisPredictor:
    """
    Reusable prediction module for early sepsis prediction.
    Loads the saved preprocessor, best model, and tuned decision threshold.
    Accepts hourly patient logs, preprocesses them, and returns sepsis probabilities and flags.
    """
    def __init__(self, models_dir=None, preprocessor_path=None, best_model_info_path=None):
        self.models_dir = models_dir if models_dir is not None else config.MODELS_DIR
        self.preprocessor_path = preprocessor_path if preprocessor_path is not None else config.PREPROCESSOR_PATH
        self.best_model_info_path = best_model_info_path if best_model_info_path is not None else config.BEST_MODEL_INFO_PATH
        
        self.pipeline = None
        self.best_model = None
        self.model_name = None
        self.tuned_threshold = 0.5
        self.feature_names = None
        
        self._load_predictor()
        
    def _load_predictor(self):
        """
        Load preprocessor and best model assets.
        """
        print(f"Loading preprocessor from {self.preprocessor_path}...")
        if not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(f"Preprocessor not found at {self.preprocessor_path}. Run preprocess.py first.")
        self.pipeline = joblib.load(self.preprocessor_path)
        
        print(f"Loading best model info from {self.best_model_info_path}...")
        if not os.path.exists(self.best_model_info_path):
            raise FileNotFoundError(f"Best model info not found at {self.best_model_info_path}. Run evaluate.py first.")
        best_info = joblib.load(self.best_model_info_path)
        
        self.model_name = best_info['metadata']['model_name']
        self.tuned_threshold = best_info['metadata']['tuned_threshold']
        self.feature_names = best_info['metadata']['feature_names']
        
        model_filename = f"{self.model_name.lower().replace(' ', '_')}.joblib"
        model_path = os.path.join(self.models_dir, model_filename)
        
        print(f"Loading best model ({self.model_name}) from {model_path}...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Run train.py first.")
        self.best_model = joblib.load(model_path)
        
        print("SepsisPredictor successfully loaded and ready for inference!")
        print(f"Active Model: {self.model_name}")
        print(f"Tuned Decision Threshold: {self.tuned_threshold:.4f}")
        
    def predict_patient(self, patient_df):
        """
        Predict sepsis risk for a single patient's hourly log sequence.
        
        Parameters:
        -----------
        patient_df : pandas.DataFrame
            DataFrame with columns matching the raw PhysioNet challenge columns.
            Can represent one hour or multiple hours of records.
            
        Returns:
        --------
        dict:
            'probabilities': list of float risk probabilities for each hour.
            'predictions': list of binary flags (0 or 1) indicating sepsis risk.
            'preprocessed_features': numpy.ndarray of scaled and imputed features (used for SHAP/LIME).
            'feature_names': list of preprocessed feature names.
        """
        # Ensure input is a DataFrame
        if not isinstance(patient_df, pd.DataFrame):
            raise TypeError("Input patient_df must be a pandas DataFrame.")
            
        # If PatientID column is missing, add a dummy one (as our preprocessor relies on it if present)
        df_in = patient_df.copy()
        if config.COL_PATIENT_ID not in df_in.columns:
            # SepsisFeatureExtractor groups by PatientID if present. If it's a single patient, 
            # we can create a dummy PatientID.
            df_in[config.COL_PATIENT_ID] = "patient_dummy"
            
        # Run through preprocessor pipeline
        # Note: the pipeline outputs a scaled/imputed numpy array
        X_preprocessed = self.pipeline.transform(df_in)
        
        # Predict probabilities
        if self.model_name == 'XGBoost':
            probs = self.best_model.predict_proba(X_preprocessed)
            y_prob = probs[:, 1] if len(probs.shape) > 1 else probs
        else:
            y_prob = self.best_model.predict_proba(X_preprocessed)[:, 1]
            
        # Predict binary classifications based on the optimized threshold
        y_pred = (y_prob >= self.tuned_threshold).astype(int)
        
        return {
            'probabilities': y_prob.tolist(),
            'predictions': y_pred.tolist(),
            'preprocessed_features': X_preprocessed,
            'feature_names': self.feature_names
        }

if __name__ == "__main__":
    # Self-test code: run prediction on a sample patient file
    print("Running self-test...")
    import glob
    raw_files = glob.glob(os.path.join(config.DATA_RAW_DIR, "*.psv"))
    if len(raw_files) > 0:
        sample_file = raw_files[0]
        print(f"Reading sample patient file: {sample_file}")
        sample_df = pd.read_csv(sample_file, sep='|')
        
        # Drop SepsisLabel to simulate actual inference where the target is unknown
        if config.COL_TARGET in sample_df.columns:
            true_labels = sample_df[config.COL_TARGET].tolist()
            sample_df = sample_df.drop(columns=[config.COL_TARGET])
        else:
            true_labels = None
            
        predictor = SepsisPredictor()
        result = predictor.predict_patient(sample_df)
        
        print("\nPrediction Results:")
        for hr in range(len(result['probabilities'])):
            prob = result['probabilities'][hr]
            pred = result['predictions'][hr]
            true_lbl = true_labels[hr] if true_labels is not None else "Unknown"
            print(f"Hour {hr+1:02d} | Risk Prob: {prob*100:5.2f}% | Pred: {pred} | True Label: {true_lbl}")
    else:
        print("No raw PSV files found to run self-test.")
