import os
import sys
from agents.base import BaseAgent
import config

# Import SepsisPredictor from Phase 1
from predict import SepsisPredictor

class PredictionAgent(BaseAgent):
    """
    Prediction Agent: Interfaces with the trained XGBoost model and preprocessor 
    to score sepsis risk probabilities for incoming patient streams.
    """
    def __init__(self):
        super().__init__("PredictionAgent")
        self.log_action("Loading SepsisPredictor model foundation...")
        
        # Load from config paths
        self.predictor = SepsisPredictor(
            models_dir=config.MODELS_DIR,
            preprocessor_path=config.PREPROCESSOR_PATH,
            best_model_info_path=config.BEST_MODEL_INFO_PATH
        )
        
    def score(self, patient_df):
        """
        Runs the sepsis prediction scoring pipeline for a patient record.
        
        Parameters:
        -----------
        patient_df : pandas.DataFrame
            DataFrame with patient metrics matching raw PhysioNet columns.
            
        Returns:
        --------
        dict:
            Contains 'probability', 'prediction' (flag), and 'preprocessed_features'.
        """
        self.log_action("Scoring current sepsis risk probability...")
        
        # Predict patient data using SepsisPredictor
        results = self.predictor.predict_patient(patient_df)
        
        # Get the latest hour's scores
        latest_prob = float(results['probabilities'][-1])
        latest_pred = int(results['predictions'][-1])
        
        self.log_action(
            f"Risk Scored: Prob = {latest_prob*100:.2f}%, "
            f"Prediction Alert = {latest_pred} (Threshold = {self.predictor.tuned_threshold:.4f})"
        )
        
        return {
            'probability': latest_prob,
            'prediction': latest_pred,
            'tuned_threshold': self.predictor.tuned_threshold,
            'all_probabilities': results['probabilities'],
            'preprocessed_features': results['preprocessed_features'],
            'feature_names': results['feature_names']
        }
