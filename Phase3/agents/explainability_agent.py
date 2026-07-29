import os
import sys
from agents.base import BaseAgent
import config

# Import SepsisExplainer from Phase 2
from explainers import SepsisExplainer

class ExplainabilityAgent(BaseAgent):
    """
    Explainability Agent: Interfaces with the SHAP and LIME explainers 
    to extract local feature contributions for predictions.
    """
    def __init__(self):
        super().__init__("ExplainabilityAgent")
        self.log_action("Loading SepsisExplainer foundation...")
        self.explainer = SepsisExplainer(use_background_data=True)
        
    def explain(self, preprocessed_features):
        """
        Generates feature-level explanations for the latest prediction hour.
        
        Parameters:
        -----------
        preprocessed_features : numpy.ndarray
            Scaled and imputed feature array from the latest transformation.
            Can represent multiple hours; we explain the latest hour (last row).
            
        Returns:
        --------
        dict:
            Structured contribution logs containing SHAP and LIME values.
        """
        self.log_action("Generating local SHAP and LIME explanations for patient risk...")
        
        # Get the preprocessed row for the latest hour (last row in array)
        latest_row = preprocessed_features[-1]
        
        # Explain the row
        explanation = self.explainer.explain_patient_hour(latest_row)
        
        # Log top 3 features by absolute SHAP impact
        top_3 = [
            f"{c['feature']} ({c['impact']:.3f})"
            for c in explanation['shap']['contributions'][:3]
        ]
        self.log_action(f"Top 3 risk drivers (SHAP): {', '.join(top_3)}")
        
        return explanation
