import pandas as pd
import numpy as np
from agents.base import BaseAgent
import config

class AlertAgent(BaseAgent):
    """
    Alert Agent: Decides if an alert is warranted based on prediction scores, 
    clinical trends, and generates pager-friendly clinical dispatches.
    """
    def __init__(self):
        super().__init__("AlertAgent")
        
    def evaluate_alert(self, prediction_results, monitoring_results, explainability_results):
        """
        Determines the alert status and formats a dispatch notification.
        
        Parameters:
        -----------
        prediction_results : dict
            Output from PredictionAgent
        monitoring_results : dict
            Output from MonitoringAgent
        explainability_results : dict
            Output from ExplainabilityAgent
            
        Returns:
        --------
        dict:
            'alert_triggered': bool
            'alert_level': str ('CRITICAL', 'WARNING', 'INFO', or 'NONE')
            'dispatch_message': str
        """
        self.log_action("Evaluating alert triggers and formatting pager dispatch...")
        
        prob = prediction_results['probability']
        threshold = prediction_results['tuned_threshold']
        model_flag = prediction_results['prediction']
        
        trend_warnings = monitoring_results['triggered_warnings']
        
        alert_triggered = False
        alert_level = "NONE"
        dispatch_message = ""
        
        # Sepsis alert trigger rules:
        # 1. Model prediction flags it (exceeds tuned threshold)
        # 2. Or clinical trends show severe dual organ/circulatory decay (e.g., 2+ trend warnings)
        if model_flag == 1 or len(trend_warnings) >= 2:
            alert_triggered = True
            
            # Determine alert severity level
            if prob >= 0.50:
                alert_level = "CRITICAL"
            elif prob >= threshold:
                alert_level = "WARNING"
            else:
                alert_level = "INFO"  # Triggered by trend deterioration only
                
        # Format the short dispatch message (pager style)
        if alert_triggered:
            # Get top 2 SHAP drivers
            top_drivers = [
                c['feature'] for c in explainability_results['shap']['contributions'][:2]
            ]
            drivers_desc = ", ".join(top_drivers)
            
            trend_desc = ""
            if len(trend_warnings) > 0:
                # Get the first word/label of the trend warning
                trend_desc = f" | Trend: {trend_warnings[0].split(':')[0]}"
                
            dispatch_message = (
                f"[{alert_level} SEPSIS ALERT] Sepsis Prob: {prob*100:.1f}% "
                f"(Thr: {threshold*100:.1f}%) | Primary Drivers: {drivers_desc}{trend_desc}"
            )
            self.log_action(f"ALERT DISPATCHED: {dispatch_message}")
        else:
            dispatch_message = "Patient stable. Sepsis alert not triggered."
            self.log_action("No alerts triggered. Patient stable.")
            
        return {
            'alert_triggered': alert_triggered,
            'alert_level': alert_level,
            'dispatch_message': dispatch_message
        }
