import os
import sys
import pandas as pd

# Initialize path settings via config import first
import config

# Import Agent classes from agents submodule
from agents.validation_agent import ValidationAgent
from agents.prediction_agent import PredictionAgent
from agents.explainability_agent import ExplainabilityAgent
from agents.monitoring_agent import MonitoringAgent
from agents.risk_assessment_agent import RiskAssessmentAgent
from agents.alert_agent import AlertAgent

class AgentOrchestrator:
    """
    Agent Orchestrator: Manages the execution sequence and message-passing interface 
    of the clinical agents. Accumulates a step-by-step whiteboard communication log.
    """
    def __init__(self):
        print("\n--- Initializing Sepsis Agent Orchestrator ---")
        
        # Instantiate the 6 specialized agents
        self.validation_agent = ValidationAgent()
        self.prediction_agent = PredictionAgent()
        self.monitoring_agent = MonitoringAgent()
        self.explainability_agent = ExplainabilityAgent()
        self.risk_assessment_agent = RiskAssessmentAgent()
        self.alert_agent = AlertAgent()
        
        print("Sepsis Agent Orchestrator initialized and ready!\n")
        
    def process_patient_record(self, patient_df):
        """
        Runs the full multi-agent pipeline on the patient record.
        
        Parameters:
        -----------
        patient_df : pandas.DataFrame
            The clinical raw vital log. The last row represents the current hour.
            
        Returns:
        --------
        dict:
            Aggregated multi-agent report with validation, prediction, explanations, 
            summaries, alerts, and whiteboard communication logs.
        """
        communication_log = []
        
        def log_message(agent, text):
            msg = f"[{agent}] {text}"
            communication_log.append(msg)
            print(f"  --> {msg}")
            
        print(f"\n[Orchestrator] Processing Patient Record hour {len(patient_df)}...")
        
        # --- Step 1: Validation Agent ---
        log_message("Orchestrator", "Routing patient vitals to ValidationAgent...")
        val_res = self.validation_agent.validate(patient_df)
        
        if not val_res['is_valid']:
            log_message("ValidationAgent", f"CRITICAL: Structural validation failed. Warnings: {val_res['critical_warnings']}")
        if len(val_res['warnings']) > 0:
            log_message("ValidationAgent", f"Noted {len(val_res['warnings'])} mild vital anomalies or missing values.")
            
        # --- Step 2: Prediction Agent ---
        log_message("Orchestrator", "Routing vitals to PredictionAgent...")
        pred_res = self.prediction_agent.score(patient_df)
        log_message(
            "PredictionAgent", 
            f"Sepsis Risk probability calculated at {pred_res['probability']*100:.2f}%. "
            f"Notification Alarm: {pred_res['prediction']} (tuned threshold: {pred_res['tuned_threshold']:.4f})."
        )
        
        # --- Step 3: Monitoring Agent ---
        log_message("Orchestrator", "Routing chronological patient logs to MonitoringAgent...")
        mon_res = self.monitoring_agent.monitor_trends(patient_df)
        if len(mon_res['triggered_warnings']) > 0:
            log_message("MonitoringAgent", f"Identified physiological deterioration: {'; '.join(mon_res['triggered_warnings'])}")
        else:
            log_message("MonitoringAgent", "Vitals trends appear stable over monitored windows.")
            
        # --- Step 4: Explainability Agent ---
        log_message("Orchestrator", "Passing scaled arrays to ExplainabilityAgent...")
        # explainability needs the preprocessed feature array output by predictor
        preprocessed_feats = pred_res['preprocessed_features']
        exp_res = self.explainability_agent.explain(preprocessed_feats)
        
        top_shap = [c['feature'] for c in exp_res['shap']['contributions'][:2]]
        log_message("ExplainabilityAgent", f"Risk explanation compiled. Primary drivers of model prediction: {', '.join(top_shap)}.")
        
        # --- Step 5: Risk Assessment Agent ---
        log_message("Orchestrator", "Synthesizing agent outcomes. Routing variables to RiskAssessmentAgent...")
        assess_res = self.risk_assessment_agent.assess_risk(
            validation_results=val_res,
            prediction_results=pred_res,
            monitoring_results=mon_res,
            explainability_results=exp_res,
            raw_row=patient_df.iloc[-1]
        )
        log_message("RiskAssessmentAgent", f"Clinical summary compiled using {assess_res['method']} engine.")
        
        # --- Step 6: Alert Agent ---
        log_message("Orchestrator", "Checking alert triggers and dispatching message via AlertAgent...")
        alert_res = self.alert_agent.evaluate_alert(
            prediction_results=pred_res,
            monitoring_results=mon_res,
            explainability_results=exp_res
        )
        
        if alert_res['alert_triggered']:
            log_message("AlertAgent", f"ALERT DISPATCHED: {alert_res['dispatch_message']}")
        else:
            log_message("AlertAgent", "Patient is stable. No alerts generated.")
            
        # Assemble final unified clinical report
        log_message("Orchestrator", "Aggregating all agent outputs into final patient report.")
        
        final_report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'current_hour': len(patient_df),
            'validation': {
                'is_valid': val_res['is_valid'],
                'warnings': val_res['warnings'],
                'critical_warnings': val_res['critical_warnings'],
                'validated_vitals': val_res['validated_vitals']
            },
            'prediction': {
                'probability': pred_res['probability'],
                'alert_triggered': pred_res['prediction'] == 1,
                'tuned_threshold': pred_res['tuned_threshold']
            },
            'monitoring': {
                'triggered_warnings': mon_res['triggered_warnings'],
                'trends': mon_res['trends']
            },
            'explainability': {
                'top_shap_contributions': exp_res['shap']['contributions'][:5],
                'top_lime_contributions': exp_res['lime']['contributions'][:5] if 'lime' in exp_res else []
            },
            'risk_assessment': {
                'clinical_summary': assess_res['clinical_summary'],
                'generation_method': assess_res['method']
            },
            'alert': alert_res,
            'whiteboard_communication_log': communication_log
        }
        
        return final_report
