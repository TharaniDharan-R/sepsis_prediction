import pandas as pd
import numpy as np
from agents.base import BaseAgent
import config

class RiskAssessmentAgent(BaseAgent):
    """
    Risk Assessment Agent: Synthesizes validation alerts, ML model probabilities, 
    vitals trends, and SHAP contributions using LLM prompt engineering.
    Falls back to a clinical rule-based summary when LLM keys are absent.
    """
    def __init__(self):
        super().__init__("RiskAssessmentAgent")
        
    def assess_risk(self, validation_results, prediction_results, monitoring_results, explainability_results, raw_row):
        """
        Generates a clinical risk summary for the patient.
        """
        self.log_action("Synthesizing patient metrics and generating clinical risk summary...")
        
        prob = prediction_results['probability'] * 100
        threshold = prediction_results['tuned_threshold'] * 100
        alarm = "TRIGGERED" if prediction_results['prediction'] == 1 else "STABLE"
        
        # 1. Format SHAP contributions for the prompt
        shap_contributions_str = ""
        shap_contribs = explainability_results['shap']['contributions'][:5]
        for c in shap_contribs:
            shap_contributions_str += f"  - Feature: {c['feature']} | Val: {c['value']:.2f} | Log-Odds Contribution: {c['impact']:.3f}\n"
            
        # 2. Format validation warnings
        validation_str = ""
        warnings_list = validation_results['warnings'] + validation_results['critical_warnings']
        if len(warnings_list) > 0:
            for w in warnings_list[:5]:
                validation_str += f"  - {w}\n"
        else:
            validation_str = "  - No vital range violations or critical data missing."
            
        # 3. Format vitals trends
        trends_str = ""
        trend_warnings = monitoring_results['triggered_warnings']
        if len(trend_warnings) > 0:
            for t in trend_warnings:
                trends_str += f"  - {t}\n"
        else:
            trends_str = "  - Vitals trends stable over the monitored window."
            
        # 4. Format current vitals
        current_vitals_str = ""
        for k, v in validation_results['validated_vitals'].items():
            if isinstance(v, float):
                current_vitals_str += f"{k}: {v:.1f} | "
            else:
                current_vitals_str += f"{k}: {v} | "
        current_vitals_str = current_vitals_str.strip(" | ")
        
        # 5. Populate LLM prompt
        prompt = config.RISK_ASSESSMENT_PROMPT_TEMPLATE.format(
            probability=prob,
            tuned_threshold=threshold,
            alarm_triggered=alarm,
            shap_contributions=shap_contributions_str,
            validation_alerts=validation_str,
            vitals_trends=trends_str,
            current_vitals=current_vitals_str
        )
        
        system_instruction = (
            "You are a clinical ICU risk assessment agent. Your goal is to write "
            "clear, professional summaries of patient sepsis risk to help doctors."
        )
        
        # 6. Attempt LLM Query
        llm_response = self.query_llm(prompt, system_instruction)
        
        if llm_response:
            self.log_action("LLM summary successfully generated.")
            return {
                'clinical_summary': llm_response,
                'method': 'LLM (Generative)'
            }
            
        # 7. Mock Fallback (Deterministic clinical rule-based generation)
        self.log_action("No LLM key configured. Generating rule-based clinical report...")
        fallback_summary = self._generate_rule_based_report(
            prob, threshold, alarm, shap_contribs, warnings_list, trend_warnings, validation_results['validated_vitals']
        )
        
        return {
            'clinical_summary': fallback_summary,
            'method': 'Rule-Based Fallback'
        }
        
    def _generate_rule_based_report(self, prob, threshold, alarm, shap_contribs, warnings_list, trend_warnings, current_vitals):
        """
        Builds a medically coherent patient report using deterministic clinical rules.
        """
        # Determine overall risk category
        if prob >= threshold:
            risk_cat = "HIGH"
        elif prob >= 10.0:
            risk_cat = "MODERATE"
        else:
            risk_cat = "LOW"
            
        summary = f"CLINICAL RISK ASSESSMENT (STATUS: {risk_cat} RISK)\n"
        summary += f"The patient is evaluated at a {prob:.1f}% risk of developing early-onset sepsis (Sepsis Alarm: {alarm}).\n\n"
        
        # Explain physiological drivers from SHAP
        summary += "PHYSIOLOGICAL DRIVERS:\n"
        positive_drivers = [c for c in shap_contribs if c['impact'] > 0]
        if len(positive_drivers) > 0:
            drivers_desc = []
            for d in positive_drivers[:3]:
                drivers_desc.append(f"{d['feature']} (value: {d['value']:.2f}, impact: +{d['impact']:.2f} log-odds)")
            summary += f"Sepsis likelihood is actively driven by: {', '.join(drivers_desc)}.\n"
        else:
            summary += "No severe physiological markers are contributing to sepsis risk.\n"
            
        # Discuss trend warnings or missing signs
        summary += "\nCLINICAL OBSERVATIONS:\n"
        obs = []
        if len(trend_warnings) > 0:
            obs.append("vitals trend logs indicate patient deterioration (" + "; ".join(trend_warnings[:2]) + ")")
        if any("missing" in w.lower() for w in warnings_list):
            missing_vitals = [w for w in warnings_list if "missing" in w.lower()]
            obs.append("monitoring values are incomplete (" + "; ".join(missing_vitals[:2]) + ")")
            
        if len(obs) > 0:
            summary += "Specifically, " + " and ".join(obs) + ".\n"
        else:
            summary += "Vital sign tracking indicates cardiorespiratory and renal homeostasis is currently maintained.\n"
            
        # Conclude with a solid recommendation
        summary += "\nCLINICAL RECOMMENDATION:\n"
        if risk_cat == "HIGH":
            summary += (
                "[CRITICAL] Sepsis notification triggered. Recommend immediately ordering blood cultures "
                "and serum lactate levels, reviewing fluid resuscitation balance, and initiating empiric "
                "broad-spectrum antibiotics within 1 hour."
            )
        elif risk_cat == "MODERATE" or len(trend_warnings) > 0:
            summary += (
                "[WARNING] Sepsis risk is borderline or show negative physiological trends. Recommend close monitoring: "
                "re-draw serum lactate levels in 2-4 hours, perform targeted head-to-toe physical assessment, "
                "and re-evaluate vitals hourly."
            )
        else:
            summary += "[INFO] Sepsis risk is low. Recommend continuing standard hourly vital checks and protocol monitoring."
            
        return summary
