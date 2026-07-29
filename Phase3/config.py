import os
import sys

# Paths setup
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PHASE1_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "Phase1"))
PHASE2_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "Phase2"))

# Add Phase 1 and Phase 2 to path to enable imports
for p in [PHASE1_PATH, PHASE2_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Phase 1 references
DATA_PROCESSED_DIR = os.path.join(PHASE1_PATH, "data", "processed")
MODELS_DIR = os.path.join(PHASE1_PATH, "models")
PREPROCESSING_DIR = os.path.join(PHASE1_PATH, "preprocessing")

# Paths to load Phase 1 assets
X_TRAIN_PATH = os.path.join(DATA_PROCESSED_DIR, "X_train.joblib")
Y_TRAIN_PATH = os.path.join(DATA_PROCESSED_DIR, "y_train.joblib")
X_TEST_PATH = os.path.join(DATA_PROCESSED_DIR, "X_test.joblib")
Y_TEST_PATH = os.path.join(DATA_PROCESSED_DIR, "y_test.joblib")
PREPROCESSOR_PATH = os.path.join(PREPROCESSING_DIR, "preprocessor.joblib")
BEST_MODEL_INFO_PATH = os.path.join(MODELS_DIR, "best_model_info.joblib")

# Biological Range Configurations for ValidationAgent
BIOLOGICAL_RANGES = {
    'HR': {'min': 30, 'max': 250, 'label': 'Heart Rate (bpm)'},
    'O2Sat': {'min': 50, 'max': 100, 'label': 'O2 Saturation (%)'},
    'Temp': {'min': 30, 'max': 45, 'label': 'Temperature (°C)'},
    'SBP': {'min': 40, 'max': 250, 'label': 'Systolic Blood Pressure (mmHg)'},
    'MAP': {'min': 30, 'max': 200, 'label': 'Mean Arterial Pressure (mmHg)'},
    'DBP': {'min': 20, 'max': 150, 'label': 'Diastolic Blood Pressure (mmHg)'},
    'Resp': {'min': 4, 'max': 60, 'label': 'Respiration Rate (breaths/min)'},
    'Age': {'min': 0, 'max': 120, 'label': 'Age (years)'}
}

# Trend Configurations for MonitoringAgent
TREND_MONITOR_WINDOWS = {
    'HR': {'window': 4, 'threshold': 15, 'direction': 'increase', 'label': 'Heart Rate Rise'},
    'Shock_Index': {'window': 3, 'threshold': 0.15, 'direction': 'increase', 'label': 'Shock Index Elevation'},
    'MAP': {'window': 4, 'threshold': -15, 'direction': 'decrease', 'label': 'MAP Drop'},
    'O2Sat': {'window': 3, 'threshold': -5, 'direction': 'decrease', 'label': 'Oxygen Desaturation'}
}

# Prompt Settings for RiskAssessmentAgent
RISK_ASSESSMENT_PROMPT_TEMPLATE = """
You are an expert ICU Clinical Risk Assessment Agent.
Review the following patient hourly observation and analyze the clinical metrics to write a structured risk assessment report.

==================================================
PATIENT CLINICAL DATA SUMMARY
==================================================
- Sepsis Probability Score (XGBoost): {probability:.2f}% (Tuned Warning Threshold: {tuned_threshold:.2f}%)
- Sepsis Alarm Class: {alarm_triggered}

Top SHAP Features explaining Sepsis Risk (Log-Odds contributions):
{shap_contributions}

Patient Vitals Validation Alerts:
{validation_alerts}

Recent Vitals Trends over past 4 hours:
{vitals_trends}

Current Physiological Vitals:
{current_vitals}

==================================================
INSTRUCTIONS
==================================================
Write a professional, concise ICU Clinical Summary (maximum 150-200 words) explaining the patient's physiological state. 
Your report must:
1. Address the primary physiological drivers of the risk score based on the SHAP contributions.
2. Discuss validation warnings or recent trend deteriorations (e.g. cardiovascular stability, respiratory work, or renal clearance).
3. Conclude with a clear recommendation (e.g. "Recommend ordering blood cultures", "Recommend lactate re-draw", "No immediate sepsis action, continue monitoring").
Do NOT use patient names or generate mock facts not specified above. Maintain a clinical, objective tone.
"""

# XAI compatibility settings
RANDOM_SEED = 42
SHAP_BACKGROUND_SAMPLES = 200

# Prediction schema settings
COL_PATIENT_ID = 'PatientID'
COL_TARGET = 'SepsisLabel'


