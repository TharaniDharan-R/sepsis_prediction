import os

# Base paths
BASE_DIR = r"D:\sepsis\Phase1"
DATA_RAW_DIR = r"C:\Users\Admin\Downloads\sepsis_dataset\physionet.org\files\challenge-2019\1.0.0\training\training_setA"
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PREPROCESSING_DIR = os.path.join(BASE_DIR, "preprocessing")
EVALUATION_DIR = os.path.join(BASE_DIR, "evaluation")
VISUALIZATIONS_DIR = os.path.join(BASE_DIR, "visualizations")

# File paths
TRAIN_PATIENTS_CSV = os.path.join(DATA_PROCESSED_DIR, "train_patients.csv")
TEST_PATIENTS_CSV = os.path.join(DATA_PROCESSED_DIR, "test_patients.csv")
PREPROCESSOR_PATH = os.path.join(PREPROCESSING_DIR, "preprocessor.joblib")
BEST_MODEL_INFO_PATH = os.path.join(MODELS_DIR, "best_model_info.joblib")

# Preprocessed NumPy data paths
X_TRAIN_PATH = os.path.join(DATA_PROCESSED_DIR, "X_train.joblib")
Y_TRAIN_PATH = os.path.join(DATA_PROCESSED_DIR, "y_train.joblib")
X_TEST_PATH = os.path.join(DATA_PROCESSED_DIR, "X_test.joblib")
Y_TEST_PATH = os.path.join(DATA_PROCESSED_DIR, "y_test.joblib")

# Feature Column Groups
COL_VITALS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
COL_LABS = [
    'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
    'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
    'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
    'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
    'Fibrinogen', 'Platelets'
]
COL_DEMOGRAPHICS = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']
COL_TARGET = 'SepsisLabel'
COL_PATIENT_ID = 'PatientID'

# Features to drop due to extreme missingness (e.g. 100% missing values)
COL_DROP = ['EtCO2']

# Derived clinical features
COL_ENGINEERED = ['Shock_Index']

# All raw columns (for initial loading)
ALL_RAW_COLUMNS = COL_VITALS + COL_LABS + COL_DEMOGRAPHICS + [COL_TARGET]

# Hyperparameters for ML Models
DT_PARAMS = {
    'max_depth': 6,
    'min_samples_split': 50,
    'min_samples_leaf': 20,
    'class_weight': 'balanced',
    'random_state': 42
}

RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 8,
    'min_samples_split': 50,
    'min_samples_leaf': 20,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1
}

XGB_PARAMS = {
    'n_estimators': 120,
    'max_depth': 5,
    'learning_rate': 0.08,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'eval_metric': 'logloss'
}

# Split and Seed settings
TRAIN_TEST_SPLIT_RATIO = 0.8  # 80% train, 20% test
RANDOM_SEED = 42

# Default target recall for threshold tuning
TARGET_RECALL = 0.85
