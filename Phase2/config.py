import os
import sys

# Add Phase 1 directory to python system path to resolve module imports like preprocessing.transformers
PHASE1_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Phase1"))
if PHASE1_PATH not in sys.path:
    sys.path.insert(0, PHASE1_PATH)

# Phase 2 Paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
VISUALIZATIONS_DIR = os.path.join(BASE_DIR, "visualizations")
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

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

# XAI specific settings
SHAP_BACKGROUND_SAMPLES = 200  # Number of samples to summarize training data for SHAP explainer
SHAP_TEST_SAMPLES = 500         # Number of test samples to calculate SHAP values for global charts
RANDOM_SEED = 42
