import os
import time
import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import config

def train_models():
    print("Loading preprocessed training data...")
    X_train = joblib.load(config.X_TRAIN_PATH)
    y_train = joblib.load(config.Y_TRAIN_PATH)
    
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"Positive samples: {np.sum(y_train == 1)} ({np.mean(y_train == 1)*100:.2f}%)")
    print(f"Negative samples: {np.sum(y_train == 0)} ({np.mean(y_train == 0)*100:.2f}%)")
    
    # Create models directory if not exists
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    # 1. Train Decision Tree
    print("\n--- Training Decision Tree Classifier ---")
    print(f"Parameters: {config.DT_PARAMS}")
    dt_model = DecisionTreeClassifier(**config.DT_PARAMS)
    
    start_time = time.time()
    dt_model.fit(X_train, y_train)
    dt_time = time.time() - start_time
    print(f"Decision Tree trained in {dt_time:.2f} seconds.")
    
    dt_path = os.path.join(config.MODELS_DIR, "decision_tree.joblib")
    joblib.dump(dt_model, dt_path)
    print(f"Saved Decision Tree to {dt_path}")
    
    # 2. Train Random Forest
    print("\n--- Training Random Forest Classifier ---")
    print(f"Parameters: {config.RF_PARAMS}")
    rf_model = RandomForestClassifier(**config.RF_PARAMS)
    
    start_time = time.time()
    rf_model.fit(X_train, y_train)
    rf_time = time.time() - start_time
    print(f"Random Forest trained in {rf_time:.2f} seconds.")
    
    rf_path = os.path.join(config.MODELS_DIR, "random_forest.joblib")
    joblib.dump(rf_model, rf_path)
    print(f"Saved Random Forest to {rf_path}")
    
    # 3. Train XGBoost
    print("\n--- Training XGBoost Classifier ---")
    # Dynamically calculate scale_pos_weight for XGBoost to handle extreme class imbalance
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count
    
    xgb_params = config.XGB_PARAMS.copy()
    xgb_params['scale_pos_weight'] = scale_pos_weight
    
    print(f"Parameters: {xgb_params}")
    xgb_model = XGBClassifier(**xgb_params)
    
    start_time = time.time()
    xgb_model.fit(X_train, y_train)
    xgb_time = time.time() - start_time
    print(f"XGBoost trained in {xgb_time:.2f} seconds.")
    
    xgb_path = os.path.join(config.MODELS_DIR, "xgboost.joblib")
    joblib.dump(xgb_model, xgb_path)
    print(f"Saved XGBoost to {xgb_path}")
    
    print("\nAll models trained and saved successfully!")

if __name__ == "__main__":
    train_models()
