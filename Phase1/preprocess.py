import os
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from preprocessing.transformers import SepsisFeatureExtractor
import config

def build_and_run_pipeline():
    print("Loading raw processed CSV files...")
    train_df = pd.read_csv(config.TRAIN_PATIENTS_CSV)
    test_df = pd.read_csv(config.TEST_PATIENTS_CSV)
    
    print(f"Train raw shape: {train_df.shape}")
    print(f"Test raw shape:  {test_df.shape}")
    
    # Extract targets
    y_train = train_df[config.COL_TARGET].values
    y_test = test_df[config.COL_TARGET].values
    
    # Build Scikit-Learn Pipeline
    print("Building the preprocessing pipeline...")
    pipeline = Pipeline([
        ('feature_extractor', SepsisFeatureExtractor(cols_to_drop=config.COL_DROP)),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Fit on training data and transform both sets
    print("Fitting and transforming training data...")
    X_train_preprocessed = pipeline.fit_transform(train_df)
    
    print("Transforming testing data...")
    X_test_preprocessed = pipeline.transform(test_df)
    
    # Extract feature names from fitted transformer
    feature_names = pipeline.named_steps['feature_extractor'].feature_names_
    print(f"\nEngineered feature list ({len(feature_names)} features):")
    print(feature_names)
    
    # Ensure directories exist
    os.makedirs(config.PREPROCESSING_DIR, exist_ok=True)
    os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)
    
    # Save the fitted pipeline
    print(f"\nSaving preprocessor pipeline to {config.PREPROCESSOR_PATH}...")
    joblib.dump(pipeline, config.PREPROCESSOR_PATH)
    
    # Save the preprocessed numpy arrays
    print(f"Saving preprocessed arrays to {config.DATA_PROCESSED_DIR}...")
    joblib.dump(X_train_preprocessed, config.X_TRAIN_PATH)
    joblib.dump(y_train, config.Y_TRAIN_PATH)
    joblib.dump(X_test_preprocessed, config.X_TEST_PATH)
    joblib.dump(y_test, config.Y_TEST_PATH)
    
    # Save feature names separately for convenience
    feature_names_path = os.path.join(config.PREPROCESSING_DIR, "feature_names.joblib")
    joblib.dump(feature_names, feature_names_path)
    
    print("Preprocessing successfully completed!")
    print(f"X_train preprocessed shape: {X_train_preprocessed.shape}")
    print(f"X_test preprocessed shape:  {X_test_preprocessed.shape}")

if __name__ == "__main__":
    build_and_run_pipeline()
