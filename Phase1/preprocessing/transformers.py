import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import config

class SepsisFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom transformer to perform patient-level forward fill, clinical feature
    engineering, and drop administrative/extremely missing columns.
    """
    def __init__(self, cols_to_drop=None):
        self.cols_to_drop = cols_to_drop if cols_to_drop is not None else []
        self.feature_names_ = None
        
    def fit(self, X, y=None):
        # Nothing to learn from X, just return self
        return self
        
    def transform(self, X):
        X_out = X.copy()
        
        # 1. Patient-level forward fill for vital signs and laboratory values
        cols_to_ffill = [
            c for c in X_out.columns 
            if c not in [config.COL_PATIENT_ID, config.COL_TARGET, 'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']
        ]
        
        if config.COL_PATIENT_ID in X_out.columns:
            # Native pandas groupby ffill is fast and keeps patient alignment
            X_out[cols_to_ffill] = X_out.groupby(config.COL_PATIENT_ID)[cols_to_ffill].ffill()
        else:
            X_out[cols_to_ffill] = X_out[cols_to_ffill].ffill()
            
        # 2. Clinical Feature Engineering
        # Calculate Mean Arterial Pressure (MAP) if missing and SBP/DBP are present: MAP = (SBP + 2*DBP)/3
        if 'MAP' in X_out.columns and 'SBP' in X_out.columns and 'DBP' in X_out.columns:
            map_missing = X_out['MAP'].isnull()
            sbp_dbp_present = X_out['SBP'].notnull() & X_out['DBP'].notnull()
            reconstruct_mask = map_missing & sbp_dbp_present
            
            X_out.loc[reconstruct_mask, 'MAP'] = (
                (X_out.loc[reconstruct_mask, 'SBP'] + 2 * X_out.loc[reconstruct_mask, 'DBP']) / 3.0
            )
            
        # Calculate Shock Index: HR / SBP
        if 'HR' in X_out.columns and 'SBP' in X_out.columns:
            X_out['Shock_Index'] = X_out['HR'] / X_out['SBP']
            # Replace infinities from division by zero with NaN
            X_out['Shock_Index'] = X_out['Shock_Index'].replace([np.inf, -np.inf], np.nan)
            
        # 3. Drop administrative columns and unusable columns
        cols_to_remove = self.cols_to_drop.copy()
        if config.COL_PATIENT_ID in X_out.columns:
            cols_to_remove.append(config.COL_PATIENT_ID)
        if config.COL_TARGET in X_out.columns:
            cols_to_remove.append(config.COL_TARGET)
            
        X_out = X_out.drop(columns=[c for c in cols_to_remove if c in X_out.columns], errors='ignore')
        
        # Store final feature names
        self.feature_names_ = list(X_out.columns)
        return X_out
