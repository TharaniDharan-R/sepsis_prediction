import pandas as pd
import numpy as np
from agents.base import BaseAgent
import config

class ValidationAgent(BaseAgent):
    """
    Validation Agent: Inspects raw patient data for structural completeness, 
    biological plausibility, and critical missing measurements.
    """
    def __init__(self):
        super().__init__("ValidationAgent")
        
    def validate(self, patient_df):
        """
        Validates the latest hour's observation of the patient record.
        
        Parameters:
        -----------
        patient_df : pandas.DataFrame
            DataFrame containing raw patient records. The last row represents the current hour.
            
        Returns:
        --------
        dict:
            'is_valid': bool (False only if critical columns are completely missing)
            'warnings': list of strings detailing mild anomalies or missing values
            'critical_warnings': list of strings detailing severe physiological violations
            'validated_vitals': dict of the current hour's key vitals
        """
        self.log_action("Validating patient vitals for current hour...")
        
        warnings = []
        critical_warnings = []
        is_valid = True
        
        if not isinstance(patient_df, pd.DataFrame):
            critical_warnings.append("Input data is not a pandas DataFrame.")
            return {'is_valid': False, 'warnings': warnings, 'critical_warnings': critical_warnings, 'validated_vitals': {}}
            
        if len(patient_df) == 0:
            critical_warnings.append("Patient DataFrame is empty.")
            return {'is_valid': False, 'warnings': warnings, 'critical_warnings': critical_warnings, 'validated_vitals': {}}
            
        # Get the current hour's data (latest row)
        current_row = patient_df.iloc[-1]
        
        validated_vitals = {}
        
        # 1. Biological Range Validations
        for col, rules in config.BIOLOGICAL_RANGES.items():
            if col not in patient_df.columns:
                warnings.append(f"Column '{col}' ({rules['label']}) is missing from schema.")
                continue
                
            val = current_row[col]
            
            # Check for missingness
            if pd.isna(val) or val is None:
                # Standard vital signs missingness check
                if col in ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'Resp']:
                    warnings.append(f"Critical vital sign '{col}' ({rules['label']}) is missing/NaN.")
                else:
                    warnings.append(f"Lab/demographic value '{col}' ({rules['label']}) is missing/NaN.")
                validated_vitals[col] = "Missing"
                continue
                
            # Range check
            val_float = float(val)
            validated_vitals[col] = val_float
            
            if val_float < rules['min'] or val_float > rules['max']:
                err_msg = (
                    f"Biological alert: {rules['label']} = {val_float} is outside "
                    f"plausible physiological limits [{rules['min']}, {rules['max']}]."
                )
                if col in ['HR', 'Temp', 'SBP', 'O2Sat']:
                    critical_warnings.append(err_msg)
                else:
                    warnings.append(err_msg)
                    
        self.log_action(f"Validation complete. Warnings: {len(warnings)}, Critical Warnings: {len(critical_warnings)}")
        
        return {
            'is_valid': len(critical_warnings) == 0,
            'warnings': warnings,
            'critical_warnings': critical_warnings,
            'validated_vitals': validated_vitals
        }
