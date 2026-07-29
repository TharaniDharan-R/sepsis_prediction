import pandas as pd
import numpy as np
from agents.base import BaseAgent
import config

class MonitoringAgent(BaseAgent):
    """
    Monitoring Agent: Analyzes patient vitals over time to identify physiological trends,
    such as cardiovascular deterioration or rapid respiratory decline.
    """
    def __init__(self):
        super().__init__("MonitoringAgent")
        
    def monitor_trends(self, patient_df):
        """
        Analyzes rates of change in vitals over historical hours.
        
        Parameters:
        -----------
        patient_df : pandas.DataFrame
            DataFrame containing raw patient records over multiple hours.
            
        Returns:
        --------
        dict:
            'trends': dict of computed differences for monitored features
            'triggered_warnings': list of warning strings detailing detected deterioration
        """
        self.log_action("Monitoring patient historical vitals trends...")
        
        trends = {}
        triggered_warnings = []
        
        if len(patient_df) < 2:
            self.log_action("Single observation hour. Insufficient data for trend monitoring.")
            return {'trends': trends, 'triggered_warnings': triggered_warnings}
            
        # Create a copy to compute derived vitals for trend checking
        df_trends = patient_df.copy()
        
        # Calculate Shock Index if missing (HR / SBP)
        if 'Shock_Index' not in df_trends.columns and 'HR' in df_trends.columns and 'SBP' in df_trends.columns:
            # Avoid division by zero
            sbp = df_trends['SBP'].replace(0, np.nan)
            df_trends['Shock_Index'] = df_trends['HR'] / sbp
            
        latest_idx = len(df_trends) - 1
        
        for col, rules in config.TREND_MONITOR_WINDOWS.items():
            if col not in df_trends.columns:
                continue
                
            window = rules['window']
            threshold = rules['threshold']
            direction = rules['direction']
            label = rules['label']
            
            # Determine start index for lookback
            lookback_idx = max(0, latest_idx - window + 1)
            actual_window = latest_idx - lookback_idx + 1
            
            latest_val = df_trends[col].iloc[latest_idx]
            past_val = df_trends[col].iloc[lookback_idx]
            
            if pd.isna(latest_val) or pd.isna(past_val):
                continue
                
            diff = float(latest_val - past_val)
            trends[col] = {
                'latest_value': float(latest_val),
                'past_value': float(past_val),
                'difference': diff,
                'lookback_hours': actual_window
            }
            
            # Evaluate trigger conditions
            is_warning = False
            if direction == 'increase' and diff >= threshold:
                is_warning = True
                warning_desc = f"{label}: increased by {diff:.2f} over the last {actual_window} hours (value: {past_val:.1f} -> {latest_val:.1f})"
            elif direction == 'decrease' and diff <= threshold:
                is_warning = True
                warning_desc = f"{label}: decreased by {abs(diff):.2f} over the last {actual_window} hours (value: {past_val:.1f} -> {latest_val:.1f})"
                
            if is_warning:
                triggered_warnings.append(warning_desc)
                self.log_action(f"ALERT: {warning_desc}")
                
        self.log_action(f"Trend monitoring complete. Triggered warnings: {len(triggered_warnings)}")
        return {
            'trends': trends,
            'triggered_warnings': triggered_warnings
        }
