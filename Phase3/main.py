import os
import sys
import glob
import json
import pandas as pd

# Add Phase 3 config and imports
import config
from orchestrator import AgentOrchestrator

def run_icu_agentic_simulation():
    print("======================================================================")
    
    # 1. Initialize Orchestrator
    orchestrator = AgentOrchestrator()
    
    # 2. Find a sample patient file
    import importlib.util
    p1_config_path = os.path.join(config.PHASE1_PATH, "config.py")
    spec = importlib.util.spec_from_file_location("p1_config", p1_config_path)
    p1_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p1_config)
    
    print(f"Searching raw PSV patient files in: {p1_config.DATA_RAW_DIR}")
    raw_files = glob.glob(os.path.join(p1_config.DATA_RAW_DIR, "*.psv"))
    
    if len(raw_files) == 0:
        print("Error: No raw patient PSV files found. Cannot run simulation.")
        return
        
    sample_file = raw_files[0]
    print(f"Selected patient file: {os.path.basename(sample_file)}")
    
    # Read patient log
    patient_df = pd.read_csv(sample_file, sep='|')
    
    # Drop SepsisLabel if it is present to simulate actual clinical monitoring
    if p1_config.COL_TARGET in patient_df.columns:
        true_labels = patient_df[p1_config.COL_TARGET].tolist()
        patient_df_inference = patient_df.drop(columns=[p1_config.COL_TARGET])
    else:
        true_labels = None
        patient_df_inference = patient_df.copy()
        
    total_hours = len(patient_df_inference)
    sim_hours = min(6, total_hours) # Simulate the first 6 hours of ICU stay
    
    print(f"Patient record contains {total_hours} hours. Simulating first {sim_hours} hours...")
    print("======================================================================")
    
    latest_report = None
    
    # Loop representing hourly ICU vital streams
    for hr in range(1, sim_hours + 1):
        print(f"\n######################################################################")
        print(f"--- CLINICAL CLOCK: HOUR {hr:02d} IN ICU ---")
        print(f"######################################################################")
        
        # Accumulate patient logs up to hour 'hr' to represent historical data availability
        accumulated_df = patient_df_inference.iloc[:hr].copy()
        
        # Process the record through the agentic whiteboard orchestrator
        latest_report = orchestrator.process_patient_record(accumulated_df)
        
        # Print Risk Assessment Summary
        print(f"\n--- CLINICAL CHART SUMMARY (HOUR {hr:02d}) ---")
        print("----------------------------------------------------------------------")
        print(latest_report['risk_assessment']['clinical_summary'])
        print("----------------------------------------------------------------------")
        
        # Print Alert Pager Dispatch if triggered
        if latest_report['alert']['alert_triggered']:
            print(f"\n--- PAGER DISPATCH: {latest_report['alert']['dispatch_message']}")
            
        true_lbl = true_labels[hr-1] if true_labels is not None else "Unknown"
        print(f"Actual Clinical Status at Hour {hr:02d}: Sepsis = {true_lbl}")
        
    # Save the final agentic report to a JSON file
    report_filename = "sample_patient_agent_report.json"
    report_path = os.path.join(config.BASE_DIR, report_filename)
    with open(report_path, 'w') as f:
        json.dump(latest_report, f, indent=4)
        
    print("\n======================================================================")
    print(f"Simulation completed. Saved final agent report to {report_path}")
    print("======================================================================")

if __name__ == "__main__":
    run_icu_agentic_simulation()
