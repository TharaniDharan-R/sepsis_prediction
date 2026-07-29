import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time
import config

def load_single_psv(filepath):
    """
    Load a single patient psv file and add PatientID.
    """
    try:
        df = pd.read_csv(filepath, sep='|')
        patient_id = os.path.basename(filepath).split('.')[0]
        df[config.COL_PATIENT_ID] = patient_id
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def load_and_merge_data():
    """
    Find all psv files, load them in parallel, and merge into one dataframe.
    """
    print(f"Scanning for PSV files in: {config.DATA_RAW_DIR}")
    psv_files = glob.glob(os.path.join(config.DATA_RAW_DIR, "*.psv"))
    total_files = len(psv_files)
    
    if total_files == 0:
        raise ValueError(f"No .psv files found in {config.DATA_RAW_DIR}. Please check the path.")
        
    print(f"Found {total_files} patient files. Loading in parallel...")
    
    start_time = time.time()
    
    # Determine number of workers (use cpu_count minus 1, minimum 1)
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {num_workers} parallel workers...")
    
    dfs = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(load_single_psv, psv_files, chunksize=100))
        
    dfs = [df for df in results if df is not None]
    
    print(f"Loaded {len(dfs)} files successfully.")
    
    print("Concatenating into a single DataFrame...")
    merged_df = pd.concat(dfs, ignore_index=True)
    
    end_time = time.time()
    print(f"Loading and merging completed in {end_time - start_time:.2f} seconds.")
    print(f"Merged shape: {merged_df.shape}")
    
    return merged_df

def split_and_save_data(df):
    """
    Split the dataset at the patient level (no row leakage) and save to processed data folder.
    """
    print("Performing patient-level train/test split...")
    
    # Get unique patient IDs
    unique_patients = df[config.COL_PATIENT_ID].unique()
    
    # Get label status for each patient (does the patient ever develop sepsis?)
    patient_labels = df.groupby(config.COL_PATIENT_ID)[config.COL_TARGET].max().loc[unique_patients]
    
    # Stratified split based on patient-level sepsis labels
    train_patients, test_patients = train_test_split(
        unique_patients,
        test_size=(1.0 - config.TRAIN_TEST_SPLIT_RATIO),
        random_state=config.RANDOM_SEED,
        stratify=patient_labels
    )
    
    # Subset the main DataFrame
    train_df = df[df[config.COL_PATIENT_ID].isin(train_patients)].copy()
    test_df = df[df[config.COL_PATIENT_ID].isin(test_patients)].copy()
    
    # Ensure processed directory exists
    os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)
    
    print(f"Saving train dataset to {config.TRAIN_PATIENTS_CSV}...")
    train_df.to_csv(config.TRAIN_PATIENTS_CSV, index=False)
    
    print(f"Saving test dataset to {config.TEST_PATIENTS_CSV}...")
    test_df.to_csv(config.TEST_PATIENTS_CSV, index=False)
    
    # Print metrics
    total_patients = len(unique_patients)
    num_train_patients = len(train_patients)
    num_test_patients = len(test_patients)
    
    sepsis_train_patients = (train_df.groupby(config.COL_PATIENT_ID)[config.COL_TARGET].max() == 1).sum()
    sepsis_test_patients = (test_df.groupby(config.COL_PATIENT_ID)[config.COL_TARGET].max() == 1).sum()
    
    print("\n" + "="*50)
    print("SPLIT SUMMARY")
    print("="*50)
    print(f"Total Patients:          {total_patients}")
    print(f"Train Patients:          {num_train_patients} ({num_train_patients/total_patients*100:.2f}%)")
    print(f"Test Patients:           {num_test_patients} ({num_test_patients/total_patients*100:.2f}%)")
    print(f"Sepsis Patients (Train): {sepsis_train_patients} ({sepsis_train_patients/num_train_patients*100:.2f}%)")
    print(f"Sepsis Patients (Test):  {sepsis_test_patients} ({sepsis_test_patients/num_test_patients*100:.2f}%)")
    print(f"Train rows:              {train_df.shape[0]} (Sepsis rows: {train_df[config.COL_TARGET].sum()}, {train_df[config.COL_TARGET].mean()*100:.2f}%)")
    print(f"Test rows:               {test_df.shape[0]} (Sepsis rows: {test_df[config.COL_TARGET].sum()}, {test_df[config.COL_TARGET].mean()*100:.2f}%)")
    print("="*50 + "\n")

if __name__ == "__main__":
    df = load_and_merge_data()
    split_and_save_data(df)
