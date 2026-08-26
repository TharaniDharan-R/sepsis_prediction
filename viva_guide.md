# Sepsis Prediction Project - Viva & Explanation Guide

This guide contains the step-by-step, complete explanation of your final year project, covering Phase 1 through 3. We will append the explanation for each file here as we progress.

---

## Phase 1: Machine Learning Pipeline

### Folder: `Phase1`

* **Why this folder exists:** It serves as the foundation for the entire Sepsis Prediction system. It is responsible for taking raw hospital data, processing it, and creating an AI model.
* **What its purpose is:** To build an end-to-end Machine Learning pipeline: loading raw time-series records, handling missing values, engineering features, training models (like Random Forest), and evaluating them.
* **Which files inside it are important:** `config.py`, `load_data.py`, `preprocess.py`, `train.py`, `evaluate.py`, and `predict.py`.
* **How it connects with the rest of the project:** The final output of Phase 1 is a saved machine learning model and a preprocessor (`.joblib` files). **Phase 2 (Explainable AI)** loads this model to explain its decisions. **Phase 3 (Agentic AI)** uses this model to generate predictions that the AI agents act upon.
* **What will happen if this folder is removed:** The entire project will collapse. Phase 2 and Phase 3 cannot function without the model created here.
* **Which other folders depend on it:** `Phase2` and `Phase3`.

---

### File: `Phase1/config.py`

1. **Why this file exists:** In software engineering, hardcoding file paths or model parameters inside scripts is a bad practice. `config.py` acts as a central control panel. If you need to change the dataset location or a model's hyperparameter, you only change it here.
2. **Where it is used:** It is used as a foundation for data loading, preprocessing, and training.
3. **Which file calls it:** It is imported by almost every script in Phase 1 (`load_data.py`, `preprocess.py`, `train.py`, `evaluate.py`, `predict.py`).
4. **Which files it imports:** The built-in `os` module.
5. **Which files import it:** `load_data.py`, `preprocess.py`, `train.py`, `evaluate.py`, `predict.py`.
6. **What happens when this file runs:** It simply declares and stores variables in memory. It doesn't "execute" any active logic or print anything.
7. **The execution flow:** Python reads it from top to bottom, assigning values to variable names.
8. **The input:** None.
9. **The output:** Constants (Strings, Lists, and Dictionaries) made available to other scripts.
10. **The dependency chain:** It sits at the very bottom. `config.py` depends on nothing, but everything depends on `config.py`.

---

#### Import Statements

```python
import os
```
* **What is `os`?** It is a built-in Python library used for interacting with the Operating System.
* **Why is `os` needed?** It is needed to create file paths dynamically (using `os.path.join`). 
* **Why is it imported as os?** Because that is its standard name; no alias is used.
* **Which functions from `os` are used later?** `os.path.join()`. This function safely connects folder names regardless of whether you are on Windows (using `\`) or Mac/Linux (using `/`).
* **What would happen if this import were removed?** The code would crash with a `NameError` as soon as it tries to use `os.path.join`.

---

#### Variables Explained

```python
# Base paths
BASE_DIR = r"D:\sepsis\Phase1"
DATA_RAW_DIR = r"C:\Users\Admin\Downloads\sepsis_dataset\physionet.org\files\challenge-2019\1.0.0\training\training_setA"
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PREPROCESSING_DIR = os.path.join(BASE_DIR, "preprocessing")
EVALUATION_DIR = os.path.join(BASE_DIR, "evaluation")
VISUALIZATIONS_DIR = os.path.join(BASE_DIR, "visualizations")
```
* **Why they were created:** To store the directory paths where raw data is read from, and where processed data/models will be saved.
* **What they store:** Absolute paths as Strings. Note the `r""` (raw string) used for `BASE_DIR` and `DATA_RAW_DIR` to prevent backslashes from being treated as escape characters in Windows.
* **How they change:** They don't change during execution (they are constants, hence written in ALL_CAPS).
* **Where they are used:** Used immediately below to define specific file paths.
* **Data type:** `str` (String). Chosen because file paths are text.

```python
# File paths
TRAIN_PATIENTS_CSV = os.path.join(DATA_PROCESSED_DIR, "train_patients.csv")
TEST_PATIENTS_CSV = os.path.join(DATA_PROCESSED_DIR, "test_patients.csv")
PREPROCESSOR_PATH = os.path.join(PREPROCESSING_DIR, "preprocessor.joblib")
BEST_MODEL_INFO_PATH = os.path.join(MODELS_DIR, "best_model_info.joblib")
```
* **Why they were created:** To point directly to the specific `.csv` and `.joblib` (saved Python objects) files.
* **What they store:** Full string paths.
* **Data type:** `str`. 

```python
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
```
* **Why they were created:** To categorize the columns of the raw dataset. Clinical data is naturally grouped into Vitals (measured hourly) and Labs (measured daily/rarely), which require different preprocessing strategies.
* **What they store:** Lists of strings (column names). `COL_TARGET` and `COL_PATIENT_ID` are just single strings.
* **Where they are used:** In `load_data.py` and `preprocess.py` to select specific columns for transformations.
* **Data type:** `list` of `str` and `str`. Chosen because a list allows iterating over multiple feature names.

```python
# Features to drop due to extreme missingness (e.g. 100% missing values)
COL_DROP = ['EtCO2']
# Derived clinical features
COL_ENGINEERED = ['Shock_Index']
# All raw columns (for initial loading)
ALL_RAW_COLUMNS = COL_VITALS + COL_LABS + COL_DEMOGRAPHICS + [COL_TARGET]
```
* **Why they were created:** `EtCO2` is dropped because it contains no data. `Shock_Index` is added because it is a vital engineered feature. `ALL_RAW_COLUMNS` creates a master list for loading.
* **What they store:** Lists of strings.

```python
# Hyperparameters for ML Models
DT_PARAMS = {
    'max_depth': 6,
    'min_samples_split': 50,
    'min_samples_leaf': 20,
    'class_weight': 'balanced',
    'random_state': 42
}
```
*(RF_PARAMS and XGB_PARAMS follow the exact same logic)*
* **Why they were created:** To store the settings (hyperparameters) for the Decision Tree (`DT`), Random Forest (`RF`), and XGBoost (`XGB`) models. 
* **What they store:** Key-value pairs matching `scikit-learn` and `xgboost` parameter arguments. Notice `class_weight: 'balanced'`—this is extremely important! It tells the model to pay extra attention to the minority class (sepsis patients) because sepsis is rare.
* **Data type:** `dict` (Dictionary). Chosen because hyperparameters are inherently key-value pairs (parameter name: parameter value).

```python
TRAIN_TEST_SPLIT_RATIO = 0.8  # 80% train, 20% test
RANDOM_SEED = 42
TARGET_RECALL = 0.85
```
* **Why they were created:** 
    * `TRAIN_TEST_SPLIT_RATIO`: Defines how much data goes to training.
    * `RANDOM_SEED`: Ensures reproducibility. If someone else runs your code, they get the exact same results.
    * `TARGET_RECALL`: **Crucial.** In medical AI, missing a sepsis diagnosis (False Negative) is fatal. This variable tells the system: "Adjust the model so it catches at least 85% of all sepsis cases, even if it makes some false alarms."
* **Data type:** `float` and `int`.

---

#### Viva Preparation for `config.py`

* **Likely Viva Question 1:** *"Why did you use a `config.py` file instead of just writing the paths inside your data loading script?"*
    * **Ideal Answer:** "Using a configuration file adheres to the DRY (Don't Repeat Yourself) principle and Separation of Concerns. It centralizes all paths, hyperparameters, and feature lists. If the dataset location changes or I want to tune a hyperparameter, I only have to change it in one place, minimizing the risk of bugs."
* **Likely Viva Question 2:** *"I see `class_weight='balanced'` in your parameters. Why is that there?"*
    * **Ideal Answer:** "The sepsis dataset is highly imbalanced—only about 2.2% of hourly records indicate sepsis. If I didn't balance the class weights, the model would become biased toward the majority class and simply predict 'No Sepsis' every time to achieve high accuracy. `class_weight='balanced'` heavily penalizes the model for missing a sepsis case during training."
* **Common mistakes students make:** Not knowing what `RANDOM_SEED = 42` does. (It simply locks the random number generator so your data splits and model initializations are identical every time you run the code. `42` is just a pop-culture reference to *The Hitchhiker's Guide to the Galaxy*).

---

### File: `Phase1/load_data.py`

1. **Why this file exists:** The raw dataset consists of tens of thousands of individual `.psv` (pipe-separated value) files, where each file represents a single patient's hourly records. This script loads all these files efficiently, merges them into one massive table, and splits them into training and testing sets.
2. **Where it is used:** It is the absolute first step in the data pipeline.
3. **Which file calls it:** It is usually executed directly from the terminal (`python Phase1/load_data.py`). 
4. **Which files it imports:** `os`, `glob`, `pandas`, `numpy`, `train_test_split`, `ProcessPoolExecutor`, `multiprocessing`, `time`, and our custom `config`.
5. **Which files import it:** None. It acts as an independent execution script.
6. **What happens when this file runs:** It finds all patient files, loads them in parallel (using multiple CPU cores to save time), adds a 'PatientID' column, merges them, performs a strict patient-level split, and saves `train_patients.csv` and `test_patients.csv`.
7. **The execution flow:** `__main__` block triggers -> `load_and_merge_data()` runs -> `split_and_save_data()` runs -> Final CSVs are saved.
8. **The input:** Thousands of `.psv` files from the raw data directory.
9. **The output:** `train_patients.csv` and `test_patients.csv` in the `data/processed` folder.
10. **The dependency chain:** Depends on `config.py`. The next script in the pipeline (`preprocess.py`) depends on the output of this script.

---

#### Import Statements

```python
import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time
import config
```

* **`glob`**: A library used to search for files that match a specific pattern (e.g., `*.psv`). Without this, you'd have to write complex loops to find the right files.
* **`pandas as pd`**: The most powerful data manipulation library in Python. It is used to load the text files into tabular structures called DataFrames. It is imported as `pd` by universal convention. Without pandas, parsing tabular data is extremely difficult.
* **`numpy as np`**: Used for numerical computations. Imported as `np` by convention.
* **`train_test_split` (from `sklearn`)**: A function that randomly shuffles and splits datasets into a training portion and a testing portion.
* **`ProcessPoolExecutor` & `multiprocessing`**: Used for parallel computing. Instead of loading 10,000 files one by one (which takes forever), it assigns chunks of files to different CPU cores simultaneously.
* **`time`**: Used to calculate how long the script takes to run.
* **`config`**: Imports your `config.py` file so this script can access the file paths and settings.

---

#### Functions Explained

##### 1. `load_single_psv(filepath)`
* **Why this function exists:** To handle the reading of a single file and attach a patient identifier to it.
* **Internal logic:** Reads the `.psv` file using pandas. Crucially, the raw files don't have a column saying "I am patient A". The patient ID is just the file's name (e.g., `p000001.psv`). This function extracts `p000001` from the filename and adds it as a new column (`config.COL_PATIENT_ID`).
* **Parameters:** `filepath` (string).
* **Return value:** A pandas DataFrame.

##### 2. `load_and_merge_data()`
* **Why this function exists:** To manage the massive data loading process.
* **Internal logic:** 
    1. Uses `glob.glob` to find all `.psv` files.
    2. Calculates the number of CPU cores available using `multiprocessing.cpu_count()`.
    3. Uses `ProcessPoolExecutor` to map the `load_single_psv` function to all file paths simultaneously.
    4. Combines the resulting list of DataFrames into one gigantic DataFrame using `pd.concat()`.
* **Why this implementation was chosen:** A standard `for` loop would take 5 to 10 minutes to load 20,000 files. Parallel processing reduces this to seconds.

##### 3. `split_and_save_data(df)`  *(CRITICAL ML LOGIC)*
* **Why this function exists:** To separate the data into data the model learns from (Train) and data the model is tested on (Test).
* **Internal logic:** 
    1. Extracts a list of `unique_patients`.
    2. Groups data by patient to determine if they **ever** got sepsis (`df.groupby(config.COL_PATIENT_ID)[config.COL_TARGET].max()`).
    3. Passes the **unique patients** into `train_test_split`, not the raw rows.
    4. Subsets the original dataset based on which patient ended up in which split.
    5. Saves to CSV.

---

#### Machine Learning Focus: Patient-Level Splitting

**Why is the data split designed this way?**
This is the most critical part of this script. The dataset has hourly rows. Patient A might have 48 rows (48 hours). 
If you just split the dataset by rows, Row 10 of Patient A might end up in the Training set, and Row 11 of Patient A might end up in the Testing set. 
**This is called Data Leakage.** The model will learn Patient A's personal baseline (e.g., Patient A naturally has low blood pressure), recognize Patient A in the test set, and "cheat" on the prediction. It will fail in the real world on *new* patients.
By splitting by `PatientID`, we guarantee that if Patient A is in the training set, **none** of their data appears in the test set.

**Why Stratification?**
The code uses `stratify=patient_labels`. Because sepsis is rare (only 8.8% of patients get it), a random split might accidentally put all the sepsis patients in the train set and none in the test set. Stratification guarantees that exactly 8.8% of the train set and 8.8% of the test set will consist of sepsis patients.

---

#### Viva Preparation for `load_data.py`

* **Likely Viva Question 1:** *"Why did you use `ProcessPoolExecutor` instead of a simple loop?"*
    * **Ideal Answer:** "The dataset consists of thousands of individual patient files. I/O operations (reading from disk) are very slow. By using `ProcessPoolExecutor`, I distribute the file reading process across multiple CPU cores, effectively parallelizing the workload and drastically reducing the data ingestion time."
* **Likely Viva Question 2:** *"Explain how you prevented Data Leakage during the train/test split."*
    * **Ideal Answer:** "I performed a strict **patient-level split** instead of a row-level split. Time-series clinical data has heavy temporal correlation—measurements from the same patient are highly related. If I randomly split rows, the model would see parts of a patient's history during training and test on the same patient's future. I extracted unique Patient IDs, split the IDs, and then assigned rows based on the ID. This ensures the model is evaluated on completely unseen patients, mimicking real-world deployment."
* **Common mistakes students make:** Splitting rows randomly using `train_test_split(df)` on a time-series/longitudinal dataset. This inflates model accuracy artificially and the model completely fails in production.

---

## Phase 2: Explainable AI (XAI) Pipeline

### Overview & Purpose

**Why Phase 2 exists:** In healthcare, black-box AI models (like XGBoost from Phase 1) are not trusted by doctors. If the model says "85% risk of sepsis", the doctor needs to know *why*. 
Phase 2 decodes the decisions of Phase 1 using **SHAP** (Shapley Additive exPlanations) and **LIME** (Local Interpretable Model-agnostic Explanations).

**Core Files:**
* `explainers.py`: Contains the `SepsisExplainer` class, which is the heart of Phase 2. It loads the saved model and computes the mathematical reasons for every prediction.
* `explain_global.py`: Evaluates the model across the whole dataset (e.g., "What are the most important features overall?").
* `explain_local.py`: Evaluates specific patients (e.g., "Why did Patient A trigger the alarm at 2 PM?").

### File: `Phase2/explainers.py`

1. **Why this file exists:** It provides an API for downstream systems (like Phase 3) to ask for explanations.
2. **How it works:**
   - **Initialization:** It loads the preprocessor and the XGBoost model saved from Phase 1.
   - **SHAP (Game Theory):** It uses `shap.TreeExplainer`. SHAP treats predicting sepsis as a cooperative game where each feature (e.g., Heart Rate, Age) is a player. It calculates how much each feature "contributed" to pushing the probability up or down from the baseline.
   - **LIME (Local Surrogates):** It uses `lime.lime_tabular`. LIME perturbs (tweaks) a patient's data slightly to see how the XGBoost model reacts, then builds a simple, understandable linear model around that specific patient to explain the decision.

#### Viva Preparation for Phase 2

* **Likely Viva Question:** *"What is the difference between SHAP and LIME, and why use both?"*
    * **Ideal Answer:** "SHAP is based on cooperative game theory. It is mathematically consistent and great for exact feature contributions, especially with TreeExplainer making it fast. LIME is a surrogate model that tests how local changes affect predictions. I used both to provide clinical robustness—if both SHAP and LIME agree that a patient's Heart Rate and Kidney function are the driving factors, the clinician can trust the alert with high confidence."

---

## Phase 3: Agentic AI Clinical Architecture

### Overview & Purpose

**Why Phase 3 exists:** A raw probability and a SHAP explanation are still just numbers on a screen. In a real hospital, a clinical team collaborates. Phase 3 wraps the ML model inside an **Agentic AI Architecture**—a team of 6 specialized AI agents that act like a digital medical staff rounding on the patient.

### The 6 Agents (`Phase3/agents/`)

1. **Validation Agent:** Checks if the incoming data is physically possible (e.g., Heart rate isn't -50).
2. **Prediction Agent:** Uses the Phase 1 model to calculate the sepsis probability.
3. **Monitoring Agent:** Looks at historical hours to calculate *trends* (e.g., "Oxygen has been dropping for 3 hours").
4. **Explainability Agent:** Uses the Phase 2 `SepsisExplainer` to get the SHAP drivers.
5. **Risk Assessment Agent:** The brain of the operation. It takes all the numbers, trends, and explanations, and drafts a structured, human-readable clinical report (using LLMs or rule-based templates).
6. **Alert Agent:** Decides if the risk is high enough to dispatch a pager notification to the doctor.

### The Orchestrator (`Phase3/main.py` & `orchestrator.py`)

* **Execution Flow:** The Orchestrator uses a "Whiteboard" pattern. It receives a new hour of patient data, sends it to the Validation Agent, then Prediction, then Monitoring, etc. Each agent writes its findings to a shared dictionary (the whiteboard). Finally, the Alert Agent reads the full whiteboard and outputs the final clinical dispatch.

#### Viva Preparation for Phase 3

* **Likely Viva Question:** *"Why did you use a Multi-Agent system instead of just printing the ML prediction?"*
    * **Ideal Answer:** "A raw ML prediction lacks clinical context. By using a multi-agent architecture, I modularized the logic. One agent handles data safety (Validation), another handles temporal tracking (Monitoring), and another synthesizes a human-readable medical summary (Risk Assessment). This architecture mimics a real clinical workflow and makes the AI output immediately actionable for a busy doctor."

---
