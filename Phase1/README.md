# Phase 1: Machine Learning Pipeline Foundation
## Early Sepsis Prediction & Clinical Decision Support System

This module establishes a production-grade machine learning foundation for predicting the risk of sepsis in ICU patients using high-frequency time-series clinical data. It implements a complete end-to-end pipeline from raw data loading to feature engineering, patient-level train-test splitting, model training, threshold optimization, and a reusable prediction module.

---

## 📌 Project Overview
Sepsis is a life-threatening organ dysfunction caused by a dysregulated host response to infection. It is a leading cause of mortality in hospitals worldwide. Early prediction (even by hours) is critical because each hour of delayed treatment increases mortality risk by 4% to 8%. 

This module processes clinical records from the **PhysioNet / Computing in Cardiology Challenge 2019 Sepsis Dataset** to predict the onset of sepsis before it occurs.

---

## 📊 Dataset Structure
The dataset consists of hourly records for **10,927 patients** in the ICU, stored in pipe-separated value (`.psv`) format. Each row represents a single hourly observation for a patient.

### Features
The clinical variables (40 columns total used) are grouped as follows:

1. **Vital Signs (8 features)**
   * `HR`: Heart Rate (beats per minute)
   * `O2Sat`: Oxygen Saturation (percentage)
   * `Temp`: Temperature (degrees Celsius)
   * `SBP`: Systolic Blood Pressure (mmHg)
   * `MAP`: Mean Arterial Pressure (mmHg)
   * `DBP`: Diastolic Blood Pressure (mmHg)
   * `Resp`: Respiration rate (breaths per minute)
   * `EtCO2`: End-tidal carbon dioxide (mmHg) - *Dropped due to 100% missingness*

2. **Laboratory Values (26 features)**
   * Blood chemistry and arterial blood gas metrics: `BaseExcess`, `HCO3`, `FiO2`, `pH`, `PaCO2`, `SaO2`, `AST`, `BUN`, `Alkalinephos`, `Calcium`, `Chloride`, `Creatinine`, `Bilirubin_direct`, `Glucose`, `Lactate`, `Magnesium`, `Phosphate`, `Potassium`, `Bilirubin_total`, `TroponinI`, `Hct`, `Hgb`, `PTT`, `WBC`, `Fibrinogen`, `Platelets`.

3. **Demographic Variables (6 features)**
   * `Age`: Patient age (years)
   * `Gender`: Gender (0 for female, 1 for male)
   * `Unit1`: ICU medical unit identifier (MICU)
   * `Unit2`: ICU surgical unit identifier (SICU)
   * `HospAdmTime`: Hours between hospital admission and ICU admission
   * `ICULOS`: ICU length of stay (hours since ICU admission)

4. **Target Variable**
   * `SepsisLabel`: Binary label (1 if patient is diagnosed with sepsis within 6 hours before or after the observation hour, 0 otherwise).

---

## ⚠️ Challenges in ICU Time-Series Data
ICU clinical data presents unique challenges that this pipeline addresses:

1. **Extreme Missingness**: Laboratory features have missing rates exceeding **90-99%** because clinicians only order lab tests periodically (e.g., once a day) while vitals are monitored hourly.
2. **Extreme Class Imbalance**:
   * Patient Level: Only **8.8%** of patients ever develop sepsis.
   * Row Level: Only **2.2%** of observations are labeled positive (`SepsisLabel = 1`).
3. **Temporal Correlations & Data Leakage**: Multiple observations belong to the same patient. Randomly splitting data by row creates severe target leakage because a model will memorize patient-specific baselines instead of learning generalizable sepsis indicators. 

---

## 🛠️ Data Preprocessing & Feature Engineering
We implement a robust, patient-level pipeline to prepare the data:

1. **Patient-Level Split**: We split the dataset strictly at the patient level (80% train, 20% test). Patient IDs are kept separate to ensure zero information leakage.
2. **Forward Fill (ffill)**: Vitals and lab measurements are forward-filled per patient. This simulates clinical practice (e.g., a platelet count remains the best estimate of a patient's platelets until a new blood draw is done).
3. **Mean Arterial Pressure (MAP) Reconstruction**: If MAP is missing but SBP and DBP are available, MAP is calculated using the formula:
   $$\text{MAP} = \frac{\text{SBP} + 2 \times \text{DBP}}{3}$$
4. **Shock Index Feature Engineering**: We engineer the **Shock Index** ($\text{HR} / \text{SBP}$), which is a crucial clinical indicator for shock and organ hypoperfusion (sepsis warning).
5. **Population Median Imputation**: Initial hours in a patient's record before any measurement has occurred are filled with population medians derived *only* from the training set.
6. **Standardization**: Numeric features are scaled using `StandardScaler` to normalize feature distributions.

---

## 📈 Model Comparison and Evaluation
We trained three models: **Decision Tree**, **Random Forest**, and **XGBoost**.

### Tuning for Clinical Goals
In a clinical setting, **Recall (Sensitivity)** is prioritized. Missing a sepsis diagnosis (False Negative) can be fatal. However, setting the decision threshold too low causes excessive False Positives, leading to "alert fatigue" among clinical staff. We set a target recall of **0.85** and optimized the decision threshold for each model to maximize precision at that recall level.

### Results Table (Test Set)
| Model | Threshold Type | Decision Threshold | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Decision Tree** | Default | 0.5000 | 0.7775 | 0.0587 | 0.5989 | 0.1069 | 0.7493 |
| **Decision Tree** | Tuned | 0.3426 | 0.3701 | 0.0304 | 0.8840 | 0.0587 | 0.7493 |
| **Random Forest** | Default | 0.5000 | 0.8825 | 0.0917 | 0.4813 | 0.1541 | **0.8079** |
| **Random Forest** | Tuned | 0.3750 | 0.6001 | 0.0455 | 0.8503 | **0.0864** | **0.8079** |
| **XGBoost** | Default | 0.5000 | 0.8671 | 0.0856 | 0.5139 | 0.1467 | 0.8019 |
| **XGBoost** | Tuned | 0.2668 | 0.5611 | 0.0417 | 0.8529 | 0.0795 | 0.8019 |

### Best Model Selection
**Random Forest** was selected as the best overall model for this baseline structure, achieving the highest **ROC-AUC of 0.8079** and the best F1-Score of **0.0864** at the tuned threshold of **0.3750** (which guarantees **85.03% Recall** on unseen patients).

* **Advantages**: Stable, handles high dimensionality and missing values effectively, less prone to overfitting on sparse clinical labs.
* **Limitations**: Low precision (4.55%) is a natural side effect of the low base rate (2.22%) when targeting high recall. This requires post-processing risk-level warnings instead of simple alarm alerts.

---

## 📁 Directory Structure
```
Phase1/
│
├── data/
│   ├── raw/                 # Put raw patient PSV files here (Optional)
│   └── processed/           # train_patients.csv, test_patients.csv, X_train.joblib
├── notebooks/               # Folder for EDA notebooks
├── preprocessing/           # Fitted preprocessor pipeline and custom classes
│   ├── transformers.py      # Custom sklearn SepsisFeatureExtractor
│   └── preprocessor.joblib  # Saved pipeline (ffill -> Impute -> Scale)
├── models/                  # Saved models
│   ├── decision_tree.joblib
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   └── best_model_info.joblib  # Metadata of selected model and tuned threshold
├── evaluation/              # Model performance summary reports
├── visualizations/          # ROC Curve, PR Curve, Confusion Matrices
├── scripts/                 # Utility scripts
│
├── load_data.py             # Merges raw files and splits train/test patients
├── preprocess.py            # Sets up the preprocessing pipeline and fits it
├── train.py                 # Trains the DT, RF, and XGBoost models
├── evaluate.py              # Optimizes thresholds, compiles metrics, and plots charts
├── predict.py               # Reusable predictor class for production inference
├── config.py                # Core configuration parameters
└── requirements.txt         # Required Python packages
```

---

## 🚀 How to Run the Pipeline

### 1. Setup Environment
Ensure Python 3.8+ is installed, and run:
```bash
pip install -r Phase1/requirements.txt
```

### 2. Run Data Pipeline
Execute the scripts in order from the root workspace directory:
```bash
# Step 1: Merge raw psv patient records and split train/test patients
python Phase1/load_data.py

# Step 2: Fit and run preprocessing pipeline on data
python Phase1/preprocess.py

# Step 3: Train all machine learning models
python Phase1/train.py

# Step 4: Evaluate and compare models, tune decision thresholds
python Phase1/evaluate.py
```

### 3. Run Inference Module (Test)
To run a self-test of the predictor module on a sample patient file:
```bash
python Phase1/predict.py
```

---

## 🔮 Integration with Next Phases
This pipeline forms the base of the entire sepsis prediction system. 
* **Phase 2 (Explainable AI)**: The saved preprocessor pipeline and `SepsisPredictor` output will be fed into SHAP and LIME algorithms to produce local and global explainability charts.
* **Phase 3 (Agentic AI)**: The `SepsisPredictor` will act as the core engine for the *Prediction Agent*, feeding predictions to the *Validation*, *Risk Assessment*, and *Alert Agents*.
* **Phase 4 (Backend)**: The prediction module will be wrapped inside a FastAPI service, backed by PostgreSQL.
