# Sepsis Prediction & Clinical Decision Support System

An end-to-end, multi-phase clinical AI system designed to predict the early onset of sepsis in Intensive Care Unit (ICU) patients. It integrates a machine learning pipeline, Explainable AI (XAI), a cooperative multi-agent architecture, and a real-time clinician dashboard.

---

## 🏛️ Project Architecture Overview

The project is structured into **5 distinct phases**:

```
                       [ Raw ICU Patient Vitals Stream ]
                                     │
                                     ▼
                  [ Phase 1: Machine Learning Pipeline ]
                  - Load & Preprocess High-Frequency Data
                  - Feature Engineering (Shock Index, MAP)
                  - Recall-Tuned Random Forest & XGBoost Models
                                     │
                                     ▼
                        [ Phase 2: Explainable AI ]
                  - Game-Theoretic Contributions (SHAP)
                  - Local Surrogates for Interpretability (LIME)
                                     │
                                     ▼
                   [ Phase 3: Agentic Clinical Team ]
                  - 6 Cooperative Agents (Validation, Monitoring,
                    Prediction, Explainer, Risk, Alert)
                  - Whiteboard Orchestration & Rounding Summaries
                                     │
                                     ▼
                      [ Phase 4: REST API Backend ]
                  - FastAPI Microservice & SQLite database
                  - Real-time patient & vital logging endpoints
                                     │
                                     ▼
                  [ Phase 5: Clinical Dashboard UI ]
                  - Premium Dark-Mode UI for ICU clinicians
                  - Trajectory charts, SHAP/LIME charts, agent logs
```

---

## 📂 Repository Structure

* **`Phase1/`**: Machine Learning Pipeline (Data loading, Preprocessing, Model Training, Threshold Tuning, and Predictor modules).
* **`Phase2/`**: Explainable AI Engine (Global summary beeswarms and Local waterfall/bar explanations using SHAP and LIME).
* **`Phase3/`**: Agentic AI Architecture (Multi-agent clinical whiteboard simulation scripts).
* **`Phase4/`**: Backend Service (FastAPI app, SQLAlchemy DB models, schemas, and bootstrapping script).
* **`Phase5/`**: Frontend Dashboard (Vanilla HTML5/CSS3/JavaScript UI serving interactive charts and vital stream simulation).
* **`project_explanation_guide.md`**: Comprehensive guide for project viva preparations and code explanations.
* **`viva_guide.md`**: Student preparation guide with common questions and ideal clinical answers.

---

## 🚀 Installation & Setup

Ensure you have Python 3.8+ installed. 

### 1. Setup Virtual Environment
Run the following commands in your terminal from the root workspace directory:

```bash
# Create a virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r Phase1/requirements.txt
pip install -r Phase2/requirements.txt
pip install -r Phase4/requirements.txt
```

---

## 💻 Running the Project

### Retrain Machine Learning Models (Optional)
Pre-trained models are already saved in the repository. To re-train them from raw PhysioNet data, execute the Phase 1 pipeline:
```bash
python Phase1/load_data.py
python Phase1/preprocess.py
python Phase1/train.py
python Phase1/evaluate.py
```

### Run the Agentic Clinical Team Simulation
To run the whiteboard simulation of the 6 cooperative agents evaluating patient telemetry hour-by-hour:
```bash
python Phase3/main.py
```

### Run the Live Clinical Web Dashboard
To start the backend API and open the visual web dashboard:
```bash
# Start backend server
python Phase4/run.py
```
Open your browser and navigate to:
👉 **[http://127.0.0.1:8000/dashboard/](http://127.0.0.1:8000/dashboard/)**

---

## 🩺 Clinical System Highlights

* **Recall-Tuned Classifiers**: Medical AI defaults to standard thresholds which miss rare, critical indicators. We tuned our classifier thresholds to hit **85% Recall** (maximizing sensitivity) to minimize dangerous false-negative predictions.
* **Clinical Explainability**: Clinicians can inspect SHAP and LIME contribution values to see exactly *which* vitals (e.g. rising Shock Index, falling oxygen saturation) triggered the risk increase.
* **Physiological Safety Nets**: An automated **Validation Agent** ensures no corrupt values trigger mock alarms, and a **Monitoring Agent** watches for sudden rates of change over multi-hour sliding windows.
