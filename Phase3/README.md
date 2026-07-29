# Phase 3: Agentic AI Clinical Architecture

This directory implements the **Agentic AI Architecture** for the Sepsis Prediction System. It wraps the raw ML model and explainability layers into a cooperative, multi-agent virtual clinical team that validates, monitors, explains, and assesses ICU patient vitals in real time.

---

## 🤖 Why Agentic AI in Healthcare?

In clinical settings, an raw ML probability (e.g., "72% risk") is insufficient. Clinicians require:
1. **Data Quality Assurances**: Verification that inputs are physiologically plausible (no sensor error).
2. **Trend Analysis**: Detection of rapid deterioration over time (e.g., heart rate spikes or blood pressure drops), which static single-hour models might miss.
3. **Medical Context**: A coherent clinical summary detailing which organ systems are failing (e.g. renal vs respiratory).
4. **Actionable Recommendations**: Standardized clinical protocols (e.g. order blood cultures, administer antibiotics).

By separating these tasks into **6 specialized cooperative agents**, we mimic a clinical team rounding on a patient.

---

## 🏛️ Multi-Agent Whiteboard Architecture

The system coordinates agents using a **Whiteboard Orchestration** pattern (`orchestrator.py`). The orchestrator coordinates data routing, accumulates message logs, and aggregates the clinical report:

```
                  [ Raw Patient Vitals Data Stream ]
                                 │
                                 ▼
                     1. Validation Agent (Assert Range Bounds)
                                 │
                                 ▼
                     2. Prediction Agent (Score XGBoost Model)
                                 │
                                 ▼
                     3. Monitoring Agent (Track Vitals Trends)
                                 │
                                 ▼
                     4. Explainability Agent (SHAP / LIME Contributions)
                                 │
                                 ▼
                     5. Risk Assessment Agent (Generates Clinical Summary)
                                 │
                                 ▼
                     6. Alert Agent (Formats Pager Dispatches)
                                 │
                                 ▼
                  [ Consolidated Clinical Patient Report ]
```

---

## 👤 Agent Roles & Design Details

### 1. Validation Agent (`agents/validation_agent.py`)
- **Role**: Asserts structural and physiological validity of incoming logs.
- **Logic**: Iterates through key variables and evaluates biological thresholds (e.g. Heart Rate between 30 and 250 bpm, Temperature between 30°C and 45°C). Flags warning flags for missing measurements or implausible values.

### 2. Prediction Agent (`agents/prediction_agent.py`)
- **Role**: Computes prediction probability.
- **Logic**: Interfaces with Phase 1's `SepsisPredictor` to score incoming records and check against the tuned recall-optimized decision threshold (**0.2668**).

### 3. Monitoring Agent (`agents/monitoring_agent.py`)
- **Role**: Computes physiological rates of change over historical ICU hours.
- **Logic**: Tracks vital trends over specific lookback windows (e.g., changes in Shock Index, Heart Rate, MAP, O2 Saturation). Triggers deterioration alerts if thresholds are exceeded (e.g., Heart Rate increasing by $\ge 15$ bpm over 4 hours).

### 4. Explainability Agent (`agents/explainability_agent.py`)
- **Role**: Translates black-box ML outputs into feature-level contributions.
- **Logic**: Queries Phase 2's `SepsisExplainer` to extract top SHAP and LIME contributions for the current hour.

### 5. Risk Assessment Agent (`agents/risk_assessment_agent.py`)
- **Role**: Synthesizes qualitative summaries and clinical assessments.
- **Logic**: Combines all quantitative outputs (vitals, trends, warnings, SHAP drivers) and prompts an LLM using clinical templates. If LLM API keys are missing, it executes a highly sophisticated **clinical rule-based generator** to draft a structured medical summary.

### 6. Alert Agent (`agents/alert_agent.py`)
- **Role**: Evaluates if clinical notifications should be dispatched.
- **Logic**: Triggers notifications if the model risk exceeds thresholds or if multiple vital trends deterioration warn of decay. Formats pager-friendly dispatches.

---

## 📝 Clinical Prompt Engineering & LLM Integration

The `RiskAssessmentAgent` uses structured prompt templates defined in `config.py` to draft reports.

### Prompt Template structure:
- **Context**: Defines the agent as an expert ICU clinical advisor.
- **Input variables**: Sepsis probability, SHAP log-odds contributions, validation warning logs, vitals trend summaries, and current vitals.
- **Formatting Guidelines**: Restricts output length (150-200 words), enforces professional clinical vocabulary, requires outlining primary drivers, and mandates concluding with actionable clinical protocol recommendations.

---

## ⏱️ Hourly ICU Vital Simulation Case Study

We simulated hourly monitoring of patient `p000001.psv`. Below is the transcript of vital changes, agent whiteboard messages, and report dispatches across selected hours:

### ICU Hour 01
* **Vitals**: Heart Rate = 99.0 bpm, O2 Saturation = 100%, Temperature = Missing.
* **Orchestrator Logs**:
  - `[Orchestrator] Routing patient vitals to ValidationAgent...`
  - `[ValidationAgent] Noted 7 mild vital anomalies or missing values.`
  - `[PredictionAgent] Risk Scored: Prob = 27.82%, Prediction Alert = 1`
  - `[ExplainabilityAgent] Top 3 risk drivers (SHAP): Resp (0.148), Unit1 (0.138)`
* **Alert Agent**: Dispatches Warning Sepsis Alert (Probability 27.82% exceeds threshold 26.68%).

### ICU Hour 05 (Clinical Deterioration)
* **Vitals**: O2 Saturation drops to **88.5%**, Temperature is missing, Heart Rate = 89.0 bpm.
* **Orchestrator Logs**:
  - `[MonitoringAgent] ALERT: Oxygen Desaturation: decreased by 10.50 over the last 3 hours (value: 99.0 -> 88.5)`
  - `[ExplainabilityAgent] Top 3 risk drivers (SHAP): Resp (0.148), Unit1 (0.138), Platelets (0.11)`
* **Risk Assessment Agent Output**:
  ```
  CLINICAL RISK ASSESSMENT (STATUS: HIGH RISK)
  The patient is evaluated at a 41.3% risk of developing early-onset sepsis (Sepsis Alarm: TRIGGERED).

  PHYSIOLOGICAL DRIVERS:
  Sepsis likelihood is actively driven by: Resp (value: 1.08, impact: +0.15 log-odds), Unit1 (value: 0.59, impact: +0.14 log-odds), Platelets (value: -0.15, impact: +0.11 log-odds).

  CLINICAL OBSERVATIONS:
  Specifically, vitals trend logs indicate patient deterioration (Oxygen Desaturation: decreased by 10.50 over the last 3 hours (value: 99.0 -> 88.5)) and monitoring values are incomplete (Critical vital sign 'Temp' is missing/NaN.).

  CLINICAL RECOMMENDATION:
  [CRITICAL] Sepsis notification triggered. Recommend immediately ordering blood cultures and serum lactate levels, reviewing fluid resuscitation balance, and initiating empiric broad-spectrum antibiotics within 1 hour.
  ```
* **Alert Agent**: Dispatches alert with trend warning: `[WARNING SEPSIS ALERT] Sepsis Prob: 41.3% | Primary Drivers: Resp, Unit1 | Trend: Oxygen Desaturation`.

---

## 🛠️ Verification & Execution

### 1. Run the Multi-Agent Simulation
To simulate ICU patient hourly streams and watch the whiteboard logs:
```bash
python Phase3/main.py
```

### 2. Inspect Programmatic Reports
The final consolidated multi-agent report is saved locally to `Phase3/sample_patient_agent_report.json` for validation.
