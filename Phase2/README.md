# Phase 2: Explainable AI (XAI) Pipeline

This directory implements the **Explainable AI (XAI)** layer of the Agentic Sepsis Prediction System. Using **SHAP** and **LIME**, it decodes the "black-box" decisions made by the XGBoost classifier from Phase 1, making risk alerts transparent, trustworthy, and actionable for ICU clinicians.

---

## 📌 Why Explainable AI (XAI) in Clinical AI?

In high-stakes clinical settings, accuracy is not enough. Clinicians must understand *why* an AI flags a patient for sepsis. XAI provides:
1. **Clinical Trust**: Enables doctors to verify model outputs against established medical pathophysiology.
2. **Actionable Insights**: Highlights specific physiological drivers (e.g., dropping blood pressure combined with high heart rate) so clinicians know where to intervene.
3. **Safety & Debugging**: Identifies when a model is latching onto confounding variables or demographic biases rather than clinical signals.
4. **Regulatory Alignment**: Satisfies "Right to Explanation" requirements under modern health data regulations (e.g., GDPR Article 22).

---

## 🔬 Mathematical & Conceptual Foundations

### 1. SHAP (Shapley Additive exPlanations)
SHAP is rooted in **cooperative game theory**. It treats each feature value of a patient record as a "player" in a coalition, and the model's prediction as the "payout". The Shapley value ($\phi_i$) represents the average marginal contribution of feature $i$ across all possible subsets of features.

The Shapley value is defined as:
$$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N| - |S| - 1)!}{|N|!} \left( v(S \cup \{i\}) - v(S) \right)$$

Where:
- $N$ is the set of all features.
- $S$ is a subset of features excluding feature $i$.
- $v(S)$ is the model prediction using only the features in $S$.

#### Key Axioms Guaranteed by SHAP:
- **Efficiency (Additivity)**: The sum of SHAP values of all features equals the difference between the local prediction and the base expected value: $\sum_{i=1}^M \phi_i = f(x) - E[f(x)]$.
- **Symmetry**: If two features contribute equally to all coalitions, their SHAP values are equal.
- **Dummy**: If a feature does not change the model output for any coalition, its SHAP value is zero.

#### TreeExplainer Optimization:
For tree ensemble models (like XGBoost), computing SHAP values normally scales exponentially with the number of features. SHAP's **TreeExplainer** optimizes this by tracking tree path statistics, reducing the computational complexity from $O(M \cdot 2^F)$ to polynomial time $O(T L D^2)$ (where $T$ is the number of trees, $L$ is the maximum number of leaves, and $D$ is the maximum depth).

---

### 2. LIME (Local Interpretable Model-agnostic Explanations)
LIME is a **surrogate-based explanation** method. It makes local perturbations around a specific patient's data point $x$, evaluates the model's outputs for these perturbed samples, and fits an interpretable linear surrogate model $g$ weighted by proximity to $x$.

LIME optimizes the following objective function:
$$\text{Explanation}(x) = \arg\min_{g \in G} \mathcal{L}(f, g, \pi_x) + \Omega(g)$$

Where:
- $f$ is the black-box model (XGBoost).
- $g$ is the interpretable model (e.g., ridge regression).
- $\pi_x(z)$ is the proximity measure of perturbed sample $z$ to $x$.
- $\mathcal{L}$ is the local fidelity loss measures how well $g$ approximates $f$ in the neighborhood.
- $\Omega(g)$ is the complexity of the explanation model (e.g., number of features allowed).

### SHAP vs. LIME Comparison
| Feature | SHAP | LIME |
| :--- | :--- | :--- |
| **Theoretical Base** | Game Theory (Axiomatically justified) | Local Surrogate optimization |
| **Consistency** | Consistent (Guaranteed) | Inconsistent (Stochastic perturbations) |
| **Global Consistency** | High (Aggregatable) | Low (Purely local) |
| **Speed** | Fast for tree models (TreeExplainer) | Medium (Requires generating perturbations) |

---

## 📊 Global Explanations & Feature Importance

Global explanations analyze the overall behavior of the model across the test set.

### Top 15 Feature Importances (SHAP vs. XGBoost Built-In)
We computed SHAP values on 500 test patient records and compared them with the XGBoost built-in importance (based on standard Gain):

| Feature | Mean Abs SHAP | XGBoost Gain | SHAP Rank | XGBoost Rank | Clinical Interpretation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **ICULOS** | 0.2271 | 0.1012 | 1.0 | 1.0 | ICU Length of Stay. Longer stays increase baseline infection risk. |
| **Unit1** | 0.2213 | 0.0519 | 2.0 | 2.0 | Medical ICU vs Surgical ICU care pathways. |
| **Creatinine** | 0.1373 | 0.0359 | 3.0 | 6.0 | Kidney function marker. Indicates early organ dysfunction. |
| **Temp** | 0.1338 | 0.0395 | 4.0 | 5.0 | Core body temperature. Fever/hypothermia are SIRS criteria. |
| **Lactate** | 0.1291 | 0.0501 | 5.0 | 4.0 | Tissue hypoperfusion. High lactate is a key sepsis marker. |
| **HospAdmTime** | 0.1236 | 0.0331 | 6.0 | 7.0 | Hours in hospital before ICU admission. |
| **Age** | 0.1193 | 0.0239 | 7.0 | 13.0 | Older patients have lower physiological reserves. |
| **Resp** | 0.0961 | 0.0279 | 8.0 | 10.0 | Respiration Rate. Tachypnea indicates pulmonary distress. |
| **WBC** | 0.0921 | 0.0236 | 9.0 | 14.0 | White Blood Cell count. Sign of active immune response. |
| **PTT** | 0.0842 | 0.0220 | 10.0 | 17.0 | Coagulation marker. Indicates sepsis-induced coagulopathy. |
| **Shock_Index** | 0.0795 | 0.0241 | 11.0 | 12.0 | Cardiovascular stress (Heart Rate / Systolic BP). |
| **HR** | 0.0689 | 0.0226 | 12.0 | 15.0 | Heart Rate. Tachycardia is a standard sepsis indicator. |
| **BUN** | 0.0666 | 0.0190 | 13.0 | 20.0 | Blood Urea Nitrogen. Reflects renal clearance. |
| **Hgb** | 0.0663 | 0.0187 | 14.0 | 21.0 | Hemoglobin level. Affects oxygen carrying capacity. |
| **Unit2** | 0.0636 | 0.0518 | 15.0 | 3.0 | Surgical ICU vs other specialty unit pathways. |

### Visualizations Saved in `visualizations/`:
1. `shap_summary_beeswarm.png`: Global beeswarm chart illustrating how high/low values of features shift predictions.
2. `shap_feature_importance_bar.png`: Standard bar chart representing the mean absolute impact of features.
3. `importance_comparison_top15.png`: Comparison bar chart contrasting SHAP ranking vs. XGBoost built-in ranks.

---

## 👤 Local Patient Case Studies (SHAP vs. LIME)

We extracted 4 representative patient hours from the test dataset to explain predictions at the local level. (Tuned threshold = **0.2668**).

### Case 1: True Positive (Index 28498)
* **True Label**: Sepsis (1) | **Predicted Sepsis Probability**: **93.72%** (Flagged 🚨)
* **Top Explanations**:
  - **ICULOS (Scaled value: 3.51)**: Highly elevated length of stay in ICU is the strongest driver boosting sepsis log-odds (+1.48 SHAP).
  - **Creatinine (Scaled value: 1.15)**: Kidney impairment contributes positively to sepsis risk (+0.27 SHAP).
  - **WBC (Scaled value: 4.39)**: Extremely high immune response sign contributes +0.21 SHAP.
  - **FiO2 (Scaled value: 3.77)**: High respiratory support requirement boosts risk (+0.20 SHAP).
  - **LIME Alignment**: LIME strongly aligns, placing `ICULOS` as the primary risk driver (+0.24 score) followed by `Creatinine` (+0.10 score).

### Case 2: True Negative (Index 72856)
* **True Label**: No Sepsis (0) | **Predicted Sepsis Probability**: **1.09%** (No Flag ✅)
* **Top Explanations**:
  - **HospAdmTime (Scaled value: 0.36)**: Short pre-ICU stay lowers prediction risk (-1.35 SHAP).
  - **Age (Scaled value: -2.74)**: Exceptionally young patient reduces baseline sepsis odds (-0.88 SHAP).
  - **Hct (Scaled value: 2.32)**: Normal hematocrit index reduces risk (-0.38 SHAP).
  - **LIME Alignment**: LIME places `HospAdmTime` (-0.23 score) and `ICULOS` (-0.12 score) as the top negative contributors, successfully mirroring SHAP.

### Case 3: False Positive (Index 54983) - Alarm triggered on stable patient
* **True Label**: No Sepsis (0) | **Predicted Sepsis Probability**: **94.95%** (Flagged 🚨)
* **Why did the model get confused?**
  - **ICULOS (Scaled value: 4.52)**: Extremely long ICU stay (+1.63 SHAP) heavily biased the model towards warning.
  - **Temp (Scaled value: 1.89)**: Elevated body temperature (fever indicator) pushed the log-odds up (+0.45 SHAP).
  - **Lactate (Scaled value: -0.57)**: Reconstructed/imputed lactate value and respiratory support `FiO2` (0.73) added minor positive weights.
  - **LIME Alignment**: LIME confirms that long `ICULOS` (+0.25) and high `Temp` (+0.07) triggered the alarm. Clinically, this patient displays inflammation markers (fever) and a long stay, making them look highly septic, explaining the false positive.

### Case 4: False Negative (Index 27323) - Sepsis prediction missed
* **True Label**: Sepsis (1) | **Predicted Sepsis Probability**: **26.66%** (No Flag ✅ - Missed by a fraction of a percent! Threshold is 26.68%)
* **Why did the model miss this?**
  - **Resp (Scaled value: 1.74)**: Elevated respiration rate was a strong positive sepsis marker (+0.32 SHAP).
  - **Age (Scaled value: 1.27)** & **HospAdmTime (Scaled value: 0.30)**: Pushed the risk score down (-0.22 and -0.19 SHAP).
  - **WBC (Scaled value: -0.68)**: Normal WBC levels counteracted the infection markers (-0.18 SHAP), keeping the probability just 0.02% below our active notification boundary.
  - **LIME Alignment**: LIME lists normal renal/liver counts like `Creatinine` (-0.04) and normal values as reasons for keeping prediction low.

---

## 🔌 API Integration Interface

The `SepsisExplainer` class in [explainers.py](file:///d:/sepsis/Phase2/explainers.py) offers a programmatic API `explain_patient_hour(preprocessed_row)` that returns a structured, JSON-friendly dictionary. Downstream microservices can extract this payload to explain clinical predictions dynamically in dashboards.

### Sample API Output JSON Payload:
```json
{
  "prediction": {
    "probability": 0.9371749758720398,
    "prediction_class": 1,
    "tuned_threshold": 0.2668378949165344
  },
  "shap": {
    "base_value": -1.0107970662735897,
    "prediction_value": 2.7025166984246605,
    "contributions": [
      {
        "feature": "ICULOS",
        "value": 3.5117,
        "impact": 1.4758,
        "importance": 1.4758
      },
      {
        "feature": "Creatinine",
        "value": 1.1542,
        "impact": 0.2706,
        "importance": 0.2706
      }
    ]
  },
  "lime": {
    "intercept": 0.0821,
    "contributions": [
      {
        "feature": "ICULOS",
        "value": 3.5117,
        "impact": 0.2380,
        "importance": 0.2380
      }
    ]
  }
}
```

---

## 🛠️ Verification & Usage

### 1. Run Global Explanations
To compute SHAP values for the test sample and save global plots:
```bash
python Phase2/explain_global.py
```

### 2. Run Local Explanations
To identify local cases (TP, TN, FP, FN), print JSON payloads, and save waterfall/bar local charts:
```bash
python Phase2/explain_local.py
```
