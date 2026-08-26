# Phase 5: Sepsis Clinical Dashboard UI

This folder contains the clinical dashboard web application. Served directly from our FastAPI backend, it provides a premium, responsive dark-mode dashboard for intensive care clinicians.

---

## 🎨 Dashboard Features

1. **ICU Patient Directory Sidebar**:
   - Lists all patients currently monitored in the ICU.
   - Includes real-time search filtering.
   - A button to register new patients with initial Age, Gender, ICU unit type, and Hospital offset parameters.

2. **Real-time Vitals Monitor Grid**:
   - Displays current vital stats (Heart Rate, O2 Saturation, Temp, Blood Pressure, Respiration Rate, and dynamically calculated Shock Index).

3. **Risk Trajectory Chart (Chart.js)**:
   - Interactive line graph displaying the sepsis probability timeline hour-by-hour.
   - Draws a dashed red line representing the tuned decision threshold (`0.2668`).
   - Allows clicking on any hour point to reload historical XAI metrics and agent rounding notes.

4. **Explainable AI (XAI) Panel**:
   - Visualizes SHAP feature drivers (log-odds impact) or LIME feature weights for the selected hour.
   - Color-coded indicators: Red bars push probability *up* (increasing risk), Blue bars pull probability *down* (stable factors).

5. **Multi-Agent Rounding Notes**:
   - Displays the cooperatively compiled clinical summary from the **Risk Assessment Agent**.
   - Displays the step-by-step whiteboard transcript detailing agent routing and validation.

6. **Interactive Vital Stream Simulator**:
   - Clinicians can log vitals hour-by-hour, instantly running the model, the agents, and triggering pager dispatches when threshold alarms trip (with audio notifications!).

---

## 🚀 Running the Dashboard

1. **Start the Backend Server**:
   Execute the boot script inside `Phase4`:
   ```bash
   python Phase4/run.py
   ```

2. **Open the Web Browser**:
   Navigate to the mounted dashboard endpoint:
   [http://127.0.0.1:8000/dashboard/](http://127.0.0.1:8000/dashboard/)
