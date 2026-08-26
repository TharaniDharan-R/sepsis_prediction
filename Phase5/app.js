// API Base URL (relative as it is hosted on the same server)
const BASE_URL = "";

// Global App State
let selectedPatientId = null;
let patients = [];
let vitalsHistory = [];
let predictionsHistory = [];
let selectedHour = null;
let explanationData = null;
let activeExplanationTab = "shap"; // "shap" or "lime"
let activeWhiteboardTab = "clinical-summary"; // "clinical-summary" or "whiteboard-logs"
let trajectoryChartInstance = null;
let whiteboardLogs = [];
let clinicalSummaryMarkdown = "";

// Initialize App
document.addEventListener("DOMContentLoaded", () => {
    initEventListeners();
    fetchPatients();
});

// Setup DOM Event Listeners
function initEventListeners() {
    // Add Patient Modal toggles
    const addBtn = document.getElementById("add-patient-btn");
    const closeBtn = document.getElementById("close-modal-btn");
    const modal = document.getElementById("add-patient-modal");
    
    addBtn.addEventListener("click", () => modal.classList.remove("hidden"));
    closeBtn.addEventListener("click", () => modal.classList.add("hidden"));
    
    // Register Patient Submit
    const addForm = document.getElementById("add-patient-form");
    addForm.addEventListener("submit", registerPatient);
    
    // Vital Simulator Form Submit
    const simulatorForm = document.getElementById("vitals-simulator-form");
    simulatorForm.addEventListener("submit", simulateHour);
    
    // Patient Search filter
    const searchInput = document.getElementById("patient-search-input");
    searchInput.addEventListener("input", filterPatients);
    
    // Explanation Tab switches
    document.getElementById("tab-shap").addEventListener("click", () => {
        switchTab("explanation", "shap");
    });
    document.getElementById("tab-lime").addEventListener("click", () => {
        switchTab("explanation", "lime");
    });
    
    // Whiteboard Tab switches
    document.getElementById("tab-clinical-summary").addEventListener("click", () => {
        switchTab("whiteboard", "clinical-summary");
    });
    document.getElementById("tab-whiteboard-logs").addEventListener("click", () => {
        switchTab("whiteboard", "whiteboard-logs");
    });
    
    // Auto-calculate MAP in simulator if SBP and DBP are changed
    const sbpInput = document.getElementById("input-SBP");
    const dbpInput = document.getElementById("input-DBP");
    const mapInput = document.getElementById("input-MAP");
    
    const updateMAP = () => {
        const sbp = parseFloat(sbpInput.value);
        const dbp = parseFloat(dbpInput.value);
        if (!isNaN(sbp) && !isNaN(dbp)) {
            mapInput.placeholder = Math.round((sbp + 2 * dbp) / 3);
        }
    };
    sbpInput.addEventListener("input", updateMAP);
    dbpInput.addEventListener("input", updateMAP);
}

// ==========================================================
// API TRANSACTIONS (FETCH METHODS)
// ==========================================================

// 1. Fetch Patients list
async function fetchPatients() {
    try {
        const r = await fetch(`${BASE_URL}/patients`);
        if (!r.ok) throw new Error("Failed to load patients.");
        patients = await r.json();
        
        // Sort patients by registration (newest first)
        patients.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
        
        renderPatientList();
    } catch (e) {
        console.error(e);
        document.getElementById("patient-list").innerHTML = `<li class="loading-placeholder text-red">Connection error to database.</li>`;
    }
}

// 2. Register Patient
async function registerPatient(e) {
    e.preventDefault();
    
    const payload = {
        PatientID: document.getElementById("new-patient-id").value.trim(),
        Age: parseFloat(document.getElementById("new-patient-age").value),
        Gender: parseInt(document.getElementById("new-patient-gender").value),
        Unit1: parseFloat(document.getElementById("new-patient-unit1").value),
        Unit2: parseFloat(document.getElementById("new-patient-unit2").value),
        HospAdmTime: parseFloat(document.getElementById("new-patient-admtime").value)
    };
    
    try {
        const r = await fetch(`${BASE_URL}/patients`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        
        if (!r.ok) {
            const err = await r.json();
            throw new Error(err.detail || "Registration failed.");
        }
        
        const newPatient = await r.json();
        
        // Close modal & reset form
        document.getElementById("add-patient-modal").classList.add("hidden");
        document.getElementById("add-patient-form").reset();
        
        // Refresh sidebar
        await fetchPatients();
        
        // Auto-select newly registered patient
        selectPatient(newPatient.PatientID);
        
    } catch (e) {
        alert(`Registration Error: ${e.message}`);
    }
}

// 3. Select Patient
async function selectPatient(patientId) {
    selectedPatientId = patientId;
    selectedHour = null;
    explanationData = null;
    whiteboardLogs = [];
    clinicalSummaryMarkdown = "";
    
    // Update sidebar UI selection
    const items = document.querySelectorAll(".patient-item");
    items.forEach(item => {
        if (item.dataset.id === patientId) {
            item.classList.add("active");
        } else {
            item.classList.remove("active");
        }
    });
    
    // Hide empty screen, show dashboard view
    document.getElementById("no-patient-view").classList.add("hidden");
    const dashView = document.getElementById("dashboard-view");
    dashView.classList.remove("hidden");
    
    // Load metadata and histories
    try {
        // A. Load Metadata
        const rMeta = await fetch(`${BASE_URL}/patients/${patientId}`);
        if (!rMeta.ok) throw new Error("Metadata load failed.");
        const metadata = await rMeta.json();
        
        document.getElementById("chart-patient-id").innerText = metadata.PatientID;
        document.getElementById("chart-age").innerText = metadata.Age;
        document.getElementById("chart-gender").innerText = metadata.Gender === 1 ? "Male" : "Female";
        
        let unitText = "None";
        if (metadata.Unit1 === 1.0) unitText = "MICU (Unit 1)";
        else if (metadata.Unit2 === 1.0) unitText = "SICU (Unit 2)";
        document.getElementById("chart-unit").innerText = unitText;
        document.getElementById("chart-adm-time").innerText = metadata.HospAdmTime;
        
        // B. Load History
        await loadHistory();
        
    } catch (e) {
        console.error(e);
        alert(`Error loading patient details: ${e.message}`);
    }
}

// 4. Load patient vitals history, draw charts, and auto-select latest hour
async function loadHistory() {
    const rHist = await fetch(`${BASE_URL}/patients/${selectedPatientId}/history`);
    if (!rHist.ok) throw new Error("History load failed.");
    vitalsHistory = await rHist.json();
    
    // Clear simulator form MAP placeholder
    document.getElementById("input-MAP").placeholder = "Auto-calculated if blank";
    
    if (vitalsHistory.length === 0) {
        // No vitals logged yet
        renderEmptyVitalsState();
        return;
    }
    
    // Vitals are present: Update cards using the latest logged hour
    const latestVital = vitalsHistory[vitalsHistory.length - 1];
    
    document.getElementById("vital-val-HR").innerText = latestVital.HR !== null ? Math.round(latestVital.HR) : "--";
    document.getElementById("vital-val-O2Sat").innerText = latestVital.O2Sat !== null ? latestVital.O2Sat.toFixed(1) : "--";
    document.getElementById("vital-val-Temp").innerText = latestVital.Temp !== null ? latestVital.Temp.toFixed(1) : "--";
    document.getElementById("vital-val-Resp").innerText = latestVital.Resp !== null ? Math.round(latestVital.Resp) : "--";
    
    if (latestVital.SBP !== null && latestVital.DBP !== null) {
        document.getElementById("vital-val-BP").innerText = `${Math.round(latestVital.SBP)}/${Math.round(latestVital.DBP)}`;
    } else {
        document.getElementById("vital-val-BP").innerText = "-- / --";
    }
    
    document.getElementById("vital-val-ShockIndex").innerText = latestVital.Shock_Index !== null ? latestVital.Shock_Index.toFixed(2) : "--";
    
    // Set simulator next hour index
    document.getElementById("form-hour").value = vitalsHistory.length + 1;
    document.getElementById("simulate-btn").innerHTML = `<i class="fa-solid fa-play"></i> Log Hour ${vitalsHistory.length + 1} & Run AI Pipeline`;
    
    // Draw/Refresh trajectory line chart
    await fetchAndDrawTrajectory();
}

// 5. Fetch Sepsis Trajectory (predictions history) and render chart
async function fetchAndDrawTrajectory() {
    try {
        const r = await fetch(`${BASE_URL}/patients/${selectedPatientId}/predictions`);
        if (!r.ok) throw new Error("Failed to load predictions.");
        predictionsHistory = await r.json();
        
        const labels = predictionsHistory.map(p => `Hour ${p.hour}`);
        const probabilities = predictionsHistory.map(p => p.probability * 100);
        
        let thresholdVal = 26.68; // Default tuned recall threshold percentage
        let latestProbText = "0.0%";
        let isAlert = false;
        
        if (predictionsHistory.length > 0) {
            const latest = predictionsHistory[predictionsHistory.length - 1];
            thresholdVal = latest.tuned_threshold * 100;
            latestProbText = `${(latest.probability * 100).toFixed(1)}%`;
            isAlert = latest.alert_triggered;
        }
        
        // Update top summary header
        const probBadge = document.getElementById("chart-risk-prob");
        probBadge.innerText = latestProbText;
        if (isAlert) {
            probBadge.className = "badge-value risk-high";
            document.getElementById("chart-alarm-badge").className = "badge-status status-alert";
            document.getElementById("chart-alarm-badge").innerText = "CRITICAL ALARM";
        } else {
            probBadge.className = "badge-value risk-low";
            document.getElementById("chart-alarm-badge").className = "badge-status status-stable";
            document.getElementById("chart-alarm-badge").innerText = "STABLE";
        }
        
        // Setup Chart.js
        const ctx = document.getElementById("trajectoryChart").getContext("2d");
        
        if (trajectoryChartInstance) {
            trajectoryChartInstance.destroy();
        }
        
        // Add threshold baseline annotation dataset
        const thresholdLine = Array(predictionsHistory.length).fill(thresholdVal);
        
        trajectoryChartInstance = new Chart(ctx, {
            type: "line",
            data: {
                labels: labels,
                datasets: [
                    {
                        label: "Sepsis Probability (%)",
                        data: probabilities,
                        borderColor: "#3b82f6",
                        backgroundColor: "rgba(59, 130, 246, 0.1)",
                        fill: true,
                        tension: 0.2,
                        pointRadius: 6,
                        pointHoverRadius: 8,
                        pointBackgroundColor: "#3b82f6",
                        pointBorderColor: "#ffffff",
                        borderWidth: 3
                    },
                    {
                        label: "AI Alarm Threshold (%)",
                        data: thresholdLine,
                        borderColor: "#ef4444",
                        borderDash: [5, 5],
                        fill: false,
                        pointRadius: 0,
                        borderWidth: 1.5
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: "#9ca3af", font: { family: "Inter" } }
                    }
                },
                scales: {
                    y: {
                        min: 0,
                        max: 100,
                        grid: { color: "rgba(255, 255, 255, 0.05)" },
                        ticks: { color: "#9ca3af" }
                    },
                    x: {
                        grid: { color: "rgba(255, 255, 255, 0.05)" },
                        ticks: { color: "#9ca3af" }
                    }
                },
                onClick: (evt, activeElems) => {
                    if (activeElems.length > 0) {
                        const index = activeElems[0].index;
                        const clickedHour = predictionsHistory[index].hour;
                        selectHour(clickedHour);
                    }
                }
            }
        });
        
        // Auto-select latest hour
        if (predictionsHistory.length > 0) {
            selectHour(predictionsHistory[predictionsHistory.length - 1].hour);
        }
        
    } catch (e) {
        console.error(e);
    }
}

// 6. Select hour and load local explanations + clinical reports
async function selectHour(hour) {
    selectedHour = hour;
    
    // Highlight active point on Chart.js (optional visual cue in list)
    console.log(`[UI] Selected Hour: ${hour}`);
    
    const explanationContainer = document.getElementById("explanation-container");
    explanationContainer.innerHTML = `<div class="loading-placeholder"><i class="fa-solid fa-spinner fa-spin"></i> Calculating local explainability...</div>`;
    
    const summaryPanel = document.getElementById("clinical-summary-text");
    summaryPanel.innerHTML = `<div class="loading-placeholder"><i class="fa-solid fa-spinner fa-spin"></i> Fetching agent rounded notes...</div>`;
    
    try {
        // A. Load XAI
        const rExp = await fetch(`${BASE_URL}/patients/${selectedPatientId}/explain/${hour}`);
        if (!rExp.ok) throw new Error("Explanation fetch failed.");
        explanationData = await rExp.json();
        renderExplanationList();
        
        // B. Load Report
        const rReport = await fetch(`${BASE_URL}/patients/${selectedPatientId}/report/${hour}`);
        if (!rReport.ok) throw new Error("Report fetch failed.");
        const report = await rReport.json();
        
        whiteboardLogs = report.whiteboard_logs;
        clinicalSummaryMarkdown = report.clinical_summary;
        
        renderWhiteboard();
        
    } catch (e) {
        console.error(e);
        explanationContainer.innerHTML = `<div class="loading-placeholder text-red">XAI details unavailable for this hour.</div>`;
        summaryPanel.innerHTML = `<div class="loading-placeholder text-red">Agent notes unavailable for this hour.</div>`;
    }
}

// 7. Simulate hour vital sign logging and process pipeline
async function simulateHour(e) {
    e.preventDefault();
    
    const hour = parseInt(document.getElementById("form-hour").value);
    
    const vitalsPayload = {
        hour: hour,
        HR: parseFloat(document.getElementById("input-HR").value),
        O2Sat: parseFloat(document.getElementById("input-O2Sat").value),
        Temp: parseFloat(document.getElementById("input-Temp").value),
        Resp: parseFloat(document.getElementById("input-Resp").value),
        SBP: parseFloat(document.getElementById("input-SBP").value),
        DBP: parseFloat(document.getElementById("input-DBP").value),
        MAP: document.getElementById("input-MAP").value ? parseFloat(document.getElementById("input-MAP").value) : null,
        ICULOS: hour
    };
    
    const submitBtn = document.getElementById("simulate-btn");
    submitBtn.disabled = true;
    submitBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Evaluating Patient...`;
    
    try {
        // A. Log vitals log
        const rVitals = await fetch(`${BASE_URL}/patients/${selectedPatientId}/vitals`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(vitalsPayload)
        });
        
        if (!rVitals.ok) {
            const err = await rVitals.json();
            throw new Error(err.detail || "Failed to log vitals.");
        }
        
        // B. Run multi-agent pipeline process
        const rProcess = await fetch(`${BASE_URL}/patients/${selectedPatientId}/process`, {
            method: "POST"
        });
        
        if (!rProcess.ok) {
            const err = await rProcess.json();
            throw new Error(err.detail || "Multi-agent process pipeline failed.");
        }
        
        const processResult = await rProcess.json();
        
        // Refresh patient details & charts
        await loadHistory();
        
        // Reset vital input values
        document.getElementById("input-HR").value = "";
        document.getElementById("input-O2Sat").value = "";
        document.getElementById("input-Temp").value = "";
        document.getElementById("input-Resp").value = "";
        document.getElementById("input-SBP").value = "";
        document.getElementById("input-DBP").value = "";
        document.getElementById("input-MAP").value = "";
        
        // Play notification sound on alert trigger
        if (processResult.alert_triggered) {
            playAlertSound();
            alert(`[CRITICAL SEPSIS ALERT]\nProbability: ${(processResult.probability*100).toFixed(1)}%\nDispatch: ${processResult.dispatch_message}`);
        }
        
    } catch (e) {
        alert(`Pipeline Error: ${e.message}`);
    } finally {
        submitBtn.disabled = false;
    }
}

// ==========================================================
// RENDERERS (DOM INJECTIONS)
// ==========================================================

// Renders the patient sidebar directory
function renderPatientList() {
    const list = document.getElementById("patient-list");
    document.getElementById("patient-count").innerText = patients.length;
    
    if (patients.length === 0) {
        list.innerHTML = `<li class="loading-placeholder">No registered patients.</li>`;
        return;
    }
    
    list.innerHTML = "";
    
    patients.forEach(p => {
        const li = document.createElement("li");
        li.className = `patient-item ${p.PatientID === selectedPatientId ? 'active' : ''}`;
        li.dataset.id = p.PatientID;
        li.addEventListener("click", () => selectPatient(p.PatientID));
        
        li.innerHTML = `
            <div class="patient-item-details">
                <span class="patient-item-id">${p.PatientID}</span>
                <span class="patient-item-sub">Age: ${p.Age} | ${p.Gender === 1 ? 'M' : 'F'}</span>
            </div>
            <div class="patient-item-badges">
                <i class="fa-solid fa-chevron-right text-muted"></i>
            </div>
        `;
        list.appendChild(li);
    });
}

// Renders empty vitals state (upon registering new patients)
function renderEmptyVitalsState() {
    document.getElementById("vital-val-HR").innerText = "--";
    document.getElementById("vital-val-O2Sat").innerText = "--";
    document.getElementById("vital-val-Temp").innerText = "--";
    document.getElementById("vital-val-BP").innerText = "-- / --";
    document.getElementById("vital-val-Resp").innerText = "--";
    document.getElementById("vital-val-ShockIndex").innerText = "--";
    
    // Form setup
    document.getElementById("form-hour").value = 1;
    document.getElementById("simulate-btn").innerHTML = `<i class="fa-solid fa-play"></i> Log Hour 1 & Run AI Pipeline`;
    
    // Reset top badges
    const probBadge = document.getElementById("chart-risk-prob");
    probBadge.innerText = "0.0%";
    probBadge.className = "badge-value risk-low";
    
    const alarm = document.getElementById("chart-alarm-badge");
    alarm.className = "badge-status status-stable";
    alarm.innerText = "STABLE";
    
    // Clear explanations and reports
    document.getElementById("explanation-container").innerHTML = `<div class="loading-placeholder">Log hour 1 vitals to start clinical analysis.</div>`;
    document.getElementById("clinical-summary-text").innerHTML = `<div class="loading-placeholder">No agent notes logged. Run the simulator.</div>`;
    document.getElementById("whiteboard-logs-list").innerHTML = `<li>No logs available.</li>`;
    
    if (trajectoryChartInstance) {
        trajectoryChartInstance.destroy();
        trajectoryChartInstance = null;
    }
}

// Renders SHAP or LIME horizontal contribution bars
function renderExplanationList() {
    const container = document.getElementById("explanation-container");
    container.innerHTML = "";
    
    if (!explanationData) return;
    
    const contributions = activeExplanationTab === "shap" 
        ? explanationData.shap_contributions 
        : explanationData.lime_contributions;
        
    if (!contributions || contributions.length === 0) {
        container.innerHTML = `<div class="loading-placeholder">No contributions computed.</div>`;
        return;
    }
    
    // Find maximum absolute contribution to scale bar widths proportionally
    const maxVal = Math.max(...contributions.map(c => Math.abs(c.impact)), 0.001);
    
    contributions.forEach(c => {
        const item = document.createElement("div");
        item.className = "xai-item";
        
        const isPos = c.impact >= 0;
        const widthPct = Math.min((Math.abs(c.impact) / maxVal) * 100, 100);
        
        item.innerHTML = `
            <div class="xai-meta">
                <span class="xai-feat-name">${c.feature}</span>
                <span class="xai-feat-val">Val: ${c.value.toFixed(1)} | Impact: ${isPos ? '+' : ''}${c.impact.toFixed(3)}</span>
            </div>
            <div class="xai-bar-track">
                <div class="xai-bar ${isPos ? 'bar-positive' : 'bar-negative'}" style="width: ${widthPct}%"></div>
            </div>
        `;
        container.appendChild(item);
    });
}

// Renders multi-agent rounded whiteboard summaries and notes
function renderWhiteboard() {
    // 1. Clinical Summary Render (convert headers to nice markup style)
    const summaryTextPanel = document.getElementById("clinical-summary-text");
    
    // Parse the rule-based clinical summary text to add visual headers
    let formattedHtml = clinicalSummaryMarkdown
        .replace(/CLINICAL RISK ASSESSMENT/g, "<h4>CLINICAL RISK ASSESSMENT</h4>")
        .replace(/PHYSIOLOGICAL DRIVERS:/g, "<h4>PHYSIOLOGICAL DRIVERS</h4>")
        .replace(/CLINICAL OBSERVATIONS:/g, "<h4>CLINICAL OBSERVATIONS</h4>")
        .replace(/CLINICAL RECOMMENDATION:/g, "<h4>CLINICAL RECOMMENDATION</h4>")
        .replace(/\[CRITICAL\]/g, "<strong class='text-red'>[CRITICAL]</strong>")
        .replace(/\[WARNING\]/g, "<strong class='text-orange'>[WARNING]</strong>")
        .replace(/\[INFO\]/g, "<strong class='text-green'>[INFO]</strong>");
        
    summaryTextPanel.innerHTML = formattedHtml;
    
    // 2. Whiteboard transcripts list
    const logList = document.getElementById("whiteboard-logs-list");
    logList.innerHTML = "";
    
    whiteboardLogs.forEach(log => {
        const li = document.createElement("li");
        
        // Determine agent CSS styling class
        let agentClass = "log-item-orchestrator";
        if (log.includes("[Orchestrator]")) agentClass = "log-item-orchestrator";
        else if (log.includes("[ValidationAgent]")) agentClass = "log-item-validation";
        else if (log.includes("[PredictionAgent]")) agentClass = "log-item-prediction";
        else if (log.includes("[MonitoringAgent]")) agentClass = "log-item-monitoring";
        else if (log.includes("[ExplainabilityAgent]")) agentClass = "log-item-explainability";
        else if (log.includes("[RiskAssessmentAgent]")) agentClass = "log-item-riskassessment";
        else if (log.includes("[AlertAgent]")) agentClass = "log-item-alert";
        
        li.className = `log-item ${agentClass}`;
        li.innerText = log;
        logList.appendChild(li);
    });
}

// ==========================================================
// INTERACTIVE DOM TAB TOGGLES
// ==========================================================
function switchTab(cardType, tabName) {
    if (cardType === "explanation") {
        activeExplanationTab = tabName;
        document.getElementById("tab-shap").classList.toggle("active", tabName === "shap");
        document.getElementById("tab-lime").classList.toggle("active", tabName === "lime");
        renderExplanationList();
    } else if (cardType === "whiteboard") {
        activeWhiteboardTab = tabName;
        document.getElementById("tab-clinical-summary").classList.toggle("active", tabName === "clinical-summary");
        document.getElementById("tab-whiteboard-logs").classList.toggle("active", tabName === "whiteboard-logs");
        
        document.getElementById("clinical-summary-panel").classList.toggle("hidden", tabName !== "clinical-summary");
        document.getElementById("whiteboard-logs-panel").classList.toggle("hidden", tabName !== "whiteboard-logs");
    }
}

// Search and filter sidebar patients
function filterPatients() {
    const query = document.getElementById("patient-search-input").value.toLowerCase();
    const items = document.querySelectorAll(".patient-item");
    
    items.forEach(item => {
        const pid = item.dataset.id.toLowerCase();
        if (pid.includes(query)) {
            item.classList.remove("hidden");
        } else {
            item.classList.add("hidden");
        }
    });
}

// Helper audio feedback for Sepsis Alarm triggers
function playAlertSound() {
    try {
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        
        const playBeep = (freq, delay, duration) => {
            setTimeout(() => {
                const osc = audioCtx.createOscillator();
                const gain = audioCtx.createGain();
                osc.connect(gain);
                gain.connect(audioCtx.destination);
                osc.type = "sine";
                osc.frequency.value = freq;
                gain.gain.setValueAtTime(0.1, audioCtx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + duration);
                osc.start();
                osc.stop(audioCtx.currentTime + duration);
            }, delay);
        };
        
        // Double critical beep pattern
        playBeep(880, 0, 0.15);
        playBeep(880, 200, 0.15);
        
    } catch (err) {
        console.log("Audio contexts blocked by browser policy.");
    }
}
