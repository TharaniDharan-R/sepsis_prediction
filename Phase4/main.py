from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from datetime import datetime
import pandas as pd
import numpy as np
import os

# Import Phase 4 settings and DB components
import config
import models
import schemas
import crud
from database import get_db, engine, Base

# Import Phase 3 & 2 Clinical Orchestrators
from orchestrator import AgentOrchestrator
from explainers import SepsisExplainer

# Initialize FastAPI App
app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description="Backend microservice for real-time Sepsis prediction, patient trend monitoring, and explainable AI insights."
)

# Mount Phase 5 static folder
dashboard_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Phase5"))
app.mount("/dashboard", StaticFiles(directory=dashboard_path, html=True), name="dashboard")

# Enable CORS for dashboard integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for clinical engine instances loaded on startup
orchestrator = None
explainer = None

@app.on_event("startup")
def startup_event():
    global orchestrator, explainer
    
    print("\n--- Starting FastAPI Sepsis Service ---")
    
    # 1. Initialize database tables (SQLite or PostgreSQL)
    print(f"Connecting to database and initializing schemas at: {config.DATABASE_URL.split('@')[-1]}...")
    Base.metadata.create_all(bind=engine)
    print("Database tables initialized successfully.")
    
    # 2. Pre-load Machine Learning and Explainability assets
    print("Initializing Multi-Agent Clinical Orchestrator...")
    orchestrator = AgentOrchestrator()
    
    print("Initializing SHAP & LIME Sepsis Explainability Engine...")
    explainer = SepsisExplainer(use_background_data=True)
    
    print("FastAPI Sepsis Service fully loaded and ready!\n")


# Helper function to convert DB model list to a raw PhysioNet format Pandas DataFrame
def build_patient_dataframe(patient: models.Patient, vitals_list: list) -> pd.DataFrame:
    data = []
    for log in vitals_list:
        row = {
            config.COL_PATIENT_ID: patient.PatientID,
            "Age": patient.Age,
            "Gender": patient.Gender,
            "Unit1": patient.Unit1,
            "Unit2": patient.Unit2,
            "HospAdmTime": patient.HospAdmTime,
            "ICULOS": log.ICULOS
        }
        
        # Add vitals
        for col in config.COL_VITALS:
            row[col] = getattr(log, col, None)
            
        # Add labs
        for col in config.COL_LABS:
            row[col] = getattr(log, col, None)
            
        data.append(row)
        
    df = pd.DataFrame(data)
    
    # Reorder columns to match the raw training column structure exactly
    raw_col_order = config.COL_VITALS + config.COL_LABS + config.COL_DEMOGRAPHICS
    if config.COL_PATIENT_ID not in raw_col_order:
        raw_col_order.append(config.COL_PATIENT_ID)
        
    # Ensure all columns exist in the DataFrame
    for col in raw_col_order:
        if col not in df.columns:
            df[col] = None
            
    return df[raw_col_order]


@app.get("/")
def read_root():
    return {
        "status": "online",
        "api_title": config.API_TITLE,
        "version": config.API_VERSION,
        "active_model": orchestrator.prediction_agent.predictor.model_name if orchestrator else "Loading...",
        "tuned_threshold": orchestrator.prediction_agent.predictor.tuned_threshold if orchestrator else 0.5
    }


# ==========================================
# Patient Routes
# ==========================================
@app.post("/patients", response_model=schemas.PatientResponse, status_code=status.HTTP_201_CREATED)
def create_patient_route(patient: schemas.PatientCreate, db: Session = Depends(get_db)):
    db_patient = crud.get_patient(db, patient_id=patient.PatientID)
    if db_patient:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Patient with ID {patient.PatientID} already registered."
        )
    return crud.create_patient(db=db, patient=patient)

@app.get("/patients/{patient_id}", response_model=schemas.PatientResponse)
def read_patient_route(patient_id: str, db: Session = Depends(get_db)):
    db_patient = crud.get_patient(db, patient_id=patient_id)
    if not db_patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient with ID {patient_id} not found."
        )
    return db_patient

@app.get("/patients", response_model=list[schemas.PatientResponse])
def list_patients_route(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return crud.get_patients(db, skip=skip, limit=limit)


# ==========================================
# Vital Log Routes
# ==========================================
@app.post("/patients/{patient_id}/vitals", response_model=schemas.VitalLogResponse, status_code=status.HTTP_201_CREATED)
def log_patient_vitals_route(patient_id: str, vital: schemas.VitalLogCreate, db: Session = Depends(get_db)):
    db_patient = crud.get_patient(db, patient_id=patient_id)
    if not db_patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient with ID {patient_id} must be registered first."
        )
    
    # Check if this hour is already logged
    existing_logs = crud.get_vitals_history(db, patient_id=patient_id)
    if any(log.hour == vital.hour for log in existing_logs):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Vitals for hour {vital.hour} already logged for patient {patient_id}."
        )
        
    return crud.create_vital_log(db=db, vital=vital, patient_id=patient_id)

@app.get("/patients/{patient_id}/history", response_model=list[schemas.VitalLogResponse])
def get_vitals_history_route(patient_id: str, db: Session = Depends(get_db)):
    db_patient = crud.get_patient(db, patient_id=patient_id)
    if not db_patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient with ID {patient_id} not found."
        )
    return crud.get_vitals_history(db, patient_id=patient_id)


# ==========================================
# Clinical Orchestrator & Processing Routes
# ==========================================
@app.post("/patients/{patient_id}/process", response_model=schemas.SepsisProcessResponse)
def process_patient_route(patient_id: str, db: Session = Depends(get_db)):
    # 1. Fetch Patient
    db_patient = crud.get_patient(db, patient_id=patient_id)
    if not db_patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient with ID {patient_id} not found."
        )
        
    # 2. Fetch all historical logs ordered by hour
    vitals_list = crud.get_vitals_history(db, patient_id=patient_id)
    if not vitals_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot process patient record. No hourly vital records have been logged."
        )
        
    # 3. Assemble chronological DataFrame matching model feature schema
    patient_df = build_patient_dataframe(db_patient, vitals_list)
    current_hour = len(vitals_list)
    
    # 4. Execute Multi-Agent clinical blackboard orchestration
    print(f"\n[API] Running multi-agent clinical team evaluation for patient {patient_id} at hour {current_hour}...")
    report = orchestrator.process_patient_record(patient_df)
    
    # 5. Commit orchestrator outputs to database
    probability = float(report['prediction']['probability'])
    alert_triggered = bool(report['prediction']['alert_triggered'])
    tuned_threshold = float(report['prediction']['tuned_threshold'])
    
    crud.create_prediction(
        db, patient_id, current_hour, 
        probability=probability, 
        alert_triggered=alert_triggered, 
        tuned_threshold=tuned_threshold
    )
    
    crud.create_agent_report(
        db, patient_id, current_hour,
        clinical_summary=report['risk_assessment']['clinical_summary'],
        generation_method=report['risk_assessment']['generation_method'],
        whiteboard_logs=report['whiteboard_communication_log']
    )
    
    db_alert = crud.create_alert(
        db, patient_id, current_hour,
        alert_triggered=report['alert']['alert_triggered'],
        dispatch_message=report['alert']['dispatch_message']
    )
    
    return schemas.SepsisProcessResponse(
        PatientID=patient_id,
        hour=current_hour,
        probability=probability,
        alert_triggered=alert_triggered,
        tuned_threshold=tuned_threshold,
        clinical_summary=report['risk_assessment']['clinical_summary'],
        generation_method=report['risk_assessment']['generation_method'],
        whiteboard_logs=report['whiteboard_communication_log'],
        dispatch_message=db_alert.dispatch_message if db_alert.alert_triggered else None,
        created_at=db_alert.created_at
    )


# ==========================================
# Alerting Routes
# ==========================================
@app.get("/alerts", response_model=list[schemas.AlertResponse])
def get_active_alerts_route(limit: int = 50, db: Session = Depends(get_db)):
    return crud.get_active_alerts(db, limit=limit)


# ==========================================
# Explainable AI (XAI) Routes
# ==========================================
@app.get("/patients/{patient_id}/explain/{hour}", response_model=schemas.LocalExplanationResponse)
def get_patient_hour_explanation_route(patient_id: str, hour: int, db: Session = Depends(get_db)):
    # 1. Fetch Patient and History
    db_patient = crud.get_patient(db, patient_id=patient_id)
    if not db_patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Patient {patient_id} not found.")
        
    vitals_list = db.query(models.VitalLog).filter(
        models.VitalLog.PatientID == patient_id,
        models.VitalLog.hour <= hour
    ).order_by(models.VitalLog.hour).all()
    
    if not vitals_list:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"No vitals logged up to hour {hour}.")
        
    # 2. Build DataFrame
    patient_df = build_patient_dataframe(db_patient, vitals_list)
    
    # 3. Transform through fitted preprocessing pipeline
    X_preprocessed = explainer.preprocessor.transform(patient_df)
    
    # 4. Extract target hour (the last row in preprocessed sequence)
    target_row = X_preprocessed[-1]
    
    # 5. Compute local SHAP and LIME explanations
    explanation_payload = explainer.explain_patient_hour(target_row)
    
    # Map features to contributions schemas
    shap_contribs = []
    for c in explanation_payload['shap']['contributions']:
        shap_contribs.append(schemas.FeatureContribution(
            feature=c['feature'],
            value=c['value'],
            impact=c['impact'],
            importance=c['importance']
        ))
        
    lime_contribs = []
    for c in explanation_payload['lime']['contributions']:
        lime_contribs.append(schemas.FeatureContribution(
            feature=c['feature'],
            value=c['value'],
            impact=c['impact'],
            importance=c['importance']
        ))
        
    return schemas.LocalExplanationResponse(
        patient_id=patient_id,
        hour=hour,
        probability=explanation_payload['prediction']['probability'],
        tuned_threshold=explanation_payload['prediction']['tuned_threshold'],
        shap_contributions=shap_contribs,
        lime_contributions=lime_contribs
    )

@app.get("/patients/{patient_id}/report/{hour}", response_model=schemas.AgentReportResponse)
def get_patient_hour_report_route(patient_id: str, hour: int, db: Session = Depends(get_db)):
    db_report = db.query(models.AgentReport).filter(
        models.AgentReport.PatientID == patient_id,
        models.AgentReport.hour == hour
    ).first()
    if not db_report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"No agent report found for patient {patient_id} at hour {hour}."
        )
    return db_report

@app.get("/patients/{patient_id}/predictions", response_model=list[schemas.PredictionResponse])
def get_patient_predictions_route(patient_id: str, db: Session = Depends(get_db)):
    db_patient = crud.get_patient(db, patient_id=patient_id)
    if not db_patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Patient with ID {patient_id} not found."
        )
    return db.query(models.PredictionRecord).filter(
        models.PredictionRecord.PatientID == patient_id
    ).order_by(models.PredictionRecord.hour).all()
