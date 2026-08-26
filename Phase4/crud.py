from sqlalchemy.orm import Session
from sqlalchemy import desc
import models
import schemas

# ==========================================
# Patient Operations
# ==========================================
def get_patient(db: Session, patient_id: str):
    return db.query(models.Patient).filter(models.Patient.PatientID == patient_id).first()

def get_patients(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Patient).offset(skip).limit(limit).all()

def create_patient(db: Session, patient: schemas.PatientCreate):
    db_patient = models.Patient(
        PatientID=patient.PatientID,
        Age=patient.Age,
        Gender=patient.Gender,
        Unit1=patient.Unit1,
        Unit2=patient.Unit2,
        HospAdmTime=patient.HospAdmTime
    )
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    return db_patient


# ==========================================
# Vital Log Operations
# ==========================================
def get_vitals_history(db: Session, patient_id: str):
    return db.query(models.VitalLog).filter(models.VitalLog.PatientID == patient_id).order_by(models.VitalLog.hour).all()

def create_vital_log(db: Session, vital: schemas.VitalLogCreate, patient_id: str):
    # Calculate Shock Index dynamically (HR / SBP)
    shock_index = None
    if vital.HR is not None and vital.SBP is not None and vital.SBP > 0:
        shock_index = float(vital.HR / vital.SBP)
        
    db_vital = models.VitalLog(
        PatientID=patient_id,
        hour=vital.hour,
        
        # Vitals
        HR=vital.HR,
        O2Sat=vital.O2Sat,
        Temp=vital.Temp,
        SBP=vital.SBP,
        MAP=vital.MAP,
        DBP=vital.DBP,
        Resp=vital.Resp,
        EtCO2=vital.EtCO2,
        
        # Labs
        BaseExcess=vital.BaseExcess,
        HCO3=vital.HCO3,
        FiO2=vital.FiO2,
        pH=vital.pH,
        PaCO2=vital.PaCO2,
        SaO2=vital.SaO2,
        AST=vital.AST,
        BUN=vital.BUN,
        Alkalinephos=vital.Alkalinephos,
        Calcium=vital.Calcium,
        Chloride=vital.Chloride,
        Creatinine=vital.Creatinine,
        Bilirubin_direct=vital.Bilirubin_direct,
        Glucose=vital.Glucose,
        Lactate=vital.Lactate,
        Magnesium=vital.Magnesium,
        Phosphate=vital.Phosphate,
        Potassium=vital.Potassium,
        Bilirubin_total=vital.Bilirubin_total,
        TroponinI=vital.TroponinI,
        Hct=vital.Hct,
        Hgb=vital.Hgb,
        PTT=vital.PTT,
        WBC=vital.WBC,
        Fibrinogen=vital.Fibrinogen,
        Platelets=vital.Platelets,
        
        # Demographics
        ICULOS=vital.ICULOS,
        
        # Calculated
        Shock_Index=shock_index
    )
    db.add(db_vital)
    db.commit()
    db.refresh(db_vital)
    return db_vital


# ==========================================
# Orchestrator Result Operations
# ==========================================
def create_prediction(db: Session, patient_id: str, hour: int, probability: float, alert_triggered: bool, tuned_threshold: float):
    db_pred = models.PredictionRecord(
        PatientID=patient_id,
        hour=hour,
        probability=probability,
        alert_triggered=alert_triggered,
        tuned_threshold=tuned_threshold
    )
    db.add(db_pred)
    db.commit()
    db.refresh(db_pred)
    return db_pred

def create_agent_report(db: Session, patient_id: str, hour: int, clinical_summary: str, generation_method: str, whiteboard_logs: list):
    db_report = models.AgentReport(
        PatientID=patient_id,
        hour=hour,
        clinical_summary=clinical_summary,
        generation_method=generation_method,
        whiteboard_logs=whiteboard_logs
    )
    db.add(db_report)
    db.commit()
    db.refresh(db_report)
    return db_report

def create_alert(db: Session, patient_id: str, hour: int, alert_triggered: bool, dispatch_message: str):
    db_alert = models.AlertRecord(
        PatientID=patient_id,
        hour=hour,
        alert_triggered=alert_triggered,
        dispatch_message=dispatch_message
    )
    db.add(db_alert)
    db.commit()
    db.refresh(db_alert)
    return db_alert

def get_active_alerts(db: Session, limit: int = 50):
    return db.query(models.AlertRecord).filter(models.AlertRecord.alert_triggered == True).order_by(desc(models.AlertRecord.created_at)).limit(limit).all()
