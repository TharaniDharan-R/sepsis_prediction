from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, ForeignKey, Text, JSON, func
from sqlalchemy.orm import relationship
from database import Base

class Patient(Base):
    __tablename__ = "patients"
    
    PatientID = Column(String, primary_key=True, index=True)
    Age = Column(Float, nullable=False)
    Gender = Column(Integer, nullable=False)
    Unit1 = Column(Float, nullable=True)
    Unit2 = Column(Float, nullable=True)
    HospAdmTime = Column(Float, nullable=False)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    vitals_logs = relationship("VitalLog", back_populates="patient", cascade="all, delete-orphan")
    predictions = relationship("PredictionRecord", back_populates="patient", cascade="all, delete-orphan")
    agent_reports = relationship("AgentReport", back_populates="patient", cascade="all, delete-orphan")
    alerts = relationship("AlertRecord", back_populates="patient", cascade="all, delete-orphan")


class VitalLog(Base):
    __tablename__ = "vitals_logs"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    PatientID = Column(String, ForeignKey("patients.PatientID"), nullable=False)
    hour = Column(Integer, nullable=False)
    
    # Vitals
    HR = Column(Float, nullable=True)
    O2Sat = Column(Float, nullable=True)
    Temp = Column(Float, nullable=True)
    SBP = Column(Float, nullable=True)
    MAP = Column(Float, nullable=True)
    DBP = Column(Float, nullable=True)
    Resp = Column(Float, nullable=True)
    EtCO2 = Column(Float, nullable=True)
    
    # Labs
    BaseExcess = Column(Float, nullable=True)
    HCO3 = Column(Float, nullable=True)
    FiO2 = Column(Float, nullable=True)
    pH = Column(Float, nullable=True)
    PaCO2 = Column(Float, nullable=True)
    SaO2 = Column(Float, nullable=True)
    AST = Column(Float, nullable=True)
    BUN = Column(Float, nullable=True)
    Alkalinephos = Column(Float, nullable=True)
    Calcium = Column(Float, nullable=True)
    Chloride = Column(Float, nullable=True)
    Creatinine = Column(Float, nullable=True)
    Bilirubin_direct = Column(Float, nullable=True)
    Glucose = Column(Float, nullable=True)
    Lactate = Column(Float, nullable=True)
    Magnesium = Column(Float, nullable=True)
    Phosphate = Column(Float, nullable=True)
    Potassium = Column(Float, nullable=True)
    Bilirubin_total = Column(Float, nullable=True)
    TroponinI = Column(Float, nullable=True)
    Hct = Column(Float, nullable=True)
    Hgb = Column(Float, nullable=True)
    PTT = Column(Float, nullable=True)
    WBC = Column(Float, nullable=True)
    Fibrinogen = Column(Float, nullable=True)
    Platelets = Column(Float, nullable=True)
    
    # Demographics / tracking
    ICULOS = Column(Float, nullable=True)
    
    # Engineered
    Shock_Index = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=func.now())
    
    # Relationship
    patient = relationship("Patient", back_populates="vitals_logs")


class PredictionRecord(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    PatientID = Column(String, ForeignKey("patients.PatientID"), nullable=False)
    hour = Column(Integer, nullable=False)
    probability = Column(Float, nullable=False)
    alert_triggered = Column(Boolean, nullable=False)
    tuned_threshold = Column(Float, nullable=False)
    created_at = Column(DateTime, default=func.now())
    
    # Relationship
    patient = relationship("Patient", back_populates="predictions")


class AgentReport(Base):
    __tablename__ = "agent_reports"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    PatientID = Column(String, ForeignKey("patients.PatientID"), nullable=False)
    hour = Column(Integer, nullable=False)
    clinical_summary = Column(Text, nullable=False)
    generation_method = Column(String, nullable=False)
    whiteboard_logs = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=func.now())
    
    # Relationship
    patient = relationship("Patient", back_populates="agent_reports")


class AlertRecord(Base):
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    PatientID = Column(String, ForeignKey("patients.PatientID"), nullable=False)
    hour = Column(Integer, nullable=False)
    alert_triggered = Column(Boolean, nullable=False)
    dispatch_message = Column(Text, nullable=False)
    created_at = Column(DateTime, default=func.now())
    
    # Relationship
    patient = relationship("Patient", back_populates="alerts")
