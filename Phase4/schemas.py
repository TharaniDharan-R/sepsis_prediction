from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime

# ==========================================
# Patient Schemas
# ==========================================
class PatientBase(BaseModel):
    PatientID: str = Field(..., description="Unique alphanumeric identifier for the patient")
    Age: float = Field(..., description="Patient age in years")
    Gender: int = Field(..., description="Gender (1 for male, 0 for female)")
    Unit1: Optional[float] = Field(None, description="ICU unit type 1 (MICU)")
    Unit2: Optional[float] = Field(None, description="ICU unit type 2 (SICU)")
    HospAdmTime: float = Field(..., description="Hours between hospital admission and ICU admission")

class PatientCreate(PatientBase):
    pass

class PatientResponse(PatientBase):
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


# ==========================================
# Vital Log Schemas
# ==========================================
class VitalLogBase(BaseModel):
    hour: int = Field(..., description="Hourly index of the patient record in ICU")
    
    # Vitals
    HR: Optional[float] = None
    O2Sat: Optional[float] = None
    Temp: Optional[float] = None
    SBP: Optional[float] = None
    MAP: Optional[float] = None
    DBP: Optional[float] = None
    Resp: Optional[float] = None
    EtCO2: Optional[float] = None
    
    # Labs
    BaseExcess: Optional[float] = None
    HCO3: Optional[float] = None
    FiO2: Optional[float] = None
    pH: Optional[float] = None
    PaCO2: Optional[float] = None
    SaO2: Optional[float] = None
    AST: Optional[float] = None
    BUN: Optional[float] = None
    Alkalinephos: Optional[float] = None
    Calcium: Optional[float] = None
    Chloride: Optional[float] = None
    Creatinine: Optional[float] = None
    Bilirubin_direct: Optional[float] = None
    Glucose: Optional[float] = None
    Lactate: Optional[float] = None
    Magnesium: Optional[float] = None
    Phosphate: Optional[float] = None
    Potassium: Optional[float] = None
    Bilirubin_total: Optional[float] = None
    TroponinI: Optional[float] = None
    Hct: Optional[float] = None
    Hgb: Optional[float] = None
    PTT: Optional[float] = None
    WBC: Optional[float] = None
    Fibrinogen: Optional[float] = None
    Platelets: Optional[float] = None
    
    # Tracking
    ICULOS: Optional[float] = None

class VitalLogCreate(VitalLogBase):
    pass

class VitalLogResponse(VitalLogBase):
    id: int
    PatientID: str
    Shock_Index: Optional[float] = None
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


# ==========================================
# Prediction and Orchestrator Reports
# ==========================================
class PredictionResponse(BaseModel):
    id: int
    PatientID: str
    hour: int
    probability: float
    alert_triggered: bool
    tuned_threshold: float
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

class AgentReportResponse(BaseModel):
    id: int
    PatientID: str
    hour: int
    clinical_summary: str
    generation_method: str
    whiteboard_logs: List[str]
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

class AlertResponse(BaseModel):
    id: int
    PatientID: str
    hour: int
    alert_triggered: bool
    dispatch_message: str
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


# ==========================================
# Explanation (XAI) Schemas
# ==========================================
class FeatureContribution(BaseModel):
    feature: str
    value: float
    impact: float
    importance: float

class LocalExplanationResponse(BaseModel):
    patient_id: str
    hour: int
    probability: float
    tuned_threshold: float
    shap_contributions: List[FeatureContribution]
    lime_contributions: List[FeatureContribution]


# ==========================================
# End-to-End Orchestration Response
# ==========================================
class SepsisProcessResponse(BaseModel):
    PatientID: str
    hour: int
    probability: float
    alert_triggered: bool
    tuned_threshold: float
    clinical_summary: str
    generation_method: str
    whiteboard_logs: List[str]
    dispatch_message: Optional[str] = None
    created_at: datetime
