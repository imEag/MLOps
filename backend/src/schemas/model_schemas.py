from pydantic import BaseModel
from typing import Optional, List, Any, Dict
from datetime import datetime

class ModelVersionResponse(BaseModel):
    model_name: str
    version: str
    run_id: str
    status: str
    current_stage: Optional[str] = None
    aliases: Optional[List[str]] = []
    source: Optional[str] = None
    creation_timestamp: Optional[Any] = None
    last_updated_timestamp: Optional[Any] = None

    class Config:
        orm_mode = True 

class TrainResponse(BaseModel):
    message: str
    flow_state: Optional[str] = None

class InputExampleResponse(BaseModel):
    example: Any

class ModelMetrics(BaseModel):
    accuracy: Optional[float] = None
    macro_avg_precision: Optional[float] = None
    macro_avg_recall: Optional[float] = None
    macro_avg_f1_score: Optional[float] = None
    weighted_avg_precision: Optional[float] = None
    weighted_avg_recall: Optional[float] = None
    weighted_avg_f1_score: Optional[float] = None
    cv_mean_accuracy: Optional[float] = None
    cv_std_accuracy: Optional[float] = None

class ModelInfo(BaseModel):
    model_name: str
    version: str
    run_id: str
    status: str
    creation_timestamp: Optional[Any] = None
    last_updated_timestamp: Optional[Any] = None
    metrics: Optional[ModelMetrics] = None
    experiment_name: Optional[str] = None
    
class ModelTrainingHistory(BaseModel):
    run_id: str
    run_name: Optional[str] = None
    start_time: Optional[Any] = None
    end_time: Optional[Any] = None
    status: str
    metrics: Optional[ModelMetrics] = None
    model_version: Optional[str] = None

class ExperimentHistory(BaseModel):
    run_id: str
    run_name: Optional[str] = None
    experiment_id: str
    experiment_name: str
    start_time: Optional[Any] = None
    end_time: Optional[Any] = None
    status: str
    metrics: Optional[ModelMetrics] = None
    artifact_uri: Optional[str] = None
    tags: Optional[Dict[str, Any]] = None

class PredictionResponse(BaseModel):
    prediction: Any
    model_name: str
    model_version: str
    timestamp: datetime
    input_data: Dict[str, Any]

class PredictionHistory(BaseModel):
    predictions: List[PredictionResponse]
    total_count: int

class ParentExperimentHistory(ExperimentHistory):
    child_runs: List[ExperimentHistory] = []

class RegisterModelResponse(BaseModel):
    message: str
    model_name: str
    version: str
    run_id: str

class GroupedExperimentHistoryResponse(BaseModel):
    runs: List[ParentExperimentHistory]
    total_count: int
    next_page: Optional[str] = None
    previous_page: Optional[str] = None

class ModelSummary(BaseModel):
    model_name: str
    production_version: Optional[str] = None
    total_versions: int
    latest_model_training: Optional[ModelTrainingHistory] = None
    current_metrics: Optional[ModelMetrics] = None
    description: Optional[str] = None

class DashboardSummary(BaseModel):
    models: List[ModelSummary]
    total_models: int
    recent_predictions_count: int
    total_predictions_count: int
    recent_trainings_count: int