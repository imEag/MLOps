from pydantic import BaseModel
from typing import Optional, List, Any

class ModelVersionResponse(BaseModel):
    model_name: str
    version: str
    run_id: str
    status: str
    current_stage: Optional[str] = None # Kept for now, may be None
    aliases: Optional[List[str]] = []
    source: Optional[str] = None
    creation_timestamp: Optional[Any] = None # Using Any for flexibility from MLflow client
    last_updated_timestamp: Optional[Any] = None # Using Any for flexibility from MLflow client

    class Config:
        orm_mode = True # or from_attributes = True for Pydantic v2

class TrainResponse(BaseModel):
    message: str
    flow_state: Optional[str] = None

class InputExampleResponse(BaseModel):
    example: Any 