from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class ProjectBase(BaseModel):
    name: str = Field(..., max_length=255)
    description: Optional[str] = None

class ProjectCreate(ProjectBase):
    pass

class ProjectResponse(ProjectBase):
    id: int
    created_at: datetime
    model_config = {"from_attributes": True}

class ExperimentBase(BaseModel):
    name: str = Field(..., max_length=255)
    status: Optional[str] = "pending"

class ExperimentCreate(ExperimentBase):
    pass

class ExperimentResponse(ExperimentBase):
    id: int
    project_id: int
    created_at: datetime
    model_config = {"from_attributes": True}

class ExperimentRunBase(BaseModel):
    model_version_id: int
    mixture_config: Optional[dict] = None
    metrics: Optional[dict] = None

class ExperimentRunCreate(ExperimentRunBase):
    pass

class ExperimentRunUpdate(BaseModel):
    metrics: Optional[dict] = None
    status: Optional[str] = None

class ExperimentRunResponse(ExperimentRunBase):
    id: int
    experiment_id: int
    created_at: datetime
    
    # Background job tracking
    status: str
    progress_samples: int
    total_samples: int
    current_stage: Optional[str] = None
    elapsed_time: float

    model_config = {"from_attributes": True}

class PredictionResponse(BaseModel):
    model_used: str
    challenge_type: str
    prediction: str
    confidence: float
    processing_latency: float
    success: bool
    error_information: Optional[str] = None
    model_version: str
