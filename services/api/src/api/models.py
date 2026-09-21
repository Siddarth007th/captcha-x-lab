from datetime import datetime
from sqlalchemy import String, ForeignKey, DateTime, JSON, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class Project(Base):
    __tablename__ = "projects"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    experiments = relationship("Experiment", back_populates="project")

class Dataset(Base):
    __tablename__ = "datasets"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    versions = relationship("DatasetVersion", back_populates="dataset")

class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    version_tag: Mapped[str] = mapped_column(String(50), nullable=False)
    metadata_: Mapped[dict | None] = mapped_column(JSON) # e.g. sample count, preprocessing config
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    dataset = relationship("Dataset", back_populates="versions")
    
class Model(Base):
    __tablename__ = "models"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    architecture_type: Mapped[str] = mapped_column(String(50), nullable=False) # vision, reasoning, audio, master
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    versions = relationship("ModelVersion", back_populates="model")

class ModelVersion(Base):
    __tablename__ = "model_versions"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("models.id"), nullable=False)
    version_tag: Mapped[str] = mapped_column(String(50), nullable=False)
    metadata_: Mapped[dict | None] = mapped_column(JSON) # e.g. hyperparameters
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    model = relationship("Model", back_populates="versions")

class Experiment(Base):
    __tablename__ = "experiments"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    project = relationship("Project", back_populates="experiments")
    runs = relationship("ExperimentRun", back_populates="experiment")

class ExperimentRun(Base):
    __tablename__ = "experiment_runs"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"), nullable=False)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"), nullable=False)
    mixture_config: Mapped[dict | None] = mapped_column(JSON)
    metrics: Mapped[dict | None] = mapped_column(JSON)
    
    # Background job tracking
    status: Mapped[str] = mapped_column(String(50), default="queued") # queued, running, completed, failed, cancelled
    progress_samples: Mapped[int] = mapped_column(default=0)
    total_samples: Mapped[int] = mapped_column(default=0)
    current_stage: Mapped[str | None] = mapped_column(String(255))
    elapsed_time: Mapped[float] = mapped_column(default=0.0)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    experiment = relationship("Experiment", back_populates="runs")
