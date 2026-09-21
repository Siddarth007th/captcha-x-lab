from celery import shared_task
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from .models import ExperimentRun
from datetime import datetime
from .database import DATABASE_URL
import time

# Create a synchronous engine for Celery workers
sync_db_url = DATABASE_URL.replace("+asyncpg", "")
engine = create_engine(sync_db_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def _update_run_status(run_id: int, status: str, progress: int = None, total: int = None, stage: str = None, elapsed: float = None, metrics: dict = None):
    with SessionLocal() as session:
        run = session.query(ExperimentRun).filter(ExperimentRun.id == run_id).first()
        if run:
            run.status = status
            if progress is not None:
                run.progress_samples = progress
            if total is not None:
                run.total_samples = total
            if stage is not None:
                run.current_stage = stage
            if elapsed is not None:
                run.elapsed_time = elapsed
            if metrics is not None:
                if run.metrics is None:
                    run.metrics = {}
                run.metrics.update(metrics)
                # Ensure JSON mutation is detected
                run.metrics = dict(run.metrics)
            session.commit()
        return run

@shared_task(bind=True)
def evaluate_model_task(self, run_id: int, model_architecture: str, config: dict):
    """
    Background job to train/evaluate a model based on architecture.
    """
    # 1. Update status to running
    _update_run_status(
        run_id, 
        status="running", 
        stage="Initializing model and dataset", 
        progress=0, 
        elapsed=0.0
    )
    
    start_time = time.time()
    
    try:
        if model_architecture == "vision":
            from ml.vision.training.train import run_training
        elif model_architecture == "reasoning":
            from ml.reasoning.training.train import run_training
        elif model_architecture == "audio":
            from ml.audio.training.train import run_training
        else:
            raise NotImplementedError(f"Architecture {model_architecture} not implemented yet.")
            
        def progress_callback(samples_processed, stage_name):
            # Check if task was cancelled
            with SessionLocal() as session:
                current_run = session.query(ExperimentRun).filter(ExperimentRun.id == run_id).first()
                if current_run and current_run.status == "cancelled":
                    return False # Cancel signal
            
            elapsed = time.time() - start_time
            _update_run_status(
                run_id,
                status="running",
                stage=stage_name,
                progress=samples_processed,
                elapsed=elapsed
            )
            return True
            
        result = run_training(config, progress_callback)
        
        if result.get("status") == "cancelled":
            return {"status": "cancelled"}
            
        elapsed = time.time() - start_time
        metrics = result.get("metrics", {})
        
        _update_run_status(
            run_id, 
            status="completed", 
            stage="Evaluation complete", 
            elapsed=elapsed,
            metrics=metrics
        )
        return {"status": "completed", "metrics": metrics}
            
    except Exception as e:
        # Handle failures robustly
        _update_run_status(run_id, status="failed", stage=f"Error: {str(e)}")
        raise e
