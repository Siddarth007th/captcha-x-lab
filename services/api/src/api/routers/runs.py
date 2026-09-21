from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from typing import List

from ..database import get_db
from ..models import ExperimentRun, Experiment
from ..schemas import ExperimentRunCreate, ExperimentRunUpdate, ExperimentRunResponse

router = APIRouter(
    prefix="/experiments",
    tags=["ExperimentRuns"]
)

@router.get("/export/all")
async def export_all_data(db: AsyncSession = Depends(get_db)):
    # Fetch all experiments and their runs
    result = await db.execute(select(Experiment))
    experiments = result.scalars().all()
    
    run_result = await db.execute(select(ExperimentRun))
    runs = run_result.scalars().all()
    
    export_data = {
        "experiments": [{"id": e.id, "project_id": e.project_id, "name": e.name, "status": e.status} for e in experiments],
        "runs": [{
            "id": r.id, 
            "experiment_id": r.experiment_id, 
            "mixture_config": r.mixture_config,
            "metrics": r.metrics,
            "status": r.status,
            "created_at": r.created_at.isoformat()
        } for r in runs]
    }
    return export_data


@router.get("/{experiment_id}/runs", response_model=List[ExperimentRunResponse])
async def list_runs(experiment_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ExperimentRun).where(ExperimentRun.experiment_id == experiment_id))
    runs = result.scalars().all()
    return runs

from ..celery_app import celery_app
from ..tasks import evaluate_model_task

@router.get("/runs/{run_id}", response_model=ExperimentRunResponse)
async def get_run(run_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ExperimentRun).where(ExperimentRun.id == run_id))
    db_run = result.scalar_one_or_none()
    if not db_run:
        raise HTTPException(status_code=404, detail="Run not found")
    return db_run

@router.post("/{experiment_id}/runs", response_model=ExperimentRunResponse)
async def create_run(experiment_id: int, run: ExperimentRunCreate, db: AsyncSession = Depends(get_db)):
    db_run = ExperimentRun(
        experiment_id=experiment_id,
        model_version_id=run.model_version_id,
        mixture_config=run.mixture_config,
        metrics=run.metrics,
        status="queued"
    )
    db.add(db_run)
    await db.commit()
    await db.refresh(db_run)
    
    # Enqueue Celery Task
    config = run.mixture_config or {}
    model_architecture = config.get("model_architecture", "vision")
    task = evaluate_model_task.delay(db_run.id, model_architecture, config)
    
    return db_run

@router.post("/runs/{run_id}/cancel")
async def cancel_run(run_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ExperimentRun).where(ExperimentRun.id == run_id))
    db_run = result.scalar_one_or_none()
    if not db_run:
        raise HTTPException(status_code=404, detail="Run not found")
        
    if db_run.status in ["completed", "failed", "cancelled"]:
        raise HTTPException(status_code=400, detail="Run is already finished")
        
    # We ideally store celery task_id on the DB model for strict revocation, 
    # but we can also use Celery app.control API if we know the task ID. 
    # For now, we update DB to cancelled, and the worker loop will check `is_aborted()` or DB state if we implement DB checking.
    db_run.status = "cancelled"
    db_run.current_stage = "Cancelled by user"
    await db.commit()
    
    return {"status": "cancelled", "run_id": run_id}

@router.patch("/runs/{run_id}", response_model=ExperimentRunResponse)
async def update_run(run_id: int, run: ExperimentRunUpdate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ExperimentRun).where(ExperimentRun.id == run_id))
    db_run = result.scalar_one_or_none()
    if not db_run:
        raise HTTPException(status_code=404, detail="Run not found")
        
    if run.metrics is not None:
        if db_run.metrics is None:
            db_run.metrics = {}
        # Merge metrics
        db_run.metrics.update(run.metrics)
        # SQLAlchemy needs to know JSON was mutated, but reassignment works best
        db_run.metrics = dict(db_run.metrics)
        
    await db.commit()
    await db.refresh(db_run)
    return db_run
