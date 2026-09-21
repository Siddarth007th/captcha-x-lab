from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List

from ..database import get_db
from ..models import Experiment
from ..schemas import ExperimentCreate, ExperimentResponse

router = APIRouter(
    prefix="/projects",
    tags=["Experiments"]
)

@router.post("/{project_id}/experiments", response_model=ExperimentResponse)
async def create_experiment(project_id: int, experiment: ExperimentCreate, db: AsyncSession = Depends(get_db)):
    db_exp = Experiment(project_id=project_id, name=experiment.name, status=experiment.status)
    db.add(db_exp)
    await db.commit()
    await db.refresh(db_exp)
    return db_exp

@router.get("/{project_id}/experiments", response_model=List[ExperimentResponse])
async def list_experiments(project_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Experiment).where(Experiment.project_id == project_id))
    exps = result.scalars().all()
    return exps
