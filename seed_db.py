import asyncio
import sys
sys.path.append("services/api")

sys.path.append("services/api/src")

from api.database import engine, async_session_maker
from api.models import Model, ModelVersion

async def seed():
    async with async_session_maker() as db:
        # Create models
        for i, (name, arch) in enumerate([("TrOCR-Baseline", "vision"), ("ViLT-Baseline", "reasoning"), ("Wav2Vec2-Baseline", "audio")]):
            model = Model(name=name, architecture_type=arch)
            db.add(model)
            await db.flush()
            version = ModelVersion(model_id=model.id, version_tag="v1.0", metadata_={})
            db.add(version)
        await db.commit()
        print("Database seeded with 3 models and versions.")

if __name__ == "__main__":
    asyncio.run(seed())
