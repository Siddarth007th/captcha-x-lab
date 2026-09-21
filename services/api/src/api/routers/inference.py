import sys
import os

# Add project root to path so we can import from ml.master
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from ..schemas import PredictionResponse
from ml.master.router.router import MasterRouter

router = APIRouter(prefix="/inference", tags=["Inference"])

# Singleton instantiation of router (which holds ModelManager)
master_router = MasterRouter()

MAX_FILE_SIZE = 5 * 1024 * 1024 # 5 MB

@router.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    text_query: str = Form(None)
):
    """
    Accepts an image or audio file and an optional text query.
    Routes the request to the appropriate specialist model.
    """
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Max size is 5MB.")

    # Fallback to reading the stream to check size if file.size is somehow not set
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Max size is 5MB.")
    
    mime_type = file.content_type or ""

    if "image" not in mime_type and "audio" not in mime_type and "wav" not in file.filename.lower():
        # Fallback to extension heuristic if mime is weird
        if file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            mime_type = "image/png"
        elif file.filename.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
            mime_type = "audio/wav"
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {mime_type}")

    # Use the ML MasterRouter to infer
    result = master_router.infer(file_bytes=file_bytes, mime_type=mime_type, text_query=text_query)

    return result
