from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="CAPTCHA-X Lab API")

class HealthResponse(BaseModel):
    status: str
    message: str

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok", message="CAPTCHA-X Lab API is running")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
