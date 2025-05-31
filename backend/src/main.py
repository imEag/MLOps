from fastapi import FastAPI
import mlflow
from dotenv import load_dotenv
import os

load_dotenv()

MLFLOW_PORT = os.getenv('MLFLOW_PORT', '5001')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', f"http://mlflow:{MLFLOW_PORT}")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_registry_uri(MLFLOW_TRACKING_URI)

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the ML API"}

# Placeholder for API routers
from .routers import models_router
app.include_router(models_router.router) 