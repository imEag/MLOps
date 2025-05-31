from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import mlflow
from dotenv import load_dotenv
import os

load_dotenv()

MLFLOW_PORT = os.getenv('MLFLOW_PORT', '5001')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', f"http://mlflow:{MLFLOW_PORT}")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_registry_uri(MLFLOW_TRACKING_URI)

app = FastAPI(
    title="MLOps API",
    description="FastAPI application for MLOps project",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to the ML API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "mlflow_uri": MLFLOW_TRACKING_URI}

# Placeholder for API routers
from .routers import models_router
app.include_router(models_router.router) 