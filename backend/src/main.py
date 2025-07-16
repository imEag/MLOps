from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import mlflow
from dotenv import load_dotenv
import os
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

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

# Add logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    logger.info(f"Request: {request.method} {request.url}")
    print(f"=== REQUEST: {request.method} {request.url} ===")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} in {process_time:.4f}s")
    print(f"=== RESPONSE: {response.status_code} in {process_time:.4f}s ===")
    
    return response

@app.get("/")
async def root():
    logger.info("Root endpoint called")
    print("=== ROOT ENDPOINT CALLED ===")
    return {"message": "Welcome to the ML API"}

@app.get("/health")
async def health_check():
    logger.info("Health check endpoint called")
    print("=== HEALTH CHECK ENDPOINT CALLED ===")
    return {"status": "healthy", "mlflow_uri": MLFLOW_TRACKING_URI}

# Placeholder for API routers
from .routers import models_router
app.include_router(models_router.router) 