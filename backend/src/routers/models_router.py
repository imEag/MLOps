from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from mlflow.exceptions import MlflowException
import pandas as pd
import os

from ..services import ml_model_service
from ..schemas.model_schemas import (
    ModelVersionResponse, TrainResponse,
    ModelInfo, ExperimentHistory, 
    PredictionHistory, DashboardSummary, 
    ModelTrainingHistory, RegisterModelResponse
)

router = APIRouter(
    prefix="/api/models",
    tags=["models"],
)

@router.post("/train", response_model=TrainResponse)
async def train_model_endpoint(background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(ml_model_service.run_ml_training_pipeline)
        return {"message": "Model training initiated in the background.", "flow_state": "STARTED"}
    except ImportError as e: # Catch potential import errors from the service layer if they occur at runtime
        raise HTTPException(status_code=500, detail=f"Error during model training (possibly import-related from service): {str(e)}")
    except Exception as e:
        # Catch other exceptions from the service layer
        raise HTTPException(status_code=500, detail=f"Error initiating model training: {str(e)}")

@router.get("/{model_name}/production-version", response_model=ModelVersionResponse)
async def get_production_model_version_endpoint(model_name: str):
    try:
        prod_model = ml_model_service.get_production_model_from_mlflow(model_name)
        return prod_model # FastAPI will serialize this Pydantic model or dict matching the schema
    except MlflowException as e:
        # Specific handling for MLflow exceptions, re-raised from the service
        if "RESOURCE_DOES_NOT_EXIST" in str(e) or \
           f"No Model Version found for model name '{model_name}' and alias 'production'" in str(e) or \
           f"Registered model '{model_name}' not found" in str(e) or \
           f"Model version with name '{model_name}' and alias 'production' not found" in str(e):
             raise HTTPException(status_code=404, detail=f"No version with alias 'production' found for model '{model_name}'. MLflow error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error communicating with MLflow: {str(e)}")
    except Exception as e:
        # Catch other exceptions from the service layer
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@router.get("/{model_name}/production-input-example")
async def get_production_input_example_endpoint(model_name: str):
    try:
        prod_model_input_example = ml_model_service.get_production_model_input_example(model_name)
        return prod_model_input_example
        
    except MlflowException as e:
        # Catching exceptions from the service layer
        error_message = str(e)
        if f"Failed to load production model '{model_name}'" in error_message or \
           f"No input example found logged with the production version of model '{model_name}'" in error_message:
            # These are considered "Not Found" scenarios for the specific resource (model or its example)
            raise HTTPException(status_code=404, detail=error_message)
        # Other MLflow exceptions might be server-side issues
        raise HTTPException(status_code=500, detail=f"MLflow related error: {error_message}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while fetching input example: {str(e)}")
      
@router.post("/{model_name}/predict")
async def predict(model_name: str, data: dict):
  try:
    prediction = ml_model_service.predict(model_name, data)
    return prediction
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"An unexpected error occurred while predicting: {str(e)}")

@router.get("/dashboard", response_model=DashboardSummary)
async def get_dashboard_summary():
    """Get dashboard summary with all registered models and recent activity."""
    try:
        summary = ml_model_service.get_dashboard_summary()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting dashboard summary: {str(e)}")

@router.get("/{model_name}/info", response_model=ModelInfo)
async def get_current_model_info(model_name: str):
    """Get detailed information about the current production model."""
    try:
        model_info = ml_model_service.get_current_model_info(model_name)
        if not model_info:
            raise HTTPException(status_code=404, detail=f"No production model found with name '{model_name}'")
        return model_info
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@router.get("/{model_name}/training-history")
async def get_model_training_history(
    model_name: str, 
    limit: int = Query(default=10, ge=1, le=100)
):
    """Get training history for a model."""
    try:
        history = ml_model_service.get_model_training_history(model_name, limit)
        return {"training_history": history, "total_count": len(history)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model training history: {str(e)}")

@router.get("/{model_name}/latest-training", response_model=ModelTrainingHistory)
async def get_latest_model_training(model_name: str):
    """Get the most recent training run for a model."""
    try:
        latest_model_training = ml_model_service.get_model_latest_training(model_name)
        if not latest_model_training:
            raise HTTPException(status_code=404, detail=f"No training history found for model '{model_name}'")
        return latest_model_training
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting latest model training: {str(e)}")

@router.get("/experiments/history")
async def get_experiments_training_history(
    limit: int = Query(default=10, ge=1, le=100)
):
    """Get training history for all experiments."""
    try:
        history = ml_model_service.get_experiment_history(limit)
        return {"training_history": history, "total_count": len(history)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting training history: {str(e)}")

@router.get("/experiments/latest", response_model=ExperimentHistory)
async def get_latest_experiment():
    """Get the most recent training run (experiment)."""
    try:
        latest_experiment = ml_model_service.get_latest_experiment()
        if not latest_experiment:
            raise HTTPException(status_code=404, detail=f"No experiment found'")
        return latest_experiment
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting latest experiment: {str(e)}")

@router.get("/predictions/history", response_model=PredictionHistory)
async def get_prediction_history(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0)
):
    """Get prediction history with pagination."""
    try:
        history = ml_model_service.get_prediction_history(limit, offset)
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting prediction history: {str(e)}")

@router.get("/predictions/stats")
async def get_prediction_stats():
    """Get prediction statistics from MLflow."""
    try:
        stats = ml_model_service.get_prediction_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting prediction stats: {str(e)}")

@router.get("/flows/recent")
async def get_recent_flow_runs(limit: int = Query(default=10, ge=1, le=50)):
    """Get recent Prefect flow runs."""
    try:
        flows = ml_model_service.get_prefect_flow_runs(limit)
        return {"flows": flows, "count": len(flows)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting recent flows: {str(e)}")

@router.get("/health/services")
async def check_services_health():
    """Check health of all connected services (MLflow, Prefect)."""
    try:
        health_status = {"services": {}}
        
        # Check MLflow
        try:
            import mlflow
            mlflow.search_experiments(max_results=1)
            health_status["services"]["mlflow"] = {"status": "healthy", "url": os.getenv('MLFLOW_TRACKING_URI')}
        except Exception as e:
            health_status["services"]["mlflow"] = {"status": "unhealthy", "error": str(e)}
        
        # Check Prefect
        try:
            flows = ml_model_service.get_prefect_flow_runs(limit=1)
            health_status["services"]["prefect"] = {"status": "healthy", "url": ml_model_service.PREFECT_API_URL}
        except Exception as e:
            health_status["services"]["prefect"] = {"status": "unhealthy", "error": str(e)}
        
        # Overall status
        all_healthy = all(
            service["status"] == "healthy" 
            for service in health_status["services"].values()
        )
        health_status["overall"] = "healthy" if all_healthy else "degraded"
        
        return health_status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking services health: {str(e)}")

@router.get("/available")
async def get_available_models():
    """Get list of all available registered models."""
    try:
        models = ml_model_service.get_available_models()
        return {"models": models, "count": len(models)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting available models: {str(e)}")

@router.get("/experiments")
async def get_experiments_summary():
    """Get summary of all MLflow experiments."""
    try:
        experiments = ml_model_service.get_experiments_summary()
        return {"experiments": experiments, "count": len(experiments)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting experiments: {str(e)}")

@router.get("/{model_name}/versions")
async def get_model_versions(model_name: str):
    """Get detailed information about all versions of a model."""
    try:
        versions = ml_model_service.get_model_versions_info(model_name)
        if not versions:
            raise HTTPException(status_code=404, detail=f"No versions found for model '{model_name}'")
        return {"model_name": model_name, "versions": versions, "count": len(versions)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model versions: {str(e)}")

@router.post("/{model_name}/promote/{version}")
async def promote_model_to_production(model_name: str, version: str):
    """Promote a specific model version to production."""
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        # Use the correct method name for MLflow 2.x
        client.set_registered_model_alias(
            name=model_name,
            alias="production",
            version=version
        )
        
        return {
            "message": f"Model {model_name} version {version} promoted to production",
            "model_name": model_name,
            "version": version,
            "alias": "production"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error promoting model: {str(e)}")

@router.post("/register", response_model=RegisterModelResponse)
async def register_model_from_run_endpoint(run_id: str, model_name: str):
    """Register a model from a specific run, if a model artifact exists."""
    try:
        result = ml_model_service.register_model_from_run(run_id, model_name)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except MlflowException as e:
        raise HTTPException(status_code=500, detail=f"MLflow error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
