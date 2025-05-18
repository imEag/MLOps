from fastapi import APIRouter, HTTPException
from mlflow.exceptions import MlflowException

# Updated imports to use services and schemas
from ..services import ml_model_service
from ..schemas.model_schemas import ModelVersionResponse, TrainResponse, InputExampleResponse

router = APIRouter(
    prefix="/api/models",
    tags=["models"],
)

@router.post("/train", response_model=TrainResponse)
async def train_model_endpoint():
    try:
        flow_state = ml_model_service.run_ml_training_pipeline()
        return {"message": "Model training initiated successfully via service", "flow_state": str(flow_state)}
    except ImportError as e: # Catch potential import errors from the service layer if they occur at runtime
        raise HTTPException(status_code=500, detail=f"Error during model training (possibly import-related from service): {str(e)}")
    except Exception as e:
        # Catch other exceptions from the service layer
        raise HTTPException(status_code=500, detail=f"Error during model training: {str(e)}")

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

@router.get("/{model_name}/production-input-example", response_model=InputExampleResponse)
async def get_production_input_example_endpoint(model_name: str):
    try:
        example_data = ml_model_service.get_production_model_input_example(model_name)
        return {"example": example_data}
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
