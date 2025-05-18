from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
import mlflow.pyfunc # Added for loading model and metadata

# Assuming these imports are still valid relative to the src directory
# If your flows/training_script are directly under src, these relative imports are correct from services/
from ..flows.training_flow import ml_pipeline_flow
from ..training_script.training_script import load_data, process_data, train_model
from ..schemas.model_schemas import ModelVersionResponse # For type hinting if needed, or direct return

def run_ml_training_pipeline():
    """Runs the ML training pipeline."""
    print("Starting the ML training pipeline flow via service...")
    flow_state = ml_pipeline_flow(
        load_data_func=load_data,
        load_data_args=(),
        load_data_kwargs={},
        process_data_func=process_data,
        process_data_args=(),
        process_data_kwargs={},
        train_model_func=train_model,
        train_model_args=(),
        train_model_kwargs={}
    )
    print("ML training pipeline flow finished in service.")
    return flow_state

def get_production_model_from_mlflow(model_name: str) -> ModelVersionResponse:
    """Retrieves the production version of a model from MLflow."""
    client = MlflowClient()
    try:
        prod_model_version = client.get_model_version_by_alias(name=model_name, alias="production")
        # Manually create a dictionary that matches ModelVersionResponse fields
        # Pydantic model will validate this structure when returned by the endpoint
        return ModelVersionResponse(
            model_name=prod_model_version.name,
            version=str(prod_model_version.version),  # Explicitly cast to string
            run_id=prod_model_version.run_id,
            status=prod_model_version.status,
            current_stage=prod_model_version.current_stage,
            aliases=prod_model_version.aliases if prod_model_version.aliases else [],
            source=prod_model_version.source,
            creation_timestamp=prod_model_version.creation_timestamp,
            last_updated_timestamp=prod_model_version.last_updated_timestamp
        )
    except MlflowException as e:
        # Let the router handle the HTTPException wrapping
        raise e 

def get_production_model_input_example(model_name: str):
  """Retrieves the input example for the production version of a model from MLflow."""
  client = MlflowClient()
  try:
    prod_model_version = client.get_model_version_by_alias(name=model_name, alias="production")
    print("prod_model_version: ", prod_model_version)
    # TODO: Return the input example
    return prod_model_version
    
  except MlflowException as e:
    raise e
  except Exception as e:
    raise e
    