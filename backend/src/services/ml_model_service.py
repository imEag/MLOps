from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
import mlflow.pyfunc
import json
import pickle
import pandas as pd 

from ..flows.training_flow import ml_pipeline_flow
from ..training_script.training_script import load_data, process_data, train_model
from ..schemas.model_schemas import ModelVersionResponse

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
        return ModelVersionResponse(
            model_name=prod_model_version.name,
            version=str(prod_model_version.version),
            run_id=prod_model_version.run_id,
            status=prod_model_version.status,
            current_stage=prod_model_version.current_stage,
            aliases=prod_model_version.aliases if prod_model_version.aliases else [],
            source=prod_model_version.source,
            creation_timestamp=prod_model_version.creation_timestamp,
            last_updated_timestamp=prod_model_version.last_updated_timestamp
        )
    except MlflowException as e:
        raise e 

def get_production_model_input_example(model_name: str):
  """Retrieves the input example for the production version of a model from MLflow."""
  client = MlflowClient()
  try:
    prod_model_version = client.get_model_version_by_alias(name=model_name, alias="production")
    
    prod_model_source = prod_model_version.source
    run_id = prod_model_source.split("/")[-3]
    run = mlflow.get_run(run_id)
    artifact_path = run.info.artifact_uri
    artifact = mlflow.artifacts.download_artifacts(f"{artifact_path}/model/serving_input_example.json")

    with open(artifact, "r") as f:
        input_example = json.load(f)
    
    return input_example
    
  except MlflowException as e:
    raise e
  except Exception as e:
    raise e
    
def predict(model_name: str, data: dict):
  """ Do a prediction using the production model """
  client = MlflowClient()
  try:
    prod_model_version = client.get_model_version_by_alias(name=model_name, alias="production")
    prod_model_source = prod_model_version.source
    run_id = prod_model_source.split("/")[-3]
    run = mlflow.get_run(run_id)
    artifact_path = run.info.artifact_uri
    artifact = mlflow.artifacts.download_artifacts(f"{artifact_path}/model/model.pkl")
    
    with open(artifact, "rb") as f:
        model = pickle.load(f)

    input_example_info = get_production_model_input_example(model_name)
    
    if not (
        input_example_info and 
        isinstance(input_example_info, dict) and
        "dataframe_split" in input_example_info and 
        isinstance(input_example_info["dataframe_split"], dict) and
        "columns" in input_example_info["dataframe_split"]
    ):
        raise ValueError("Could not retrieve valid input example columns for the model from MLflow.")
        
    columns_ordered = input_example_info["dataframe_split"]["columns"]
    
    try:
        df_to_predict = pd.DataFrame([data], columns=columns_ordered)
    except Exception as e:
        raise ValueError(f"Error creating DataFrame from input data: {str(e)}. Ensure data is a flat dictionary of features.")

    prediction_result = model.predict(df_to_predict)
    
    if hasattr(prediction_result, 'tolist'):
        output = prediction_result.tolist()
    else:
        output = prediction_result
        
    if isinstance(output, list) and len(output) == 1:
        return output[0]
       
    return output
  except MlflowException as e:
    raise e
  except Exception as e:
    raise e