from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
import mlflow
import mlflow.pyfunc
import json
import pickle
import pandas as pd 
from datetime import datetime
from typing import List, Optional, Callable, Tuple
import os
import requests
import tempfile

from ..flows.training_flow import ml_pipeline_flow
from ..flows import ml_prediction_flow
from ..custom_scripts.training_script import load_data, process_data, train_model
from ..schemas.model_schemas import (
    ModelVersionResponse, ModelMetrics, ModelInfo, ExperimentHistory, ModelTrainingHistory,
    PredictionResponse, PredictionHistory, DashboardSummary, ModelSummary, ParentExperimentHistory,
    IndividualPrediction
)
from ..custom_scripts.data_preprocessing_script import process_data as process_data_for_prediction

# Prediction experiment name for MLflow
PREDICTION_EXPERIMENT_NAME = "Model_Predictions"

# Prefect configuration
PREFECT_API_URL = os.getenv('PREFECT_API_URL', 'http://prefect:4200/api')
print(f"PREFECT_API_URL: {PREFECT_API_URL}")

def _ensure_prediction_experiment():
    """Ensure the prediction experiment exists in MLflow."""
    try:
        experiment = mlflow.get_experiment_by_name(PREDICTION_EXPERIMENT_NAME)
        if experiment is None:
            experiment_id = mlflow.create_experiment(PREDICTION_EXPERIMENT_NAME)
            return experiment_id
        return experiment.experiment_id
    except Exception as e:
        print(f"Error creating/getting prediction experiment: {e}")
        return None

def _get_total_predictions_count() -> int:
    """Get total count of parent prediction runs from MLflow."""
    try:
        experiment_id = _ensure_prediction_experiment()
        if experiment_id is None:
            return 0
        
        runs_df = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string="tags.prediction_batch = 'true'",
            max_results=50000 
        )
        
        if runs_df.empty:
            return 0

        try:
            # Filter for parent runs in pandas by checking for null in the parentRunId tag
            parent_runs_df = runs_df[runs_df['tags.mlflow.parentRunId'].isnull()]
            return len(parent_runs_df)
        except KeyError:
            # If 'tags.mlflow.parentRunId' column does not exist, it means no run is a child run.
            # Therefore, all runs are parent runs.
            return len(runs_df)
        
    except Exception as e:
        print(f"Error counting predictions from MLflow: {e}")
        return 0

def _get_recent_predictions_count(hours: int = 24) -> int:
    """Get count of recent parent prediction runs from MLflow."""
    try:
        experiment_id = _ensure_prediction_experiment()
        if experiment_id is None:
            return 0
        
        from datetime import timedelta
        threshold_time = datetime.now() - timedelta(hours=hours)
        threshold_timestamp = int(threshold_time.timestamp() * 1000)
        
        runs_df = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.prediction_batch = 'true' and attribute.start_time >= {threshold_timestamp}",
            max_results=50000
        )
        
        if runs_df.empty:
            return 0
            
        try:
            # Filter for parent runs in pandas by checking for null in the parentRunId tag
            parent_runs_df = runs_df[runs_df['tags.mlflow.parentRunId'].isnull()]
            return len(parent_runs_df)
        except KeyError:
            # If 'tags.mlflow.parentRunId' column does not exist, all runs are parents.
            return len(runs_df)
        
    except Exception as e:
        print(f"Error counting recent predictions from MLflow: {e}")
        return 0

def get_prediction_history(limit: int = 50, offset: int = 0) -> PredictionHistory:
    """Retrieve prediction history from MLflow with nested runs."""
    try:
        experiment_id = _ensure_prediction_experiment()
        if experiment_id is None:
            return PredictionHistory(predictions=[], total_count=0)

        # Search for prediction runs (batches)
        parent_runs_df = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string="tags.prediction_batch = 'true'",
            order_by=["start_time DESC"]
        )

        if parent_runs_df.empty:
            return PredictionHistory(predictions=[], total_count=0)

        try:
            # Filter for parent runs in pandas by checking for null in the parentRunId tag
            parent_runs_df = parent_runs_df[parent_runs_df['tags.mlflow.parentRunId'].isnull()]
        except KeyError:
            # If 'tags.mlflow.parentRunId' column does not exist, all runs are parents.
            pass

        total_count = len(parent_runs_df)
        
        # Apply pagination to parent runs
        paginated_parent_runs = parent_runs_df.iloc[offset:offset + limit]

        prediction_batches = []
        for _, parent_run_row in paginated_parent_runs.iterrows():
            parent_run_id = parent_run_row['run_id']
            parent_run = mlflow.get_run(parent_run_id)
            
            # Search for child runs of the current parent
            child_runs_df = mlflow.search_runs(
                experiment_ids=[experiment_id],
                filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
                order_by=["start_time ASC"]
            )
            
            individual_predictions = []
            if not child_runs_df.empty:
                for _, child_run_row in child_runs_df.iterrows():
                    child_run_id = child_run_row['run_id']
                    child_run = mlflow.get_run(child_run_id)
                    
                    inputs = {k: v for k, v in child_run.data.params.items()}
                    prediction_value = child_run.data.metrics.get('prediction_value', 'N/A')

                    individual_predictions.append(IndividualPrediction(
                        run_id=child_run_id,
                        start_time=datetime.fromtimestamp(child_run.info.start_time / 1000),
                        status=child_run.info.status,
                        inputs=inputs,
                        prediction=prediction_value
                    ))

            prediction_batches.append(PredictionResponse(
                run_id=parent_run_id,
                model_name=parent_run.data.params.get('model_name', 'unknown'),
                model_version=parent_run.data.params.get('model_version', 'unknown'),
                start_time=datetime.fromtimestamp(parent_run.info.start_time / 1000),
                status=parent_run.info.status,
                num_records=len(individual_predictions),
                predictions=individual_predictions
            ))

        return PredictionHistory(predictions=prediction_batches, total_count=total_count)

    except Exception as e:
        print(f"Error retrieving prediction history from MLflow: {e}")
        return PredictionHistory(predictions=[], total_count=0)

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

def _preprocess_to_df_path(input_path: str) -> str:
    """Default preprocess adapter: uses existing process_data_for_prediction, materializes a DataFrame to a temp file, and returns its path.

    This keeps the new prediction flow contract (function returns a path to a DataFrame) without changing existing preprocessing code.
    """
    data = process_data_for_prediction(input_path)
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.DataFrame(data).copy()

    # Persist to a temp pickle to preserve dtypes
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
    tmp.close()
    df.to_pickle(tmp.name)
    return tmp.name

def run_ml_prediction_pipeline(model_name: str, input_path: str, preprocess_func: Optional[Callable] = None, preprocess_args: Tuple = (), preprocess_kwargs: Optional[dict] = None):
    """Runs the ML prediction flow using Prefect.

    - preprocess_func must accept input_path= and return a path to a DataFrame file. If None, uses the default adapter that wraps process_data_for_prediction.
    - Returns the flow's small payload with prediction, run_id, and model_version.
    """
    if preprocess_func is None:
        preprocess_func = _preprocess_to_df_path
    preprocess_kwargs = preprocess_kwargs or {}

    return ml_prediction_flow(
        preprocess_func=preprocess_func,
        input_path=input_path,
        model_name=model_name,
        preprocess_args=preprocess_args,
        preprocess_kwargs=preprocess_kwargs,
    )

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

def predict(model_name: str, data_path: str):
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

    data = process_data_for_prediction(data_path)

    try:
        # Build a DataFrame from the processed input (handle dict, list of dicts, DataFrame, array)
        # Ensure we have a pandas DataFrame to work with
        if isinstance(data, pd.DataFrame):
          df = data.copy()
        else:
          df = pd.DataFrame(data).copy()
        
        # Ensure all expected columns exist — create missing ones filled with zeros
        for col in columns_ordered:
            if col not in df.columns:
              df[col] = 0

        # Reorder/keep only the expected columns
        df_to_predict = df[columns_ordered]
        print(f"DataFrame to predict: {df_to_predict}")
    except Exception as e:
        raise ValueError(f"Error creating DataFrame from input data: {str(e)}. Ensure data is a flat dictionary of features.")

    print(f"DataFrame to predict values: {df_to_predict.values}")
    prediction_result = model.predict(df_to_predict.values)
    
    if hasattr(prediction_result, 'tolist'):
        output = prediction_result.tolist()
    else:
        output = prediction_result
        
    if isinstance(output, list) and len(output) == 1:
        prediction = output[0]
    else:
        prediction = output
    
    # This function is now handled by the prediction flow, so this call is removed.
    # _log_prediction_to_mlflow(prediction, model_name, str(prod_model_version.version), data)

    return prediction
    
  except MlflowException as e:
    raise e
  except Exception as e:
    raise e

def get_model_metrics(run_id: str) -> Optional[ModelMetrics]:
    """Extract metrics from an MLflow run."""
    try:
        run = mlflow.get_run(run_id)
        metrics = run.data.metrics
        
        return ModelMetrics(
            accuracy=metrics.get('accuracy'),
            macro_avg_precision=metrics.get('macro_avg_precision'),
            macro_avg_recall=metrics.get('macro_avg_recall'),
            macro_avg_f1_score=metrics.get('macro_avg_f1_score'),
            weighted_avg_precision=metrics.get('weighted_avg_precision'),
            weighted_avg_recall=metrics.get('weighted_avg_recall'),
            weighted_avg_f1_score=metrics.get('weighted_avg_f1_score'),
            cv_mean_accuracy=metrics.get('cv_mean_accuracy'),
            cv_std_accuracy=metrics.get('cv_std_accuracy')
        )
    except Exception as e:
        print(f"Error getting metrics for run {run_id}: {e}")
        return None

def get_current_model_info(model_name: str) -> Optional[ModelInfo]:
    """Get detailed information about the current production model."""
    try:
        prod_model = get_production_model_from_mlflow(model_name)
        metrics = get_model_metrics(prod_model.run_id)
        
        run = mlflow.get_run(prod_model.run_id)
        experiment = mlflow.get_experiment(run.info.experiment_id)
        
        return ModelInfo(
            model_name=prod_model.model_name,
            version=prod_model.version,
            run_id=prod_model.run_id,
            status=prod_model.status,
            creation_timestamp=prod_model.creation_timestamp,
            last_updated_timestamp=prod_model.last_updated_timestamp,
            metrics=metrics,
            experiment_name=experiment.name if experiment else None
        )
    except Exception as e:
        print(f"Error getting current model info: {e}")
        return None

def get_model_training_history(model_name: str, limit: int = 10) -> List[ModelTrainingHistory]:
    """Get training history for a model."""
    client = MlflowClient()
    model_training_history = []
    
    try:
        model_versions = client.search_model_versions(f"name='{model_name}'")
        
        model_versions.sort(key=lambda x: int(x.version), reverse=True)
        
        for version in model_versions[:limit]:
            try:
                run = mlflow.get_run(version.run_id)
                metrics = get_model_metrics(version.run_id)
                
                model_training_history.append(ModelTrainingHistory(
                    run_id=version.run_id,
                    run_name=run.data.tags.get('mlflow.runName'),
                    start_time=run.info.start_time,
                    end_time=run.info.end_time,
                    status=run.info.status,
                    metrics=metrics,
                    model_version=version.version
                ))
            except Exception as e:
                print(f"Error processing version {version.version}: {e}")
                continue
    except Exception as e:
        print(f"Error getting model training history: {e}")
    
    return model_training_history

def get_model_latest_training(model_name: str) -> Optional[ModelTrainingHistory]:
    """Get the most recent training run for a model."""
    history = get_model_training_history(model_name, limit=1)
    return history[0] if history else None

def get_experiment_history(limit: int = 10, offset: int = 0, experiment_name: Optional[str] = None) -> dict:
    """Get and group training history from MLflow experiment runs."""
    try:
        experiment_ids = []
        experiment_map = {}

        if experiment_name:
            exp = mlflow.get_experiment_by_name(experiment_name)
            if not exp or exp.name == PREDICTION_EXPERIMENT_NAME:
                return {"runs": [], "total_count": 0}
            experiment_ids = [exp.experiment_id]
            experiment_map[exp.experiment_id] = exp.name
        else:
            all_experiments = mlflow.search_experiments()
            for exp in all_experiments:
                if exp.name != PREDICTION_EXPERIMENT_NAME:
                    experiment_ids.append(exp.experiment_id)
                    experiment_map[exp.experiment_id] = exp.name

        if not experiment_ids:
            return {"runs": [], "total_count": 0}

        # Fetch all runs to build the hierarchy, this could be memory intensive
        all_runs = mlflow.search_runs(
            experiment_ids=experiment_ids,
            output_format="list",
            max_results=50000  # A high number to get all runs
        )

        parent_runs = {}
        child_runs = []

        for run in all_runs:
            if 'mlflow.parentRunId' not in run.data.tags:
                metrics = get_model_metrics(run.info.run_id)
                parent_run = ParentExperimentHistory(
                    run_id=run.info.run_id,
                    run_name=run.data.tags.get('mlflow.runName', ''),
                    experiment_id=run.info.experiment_id,
                    experiment_name=experiment_map.get(run.info.experiment_id, 'Unknown'),
                    start_time=run.info.start_time,
                    end_time=run.info.end_time,
                    status=run.info.status,
                    metrics=metrics,
                    artifact_uri=run.info.artifact_uri,
                    tags=run.data.tags,
                    child_runs=[]
                )
                parent_runs[run.info.run_id] = parent_run
            else:
                child_runs.append(run)

        for run in child_runs:
            parent_id = run.data.tags.get('mlflow.parentRunId')
            if parent_id in parent_runs:
                metrics = get_model_metrics(run.info.run_id)
                child_run_obj = ExperimentHistory(
                    run_id=run.info.run_id,
                    run_name=run.data.tags.get('mlflow.runName', ''),
                    experiment_id=run.info.experiment_id,
                    experiment_name=experiment_map.get(run.info.experiment_id, 'Unknown'),
                    start_time=run.info.start_time,
                    end_time=run.info.end_time,
                    status=run.info.status,
                    metrics=metrics,
                    artifact_uri=run.info.artifact_uri,
                    tags=run.data.tags
                )
                parent_runs[parent_id].child_runs.append(child_run_obj)
        
        # Sort parent runs by start time
        sorted_parent_runs = sorted(parent_runs.values(), key=lambda x: x.start_time, reverse=True)
        
        # Paginate
        total_count = len(sorted_parent_runs)
        paginated_runs = sorted_parent_runs[offset : offset + limit]

        return {"runs": paginated_runs, "total_count": total_count}

    except Exception as e:
        print(f"Error getting experiment history: {e}")
        return {"runs": [], "total_count": 0}

def get_latest_experiment() -> Optional[ExperimentHistory]:
    """Get the most recent experiment run."""
    history_data = get_experiment_history(limit=1)
    return history_data["runs"][0] if history_data["runs"] else None

def get_dashboard_summary() -> DashboardSummary:
    """Get summary information for the dashboard with all registered models."""
    from datetime import datetime
    import requests
    import os
    client = MlflowClient()
    
    try:
        try:
            mlflow_host = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5001')
            api_endpoint = f"{mlflow_host}/api/2.0/mlflow/registered-models/search"
            
            response = requests.get(api_endpoint, params={'max_results': 100})
            
            if response.status_code == 200:
                rest_data = response.json()
                registered_models_rest = rest_data.get('registered_models', [])
                
                registered_models = []
                for model_data in registered_models_rest:
                    try:
                        model_obj = client.get_registered_model(model_data['name'])
                        registered_models.append(model_obj)
                    except Exception as e:
                        print(f"Error converting model {model_data.get('name')}: {e}")
                        
            else:
                registered_models = []
                
        except Exception as e:
            registered_models = client.search_registered_models()
        
        models_summary = []
        recent_trainings_count = 0
        
        for model in registered_models:
            try:
                production_version = None
                current_metrics = None
                latest_model_training = None
                
                try:
                    prod_model = client.get_model_version_by_alias(name=model.name, alias="production")
                    production_version = prod_model.version
                    current_metrics = get_model_metrics(prod_model.run_id)
                except:
                    pass
                
                latest_model_training = get_model_latest_training(model.name)
                if latest_model_training:
                    if latest_model_training.start_time:
                        try:
                            start_time_seconds = latest_model_training.start_time
                            if start_time_seconds > 1e12:
                                start_time_seconds = start_time_seconds / 1000
                            start_time = datetime.fromtimestamp(start_time_seconds)
                            if (datetime.now() - start_time).days <= 7:
                                recent_trainings_count += 1
                        except:
                            pass
                
                versions = client.search_model_versions(f"name='{model.name}'")
                
                models_summary.append(ModelSummary(
                    model_name=model.name,
                    production_version=production_version,
                    total_versions=len(versions),
                    latest_model_training=latest_model_training,
                    current_metrics=current_metrics,
                    description=model.description
                ))
                
            except Exception as e:
                print(f"Error processing model {model.name}: {e}")
                continue
        
        return DashboardSummary(
            models=models_summary,
            total_models=len(models_summary),
            recent_predictions_count=_get_recent_predictions_count(24),
            total_predictions_count=_get_total_predictions_count(),
            recent_trainings_count=recent_trainings_count
        )
        
    except Exception as e:
        print(f"Error getting dashboard summary: {e}")
        return DashboardSummary(
            models=[],
            total_models=0,
            recent_predictions_count=0,
            total_predictions_count=0,
            recent_trainings_count=0
        )

def get_prefect_flow_runs(limit: int = 10) -> List[dict]:
    """Get recent Prefect flow runs using the correct Prefect 2.x API."""
    try:
        # Use POST with filter for Prefect 2.x API
        response = requests.post(
            f"{PREFECT_API_URL}/flow_runs/filter",
            headers={"Content-Type": "application/json"},
            json={
                "limit": limit,
                "sort": "START_TIME_DESC"  # Use correct enum value from Prefect API
            }
        )
        if response.status_code == 200:
            flow_runs = response.json()
            # Convert to list format if needed
            return flow_runs if isinstance(flow_runs, list) else []
        else:
            print(f"Error fetching flow runs: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"Error connecting to Prefect API: {e}")
        return []

def get_training_flow_status(flow_run_id: str = None) -> dict:
    """Get the status of training flows using the correct Prefect 2.x API."""
    try:
        if flow_run_id:
            # Get specific flow run
            response = requests.get(f"{PREFECT_API_URL}/flow_runs/{flow_run_id}")
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error fetching specific flow run: {response.status_code} - {response.text}")
                return {"error": f"Flow run {flow_run_id} not found"}
        else:
            # Get latest training flows
            flows = get_prefect_flow_runs(limit=5)
            print(f"flows: {flows}")
            training_flows = [
                flow for flow in flows 
                if flow.get('flow_name', '').lower().find('training') != -1 or 
                   flow.get('flow_name', '').lower().find('ml') != -1 or
                   'train' in flow.get('name', '').lower() or
                   'ml' in flow.get('name', '').lower()
            ]
            print(f"training_flows: {training_flows}")
            return {"recent_training_flows": training_flows}
    except Exception as e:
        print(f"Error getting flow status: {e}")
        return {"error": str(e)}

def get_available_models() -> List[dict]:
    """Get list of all available registered models using the same approach as dashboard."""
    client = MlflowClient()
    try:
        # Use the same hybrid approach as dashboard to get registered models
        try:
            mlflow_host = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5001')
            api_endpoint = f"{mlflow_host}/api/2.0/mlflow/registered-models/search"
            
            response = requests.get(api_endpoint, params={'max_results': 100})
            
            if response.status_code == 200:
                rest_data = response.json()
                registered_models_rest = rest_data.get('registered_models', [])
                
                registered_models = []
                for model_data in registered_models_rest:
                    try:
                        model_obj = client.get_registered_model(model_data['name'])
                        registered_models.append(model_obj)
                    except Exception as e:
                        print(f"Error converting model {model_data.get('name')}: {e}")
                        
            else:
                registered_models = []
                
        except Exception as e:
            print(f"Error with REST API, falling back to client: {e}")
            registered_models = client.search_registered_models()
        
        models_info = []
        
        for model in registered_models:
            try:
                prod_version = None
                try:
                    prod_model = client.get_model_version_by_alias(name=model.name, alias="production")
                    prod_version = {
                        "version": prod_model.version,
                        "run_id": prod_model.run_id,
                        "creation_timestamp": prod_model.creation_timestamp
                    }
                except:
                    pass
                
                versions = client.search_model_versions(f"name='{model.name}'")
                
                models_info.append({
                    "name": model.name,
                    "description": model.description,
                    "creation_timestamp": model.creation_timestamp,
                    "last_updated_timestamp": model.last_updated_timestamp,
                    "production_version": prod_version,
                    "total_versions": len(versions),
                    "tags": model.tags if hasattr(model, 'tags') else {}
                })
            except Exception as e:
                print(f"Error processing model {model.name}: {e}")
                continue
                
        return models_info
    except Exception as e:
        print(f"Error getting available models: {e}")
        return []

def get_experiments_summary() -> List[dict]:
    """Get summary of all MLflow experiments."""
    try:
        experiments = mlflow.search_experiments()
        experiments_info = []
        
        for experiment in experiments:
            # Get runs count
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            
            experiments_info.append({
                "experiment_id": experiment.experiment_id,
                "name": experiment.name,
                "artifact_location": experiment.artifact_location,
                "lifecycle_stage": experiment.lifecycle_stage,
                "creation_time": experiment.creation_time,
                "last_update_time": experiment.last_update_time,
                "runs_count": len(runs),
                "tags": experiment.tags if hasattr(experiment, 'tags') else {}
            })
            
        return experiments_info
    except Exception as e:
        print(f"Error getting experiments summary: {e}")
        return []

def get_model_versions_info(model_name: str) -> List[dict]:
    """Get detailed information about all versions of a model."""
    client = MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        versions_info = []
        
        for version in versions:
            try:
                # Get metrics from the run
                metrics = get_model_metrics(version.run_id)
                
                # Get run info
                run = mlflow.get_run(version.run_id)
                
                versions_info.append({
                    "version": version.version,
                    "run_id": version.run_id,
                    "status": version.status,
                    "current_stage": version.current_stage,
                    "aliases": version.aliases if version.aliases else [],
                    "creation_timestamp": version.creation_timestamp,
                    "last_updated_timestamp": version.last_updated_timestamp,
                    "source": version.source,
                    "description": version.description,
                    "metrics": metrics.__dict__ if metrics else None,
                    "run_name": run.data.tags.get('mlflow.runName', ''),
                    "run_status": run.info.status,
                    "run_start_time": run.info.start_time,
                    "run_end_time": run.info.end_time,
                    "tags": version.tags if hasattr(version, 'tags') else {}
                })
            except Exception as e:
                print(f"Error processing version {version.version}: {e}")
                continue
                
        return sorted(versions_info, key=lambda x: int(x['version']), reverse=True)
    except Exception as e:
        print(f"Error getting model versions info: {e}")
        return []

def register_model_from_run(run_id: str, model_name: str):
    """Register a model from a specific run if a model artifact exists."""
    client = MlflowClient()
    try:
        # Check if a model was logged in the run
        artifacts = client.list_artifacts(run_id, path="model")
        if not any(artifact.path.endswith("MLmodel") for artifact in artifacts):
            raise ValueError(f"No model found in run {run_id}. Cannot register.")

        # Register the model
        model_uri = f"runs:/{run_id}/model"
        model_version = mlflow.register_model(model_uri, model_name)
        
        return {
            "message": "Model registered successfully",
            "model_name": model_version.name,
            "version": model_version.version,
            "run_id": run_id
        }
    except MlflowException as e:
        # Handle cases where the model name already exists but you might want to create a new version
        if "RESOURCE_ALREADY_EXISTS" in str(e):
            model_uri = f"runs:/{run_id}/model"
            model_version = mlflow.register_model(model_uri, model_name)
            return {
                "message": f"New version for model '{model_name}' registered.",
                "model_name": model_version.name,
                "version": model_version.version,
                "run_id": run_id,
            }
        raise e
    except Exception as e:
        print(f"Error registering model from run {run_id}: {e}")
        raise e

def get_prediction_stats():
    """Get prediction statistics from MLflow."""
    try:
        total_predictions = _get_total_predictions_count()
        last_24h = _get_recent_predictions_count(24)
        last_7d = _get_recent_predictions_count(24 * 7)
        last_30d = _get_recent_predictions_count(24 * 30)
        
        return {
            "total_predictions": total_predictions,
            "last_24h": last_24h,
            "last_7d": last_7d,
            "last_30d": last_30d
        }
    except Exception as e:
        print(f"Error getting prediction stats: {e}")
        return {
            "total_predictions": 0,
            "last_24h": 0,
            "last_7d": 0,
            "last_30d": 0
        }
