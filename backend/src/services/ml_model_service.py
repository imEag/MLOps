from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
import mlflow
import mlflow.pyfunc
import json
import pickle
import pandas as pd 
from datetime import datetime
from typing import List, Optional
import os
import requests

from ..flows.training_flow import ml_pipeline_flow
from ..training_script.training_script import load_data, process_data, train_model
from ..schemas.model_schemas import (
    ModelVersionResponse, ModelMetrics, ModelInfo, ExperimentHistory, ModelTrainingHistory,
    PredictionResponse, PredictionHistory, DashboardSummary, ModelSummary
)

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

def _log_prediction_to_mlflow(prediction, model_name, model_version, input_data):
    """Log a prediction to MLflow as a run."""
    try:
        experiment_id = _ensure_prediction_experiment()
        if experiment_id is None:
            print("Failed to create/get prediction experiment")
            return None
            
        with mlflow.start_run(experiment_id=experiment_id):
            # Log prediction metadata as parameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("model_version", model_version)
            mlflow.log_param("prediction_type", type(prediction).__name__)
            
            # Log prediction value as metric if it's numeric
            if isinstance(prediction, (int, float)):
                mlflow.log_metric("prediction_value", float(prediction))
            else:
                mlflow.log_param("prediction_value", str(prediction))
            
            # Log input data as parameters (flatten if necessary)
            for key, value in input_data.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"input_{key}", float(value))
                else:
                    mlflow.log_param(f"input_{key}", str(value))
            
            # Add tags for easier filtering
            mlflow.set_tag("mlflow.runName", f"Prediction_{model_name}")
            mlflow.set_tag("prediction_run", "true")
            mlflow.set_tag("model_name", model_name)
            
            run_info = mlflow.active_run().info
            return run_info.run_id
            
    except Exception as e:
        print(f"Error logging prediction to MLflow: {e}")
        return None

def _get_predictions_from_mlflow(limit: int = 50, offset: int = 0) -> List[PredictionResponse]:
    """Retrieve predictions from MLflow runs."""
    try:
        experiment_id = _ensure_prediction_experiment()
        if experiment_id is None:
            return []
        
        # Search for prediction runs
        runs_df = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string="tags.prediction_run = 'true'",
            order_by=["start_time DESC"],
            max_results=limit + offset
        )
        
        if runs_df.empty:
            return []
        
        # Apply pagination
        paginated_runs = runs_df.iloc[offset:offset + limit]
        
        predictions = []
        for _, run_row in paginated_runs.iterrows():
            try:
                # Extract prediction data from the run
                run_id = run_row['run_id']
                run = mlflow.get_run(run_id)
                
                model_name = run.data.params.get('model_name', 'unknown')
                model_version = run.data.params.get('model_version', 'unknown')
                
                # Get prediction value
                prediction_value = run.data.metrics.get('prediction_value')
                if prediction_value is None:
                    prediction_value = run.data.params.get('prediction_value', 'unknown')
                
                # Extract input data
                input_data = {}
                for key, value in run.data.params.items():
                    if key.startswith('input_'):
                        input_key = key.replace('input_', '')
                        input_data[input_key] = value
                
                # Add metric inputs
                for key, value in run.data.metrics.items():
                    if key.startswith('input_'):
                        input_key = key.replace('input_', '')
                        input_data[input_key] = value
                
                # Create prediction response
                prediction_response = PredictionResponse(
                    prediction=prediction_value,
                    model_name=model_name,
                    model_version=model_version,
                    timestamp=datetime.fromtimestamp(run.info.start_time / 1000),
                    input_data=input_data
                )
                
                predictions.append(prediction_response)
                
            except Exception as e:
                print(f"Error processing prediction run {run_id}: {e}")
                continue
                
        return predictions
        
    except Exception as e:
        print(f"Error retrieving predictions from MLflow: {e}")
        return []

def _get_total_predictions_count() -> int:
    """Get total count of predictions from MLflow."""
    try:
        experiment_id = _ensure_prediction_experiment()
        if experiment_id is None:
            return 0
        
        runs_df = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string="tags.prediction_run = 'true'",
            max_results=10000  # Set high limit to count all
        )
        
        return len(runs_df) if not runs_df.empty else 0
        
    except Exception as e:
        print(f"Error counting predictions from MLflow: {e}")
        return 0

def _get_recent_predictions_count(hours: int = 24) -> int:
    """Get count of recent predictions from MLflow."""
    try:
        experiment_id = _ensure_prediction_experiment()
        if experiment_id is None:
            return 0
        
        # Calculate timestamp threshold
        from datetime import timedelta
        threshold_time = datetime.now() - timedelta(hours=hours)
        threshold_timestamp = int(threshold_time.timestamp() * 1000)
        
        runs_df = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.prediction_run = 'true' and attribute.start_time >= {threshold_timestamp}",
            max_results=10000
        )
        
        return len(runs_df) if not runs_df.empty else 0
        
    except Exception as e:
        print(f"Error counting recent predictions from MLflow: {e}")
        return 0

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
        prediction = output[0]
    else:
        prediction = output
    
    # Store prediction in memory (replace with database in production)
    prediction_record = PredictionResponse(
        prediction=prediction,
        model_name=model_name,
        model_version=str(prod_model_version.version),
        timestamp=datetime.now(),
        input_data=data
    )
    _log_prediction_to_mlflow(prediction, model_name, str(prod_model_version.version), data)
    
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


def get_experiment_history(limit: int = 10, experiment_name: Optional[str] = None) -> List[ExperimentHistory]:
    """Get training history from MLflow experiment runs."""
    experiment_history = []
    
    try:
        experiment_ids = []
        experiment_map = {}

        if experiment_name:
            exp = mlflow.get_experiment_by_name(experiment_name)
            if not exp or exp.name == PREDICTION_EXPERIMENT_NAME:
                return []
            experiment_ids = [exp.experiment_id]
            experiment_map[exp.experiment_id] = exp.name
        else:
            all_experiments = mlflow.search_experiments()
            for exp in all_experiments:
                if exp.name != PREDICTION_EXPERIMENT_NAME:
                    experiment_ids.append(exp.experiment_id)
                    experiment_map[exp.experiment_id] = exp.name

        if not experiment_ids:
            return []

        runs = mlflow.search_runs(
            experiment_ids=experiment_ids,
            order_by=["start_time DESC"],
            max_results=limit,
            output_format="list"
        )
        
        print(f"runs: {runs}")
        for run in runs:
            try:
                metrics = get_model_metrics(run.info.run_id)
                
                experiment_history.append(ExperimentHistory(
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
                ))
            except Exception as e:
                print(f"Error processing run {run.info.run_id}: {e}")
                continue
    except Exception as e:
        print(f"Error getting training history: {e}")
    
    return experiment_history

def get_latest_experiment() -> Optional[ExperimentHistory]:
    """Get the most recent experiment run."""
    history = get_experiment_history(limit=1)
    return history[0] if history else None

def get_prediction_history(limit: int = 50, offset: int = 0) -> PredictionHistory:
    """Get prediction history from storage."""
    predictions = _get_predictions_from_mlflow(limit, offset)
    
    return PredictionHistory(
        predictions=predictions,
        total_count=_get_total_predictions_count()
    )

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
