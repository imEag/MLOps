import os
import json
import pickle
from typing import Callable, Tuple, Dict, Any, Optional, List

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from prefect import task, flow, get_run_logger


# --- MLflow Configuration ---
MLFLOW_PORT = os.getenv('MLFLOW_PORT', '5001')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', f"http://mlflow:{MLFLOW_PORT}")

# Configure MLflow to ensure consistent server usage
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_registry_uri(MLFLOW_TRACKING_URI)


# --- Constants ---
PREDICTION_EXPERIMENT_NAME = "Model_Predictions"


# --- Helpers (mirroring service behavior) ---
def _ensure_prediction_experiment() -> Optional[str]:
    """Ensure the prediction experiment exists in MLflow and return its ID."""
    try:
        experiment = mlflow.get_experiment_by_name(PREDICTION_EXPERIMENT_NAME)
        if experiment is None:
            return mlflow.create_experiment(PREDICTION_EXPERIMENT_NAME)
        return experiment.experiment_id
    except Exception as e:
        print(f"Error creating/getting prediction experiment: {e}")
        return None


def _log_prediction_to_mlflow(
    prediction: Any,
    model_name: str,
    model_version: str,
    input_data: List[Dict[str, Any]],
) -> Optional[str]:
    """Log each prediction and its corresponding input data into a nested MLflow run."""
    try:
        experiment_id = _ensure_prediction_experiment()
        if experiment_id is None:
            print("Failed to create/get prediction experiment")
            return None

        # Start a parent run for the entire batch of predictions
        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=f"Prediction Batch - {model_name} v{model_version}",
        ) as parent_run:
            parent_run_id = parent_run.info.run_id
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("model_version", model_version)
            mlflow.log_param("num_records", len(input_data))
            mlflow.set_tag("prediction_batch", "true")

            # Pair each input row with a prediction
            if hasattr(prediction, "tolist"):
                preds_list = prediction.tolist()
            elif isinstance(prediction, (list, tuple)):
                preds_list = list(prediction)
            else:
                # If prediction is a scalar, broadcast it to all input rows
                preds_list = [prediction] * len(input_data)

            # Adjust lengths if mismatched, logging a warning
            if len(preds_list) != len(input_data):
                print(
                    f"Warning: Mismatch between number of predictions ({len(preds_list)}) and inputs ({len(input_data)}). Truncating to match inputs."
                )
                preds_list = preds_list[: len(input_data)]

            # Iterate over each prediction and its corresponding input data
            for i, (pred_val, input_row) in enumerate(zip(preds_list, input_data)):
                # Create a nested run for each individual prediction.
                # Using nested=True automatically links it to the parent run.
                with mlflow.start_run(
                    experiment_id=experiment_id, run_name=f"Prediction Row {i}", nested=True
                ) as child_run:
                    mlflow.set_tag("prediction_row", i)

                    # Log the input features as parameters
                    for key, value in input_row.items():
                        mlflow.log_param(f"input_{key}", str(value))

                    # Log the prediction as a metric
                    if isinstance(pred_val, (int, float)):
                        mlflow.log_metric("prediction_value", float(pred_val))
                    else:
                        # Attempt to convert to float, otherwise log as param
                        try:
                            mlflow.log_metric("prediction_value", float(pred_val))
                        except (ValueError, TypeError):
                            mlflow.log_param("prediction_value", str(pred_val))

            return parent_run_id

    except Exception as e:
        print(f"Error logging prediction to MLflow: {e}")
        return None


# --- Prefect Tasks ---
@task(name="Preprocess Input")
def preprocess_input_task(preprocess_func: Callable, input_path: str, *args, **kwargs) -> str:
    """Call user-provided preprocess function that returns a path to a persisted pandas DataFrame."""
    logger = get_run_logger()
    logger.info("Running preprocess function to produce a DataFrame file path")

    # Ensure the function is called with input_path named argument
    result_path = preprocess_func(input_path=input_path, *args, **kwargs)

    if not isinstance(result_path, str):
        raise TypeError("The preprocess function must return a file path (str) to a pandas DataFrame.")
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Preprocess function returned non-existing path: {result_path}")

    logger.info(f"Preprocess produced file: {result_path}")
    return result_path


@task(name="Load Production Model")
def load_model_task(model_name: str) -> Dict[str, Any]:
    """Load the production model and its expected input columns from MLflow artifacts."""
    logger = get_run_logger()
    logger.info(f"Loading production model for: {model_name}")

    client = MlflowClient()
    prod_model_version = client.get_model_version_by_alias(name=model_name, alias="production")

    prod_model_source = prod_model_version.source
    # Extract run_id from the source path (same approach as in the service)
    run_id = prod_model_source.split("/")[-3]
    run = mlflow.get_run(run_id)
    artifact_path = run.info.artifact_uri

    # Download model and input example
    model_artifact = mlflow.artifacts.download_artifacts(f"{artifact_path}/model/model.pkl")
    with open(model_artifact, "rb") as f:
        model = pickle.load(f)

    input_example_art = mlflow.artifacts.download_artifacts(
        f"{artifact_path}/model/serving_input_example.json"
    )
    with open(input_example_art, "r") as f:
        input_example = json.load(f)

    # Validate and extract expected columns
    if not (
        input_example and isinstance(input_example, dict) and
        "dataframe_split" in input_example and isinstance(input_example["dataframe_split"], dict) and
        "columns" in input_example["dataframe_split"]
    ):
        raise ValueError("Could not retrieve valid input example columns from MLflow.")

    columns_ordered = input_example["dataframe_split"]["columns"]

    logger.info(
        f"Loaded production model v{prod_model_version.version} and {len(columns_ordered)} expected columns"
    )

    return {
        "model": model,
        "model_version": str(prod_model_version.version),
        "run_id": run_id,
        "columns": columns_ordered,
    }


@task(name="Load DataFrame")
def load_dataframe_task(dataframe_path: str) -> pd.DataFrame:
    """Load a pandas DataFrame from the given path supporting common formats (csv, parquet, pkl)."""
    ext = os.path.splitext(dataframe_path)[1].lower()
    if ext in [".csv"]:
        df = pd.read_csv(dataframe_path)
    elif ext in [".parquet"]:
        df = pd.read_parquet(dataframe_path)
    elif ext in [".pkl", ".pickle"]:
        df = pd.read_pickle(dataframe_path)
    elif ext in [".feather"]:
        # pandas.read_feather requires pyarrow at runtime
        df = pd.read_feather(dataframe_path)
    else:
        raise ValueError(f"Unsupported DataFrame file extension: {ext}")
    return df.copy()


@task(name="Predict")
def predict_task(model_info: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    """Align columns and run prediction. Returns prediction and first-row input mapping used for logging."""
    model = model_info["model"]
    columns_ordered = model_info["columns"]

    # Ensure all expected columns exist; create missing with zeros
    for col in columns_ordered:
        if col not in df.columns:
            df[col] = 0

    # Keep only expected columns in order
    df_to_predict = df[columns_ordered]

    # Execute prediction using numpy array values
    prediction_result = model.predict(df_to_predict.values)
    if hasattr(prediction_result, 'tolist'):
        output = prediction_result.tolist()
    else:
        output = prediction_result

    # Normalize
    if isinstance(output, list) and len(output) == 1:
        prediction = output[0]
    else:
        prediction = output

    # Build input_data list of dicts, one per row used for prediction
    if len(df_to_predict) > 0:
      records = df_to_predict.to_dict(orient="records")
    else:
      # create an empty pd.Series with the expected columns (preserve previous behavior)
      empty_row = pd.Series(index=columns_ordered, dtype=float)
      records = [empty_row.to_dict()]

    input_data = [
      {k: (v.item() if hasattr(v, "item") else v) for k, v in rec.items()}
      for rec in records
    ]

    return {"prediction": prediction, "input_data": input_data}


@task(name="Log Prediction")
def log_prediction_task(prediction_payload: Dict[str, Any], model_name: str, model_info: Dict[str, Any]) -> Optional[str]:
    """Log the prediction to MLflow's Model_Predictions experiment and return the run_id."""
    prediction = prediction_payload["prediction"]
    input_data = prediction_payload["input_data"]
    model_version = model_info.get("model_version", "unknown")
    return _log_prediction_to_mlflow(prediction, model_name, model_version, input_data)


# --- Prefect Flow ---
@flow(name="ML Prediction Pipeline")
def ml_prediction_flow(
    preprocess_func: Callable,
    input_path: str,
    model_name: str,
    preprocess_args: Tuple = (),
    preprocess_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Orchestrates a prediction pipeline:
    - Calls a user-provided preprocess function that must accept `input_path=` and return a path to a DataFrame file.
    - Loads the current production model and expected input columns from MLflow artifacts.
    - Loads the DataFrame, aligns columns, performs prediction.
    - Logs the prediction into the existing `Model_Predictions` experiment (same schema as current service).

    Returns a dict with: { 'prediction': Any, 'run_id': Optional[str], 'model_version': str }
    """
    logger = get_run_logger()
    preprocess_kwargs = preprocess_kwargs or {}

    # 1) Produce the DataFrame file via user function
    df_path_future = preprocess_input_task.submit(preprocess_func, input_path, *preprocess_args, **preprocess_kwargs)

    # 2) Load production model + columns
    model_info_future = load_model_task.submit(model_name)

    # 3) Load DataFrame
    df_future = load_dataframe_task.submit(df_path_future)

    # 4) Predict
    prediction_payload_future = predict_task.submit(model_info_future, df_future)

    # 5) Log prediction (standalone run in Model_Predictions)
    run_id_future = log_prediction_task.submit(prediction_payload_future, model_name, model_info_future)

    # Return a small payload
    prediction_payload = prediction_payload_future.result()
    run_id = run_id_future.result()
    model_version = model_info_future.result()["model_version"]

    logger.info("Prediction complete. run_id=%s, model_version=%s", run_id, model_version)
    return {
        "prediction": prediction_payload["prediction"],
        "run_id": run_id,
        "model_version": model_version,
    }
