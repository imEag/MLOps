import os
import pandas as pd
import numpy as np
# Removed matplotlib imports as they are not used in the refactored structure yet
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime
from dotenv import load_dotenv
from prefect import task, flow, get_run_logger # Import Prefect components
from pydantic import SkipValidation # Import SkipValidation
from typing import Callable, Tuple # Import Callable and Tuple
from pandas import Series # Import Series for type hinting
import mlflow.sklearn

# Load environment variables from .env file
load_dotenv()

# --- MLflow Configuration ---
MLFLOW_PORT = os.getenv('MLFLOW_PORT', '5000') # Default to 5000 if not set
MLFLOW_TRACKING_URI = f"http://localhost:{MLFLOW_PORT}"

TIME_NOW = datetime.now().strftime('%Y%m%d_%H%M%S')

# --- Helper for Logging ---
# Define no-op loggers for when functions are called outside the flow context
def _noop(*args, **kwargs): pass

noop_loggers = {
    "log_param": _noop,
    "log_metric": _noop,
    "log_artifact": _noop,
    "log_model": _noop,
}

# --- Prefect Tasks ---
@task(name="Load Data")
def load_data_task(load_data_func: SkipValidation[Callable], parent_run_id: str, *args, **kwargs):
    """
    Prefect task to load data using a provided function.
    It is assumed that the loaded DataFrame will have the target variable as its last column,
    as downstream tasks (e.g., model training for input example generation) may rely on this convention.

    Args:
        load_data_func (Callable): The function to execute for loading data.
        parent_run_id (str): The run_id of the parent MLflow run (from the flow).
        *args: Positional arguments for the load_data_func.
        **kwargs: Keyword arguments for the load_data_func.

    Returns:
        pd.DataFrame: The loaded data.

    Raises:
        TypeError: If the load_data_func does not return a pandas DataFrame.
        Exception: Propagates exceptions from the load_data_func.
    """
    logger = get_run_logger()
    logger.info("--- Loading Data Task ---")

    with mlflow.start_run(run_id=parent_run_id):
        with mlflow.start_run(run_name="Load Data Task", nested=True) as nested_run:
            try:
                # Define loggers specific to this nested run
                task_loggers = {
                    "log_param": mlflow.log_param,
                    "log_metric": mlflow.log_metric,
                    "log_artifact": mlflow.log_artifact,
                    # Add log_model if needed for this task, typically not
                }

                mlflow.log_param("task_name", "Load Data")
                mlflow.log_param("loader_function_task", getattr(load_data_func, '__name__', repr(load_data_func)))

                # Pass loggers to the user function
                data = load_data_func(*args, **kwargs, mlflow_loggers=task_loggers)

                if not isinstance(data, pd.DataFrame):
                    raise TypeError("The load_data_func must return a pandas DataFrame.")

                logger.info(f"Data loaded successfully: {data.shape[0]} samples, {data.shape[1]} features")

                mlflow.log_param("num_samples_raw", data.shape[0])
                mlflow.log_param("num_features_raw", data.shape[1])
                mlflow.log_param("data_loading_status", "success")

                logger.info("--- Data Loading Task Complete --- Nested Run Finished ---")
                return data

            except Exception as e:
                mlflow.log_param("data_loading_status", "failed")
                mlflow.log_param("data_loading_error", str(e))
                raise # Re-raise the exception for Prefect to handle

@task(name="Process Data")
def process_data_task(process_data_func: SkipValidation[Callable], data: pd.DataFrame, parent_run_id: str, *args, **kwargs):
    """
    Prefect task to process data using a provided function.

    Args:
        process_data_func (Callable): The function to execute for processing. It must accept a pandas DataFrame as input (first argument) and return a pandas DataFrame as output.
        data (pd.DataFrame): The data to process.
        parent_run_id (str): The run_id of the parent MLflow run (from the flow).
        *args: Positional arguments for the process_data_func.
        **kwargs: Keyword arguments for the process_data_func.

    Returns: pd.DataFrame: The processed data.
        
    Raises:
        TypeError: If the process_data_func does not return a pandas DataFrame.
        Exception: Propagates exceptions from the process_data_func.
    """
    logger = get_run_logger()
    logger.info("--- Processing Data Task ---")
    
    with mlflow.start_run(run_id=parent_run_id):
        with mlflow.start_run(run_name="Process Data Task", nested=True) as nested_run:
            try:
              # Define loggers specific to this nested run
              task_loggers = {
                  "log_param": mlflow.log_param,
                  "log_metric": mlflow.log_metric,
                  "log_artifact": mlflow.log_artifact,
                  # Add log_model if needed for this task
              }

              mlflow.log_param("task_name", "Process Data")
              mlflow.log_param("processor_function_task", getattr(process_data_func, '__name__', repr(process_data_func)))

              # Pass loggers to the user function
              processed_data = process_data_func(data, *args, **kwargs, mlflow_loggers=task_loggers)

              if not isinstance(processed_data, pd.DataFrame):
                raise TypeError("The process_data_func must return a pandas DataFrame.")

              logger.info(f"Data processed successfully.")
              mlflow.log_param("num_samples_processed", processed_data.shape[0])
              mlflow.log_param("num_features_processed", processed_data.shape[1])
              mlflow.log_param("data_processing_status", "success")

              logger.info("--- Data Processing Task Complete --- Nested Run Finished ---")
              return processed_data

            except Exception as e:
                mlflow.log_param("data_processing_status", "failed")
                mlflow.log_param("data_processing_error", str(e))
                raise # Re-raise the exception for Prefect to handle

@task(name="Train Model")
def train_model_task(train_model_func: SkipValidation[Callable], data: pd.DataFrame, parent_run_id: str, *args, **kwargs):
    """
    Prefect task to train a model using a provided function.
    The training function IS REQUIRED to return a tuple: (trained_model, metrics_dict).
    
    The input DataFrame 'data' is expected to have the target variable as its last column.
    This convention is used, for example, when generating the 'input_example' for MLflow logging,
    where all columns except the last one are considered features.

    The metrics_dict MUST contain the following keys with their corresponding float values:
    - 'accuracy'
    - 'macro_avg_precision'
    - 'macro_avg_recall'
    - 'macro_avg_f1_score'
    - 'weighted_avg_precision'
    - 'weighted_avg_recall'
    - 'weighted_avg_f1_score'

    Args:
        train_model_func (Callable): The function to use for training the model. 
                                     It must accept a pandas DataFrame as input (first argument)
                                     and return a tuple (trained_model, metrics_dict as specified above).
        data (pd.DataFrame): The data to train the model on.
        parent_run_id (str): The run_id of the parent MLflow run (from the flow).
        *args: Positional arguments for the train_model_func.
        **kwargs: Keyword arguments for the train_model_func.

    Returns:
        Tuple[Any, dict]: A tuple containing the trained model and the validated dictionary of metrics.
    
    Raises:
        ValueError: If metrics_dict is not returned, is not a dictionary, or is missing any of the required metric keys.
        TypeError: If train_model_func does not return a tuple of length 2.
    """
    
    logger = get_run_logger()
    logger.info("--- Training Model Task ---")
    
    with mlflow.start_run(run_id=parent_run_id):
        with mlflow.start_run(run_name="Train Model Task", nested=True) as nested_run:
            try:
              task_loggers = {
                  "log_param": mlflow.log_param,
                  "log_metric": mlflow.log_metric,
                  "log_artifact": mlflow.log_artifact,
                  "log_model": mlflow.sklearn.log_model, # Use the sklearn flavor
              }

              mlflow.log_param("task_name", "Train Model")
              mlflow.log_param("trainer_function_task", getattr(train_model_func, '__name__', repr(train_model_func)))

              # Pass loggers to the user function
              result = train_model_func(data, *args, **kwargs, mlflow_loggers=task_loggers)
              
              if not isinstance(result, tuple) or len(result) != 2:
                  raise TypeError(f"The train_model_func is expected to return a tuple (model, metrics_dict), but got {type(result)}")
              
              trained_model, metrics_dict = result

              print(f"Trained model: {trained_model}")
              print(f"Metrics received: {metrics_dict}")

              mlflow.log_param("training_status", "success")

              mandatory_metrics = [
                  'accuracy', 'macro_avg_precision', 'macro_avg_recall', 'macro_avg_f1_score',
                  'weighted_avg_precision', 'weighted_avg_recall', 'weighted_avg_f1_score'
              ]

              if not isinstance(metrics_dict, dict):
                  raise ValueError(f"Expected metrics_dict to be a dictionary, but got {type(metrics_dict)}")

              missing_metrics = [key for key in mandatory_metrics if key not in metrics_dict]
              if missing_metrics:
                  raise ValueError(f"Missing required metrics in metrics_dict: {', '.join(missing_metrics)}")
              
              for metric_name in mandatory_metrics: 
                  metric_value = metrics_dict[metric_name]
                  if metric_value is not None:
                      mlflow.log_metric(metric_name, float(metric_value)) 
                  else:
                      logger.warning(f"Metric '{metric_name}' has a None value. Logging as is or skipping if needed.")
              
              logger.info(f"Successfully logged validated metrics: {mandatory_metrics}")

              # Set the input example for the model
              input_example = data.iloc[:, :-1].head()

              if trained_model: 
                  task_loggers["log_model"](
                      trained_model,
                      "model",
                      input_example=input_example
                  )
              else:
                   logger.warning("Training function did not return a model to log.")

              logger.info("--- Training Model Task Complete --- Nested Run Finished ---")
              return trained_model, metrics_dict

            except Exception as e:
                mlflow.log_param("training_status", "failed")
                mlflow.log_param("training_error", str(e))
                raise # Re-raise the exception for Prefect to handle

# --- Prefect Flow ---
@flow(name="ML Training Pipeline")
def ml_pipeline_flow(
    load_data_func: SkipValidation[Callable],
    load_data_args: tuple,
    load_data_kwargs: dict,
    process_data_func: SkipValidation[Callable],
    process_data_args: tuple,
    process_data_kwargs: dict,
    train_model_func: SkipValidation[Callable],
    train_model_args: tuple,
    train_model_kwargs: dict
    ):
    """
    Prefect flow orchestrating the ML pipeline steps.

    Args:
        load_data_func (Callable): The function to use for loading data.
        load_data_args (tuple): Positional arguments for the load function.
        load_data_kwargs (dict): Keyword arguments for the load function.
        process_data_func (Callable): The function to use for processing the data.
        process_data_args (tuple): Positional arguments for the process function.
        process_data_kwargs (dict): Keyword arguments for the process function.
        train_model_func (Callable): The function to use for training the model.
        train_model_args (tuple): Positional arguments for the train function.
        train_model_kwargs (dict): Keyword arguments for the train function.
    """
    logger = get_run_logger()

    if mlflow.active_run():
        print("🚩 Ending active run")
        mlflow.end_run()
    else:
        print("🍏 No active run to end")
    
    # Set MLflow Tracking URI and Experiment Name
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI,)
    experiment_name = "ML Training Pipeline"
    mlflow.set_experiment(experiment_name)

    # Start MLflow run for the entire flow
    with mlflow.start_run(run_name=f"Prefect Flow - {TIME_NOW}") as parent_run: # No nested=True needed here
        parent_run_id = parent_run.info.run_id
        mlflow.log_param("mlflow_parent_run_id", parent_run_id) # Log parent ID for clarity

        mlflow.log_param("prefect_flow_name", f"ML Training Pipeline - {TIME_NOW}")
        mlflow.log_param("mlflow_tracking_uri", MLFLOW_TRACKING_URI)
        
        # Log the load data functions and their arguments
        mlflow.log_param("loader_function", getattr(load_data_func, '__name__', repr(load_data_func)))
        if load_data_args:
             mlflow.log_param("load_data_args", str(load_data_args))
        if load_data_kwargs:
             mlflow.log_param("load_data_kwargs", str(load_data_kwargs))
             
        # Log the process data functions and their arguments
        mlflow.log_param("processor_function", getattr(process_data_func, '__name__', repr(process_data_func)))
        if process_data_args:
             mlflow.log_param("process_data_args", str(process_data_args))
        if process_data_kwargs:
             mlflow.log_param("process_data_kwargs", str(process_data_kwargs))

        logger.info("--- Starting Prefect ML Pipeline Flow --- parent run_id: %s", parent_run_id)

        # --- Execute tasks ---
        raw_data = load_data_task.submit(load_data_func, parent_run_id, *load_data_args, **load_data_kwargs)
        
        processed_data = process_data_task.submit(process_data_func, raw_data, parent_run_id, *process_data_args, **process_data_kwargs)

        model_and_metrics_future = train_model_task.submit(train_model_func, processed_data, parent_run_id, *train_model_args, **train_model_kwargs)

        #validation_results = validate_model_task.submit(model, processed_data)

        # You can retrieve results if needed:
        # model, metrics = model_and_metrics_future.result()
        # logger.info(f"Model: {model}")
        # logger.info(f"Metrics from flow: {metrics}")
        #logger.info(f"Validation results (task future): {validation_results}")


        logger.info("--- Prefect ML Pipeline Flow Complete ---")
