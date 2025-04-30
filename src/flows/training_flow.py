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

# Load environment variables from .env file
load_dotenv()

# --- MLflow Configuration ---
MLFLOW_PORT = os.getenv('MLFLOW_PORT', '5000') # Default to 5000 if not set
MLFLOW_TRACKING_URI = f"http://localhost:{MLFLOW_PORT}"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

TIME_NOW = datetime.now().strftime('%Y%m%d_%H%M%S')

# --- Prefect Tasks ---
@task(name="Load Data")
def load_data_task(load_data_func: callable, *args, **kwargs):
    """
    Prefect task to load data using a provided function and log details to MLflow.

    Args:
        load_data_func (callable): The function to execute for loading data.
        *args: Positional arguments for the load_data_func.
        **kwargs: Keyword arguments for the load_data_func.

    Returns:
        pd.DataFrame: The loaded data.

    Raises:
        TypeError: If the load_data_func does not return a pandas DataFrame.
        Exception: Propagates exceptions from the load_data_func.
    """
    logger = get_run_logger()
    logger.info("--- Loading Data ---")

    try:
        data = load_data_func(*args, **kwargs)

        if not isinstance(data, pd.DataFrame):
            raise TypeError("The load_data_func must return a pandas DataFrame.")

        logger.info(f"Data loaded successfully: {data.shape[0]} samples, {data.shape[1]} features")

        mlflow.log_param("num_samples_raw", data.shape[0])
        mlflow.log_param("num_features_raw", data.shape[1])

        logger.info("--- Data Loading Complete ---")
        return data

    except Exception as e:
        logger.error(f"Error during data loading: {e}")
        mlflow.log_param("data_loading_status", "failed")
        mlflow.log_param("data_loading_error", str(e))
        # Prefect automatically tracks task failure states
        raise # Re-raise the exception for Prefect to handle


# --- Prefect Flow ---
@flow(name="ML Training Pipeline")
def ml_pipeline_flow(load_data_func: callable, load_args: tuple = None, load_kwargs: dict = None):
    """
    Prefect flow orchestrating the ML pipeline steps.

    Args:
        load_data_func (callable): The function to use for loading data.
        load_args (tuple, optional): Positional arguments for the load function. Defaults to None.
        load_kwargs (dict, optional): Keyword arguments for the load function. Defaults to None.
    """
    logger = get_run_logger()
    if load_args is None:
        load_args = ()
    if load_kwargs is None:
        load_kwargs = {}

    # Start MLflow run for the entire flow
    # Prefect automatically integrates with MLflow if configured,
    # but explicit start_run gives more control over run naming.
    with mlflow.start_run(run_name=f"PrefectFlow_{TIME_NOW}"):
        # Log parameters specific to the flow run
        mlflow.log_param("prefect_flow_name", f"ML Training Pipeline - {TIME_NOW}")
        mlflow.log_param("mlflow_tracking_uri", MLFLOW_TRACKING_URI)
        mlflow.log_param("loader_function", getattr(load_data_func, '__name__', repr(load_data_func)))
        if load_args:
             mlflow.log_param("load_data_args", str(load_args))
        if load_kwargs:
             mlflow.log_param("load_data_kwargs", str(load_kwargs))

        logger.info("--- Starting Prefect ML Pipeline Flow ---")

        # --- Execute tasks ---
        raw_data = load_data_task.submit(load_data_func, *load_args, **load_kwargs) # Use .submit for async execution if desired

        # Wait for raw_data before proceeding (if using .submit)
        #processed_data = process_data_task.submit(raw_data)

        #model = train_model_task.submit(processed_data)

        #validation_results = validate_model_task.submit(model, processed_data)

        # You can retrieve results if needed, e.g., final_metrics = validation_results.result()
        #logger.info(f"Validation results (task future): {validation_results}")


        logger.info("--- Prefect ML Pipeline Flow Complete ---")
        # MLflow run ends automatically when 'with' block exits
