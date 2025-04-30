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

TIME_NOW = datetime.now().strftime('%Y%m%d_%H%M%S')

# --- Prefect Tasks ---
@task(name="Load Data")
def load_data_task(load_data_func: callable, parent_run_id: str, *args, **kwargs):
    """
    Prefect task to load data using a provided function.

    Args:
        load_data_func (callable): The function to execute for loading data.
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
    logger.info("--- Loading Data Task --- Received parent run_id: %s", parent_run_id)

    with mlflow.start_run(run_id=parent_run_id):
        with mlflow.start_run(run_name="Load Data Task", nested=True):
            logger.info("--- Starting Nested Run for Load Data Task ---")
            try:
                mlflow.log_param("task_name", "Load Data")
                mlflow.log_param("loader_function_task", getattr(load_data_func, '__name__', repr(load_data_func)))

                data = load_data_func(*args, **kwargs)

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

    # if theres a running run, end it
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
        mlflow.log_param("loader_function", getattr(load_data_func, '__name__', repr(load_data_func)))
        if load_args:
             mlflow.log_param("load_data_args", str(load_args))
        if load_kwargs:
             mlflow.log_param("load_data_kwargs", str(load_kwargs))

        logger.info("--- Starting Prefect ML Pipeline Flow --- parent run_id: %s", parent_run_id)

        # --- Execute tasks ---
        raw_data = load_data_task.submit(load_data_func, parent_run_id, *load_args, **load_kwargs)
        
        # Wait for raw_data before proceeding (if using .submit)
        #processed_data = process_data_task.submit(raw_data)

        #model = train_model_task.submit(processed_data)

        #validation_results = validate_model_task.submit(model, processed_data)

        # You can retrieve results if needed, e.g., final_metrics = validation_results.result()
        #logger.info(f"Validation results (task future): {validation_results}")


        logger.info("--- Prefect ML Pipeline Flow Complete ---")
