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

# --- Prefect Tasks ---
@task(name="Load Data")
def load_data_task(load_data_func: SkipValidation[Callable], parent_run_id: str, *args, **kwargs):
    """
    Prefect task to load data using a provided function.

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
        with mlflow.start_run(run_name="Load Data Task", nested=True):
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
        with mlflow.start_run(run_name="Process Data Task", nested=True):
            try:
              mlflow.log_param("task_name", "Process Data")
              mlflow.log_param("processor_function_task", getattr(process_data_func, '__name__', repr(process_data_func)))

              processed_data = process_data_func(data, *args, **kwargs)
              
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

    Args:
        train_model_func (Callable): The function to use for training the model. It must accept a pandas DataFrame as input (first argument).
        data (pd.DataFrame): The data to train the model on.
        parent_run_id (str): The run_id of the parent MLflow run (from the flow).
        *args: Positional arguments for the train_model_func.
        **kwargs: Keyword arguments for the train_model_func.
    """
    
    logger = get_run_logger()
    logger.info("--- Training Model Task ---")
    
    with mlflow.start_run(run_id=parent_run_id):
        with mlflow.start_run(run_name="Train Model Task", nested=True):
            try:
              mlflow.log_param("task_name", "Train Model")
              mlflow.log_param("trainer_function_task", getattr(train_model_func, '__name__', repr(train_model_func)))

              trained_model = train_model_func(data, *args, **kwargs)
              print(f"Trained model: {trained_model}")
              
              mlflow.log_param("training_status", "success")
              
              input_example = data.head() if isinstance(data, pd.DataFrame) else data[:5]
              
              mlflow.sklearn.log_model(
                  trained_model, 
                  "model",
                  input_example=input_example
              )

              logger.info("--- Training Model Task Complete --- Nested Run Finished ---")
              return trained_model

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

        model = train_model_task.submit(train_model_func, processed_data, parent_run_id, *train_model_args, **train_model_kwargs)

        #validation_results = validate_model_task.submit(model, processed_data)

        # You can retrieve results if needed, e.g., final_metrics = validation_results.result()
        #logger.info(f"Validation results (task future): {validation_results}")


        logger.info("--- Prefect ML Pipeline Flow Complete ---")
