from prefect import flow, task
from pathlib import Path
import mlflow
from train import load_data, train_model

@task
def prepare_data(data_path):
    """Task to prepare the data."""
    return load_data(data_path)

@task
def train_and_log_model(X_train, y_train, X_test, y_test):
    """Task to train the model and log with MLflow."""
    return train_model(X_train, y_train, X_test, y_test)

@flow(name="Training Pipeline")
def training_pipeline(data_path: str = "data/raw/dataset.csv"):
    """Main training pipeline flow."""
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(data_path)
    
    # Train model
    model, accuracy = train_and_log_model(X_train, y_train, X_test, y_test)
    
    return model, accuracy

if __name__ == "__main__":
    training_pipeline() 