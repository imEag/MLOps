import sys
import os

# Add the src directory to the Python path
# This allows importing modules from src like src.flows.training_flow
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.flows.training_flow import ml_pipeline_flow
from src.training_script.training_script import load_data, process_data, train_model

if __name__ == "__main__":
    print("Starting the ML training pipeline flow...")
    # Run the Prefect flow, passing the load_data function
    # from training_script.py as the data loading mechanism.
    flow_state = ml_pipeline_flow(
      load_data_func=load_data, 
      load_data_args=(), 
      load_data_kwargs={},
      process_data_func=process_data, 
      process_data_args=(), 
      process_data_kwargs={},
      train_model_func=train_model,
      train_model_args=(),
      train_model_kwargs={})
    print("ML training pipeline flow finished.")
    print(f"Final flow state: {flow_state}") # Optional: Print the final state for debugging 