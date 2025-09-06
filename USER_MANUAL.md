# User Manual

## 1. Introduction

This manual provides a detailed guide for the installation, configuration, and use of the MLOps platform developed in this project. The objective is to enable researchers, students, and developers to deploy and operate the system in a local environment, taking advantage of its capabilities to automate the lifecycle of Machine Learning models applied to the analysis of neuroscience data.

The platform has been designed to be portable and self-contained, minimizing configuration complexity. Through the following chapters, it will be explained step-by-step how to launch the complete infrastructure, interact with the user interface to manage models, and execute predictions.

It is important to note that, while this project is delivered with a pre-configured pipeline for EEG signal analysis, the underlying architecture is model and domain-agnostic. It has been intentionally designed as a generic and reusable "meta-pipeline". This means that developers can adapt the platform to orchestrate their own models, using different data loading, preprocessing, and training functions without needing to modify the main orchestration logic. How to perform these customizations will be detailed later in this manual.

## 2. System Requirements

Before proceeding with the installation, it is necessary to ensure that the host system meets the following minimum requirements:

*   **Software:**
    *   **Docker Engine:** Version 20.10.0 or higher.
    *   **Docker Compose:** Version 1.29.0 or higher (or the version integrated into Docker Desktop).
    *   **Git:** To clone the project repository.

*   **Hardware (Recommended):**
    *   **Operating System:** Linux, macOS, or Windows with WSL2 (Windows Subsystem for Linux).
    *   **RAM:** 8 GB or more, especially for handling the processing of EEG datasets and the simultaneous execution of all services.
    *   **Disk Space:** 40 GB of free space to store Docker images, model artifacts, metadata, and test datasets.

## 3. Installation and Initial Configuration

The platform uses Docker to encapsulate each of its components (backend, frontend, database, etc.) in containers, and Docker Compose to orchestrate their deployment and communication. This approach ensures a consistent and reproducible execution environment.

### 3.1. Obtaining the Source Code

The first step is to download the project's source code from its official repository. Open a terminal and run the following command:

```bash
git clone https://github.com/imEag/MLOps.git
cd MLOps
```

This command will create a folder named `MLOps` in your current directory and navigate into it.

### 3.2. Environment Configuration

The platform is configured using environment variables, which allows adjusting parameters such as network ports and database credentials without modifying the code. The `docker-compose.yml` file is prepared to read these variables from a `.env` file.

1.  **Create the configuration file:**
    The project includes an example file named `.env.example` inside the `backend/` folder. Copy it to create your local configuration file:

    ```bash
    cp backend/.env.example backend/.env
    ```

2.  **Review the environment variables (Optional):**
    Open the `backend/.env` file with a text editor. For a first run in a local environment, the default values are usually sufficient. However, it is useful to know the main variables:

    *   `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`: Credentials for the PostgreSQL database where MLflow and Prefect metadata will be stored.
    *   `MLFLOW_PORT`: Port to access the MLflow user interface (default: `5001`).
    *   `PREFECT_PORT`: Port for the Prefect user interface (default: `4200`).
    *   `FASTAPI_PORT`: Port on which the backend API will run (default: `8000`).
    *   `FRONTEND_PORT`: Port for the frontend web application (default: `3000`).

    If any of the default ports are already in use on your system, you can change them in this file before starting the application.

## 4. Starting and Verifying the Application

Once the environment is configured, you can start the entire stack of services with a single command.

### 4.1. Building and Starting the Services

From the root of the project (`MLOps/`), run the following command in your terminal:

```bash
docker compose up -d --build
```

*   **Command analysis:**
    *   `docker compose up`: Reads the `docker-compose.yml` file and starts all defined services (PostgreSQL, MLflow, Prefect, FastAPI, Frontend).
    *   `-d` (or `--detach`): Runs the containers in the background ("detached" mode), freeing up the terminal.
    *   `--build`: Forces the build of Docker images from the corresponding `Dockerfile`s (`backend/Dockerfile`, `frontend/Dockerfile`, etc.) before starting them. This step is essential on the first run or after making changes to the source code.

The build process may take several minutes the first time, as Docker will download the base images and install all software dependencies.

### 4.2. Verifying the Status of the Services

To confirm that all containers are running correctly, you can use the command:

```bash
docker compose ps
```

You should see a list of the five services (`postgres`, `mlflow`, `prefect`, `fastapi`, `frontend`) with the status `running` or `healthy`.

### 4.3. Accessing the User Interfaces

If the installation and startup were successful, the platform will be accessible through your web browser. Open the following URLs to verify each component:

*   **Main Platform (Frontend):**
    *   URL: `http://localhost:3000`
    *   Description: This is the main interface of the application, from where you can manage models and make predictions.

*   **Backend API (Interactive Documentation):**
    *   URL: `http://localhost:8000/docs`
    *   Description: FastAPI automatically generates an interactive documentation page (Swagger UI) where you can explore and test all API endpoints.

*   **MLflow Interface:**
    *   URL: `http://localhost:5001`
    *   Description: Here you can explore in detail the experiments, runs, metrics, and artifacts registered by the system.

*   **Prefect Interface:**
    *   URL: `http://localhost:4200`
    *   Description: The Prefect dashboard allows you to monitor the execution of the training and prediction workflows (pipelines).

If all these pages load correctly, the platform is installed and running.

### 4.4. Stopping the Application

To stop all platform services, run the following command from the root of the project:

```bash
docker compose down
```

This command will stop and remove the containers, but persistent data (such as the database and MLflow artifacts) will be preserved in Docker volumes, ready for the next time you start the application.

## 5. Platform Usage Guide

Once the platform is up and running, you can interact with it through the main user interface available at `http://localhost:3000`.

### 5.1. Dashboard

The Dashboard is the home page and the entry point to the main functionalities. When you first access it, you will see a summary of the system status, which will initially be empty (Figure 1). From here, you can navigate to the two main sections:

*   **Model Management:** To train, register, version, and promote models to production.
*   **Predictions:** To use models in production and perform inference on new data.

In the application header, you will find shortcuts to the **MLflow** and **Prefect** interfaces, essential tools for advanced system monitoring.

*<p align="center">Figure 1: Initial view of the platform's Dashboard.</p>*
<p align="center">
  <img src="screenshots/dashboard empty.png" width="800">
</p>

### 5.2. Model Lifecycle Management

This section is the control center for the entire model lifecycle. When you enter for the first time, the page will be empty (Figure 2).

*<p align="center">Figure 2: Model Management page in its initial state.</p>*
<p align="center">
  <img src="screenshots/model management empty.png" width="800">
</p>

The typical workflow is as follows:

**Step 1: Start a New Training**
1.  Click the **"Start New Training"** button. This action will start the automated training pipeline. By default, the pipeline will use the training dataset located at `backend/data/processed/`.
2.  You can monitor the progress of the execution in real-time by accessing the **Prefect** interface (to see the status of tasks, as in Figure 3) and **MLflow** (to see the list of experiments and their runs, as in Figure 4).

*<p align="center">Figure 3: Monitoring a training pipeline in the Prefect interface.</p>*
<p align="center">
  <img src="screenshots/prefect training flow.png" width="800">
</p>

*<p align="center">Figure 4: List of training runs in the MLflow interface.</p>*
<p align="center">
  <img src="screenshots/mlflow training pipeline experiment list.png" width="800">
</p>

**Step 2: Explore the Experiment History**
*   Once the training is finished, the **"Experiment History"** table in the main interface will show a record of the run (Figure 5).
*   Each row represents a "parent run" in MLflow and is expandable to show the "child runs" or nested tasks (load data, process, train). This allows for complete traceability, showing the status and duration of each step of the pipeline.

*<p align="center">Figure 5: Experiment history after a successful training.</p>*
<p align="center">
  <img src="screenshots/model management after training before registering.png" width="800">
</p>

**Step 3: Register a Model**
1.  In the row of a successfully completed training run (`successful`), click the **"Register Model"** button.
2.  A modal will open asking you for a name for the model (e.g., `EEG_Cognitive_Decline_Classifier`), as seen in Figure 6.
3.  This action formally registers the trained model and its artifacts in the MLflow **Model Registry**, creating its first version.

*<p align="center">Figure 6: Modal for registering a new model.</p>*
<p align="center">
  <img src="screenshots/model management model registering.png" width="800">
</p>

**Step 4: Manage Model Versions**
*   Once a model is registered, it will appear in the dropdown selector at the top of the page. Selecting it will populate the interface with specific information for that model (Figure 7).
    *   **Current Production Model:** Shows a summary of the version currently in production (initially empty).
    *   **Training History:** A filtered table showing only the runs that resulted in a registered version for the selected model.
    *   **Model Versions:** A table listing all existing versions of the model.

*<p align="center">Figure 7: View of a registered model before promoting a version to production.</p>*
<p align="center">
  <img src="screenshots/model management registered model no production version.png" width="800">
</p>

**Step 5: Promote a Model to Production**
1.  In the **"Model Versions"** table, locate the version you want to deploy for the inference service.
2.  Click the **"Promote to Production"** button. A confirmation modal will open (Figure 8).
3.  This action assigns the `"production"` alias to that specific version in the MLflow registry. The model with this alias will be the one used by default in the Predictions section. The interface will update to reflect the new status (Figure 9).

*<p align="center">Figure 8: Confirmation to promote a model to production.</p>*
<p align="center">
  <img src="screenshots/model management promoting to production.png" width="800">
</p>

*<p align="center">Figure 9: Final state with a registered model and a version in production.</p>*
<p align="center">
  <img src="screenshots/model management model registered with production version.png" width="800">
</p>

### 5.3. Making Predictions

This module allows you to use a model in production to perform inference on new data. When you first access it, the page will be empty (Figure 10).

*<p align="center">Figure 10: Predictions page in its initial state.</p>*
<p align="center">
  <img src="screenshots/predictions empty.png" width="800">
</p>

**Step 1: Prepare and Upload the Data**
1.  With the current configuration, the system is designed to process raw EEG data in **BIDS (Brain Imaging Data Structure)** format. Compress your BIDS dataset folder into a single `.zip` file.
2.  Use the file upload component in the interface to upload your `.zip` file to the server. Once uploaded, the contents of the file will be unzipped and will appear in the interactive file explorer (Figure 11).

*<p align="center">Figure 11: View after uploading a data file for prediction.</p>*
<p align="center">
  <img src="screenshots/predictions before prediction.png" width="800">
</p>

**Step 2: Select Data and Run the Prediction**
1.  Navigate through the folder structure and select the file or the root folder of the subject on which you want to perform inference.
2.  Click the **"Make Prediction"** button.
3.  A modal will open where you must select the registered model you want to use (Figure 12). Typically, you will select the model you previously promoted to production.

*<p align="center">Figure 12: Modal to select the model and run a prediction.</p>*
<p align="center">
  <img src="screenshots/prediction making a prediction.png" width="800">
</p>

**Step 3: Consult the Prediction History**
*   The prediction runs asynchronously in the backend. You can monitor its progress in the **Prefect** (Figure 13) and **MLflow** (Figure 14) interfaces.
*   Once completed, the **"Prediction History"** table in the main interface will be updated to show the new run with its status, date, and result (Figure 15).
*   Each row is expandable to show additional details, such as the preprocessed input data used for inference (Figure 16).
*   For auditing purposes, each prediction is also logged as a run in a dedicated experiment (`Model_Predictions`) in MLflow, where all its details can be consulted (Figure 17).

*<p align="center">Figure 13: Monitoring a prediction pipeline in Prefect.</p>*
<p align="center">
  <img src="screenshots/prefect prediction flow.png" width="800">
</p>

*<p align="center">Figure 14: List of prediction runs in MLflow.</p>*
<p align="center">
  <img src="screenshots/mlflow model predictions list.png" width="800">
</p>

*<p align="center">Figure 15: Prediction history with a visible result.</p>*
<p align="center">
  <img src="screenshots/predictions after prediction.png" width="800">
</p>

*<p align="center">Figure 16: Modal showing the input data of a previous prediction.</p>*
<p align="center">
  <img src="screenshots/prediction input details.png" width="800">
</p>

*<p align="center">Figure 17: Details of a prediction run in MLflow.</p>*
<p align="center">
  <img src="screenshots/mlflow prediction details.png" width="800">
</p>

### 6. Customization and Advanced Development

As mentioned in the introduction, the heart of the platform is a generic and reusable "meta-pipeline" for training. The orchestration flow, defined in `backend/src/flows/training_flow.py`, does not contain business logic specific to a model. Instead, it is designed to receive three interchangeable (*pluggable*) functions that define the ML lifecycle:

1.  A **data loading** function (`load_data_func`)
2.  A **data preprocessing** function (`process_data_func`)
3.  A **model training** function (`train_model_func`)

This design allows developers to adapt the platform to train virtually any type of model (Scikit-learn, TensorFlow, PyTorch, etc.) on any type of data, simply by implementing their own functions and connecting them to the orchestrator.

#### 6.1. Guide to Implementing a Custom Training Pipeline

To replace the example pipeline with your own, follow these steps:

**Step 1: Prepare the Data and Create a New Training Script**

1.  **Place your data:** Add your training dataset in a subfolder within `backend/data/`. For example: `backend/data/my_custom_data/`.
2.  **Create a new script:** Inside the `backend/src/custom_scripts/` folder, create a new Python file. For example: `my_training_script.py`. This file will contain the logic of your model.

**Step 2: Implement the Custom Functions**

Within your new script (`my_training_script.py`), you must define three functions that comply with a specific "contract" so that the orchestrator can execute them correctly.

1.  **`load_data` function:**
    *   **Purpose:** Load your data from a file and return it as a pandas DataFrame.
    *   **Contract:** Must return a `pandas.DataFrame`.
    *   **Example:**
        ```python
        import pandas as pd

        def load_data():
            # Logic to load your data
            file_path = '/app/data/my_custom_data/my_dataset.csv'
            df = pd.read_csv(file_path)
            return df
        ```

2.  **`process_data` function:**
    *   **Purpose:** Perform preprocessing, cleaning, or feature extraction.
    *   **Contract:** Must accept a `pandas.DataFrame` as input and return a `pandas.DataFrame` as output.
    *   **Example:**
        ```python
        import pandas as pd

        def process_data(data: pd.DataFrame):
            # Preprocessing logic
            processed_df = data.dropna()
            return processed_df
        ```

3.  **`train_model` function:**
    *   **Purpose:** Train the model, evaluate its performance, and return the model object along with its metrics.
    *   **Contract:** Must accept a `pandas.DataFrame` and return a tuple containing `(model_object, metrics_dict)`. The metrics dictionary **must** include the keys required by the pipeline for logging in MLflow.
    *   **Example:**
        ```python
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        def train_model(data: pd.DataFrame):
            # Assume the last column is the target
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            # Train the model
            model = RandomForestClassifier(n_estimators=100)
            model.fit(X_train, y_train)

            # Evaluate metrics
            preds = model.predict(X_test)
            accuracy = accuracy_score(y_test, preds)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average='macro')

            # Create the metrics dictionary (mandatory keys!)
            metrics = {
                'accuracy': accuracy,
                'macro_avg_precision': precision,
                'macro_avg_recall': recall,
                'macro_avg_f1_score': f1,
                # You can add more metrics if you wish
            }

            return model, metrics
        ```

**Step 3: Connect the New Pipeline**

The last step is to tell the system to use your new functions instead of the default ones.

1.  Open the file `backend/src/services/ml_model_service.py`.
2.  Locate the `run_ml_training_pipeline` function.
3.  Modify the imports to point to your new script and change the functions passed to the `ml_pipeline_flow`.

    ```python
    # In backend/src/services/ml_model_service.py

    # ... other imports ...
    from ..flows.training_flow import ml_pipeline_flow
    # from ..custom_scripts.training_script import load_data, process_data, train_model # Comment out or delete the original line

    # Import the new functions
    from ..custom_scripts.my_training_script import load_data, process_data, train_model

    def run_ml_training_pipeline():
        """Runs the ML training pipeline."""
        print("Starting the CUSTOM ML training pipeline flow via service...")
        flow_state = ml_pipeline_flow(
            load_data_func=load_data,      # Your new function
            load_data_args=(),
            load_data_kwargs={},
            process_data_func=process_data,  # Your new function
            process_data_args=(),
            process_data_kwargs={},
            train_model_func=train_model,    # Your new function
            train_model_args=(),
            train_model_kwargs={}
        )
        print("CUSTOM ML training pipeline flow finished in service.")
        return flow_state
    ```

**Step 4: Rebuild the Docker Image**

After making changes to the backend code, you need to rebuild the Docker image of the `fastapi` service for the changes to take effect.

Run the following command from the root of the project:

```bash
docker compose up -d --build fastapi
```

Done! The next time you press the **"Start New Training"** button in the user interface, the platform will run your custom pipeline.

#### 6.2. Customizing the Prediction Pipeline

The prediction pipeline follows a similar principle. If you need to process raw input data with a format different from BIDS, you will need to modify the preprocessing function used by the prediction flow.

1.  **Function to modify:** `process_data` in the file `backend/src/custom_scripts/data_preprocessing_script.py`.
2.  **Contract:** The function must accept a path to the input data and return a `pandas.DataFrame` with the features that the model expects to receive.
3.  **Rebuild:** As with the training pipeline, you will need to rebuild the image with `docker compose up -d --build fastapi` after making the changes.
