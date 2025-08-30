# MLOps "NeuroOps" Platform

An end-to-end Machine Learning Operations (MLOps) platform that implements the core ML lifecycle: training, tracking, versioning, promotion to production and serving. The platform makes it easy to start training runs, persist parameters/metrics/artifacts, manage model versions and promote a model to production for serving.

The system supports predictions from raw data (for example, EEG in BIDS format): uploads are preprocessed, features are extracted, inference is executed with the selected production model, and inputs/outputs are logged for auditability.

Although this repository includes an EEG-ready preprocessing pipeline (bundled `sovaharmony`), the architecture is model- and domain-agnostic. Developers can plug different preprocessing or training functions without changing the orchestration code — see `backend/src/flows/training_flow.py`, `backend/src/flows/prediction_flow.py` and `backend/src/services/ml_model_service.py` for the pluggable pipeline hooks.

## Key components

- Backend: FastAPI application that exposes REST endpoints for training, prediction and file management. Integrates with MLflow (tracking & registry), Prefect (orchestration) and PostgreSQL (metadata).
- Frontend: React + Vite single-page application that provides a dashboard for model management and prediction workflows.
- Orchestration & packaging: Docker images and `docker-compose.yml` bring up the full stack (Postgres, MLflow, Prefect, FastAPI, Frontend).
- Preprocessing: EEG harmonization and feature extraction provided through the included `sovaharmony` package (bundled as a wheel in `backend/vendor/`).

## Repository layout

Top-level overview (relevant folders):

```
.
├── backend/                # Backend application, Prefect flows, MLflow artifacts and Dockerfiles
│   ├── src/                # FastAPI app, routers, services, flows and scripts
│   ├── artifacts/          # ML artifacts (models, figures) persisted by MLflow
│   ├── mlruns/             # MLflow local runs storage
│   └── Dockerfile*         # Dockerfiles for building backend-related images
├── frontend/               # React application (UI for dashboard, model management & predictions)
├── docker-compose.yml     # Full-stack local orchestration
├── README.md              # This document
└── LICENSE

*See backend/ for Dockerfile, Dockerfile.fastapi and Dockerfile.mlflow
```

## Quick start (recommended)

This project is packaged with Docker and a `docker-compose.yml` that will start all required services. The quickest way to run the full stack locally is:

1. Ensure Docker and Docker Compose are installed and running on your machine.
2. From the project root run:

```bash
docker compose up -d --build
```

3. Wait for services to start. Typical service ports (configurable via `.env`):

- FastAPI (backend): http://localhost:8000
- Frontend (UI): http://localhost:3000
- MLflow UI: http://localhost:5001
- Prefect UI: http://localhost:4200

Open the frontend in a browser (default: `http://localhost:3000`) to access the dashboard.

Notes:
- The `docker compose` command above will build images using the Dockerfiles in `backend/` and `frontend/`.
- If you prefer the legacy `docker-compose` binary, the command is `docker-compose up -d --build`.

## Environment configuration

Environment variables can be configured in the `backend/.env` file (or at project root if you prefer). Common variables include:

- `MLFLOW_PORT` (default: 5001)
- `PREFECT_PORT` (default: 4200)
- `POSTGRES_PORT` (default: 5432)
- `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB`
- `FASTAPI_PORT` (default: 8000)

Adjust ports and credentials before first run if needed.

## How the system works (high level)

1. Training flows are defined as Prefect flows in `backend/src/flows/`. Each training execution is tracked in MLflow. Subtasks in the pipeline are recorded as nested runs in MLflow and store parameters, metrics and artifacts (plots, confusion matrices, models).
2. The FastAPI backend exposes endpoints to start training (`/api/models/train`), list models, promote versions and run predictions. It delegates workflow execution to Prefect and logs results into MLflow.
3. The frontend consumes the backend API to provide the user a model management UI (train, register, promote) and a prediction UI (upload BIDS zips, run prediction, view history).
4. Prediction flow: user uploads a `.zip` with a BIDS dataset. Backend extracts files into `backend/data/uploads/`, the Prefect prediction flow executes preprocessing (using `sovaharmony`), loads the production model from MLflow and runs inference. Prediction runs are also logged in MLflow under a dedicated experiment.

### Model-agnostic training pipeline

The project implements a generic, reusable "meta-pipeline" for training that is intentionally model-agnostic. The orchestrator function `ml_pipeline_flow` (see `backend/src/flows/training_flow.py`) receives three pluggable callables: a data loader (`load_data_func`), a preprocessing function (`process_data_func`) and a training function (`train_model_func`), together with their arguments. This design allows any model implementation that follows the simple contract (load → process → train) to be executed by the same Prefect flow without changing orchestration logic.

During execution the pipeline creates a parent MLflow run and records each pipeline stage as a nested run (sub-run). Each nested run logs parameters (`mlflow.log_params`), metrics (`mlflow.log_metrics`) and artifacts (`mlflow.log_artifact`) and finally serializes and registers the trained model (`mlflow.sklearn.log_model` or equivalent). The model entry in MLflow includes example inputs and an environment specification (e.g. `conda.yaml`) to guarantee reproducibility. Developers can therefore adapt the code to train different model types (scikit-learn, PyTorch, TensorFlow, etc.) by providing compatible `train_model_func` implementations and wiring them into the pipeline.

## Using the platform

Basic user story:

1. Start the stack with Docker Compose.
2. Open the frontend and go to the Model Management page.
3. If the system is empty, start a new training run (it will use the training dataset defined in the training script and stored under `backend/data/processed` by default).
4. After training completes, register and promote a model version to `production` via the web UI.
5. Go to Predictions, upload a `.zip` that contains a BIDS-formatted EEG dataset, select the `production` model and run a prediction. Results will appear in the Prediction History and will be visible in MLflow and Prefect.

## Developer workflow

Backend (Python / Prefect / FastAPI)

1. Enter backend folder:

```bash
cd backend
```

2. Development with Poetry (optional): install dependencies with Poetry and run the FastAPI app locally for debugging.

Frontend (React)

1. Enter the frontend folder:

```bash
cd frontend
npm install
npm run dev
```

2. The frontend runs on Vite (default `http://localhost:3000`). Configure `VITE_API_URL` to point to the backend API if necessary.

## Notes about preprocessing and sovaharmony

The EEG preprocessing and feature extraction are provided by the `sovaharmony` package. Because `sovaharmony` has a complex dependency tree, a wheel (`.whl`) is included in `backend/vendor/` and is installed during backend image build. This avoids installing conflicting dependencies system-wide and ensures the required preprocessing functions are available to the prediction and training flows.

If you modify or update the `sovaharmony` code in `backend/vendor/sovaharmony/`, rebuild the backend image so the updated package is installed in the container.

## Common commands

Start full stack (build images):

```bash
docker compose up -d --build
```

View logs for a specific service (example: fastapi):

```bash
docker compose logs -f fastapi
```

Stop and remove containers:

```bash
docker compose down
```

## Screenshots

Below are a few screenshots showing the UI and monitoring tools used in this project. See the `screenshots/` folder for the full set of images.

### NeurOps GUI
![Model management](screenshots/model%20management%20model%20registered%20with%20production%20version.png)

![Model management (Register model)](screenshots/model%20management%20model%20registering.png)

![Predictions](screenshots/predictions%20after%20prediction.png)

![Making a prediction](screenshots/prediction%20making%20a%20prediction.png)

### Prefect GUI
![Prefect predictions pipeline](screenshots/prefect%20prediction%20flow.png)

### MLFlow GUI
![MLflow training pipeline (experiment list)](screenshots/mlflow%20training%20pipeline%20experiment%20list.png)

![MLflow training pipeline (experiment datails)](screenshots/mlflow%20training%20pipeline%20experiment%20details.png)

[View all screenshots in the repository](screenshots/)

## License

This project is released under the MIT License. See the `LICENSE` file for details.

## Acknowledgements

This work was carried out by an Bioengineering student at the University of Antioquia as part of an academic internship conducted as a research project with the Grupo de Neurociencias de Antioquia (GNA). The repository also includes a wheel (.whl) of the sovaharmony package for EEG preprocessing.
