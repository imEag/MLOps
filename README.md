# MLOps Project

This project implements a machine learning pipeline using MLflow for experiment tracking, Prefect for workflow orchestration, and Docker for containerization.

## Project Structure

```
.
├── data/
│   └── raw/           # Raw dataset
├── src/
│   ├── flows/         # Prefect flows
│   └── train.py       # Training script
├── mlruns/            # MLflow experiment tracking
├── Dockerfile
├── docker-compose.yml
├── .env              # Environment variables
├── .env.example      # Example environment variables
└── pyproject.toml     # Poetry dependencies
```

## Setup

1. Install Poetry:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies:
```bash
poetry install
```

3. Configure ports (optional):
   - Edit the `.env` file to change default ports:
     ```
     MLFLOW_PORT=5001
     PREFECT_PORT=4201
     ```
   - If not specified, defaults are:
     - MLflow: 5000
     - Prefect: 4200

4. Start the services:
```bash
docker-compose up -d
```

## Usage

1. Run the training pipeline:
```bash
poetry run python src/flows/training_flow.py
```

2. Access MLflow UI:
- Open http://localhost:${MLFLOW_PORT:-5000} in your browser

3. Access Prefect UI:
- Open http://localhost:${PREFECT_PORT:-4200} in your browser

## Development

- The training script is located in `src/train.py`
- Prefect flows are in `src/flows/`
- MLflow tracks experiments in the `mlruns/` directory

## Environment Variables

- `MLFLOW_PORT`: Port for MLflow server (default: 5000)
- `PREFECT_PORT`: Port for Prefect server (default: 4200)
- `POSTGRES_PORT`: Port for PostgreSQL database (default: 5432)
- `POSTGRES_USER`: PostgreSQL username (default: postgres)
- `POSTGRES_PASSWORD`: PostgreSQL password (default: postgres)
- `POSTGRES_DB`: PostgreSQL database name (default: mlops)
