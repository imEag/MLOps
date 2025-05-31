# MLOps EEG Platform

A comprehensive MLOps platform for automated training and serving of Machine Learning models focused on neurological disease classification using EEG signals.

## Project Architecture

This project implements a full-stack MLOps solution with:
- **Backend**: Python FastAPI with MLflow tracking, Prefect orchestration, and PostgreSQL
- **Frontend**: React + Vite
- **Infrastructure**: Docker containerization for easy deployment

## Project Structure

```
.
├── backend/
│   ├── src/
│   │   ├── flows/         # Prefect flows for ML pipelines
│   │   ├── services/      # Business logic services
│   │   ├── routers/       # FastAPI route handlers
│   │   ├── schemas/       # Pydantic models
│   │   └── training_script/  # ML training scripts
│   ├── data/
│   │   └── raw/           # Raw EEG datasets
│   ├── mlruns/            # MLflow experiment tracking
│   ├── docker-compose.yml # Backend services orchestration
│   ├── Dockerfile         # Backend application container
│   └── pyproject.toml     # Python dependencies (Poetry)
├── frontend/              # React application (to be created)
│   ├── src/
│   ├── package.json
│   └── vite.config.ts
├── docker-compose.yml     # Full-stack orchestration (optional)
└── README.md             # This file
```

## Features

- 🤖 **Automated ML Pipeline**: Continuous training and model updates
- 📊 **Real-time Monitoring**: MLflow experiment tracking and model registry
- 🌐 **REST API**: FastAPI backend for model serving and data management
- 📱 **Web Interface**: React dashboard for model management and predictions
- 🐳 **Containerized**: Docker-based deployment for all services
- 📈 **EEG Analysis**: Specialized in neurological disease classification

## Setup

### Prerequisites

- Docker & Docker Compose
- Poetry (for backend development)
- Node.js & npm (for frontend development)

### Backend Setup

1. Navigate to backend directory:
```bash
cd backend
```

2. Install Poetry:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Install Python dependencies:
```bash
poetry install
```

4. Configure environment (optional):
   - Edit the `.env` file in backend/ to change default ports:
     ```
     MLFLOW_PORT=5001
     PREFECT_PORT=4201
     POSTGRES_PORT=5433
     ```
   - If not specified, defaults are:
     - MLflow: 5000
     - Prefect: 4200
     - PostgreSQL: 5432

5. Start backend services:
```bash
cd backend
docker-compose up -d
```

### Frontend Setup

```

## Usage

### Backend Services

1. **FastAPI Application**:
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

2. **MLflow UI**:
   - Open http://localhost:5000 (or your configured port)
   - Track experiments and model versions

3. **Prefect UI**:
   - Open http://localhost:4200 (or your configured port)
   - Monitor ML pipeline execution

4. **PostgreSQL Database**:
   - Host: localhost:5432 (or your configured port)
   - Database: mlops

### Running ML Pipelines

```bash
cd backend
poetry run python src/flows/training_flow.py
```

## Development

### Backend Development

- **FastAPI App**: `backend/src/main.py`
- **ML Flows**: `backend/src/flows/`
- **Training Scripts**: `backend/src/training_script/`
- **API Routes**: `backend/src/routers/`

### Frontend Development

- **React Components**: `frontend/src/components/`
- **Pages**: `frontend/src/pages/`
- **Services**: `frontend/src/services/`

## Environment Variables

### Backend (.env in backend/)

- `MLFLOW_PORT`: Port for MLflow server (default: 5000)
- `PREFECT_PORT`: Port for Prefect server (default: 4200)
- `POSTGRES_PORT`: Port for PostgreSQL database (default: 5432)
- `POSTGRES_USER`: PostgreSQL username (default: postgres)
- `POSTGRES_PASSWORD`: PostgreSQL password (default: postgres)
- `POSTGRES_DB`: PostgreSQL database name (default: mlops)

## API Documentation

Once the backend is running, visit http://localhost:8000/docs for interactive API documentation.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
