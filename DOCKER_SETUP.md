# Docker Setup Guide

This project has been configured to run all services (backend and frontend) using Docker Compose from the root directory.

## Architecture Overview

The MLOps platform consists of multiple interconnected services:

- **PostgreSQL**: Database service
- **MLflow**: ML experiment tracking and model registry
- **Prefect**: Workflow orchestration server
- **FastAPI**: Backend API service
- **Frontend**: React application

## Quick Start

1. From the root MLOps directory, run:
```bash
docker-compose up -d
```

2. Wait for all services to start (PostgreSQL needs to initialize first)

3. Access the services:
- **Frontend**: http://localhost:3000
- **FastAPI API**: http://localhost:8000
- **MLflow UI**: http://localhost:5001
- **Prefect UI**: http://localhost:4200
- **PostgreSQL**: localhost:5432

## Service Details

### PostgreSQL Database
- **Purpose**: Primary database for all services
- **Initialization**: Automatically creates schemas and tables via `init-db/01-init.sql`
- **Databases**: `mlops` (main), `prefect_db` (Prefect workflows)
- **Schemas**: `mlflow`, `predictions`, `training_data`, `prefect`
- **Health Check**: Built-in health monitoring

### MLflow Server
- **Purpose**: Experiment tracking, model registry, and artifact storage
- **Backend**: PostgreSQL for metadata storage
- **Artifacts**: Local file system storage (`/mlflow/artifacts`)
- **Features**: Artifact serving enabled, model registry
- **UI**: Full MLflow tracking interface

### Prefect Server
- **Purpose**: Workflow orchestration and scheduling
- **Database**: Dedicated `prefect_db` database
- **API**: REST API for workflow management
- **UI**: Flow monitoring and execution dashboard

### FastAPI Service
- **Purpose**: ML model serving and API endpoints
- **Integration**: Connected to MLflow for model loading
- **Database**: PostgreSQL for predictions logging
- **Development**: Hot reload enabled with volume mounts

### Frontend Application
- **Purpose**: User interface for the MLOps platform
- **Technology**: React with Vite development server
- **Development**: Hot reload enabled
- **API Connection**: Configured to connect to FastAPI backend

## Environment Variables

You can customize the configuration by creating a `.env` file in the root directory:

```env
# Database Configuration
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=mlops
POSTGRES_PORT=5432

# MLflow Configuration
MLFLOW_PORT=5001

# Prefect Configuration
PREFECT_PORT=4200

# FastAPI Configuration
FASTAPI_PORT=8000

# Frontend Configuration
FRONTEND_PORT=3000
```

## Development Features

- **Volume Mounts**: Source code changes reflect immediately
- **Network Isolation**: All services communicate via `mlops-network`
- **Data Persistence**: PostgreSQL data persists in named volume
- **Artifact Storage**: MLflow artifacts stored in local backend directory

## Service Dependencies

Services start in the following order due to dependencies:
1. **PostgreSQL** (with health check)
2. **MLflow** (depends on PostgreSQL)
3. **Prefect** (depends on PostgreSQL and MLflow)
4. **FastAPI** (depends on all previous services)
5. **Frontend** (depends on FastAPI)

## Managing Services

### Start all services:
```bash
docker-compose up -d
```

### View logs:
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f mlflow
```

### Stop services:
```bash
docker-compose down
```

### Stop and remove all data:
```bash
docker-compose down -v
```

### Rebuild services:
```bash
docker-compose up -d --build
```

### Checking Service Health

```bash
# Check running services
docker-compose ps

# Check service logs
docker-compose logs [service-name]

# Access service shell
docker-compose exec [service-name] /bin/bash
```

## Data Persistence

- **PostgreSQL Data**: Stored in `postgres_data` Docker volume
- **MLflow Artifacts**: Stored in `./backend/artifacts` directory
- **MLflow Runs**: Stored in `./backend/mlruns` directory

To backup your data, ensure you copy both the Docker volume and the local directories.

## Network Configuration

All services communicate through the `mlops-network` bridge network, allowing:
- Service-to-service communication using service names as hostnames
- Isolated network environment
- Configurable external port access 