#!/bin/bash

# Set defaults for environment variables
POSTGRES_USER=${POSTGRES_USER:-postgres}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-postgres}
POSTGRES_HOST=${POSTGRES_HOST:-postgres}
POSTGRES_PORT=${POSTGRES_PORT:-5432}
POSTGRES_DB=${POSTGRES_DB:-mlops}
MLFLOW_PORT=${MLFLOW_PORT:-5000}

# Construct the database URI
BACKEND_STORE_URI="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}"

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
until pg_isready -h $POSTGRES_HOST -p $POSTGRES_PORT -U $POSTGRES_USER; do
  echo "PostgreSQL is unavailable - sleeping"
  sleep 2
done

echo "PostgreSQL is ready!"

# Create artifacts directory if it doesn't exist
mkdir -p /mlflow/artifacts

# Start MLflow server with artifact serving enabled
echo "Starting MLflow server on port ${MLFLOW_PORT}..."
exec mlflow server \
    --host 0.0.0.0 \
    --port $MLFLOW_PORT \
    --backend-store-uri "$BACKEND_STORE_URI" \
    --default-artifact-root /mlflow/artifacts \
    --serve-artifacts