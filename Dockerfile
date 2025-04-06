FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-tk \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install --no-cache-dir poetry==1.4.2

# Copy only dependency files first
COPY pyproject.toml poetry.lock* ./

# Configure poetry and install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

# Copy the rest of the application
COPY . .

ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=http://localhost:5000

CMD ["poetry", "run", "prefect", "server", "start", "--host", "0.0.0.0", "--port", "4200"] 