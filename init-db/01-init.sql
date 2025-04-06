-- Create schemas
CREATE SCHEMA IF NOT EXISTS mlflow;
CREATE SCHEMA IF NOT EXISTS predictions;
CREATE SCHEMA IF NOT EXISTS training_data;

-- Create tables for predictions
CREATE TABLE IF NOT EXISTS predictions.prediction_logs (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(50),
    input_data JSONB,
    prediction_result JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Create table for training data
CREATE TABLE IF NOT EXISTS training_data.dataset (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    -- Add the columns here
    -- Example:
    -- column1 FLOAT,
    -- column2 FLOAT,
    -- ...
    metadata JSONB
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_prediction_logs_created_at ON predictions.prediction_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_dataset_created_at ON training_data.dataset(created_at);

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA mlflow TO postgres;
GRANT ALL PRIVILEGES ON SCHEMA predictions TO postgres;
GRANT ALL PRIVILEGES ON SCHEMA training_data TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA predictions TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA training_data TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA predictions TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA training_data TO postgres; 