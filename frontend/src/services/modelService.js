import api from './api';

// Models API service
export const modelService = {
  // Get dashboard summary
  getDashboardSummary: async () => {
    const response = await api.get('/api/models/dashboard');
    return response.data;
  },

  // Get model info
  getModelInfo: async (modelName) => {
    const response = await api.get(`/api/models/${modelName}/info`);
    return response.data;
  },

  // Get latest training for a model
  getLatestTraining: async (modelName) => {
    const response = await api.get(`/api/models/${modelName}/latest-training`);
    return response.data;
  },

  // Get training history
  getTrainingHistory: async (modelName, limit = 10) => {
    const response = await api.get(`/api/models/${modelName}/training-history`, {
      params: { limit },
    });
    return response.data;
  },

  getExperimentHistory: async (limit = 10) => {
    const response = await api.get('/api/models/experiments/history', {
      params: { limit },
    });
    return response.data;
  },

  // Start model training
  trainModel: async () => {
    const response = await api.post('/api/models/train');
    return response.data;
  },

  // Get production model version
  getProductionModelVersion: async (modelName) => {
    const response = await api.get(`/api/models/${modelName}/production-version`);
    return response.data;
  },

  // Get model versions
  getModelVersions: async (modelName) => {
    const response = await api.get(`/api/models/${modelName}/versions`);
    return response.data;
  },

  // Promote model to production
  promoteModelToProduction: async (modelName, version) => {
    const response = await api.post(`/api/models/${modelName}/promote/${version}`);
    return response.data;
  },

  registerModel: async (runId, modelName) => {
    const response = await api.post(
      `/api/models/register?run_id=${runId}&model_name=${modelName}`,
    );
    return response.data;
  },

  // Get available models
  getAvailableModels: async () => {
    const response = await api.get('/api/models/available');
    return response.data;
  },

  // Make prediction
  predict: async (modelName, data) => {
    const response = await api.post(`/api/models/${modelName}/predict`, data);
    return response.data;
  },

  // Get prediction history
  getPredictionHistory: async (limit = 50, offset = 0) => {
    const response = await api.get('/api/models/predictions/history', {
      params: { limit, offset },
    });
    return response.data;
  },

  // Get prediction stats
  getPredictionStats: async () => {
    const response = await api.get('/api/models/predictions/stats');
    return response.data;
  },

  // Get recent flow runs
  getRecentFlowRuns: async (limit = 10) => {
    const response = await api.get('/api/models/flows/recent', {
      params: { limit },
    });
    return response.data;
  },

  // Check services health
  checkServicesHealth: async () => {
    const response = await api.get('/api/models/health/services');
    return response.data;
  },

  // Get experiments summary
  getExperimentsSummary: async () => {
    const response = await api.get('/api/models/experiments');
    return response.data;
  },
};
