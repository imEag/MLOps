import api from './api';

export const predictionService = {
  uploadFile: (file) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post('/predictions/upload/', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 0, // Disable timeout for file uploads
    });
  },
  getFiles: () => {
    return api.get('/predictions/files/');
  },
  deleteFile: (path) => {
    return api.delete('/predictions/files/', { params: { path } });
  },
  makePrediction: (path) => {
    return api.post('/predictions/predict/', { path });
  },
};
