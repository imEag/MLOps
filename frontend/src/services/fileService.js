import api from './api';

export const fileService = {
  uploadFile: (file) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post('/files/', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 0, // Disable timeout for file uploads
    });
  },
  getFiles: () => {
    return api.get('/files/');
  },
  deleteFile: (path) => {
    return api.delete('/files/', { params: { path } });
  },
  makePrediction: (path) => {
    return api.post('/predictions/predict/', { path });
  },
};
