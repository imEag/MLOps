import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { ConfigProvider } from 'antd';
import { Provider } from 'react-redux';
import Dashboard from './pages/Dashboard';
import ModelManagement from './pages/ModelManagement';
import Predictions from './pages/Predictions';
import Layout from './components/Layout';
import { themeConfig } from './config/theme';
import { store } from './store';

function App() {
  return (
    <Provider store={store}>
      <ConfigProvider theme={themeConfig}>
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Layout />}>
              <Route index element={<Navigate to="/dashboard" replace />} />
              <Route path="dashboard" element={<Dashboard />} />
              <Route path="model-management" element={<ModelManagement />} />
              <Route path="predictions" element={<Predictions />} />
              <Route path="*" element={<Navigate to="/dashboard" replace />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </ConfigProvider>
    </Provider>
  );
}

export default App;
