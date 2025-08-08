import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { ConfigProvider } from 'antd';
import App from './App.jsx';
import { ThemeProvider, useTheme } from './config/ThemeContext.jsx';

import 'antd/dist/reset.css';
import './assets/scss/reset.css';
// import './assets/scss/index.scss';

const AppWithTheme = () => {
  const { currentTheme } = useTheme();

  return (
    <ConfigProvider theme={currentTheme}>
      <App />
    </ConfigProvider>
  );
};

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ThemeProvider>
      <AppWithTheme />
    </ThemeProvider>
  </StrictMode>,
);
