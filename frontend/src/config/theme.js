import { theme } from 'antd';

// Light theme configuration
export const lightTheme = {
  algorithm: theme.defaultAlgorithm,
  token: {
    colorPrimary: '#1890ff',
    colorBgContainer: '#ffffff',
    colorBgLayout: '#f5f5f5',
    colorText: '#000000',
    colorTextSecondary: '#666666',
    colorBorder: '#d9d9d9',
    colorBorderSecondary: '#f0f0f0',
  },
};

// Dark theme configuration
export const darkTheme = {
  algorithm: theme.darkAlgorithm,
  token: {
    colorPrimary: '#1890ff',
    colorBgContainer: '#141414',
    colorBgLayout: '#000000',
    colorText: '#ffffff',
    colorTextSecondary: '#a6a6a6',
    colorBorder: '#303030',
    colorBorderSecondary: '#1f1f1f',
  },
};
