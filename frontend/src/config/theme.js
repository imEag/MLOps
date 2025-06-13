import { theme } from 'antd';

const currentTheme = theme.defaultAlgorithm;

export const themeConfig = {
  algorithm: currentTheme,
  token: {
    ...(currentTheme === theme.defaultAlgorithm && {
      borderRadius: 6,
      borderRadiusXS: 2,
      borderRadiusSM: 4,
      borderRadiusLG: 8,
      borderRadiusOuter: 4,

      lineWidth: 1,
      lineWidthBold: 2,
      lineWidthFocus: 4,

      colorBorder: '#f0f0f0',
      colorBorderSecondary: '#d9d9d9',
      colorBorderBg: '#ffffff',

      colorErrorBorder: '#ff4d4f',
      colorWarningBorder: '#faad14',
      colorSuccessBorder: '#52c41a',
      colorInfoBorder: '#1890ff',
    }),
  },
  ...(currentTheme === theme.defaultAlgorithm && {
    components: {
      Layout: {
        headerBg: '#ffffff',
        headerHeight: 64,
        headerPadding: '0 24px',
        triggerBg: '#d9d9d9',
        triggerColor: '#000',
      },
    },
  }),
};
