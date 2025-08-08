import React from 'react';
import { Layout, Typography, Button } from 'antd';
import { BulbOutlined, BulbFilled } from '@ant-design/icons';
import { theme } from 'antd';
import brainIcon from '../../assets/images/brain.svg';
import { useTheme } from '../../config/ThemeContext.jsx';
const { Header } = Layout;
const { Title } = Typography;
import { useIsMobile } from '@/hooks/useBreakpoint';

const HeaderComponent = () => {
  const {
    token: { lineWidth, colorBorder },
  } = theme.useToken();
  const isMobile = useIsMobile();
  const { isDarkMode, toggleTheme } = useTheme();

  return (
    <Header
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        borderBottom: `${lineWidth}px solid ${colorBorder}`,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center' }}>
        <img
          src={brainIcon}
          alt="Brain Icon"
          style={{
            width: isMobile ? '16px' : '24px',
            height: isMobile ? '16px' : '24px',
            marginRight: '12px',
            marginLeft: '0px',
          }}
        />
        <Title
          level={3}
          style={{
            margin: 0,
            color: '#1890ff',
            fontWeight: 'bold',
            fontSize: isMobile ? '18px' : '24px',
          }}
        >
          NeurOps
        </Title>
      </div>

      <Button
        type="text"
        icon={isDarkMode ? <BulbFilled /> : <BulbOutlined />}
        onClick={toggleTheme}
        size={isMobile ? 'small' : 'middle'}
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
        title={isDarkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
      />
    </Header>
  );
};

export default HeaderComponent;
