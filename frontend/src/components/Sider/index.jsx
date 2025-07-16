import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  DashboardOutlined,
  SettingOutlined,
  LineChartOutlined,
} from '@ant-design/icons';
import { Layout, Menu, theme } from 'antd';
import { useIsMobile } from '@/hooks/useBreakpoint';
const { Sider } = Layout;

const items2 = [
  {
    key: 'dashboard',
    icon: React.createElement(DashboardOutlined),
    label: 'Dashboard',
  },
  {
    key: 'model-management',
    icon: React.createElement(SettingOutlined),
    label: 'Model management',
  },
  {
    key: 'predictions',
    icon: React.createElement(LineChartOutlined),
    label: 'Predictions',
  },
];

const SiderComponent = () => {
  const {
    token: { colorBgContainer, lineWidth, colorBorder, paddingLG, paddingSM },
  } = theme.useToken();
  const isMobile = useIsMobile();
  const [collapsed, setCollapsed] = useState(isMobile);

  useEffect(() => {
    setCollapsed(isMobile);
  }, [isMobile]);

  const location = useLocation();
  const navigate = useNavigate();

  // Get the current path and map it to menu key
  const getCurrentMenuKey = () => {
    const pathname = location.pathname;
    if (pathname.startsWith('/dashboard')) return ['dashboard'];
    if (pathname.startsWith('/model-management')) return ['model-management'];
    if (pathname.startsWith('/predictions')) return ['predictions'];
    return ['dashboard']; // default
  };

  // Handle menu item selection
  const handleMenuSelect = ({ key }) => {
    navigate(`/${key}`);
  };

  return (
    <Sider
      width={230}
      collapsedWidth={isMobile ? 70 : 80} // Reduced collapsed width only for mobile
      style={{
        background: colorBgContainer,
        height: '100%',
        paddingTop: `${paddingLG}px`,
        paddingBottom: `${paddingLG}px`,
      }}
      collapsible={!isMobile}
      collapsed={collapsed}
      onCollapse={setCollapsed}
    >
      <Menu
        mode="inline"
        selectedKeys={getCurrentMenuKey()}
        onSelect={handleMenuSelect}
        style={{
          height: '100%',
          borderRight: `${lineWidth}px solid ${colorBorder}`,
          paddingLeft: `${paddingSM}px`,
          paddingRight: `10px`,
        }}
        items={items2}
      />
    </Sider>
  );
};

export default SiderComponent;
