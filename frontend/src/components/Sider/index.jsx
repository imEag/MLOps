import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  DashboardOutlined,
  SettingOutlined,
  LineChartOutlined,
  ExperimentOutlined,
} from '@ant-design/icons';
import { Layout, Menu, Button, Divider, theme } from 'antd';
import { useIsMobile } from '@/hooks/useBreakpoint';
import externalLinkIcon from '../../assets/images/external-link.svg';
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

  const handleMLFlowClick = () => {
    window.open('http://localhost:5001', '_blank', 'noopener,noreferrer');
  };

  const handlePrefectClick = () => {
    window.open('http://localhost:4200', '_blank', 'noopener,noreferrer');
  };

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

  const externalButtonStyle = {
    display: 'flex',
    alignItems: 'center',
    gap: '4px',
    width: collapsed ? '100%' : 'auto',
    justifyContent: collapsed ? 'center' : 'flex-start',
    marginBottom: '8px',
  };

  const externalLinkStyle = {
    width: '12px',
    height: '12px',
    transition: 'filter 0.2s ease',
  };

  return (
    <>
      <style>
        {`
          .external-button:hover .external-icon {
            filter: invert(24%) sepia(100%) saturate(1352%) hue-rotate(204deg) brightness(95%) contrast(106%);
          }
        `}
      </style>
      <Sider
        width={230}
        collapsedWidth={isMobile ? 70 : 80}
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
            height: 'calc(100% - 120px)', // Increased space for external buttons
            borderRight: `${lineWidth}px solid ${colorBorder}`,
            paddingLeft: `${paddingSM}px`,
            paddingRight: `10px`,
          }}
          items={items2}
        />

        {/* External Tools Section */}
        <div style={{
          padding: `${paddingSM}px`,
          paddingBottom: `${paddingLG * 3}px`, // Triple the bottom padding
          borderTop: `${lineWidth}px solid ${colorBorder}`,
          position: 'absolute',
          bottom: '30px', // Position above the collapse button
          left: 0,
          right: 0,
          background: colorBgContainer,
        }}>
          <Divider style={{ margin: '8px 0', fontSize: collapsed ? '10px' : '12px' }}>
            {collapsed ? 'Tools' : 'External Tools'}
          </Divider>

          <Button
            type="default"
            icon={<ExperimentOutlined />}
            onClick={handleMLFlowClick}
            className="external-button"
            style={externalButtonStyle}
            size="small"
            title={collapsed ? 'MLFlow' : undefined}
          >
            {!collapsed && (
              <>
                MLFlow
                <img
                  src={externalLinkIcon}
                  alt="External Link"
                  className="external-icon"
                  style={externalLinkStyle}
                />
              </>
            )}
          </Button>

          <Button
            type="default"
            icon={<SettingOutlined />}
            onClick={handlePrefectClick}
            className="external-button"
            style={{
              ...externalButtonStyle,
              marginBottom: '0px', // Remove bottom margin from last button
            }}
            size="small"
            title={collapsed ? 'Prefect' : undefined}
          >
            {!collapsed && (
              <>
                Prefect
                <img
                  src={externalLinkIcon}
                  alt="External Link"
                  className="external-icon"
                  style={externalLinkStyle}
                />
              </>
            )}
          </Button>
        </div>
      </Sider>
    </>
  );
};

export default SiderComponent;
