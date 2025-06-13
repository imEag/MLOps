import React from 'react';
import { Layout, Typography, Button, Space } from 'antd';
import { ExperimentOutlined, SettingOutlined } from '@ant-design/icons';
import { theme } from 'antd';
import brainIcon from '../../assets/images/brain.svg';
import externalLinkIcon from '../../assets/images/external-link.svg';
const { Header } = Layout;
const { Title } = Typography;

const HeaderComponent = () => {
  const {
    token: { lineWidth, colorBorder },
  } = theme.useToken();

  const handleMLFlowClick = () => {
    window.open('http://localhost:5001', '_blank', 'noopener,noreferrer');
  };

  const handlePrefectClick = () => {
    window.open('http://localhost:4200', '_blank', 'noopener,noreferrer');
  };

  const buttonStyle = {
    display: 'flex',
    alignItems: 'center',
    gap: '4px',
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
              width: '24px',
              height: '24px',
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
            }}
          >
            NeuroFlow
          </Title>
        </div>

        <Space size="middle">
          <Button
            type="default"
            icon={<ExperimentOutlined />}
            onClick={handleMLFlowClick}
            className="external-button"
            style={buttonStyle}
          >
            MLFlow
            <img
              src={externalLinkIcon}
              alt="External Link"
              className="external-icon"
              style={externalLinkStyle}
            />
          </Button>
          <Button
            type="default"
            icon={<SettingOutlined />}
            onClick={handlePrefectClick}
            className="external-button"
            style={buttonStyle}
          >
            Prefect
            <img
              src={externalLinkIcon}
              alt="External Link"
              className="external-icon"
              style={externalLinkStyle}
            />
          </Button>
        </Space>
      </Header>
    </>
  );
};

export default HeaderComponent;
