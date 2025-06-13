import React from 'react';
import { Layout, Typography } from 'antd';
import { theme } from 'antd';
import brainIcon from '../../assets/images/brain.svg';
const { Header } = Layout;
const { Title } = Typography;

const HeaderComponent = () => {
  const {
    token: { lineWidth, colorBorder },
  } = theme.useToken();

  return (
    <Header
      style={{
        display: 'flex',
        alignItems: 'center',
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
    </Header>
  );
};

export default HeaderComponent;
