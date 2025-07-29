import React from 'react';
import { Layout } from 'antd';
const { Footer } = Layout;

const FooterComponent = () => {
  return (
    <Footer style={{ textAlign: 'center' }}>
      MLOps Dashboard ©{new Date().getFullYear()} Created with Ant Design
    </Footer>
  );
};

export default FooterComponent;
