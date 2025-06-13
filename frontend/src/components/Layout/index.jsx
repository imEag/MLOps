import { Outlet } from 'react-router-dom';
import React from 'react';
import { Breadcrumb, Layout, theme } from 'antd';
import { useLocation } from 'react-router-dom';
import HeaderComponent from '../Header';
import SiderComponent from '../Sider';
// import FooterComponent from '../Footer'; // Uncomment if you want to use the footer

const { Content } = Layout;

const LayoutComponent = () => {
  const {
    token: { colorBgContainer, lineWidth, colorBorder },
  } = theme.useToken();

  const location = useLocation();

  // Function to generate breadcrumb items based on current path
  const getBreadcrumbItems = () => {
    const pathname = location.pathname;
    const pathSegments = pathname
      .split('/')
      .filter((segment) => segment !== '');

    const breadcrumbItems = [{ title: 'Home' }];

    // Map path segments to readable titles
    const pathTitleMap = {
      dashboard: 'Dashboard',
      'model-management': 'Model Management',
      predictions: 'Predictions',
    };

    pathSegments.forEach((segment) => {
      const title =
        pathTitleMap[segment] ||
        segment.charAt(0).toUpperCase() + segment.slice(1);
      breadcrumbItems.push({ title });
    });

    return breadcrumbItems;
  };

  return (
    <Layout style={{ minHeight: '100vh', height: '100vh' }}>
      <HeaderComponent />
      <Layout style={{ height: 'calc(100vh - 64px)' }}>
        <SiderComponent />
        <Layout
          style={{
            padding: '0 24px 24px',
            height: '100%',
            overflow: 'auto',
            backgroundColor: colorBgContainer,
          }}
        >
          <Breadcrumb
            items={getBreadcrumbItems()}
            style={{ margin: '16px 0' }}
          />
          <Content
            style={{
              padding: 24,
              margin: 0,
              flex: 1,
              overflow: 'auto',
              backgroundColor: colorBgContainer,
              borderTop: `${lineWidth}px solid ${colorBorder}`,
              borderRadius: 0,
            }}
          >
            <Outlet />
          </Content>
        </Layout>
      </Layout>
    </Layout>
  );
};

export default LayoutComponent;
