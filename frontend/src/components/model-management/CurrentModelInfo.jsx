import React, { useEffect } from 'react';
import {
  Card,
  Typography,
  Spin,
  Alert,
  Row,
  Col,
  Statistic,
  Tag,
  Space,
} from 'antd';
import { TrophyOutlined, CalendarOutlined, ExperimentOutlined, CheckCircleOutlined } from '@ant-design/icons';
import { useAppDispatch, useAppSelector } from '@/hooks/redux';
import { fetchModelInfo } from '@/store/slices/modelSlice';
import { formatDate } from '@/utils/dateFormatter';
import PropTypes from 'prop-types';

const { Title, Text } = Typography;

const CurrentModelInfo = ({ modelName }) => {
  const dispatch = useAppDispatch();
  const { currentModel, loading, error } = useAppSelector(
    (state) => state.model,
  );

  useEffect(() => {
    if (modelName) {
      dispatch(fetchModelInfo(modelName));
    }
  }, [modelName, dispatch]);

  if (loading) {
    return (
      <Card title="Current Production Model">
        <div style={{ textAlign: 'center', padding: '50px 0' }}>
          <Spin size="large" />
        </div>
      </Card>
    );
  }

  if (error) {
    return (
      <Card title="Current Production Model">
        <Alert
          message="Error fetching model information"
          description={error.detail || error.message || 'An unknown error occurred.'}
          type="error"
          showIcon
        />
      </Card>
    );
  }

  if (!currentModel) {
    return (
      <Card title="Current Production Model">
        <Text>No production model found for &quot;{modelName}&quot;.</Text>
      </Card>
    );
  }

  return (
    <Card
      title={
        <Space>
          <TrophyOutlined />
          <span>Current Production Model</span>
        </Space>
      }
      style={{ height: '100%' }}
    >
      <Row gutter={[16, 24]}>
        <Col span={24}>
          <Title level={4} style={{ color: '#1890ff' }}>
            {currentModel.model_name}
          </Title>
          <Tag color="cyan">Version: {currentModel.version}</Tag>
          <Tag color="blue">Run ID: {currentModel.run_id}</Tag>
        </Col>

        <Col xs={24} sm={12}>
          <Statistic
            title="Status"
            value={currentModel.status}
            prefix={<CheckCircleOutlined />}
            valueStyle={{ color: '#52c41a' }}
          />
        </Col>
        <Col xs={24} sm={12}>
          <Statistic
            title="Last Updated"
            value={formatDate(currentModel.last_updated_timestamp)}
            prefix={<CalendarOutlined />}
          />
        </Col>
        <Col span={24}>
            <Statistic
                title="Experiment Name"
                value={currentModel.experiment_name || 'N/A'}
                prefix={<ExperimentOutlined />}
            />
        </Col>

        {currentModel.metrics && (
          <Col span={24}>
            <Title level={5} style={{ marginTop: 16 }}>
              Production Model Metrics
            </Title>
            <Row gutter={16}>
              {Object.entries(currentModel.metrics).map(([key, value]) => (
                <Col xs={12} sm={8} key={key}>
                  <Statistic
                    title={key.replace(/_/g, ' ').replace('avg', 'Avg.').replace(/\b\w/g, c => c.toUpperCase())}
                    value={typeof value === 'number' ? (value * 100).toFixed(2) + '%' : 'N/A'}
                    precision={2}
                  />
                </Col>
              ))}
            </Row>
          </Col>
        )}
      </Row>
    </Card>
  );
};

CurrentModelInfo.propTypes = {
  modelName: PropTypes.string.isRequired,
};

export default CurrentModelInfo;
