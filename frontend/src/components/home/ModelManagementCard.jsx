import React, { useEffect, useState } from 'react';
import {
  Card,
  Button,
  Statistic,
  Tag,
  Space,
  Spin,
  Alert,
  Row,
  Col,
  Typography,
  Divider,
  Badge,
} from 'antd';
import {
  ExperimentOutlined,
  PlayCircleOutlined,
  TrophyOutlined,
  CalendarOutlined,
  ArrowRightOutlined,
} from '@ant-design/icons';
import checkIcon from '../../assets/images/check.svg';
import exclamationIcon from '../../assets/images/exclamation-circle.svg';
import circleDottedIcon from '../../assets/images/circle-dotted.svg';
import { useNavigate } from 'react-router-dom';
import { useAppDispatch, useAppSelector } from '../../hooks/redux';
import { fetchDashboardSummary } from '../../store/slices/dashboardSlice';
import { startTraining } from '../../store/slices/modelSlice';
import { formatDate } from '../../utils/dateFormatter';

const { Title, Text } = Typography;

const ModelManagementCard = () => {
  const navigate = useNavigate();
  const dispatch = useAppDispatch();

  const { summary, loading, error } = useAppSelector(
    (state) => state.dashboard,
  );
  const { trainingLoading, trainingMessage } = useAppSelector(
    (state) => state.model,
  );

  const [selectedModel, setSelectedModel] = useState(null);

  useEffect(() => {
    dispatch(fetchDashboardSummary());
  }, [dispatch]);

  useEffect(() => {
    if (summary.models && summary.models.length > 0) {
      // Select the first model with production version, or just the first one
      const productionModel = summary.models.find(
        (model) => model.production_version,
      );
      setSelectedModel(productionModel || summary.models[0]);
    }
  }, [summary.models]);

  const handleStartTraining = () => {
    dispatch(startTraining());
  };

  const handleGoToModelManagement = () => {
    navigate('/model-management');
  };

  const getStatusBadge = (status) => {
    const statusLower = status?.toLowerCase();

    switch (statusLower) {
      case 'finished':
      case 'completed':
        return (
          <Badge
            color="green"
            text={
              <Space size={4}>
                <img
                  src={checkIcon}
                  alt="Success"
                  style={{
                    width: 14,
                    height: 14,
                    filter: 'invert(42%) sepia(93%) saturate(1352%) hue-rotate(87deg) brightness(119%) contrast(119%)'
                  }}
                />
                <span style={{ color: '#52c41a', fontWeight: 500, fontSize: '16px' }}>Finished</span>
              </Space>
            }
          />
        );
      case 'failed':
      case 'error':
        return (
          <Badge
            color="red"
            text={
              <Space size={4}>
                <img
                  src={exclamationIcon}
                  alt="Failed"
                  style={{
                    width: 14,
                    height: 14,
                    filter: 'invert(27%) sepia(51%) saturate(2878%) hue-rotate(346deg) brightness(104%) contrast(97%)'
                  }}
                />
                <span style={{ color: '#ff4d4f', fontWeight: 500, fontSize: '16px' }}>Failed</span>
              </Space>
            }
          />
        );
      case 'running':
      case 'in_progress':
        return (
          <Badge
            color="orange"
            text={
              <Space size={4}>
                <img
                  src={circleDottedIcon}
                  alt="Running"
                  style={{
                    width: 14,
                    height: 14,
                    filter: 'invert(64%) sepia(88%) saturate(400%) hue-rotate(16deg) brightness(105%) contrast(92%)'
                  }}
                />
                <span style={{ color: '#fa8c16', fontWeight: 500, fontSize: '16px' }}>Running</span>
              </Space>
            }
          />
        );
      default:
        return (
          <Badge
            color="default"
            text={
              <Space size={4}>
                <span style={{ color: '#8c8c8c', fontWeight: 500, fontSize: '16px' }}>{status || 'Unknown'}</span>
              </Space>
            }
          />
        );
    }
  };

  if (loading) {
    return (
      <Card
        title={
          <Space>
            <ExperimentOutlined />
            <span>Model Management</span>
          </Space>
        }
        style={{ height: '100%' }}
        styles={{ body: { height: '100%' } }}
      >
        <div
          style={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            height: 200,
          }}
        >
          <Spin size="large" />
        </div>
      </Card>
    );
  }

  if (error) {
    return (
      <Card
        title={
          <Space>
            <ExperimentOutlined />
            <span>Model Management</span>
          </Space>
        }
        style={{ height: '100%' }}
      >
        <Alert
          message="Error loading model data"
          description={error}
          type="error"
          showIcon
          action={
            <Button
              size="small"
              onClick={() => dispatch(fetchDashboardSummary())}
            >
              Retry
            </Button>
          }
        />
      </Card>
    );
  }

  return (
    <Card
      title={
        <Space>
          <ExperimentOutlined />
          <span>Model Management</span>
        </Space>
      }
      extra={
        <Button
          type="link"
          icon={<ArrowRightOutlined />}
          onClick={handleGoToModelManagement}
        >
          View Details
        </Button>
      }
      style={{ height: '100%' }}
      styles={{ body: { height: 'calc(100% - 57px)' } }}
    >
      {!selectedModel ? (
        <div style={{ textAlign: 'center', padding: '40px 0' }}>
          <Text type="secondary">No models available</Text>
          <br />
          <Button
            type="primary"
            icon={<PlayCircleOutlined />}
            onClick={handleStartTraining}
            loading={trainingLoading}
            style={{ marginTop: 16 }}
          >
            Start First Training
          </Button>
        </div>
      ) : (
        <div
          style={{ height: '100%', display: 'flex', flexDirection: 'column' }}
        >
          {/* Current Model Info */}
          <div style={{ marginBottom: 16 }}>
            <Row gutter={[16, 16]} align="middle">
              <Col span={12}>
                <Space direction="vertical" size={4}>
                  <Text strong>Current Model</Text>
                  <Title level={4} style={{ margin: 0, color: '#1890ff' }}>
                    {selectedModel.model_name}
                  </Title>
                </Space>
              </Col>
              <Col span={12} style={{ textAlign: 'right' }}>
                <Space direction="vertical" size={4} align="end">
                  <Text type="secondary">Production Version</Text>
                  <Tag color="green" icon={<TrophyOutlined />}>
                    v{selectedModel.production_version || 'None'}
                  </Tag>
                </Space>
              </Col>
            </Row>
          </div>

          <Divider style={{ margin: '12px 0' }} />

          {/* Latest Training Info */}
          {selectedModel.latest_training && (
            <div style={{ marginBottom: 16, flex: 1 }}>
              <Text strong style={{ marginBottom: 8, display: 'block' }}>
                Latest Training
              </Text>
              <Row gutter={[16, 8]}>
                <Col span={12}>
                  <Space direction="vertical" size={4}>
                    <Text type="secondary" style={{ fontSize: '12px' }}>
                      Status
                    </Text>
                    {getStatusBadge(selectedModel.latest_training.status)}
                  </Space>
                </Col>
                <Col span={12}>
                  <Space direction="vertical" size={2}>
                    <Text type="secondary" style={{ fontSize: '12px' }}>
                      <CalendarOutlined /> Last Updated
                    </Text>
                    <Text style={{ fontSize: '14px' }}>
                      {formatDate(selectedModel.latest_training.end_time)}
                    </Text>
                  </Space>
                </Col>
              </Row>

              {/* Metrics */}
              {selectedModel.latest_training.metrics && (
                <div style={{ marginTop: 12 }}>
                  <Row gutter={[8, 8]}>
                    {selectedModel.latest_training.metrics.accuracy && (
                      <Col span={8}>
                        <Statistic
                          title="Accuracy"
                          value={(
                            selectedModel.latest_training.metrics.accuracy * 100
                          ).toFixed(1)}
                          suffix="%"
                          precision={1}
                          valueStyle={{ fontSize: '14px' }}
                        />
                      </Col>
                    )}
                    {selectedModel.latest_training.metrics
                      .macro_avg_f1_score && (
                      <Col span={8}>
                        <Statistic
                          title="F1 Score"
                          value={(
                            selectedModel.latest_training.metrics
                              .macro_avg_f1_score * 100
                          ).toFixed(1)}
                          suffix="%"
                          precision={1}
                          valueStyle={{ fontSize: '14px' }}
                        />
                      </Col>
                    )}
                    {selectedModel.latest_training.metrics
                      .macro_avg_precision && (
                      <Col span={8}>
                        <Statistic
                          title="Precision"
                          value={(
                            selectedModel.latest_training.metrics
                              .macro_avg_precision * 100
                          ).toFixed(1)}
                          suffix="%"
                          precision={1}
                          valueStyle={{ fontSize: '14px' }}
                        />
                      </Col>
                    )}
                  </Row>
                </div>
              )}
            </div>
          )}

          {/* Training Message */}
          {trainingMessage && (
            <Alert
              message={trainingMessage}
              type="success"
              showIcon
              closable
              style={{ marginBottom: 16 }}
            />
          )}

          {/* Action Button */}
          <div style={{ marginTop: 'auto', paddingTop: 16 }}>
            <Button
              type="primary"
              icon={<PlayCircleOutlined />}
              onClick={handleStartTraining}
              loading={trainingLoading}
              block
              size="large"
            >
              {trainingLoading ? 'Starting Training...' : 'Start New Training'}
            </Button>
          </div>
        </div>
      )}
    </Card>
  );
};

export default ModelManagementCard;
