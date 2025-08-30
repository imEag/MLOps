import React, { useEffect } from 'react';
import {
  Card,
  Button,
  Statistic,
  Spin,
  Alert,
  Row,
  Col,
  Typography,
  Space,
} from 'antd';
import {
  PlayCircleOutlined,
  ArrowRightOutlined,
  HistoryOutlined,
} from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import { useAppDispatch, useAppSelector } from '../../hooks/redux';
import { fetchDashboardSummary } from '../../store/slices/dashboardSlice';

const { Text } = Typography;

const PredictionsCard = () => {
  const navigate = useNavigate();
  const dispatch = useAppDispatch();

  const { summary, loading, error } = useAppSelector(
    (state) => state.dashboard,
  );

  useEffect(() => {
    // Fetch summary if it's not already loaded
    if (!summary.total_models) {
      dispatch(fetchDashboardSummary());
    }
  }, [dispatch, summary.total_models]);

  const handleGoToPredictions = () => {
    navigate('/predictions');
  };

  if (loading && !summary.total_models) {
    return (
      <Card
        title={
          <Space>
            <PlayCircleOutlined />
            <span>Predictions</span>
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
            <PlayCircleOutlined />
            <span>Predictions</span>
          </Space>
        }
        style={{ height: '100%' }}
      >
        <Alert
          message="Error loading prediction data"
          description={error.message || 'Could not fetch summary'}
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
          <PlayCircleOutlined />
          <span>Predictions</span>
        </Space>
      }
      extra={
        <Button
          type="link"
          icon={<ArrowRightOutlined />}
          onClick={handleGoToPredictions}
        >
          Go to Predictions
        </Button>
      }
      style={{ height: '100%' }}
      styles={{ body: { height: 'calc(100% - 57px)' } }}
    >
      <div
        style={{ height: '100%', display: 'flex', flexDirection: 'column' }}
      >
        <div style={{ flex: 1 }}>
          <Text strong style={{ marginBottom: 8, display: 'block' }}>
            Prediction Statistics
          </Text>
          <Row gutter={[16, 16]}>
            <Col span={24}>
              <Statistic
                title="Total Predictions Made"
                value={summary.total_predictions_count}
                prefix={<HistoryOutlined />}
              />
            </Col>
          </Row>
        </div>

        <div style={{ marginTop: 'auto', paddingTop: 16 }}>
          <Button
            type="primary"
            icon={<PlayCircleOutlined />}
            onClick={handleGoToPredictions}
            block
            size="large"
          >
            Run new Prediction
          </Button>
        </div>
      </div>
    </Card>
  );
};

export default PredictionsCard;


