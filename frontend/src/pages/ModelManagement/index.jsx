import React, { useEffect } from 'react';
import {
  Typography,
  Select,
  Row,
  Col,
  Spin,
  Alert,
  Empty,
  Card,
  Button,
  Space,
  App,
} from 'antd';
import { PlayCircleOutlined } from '@ant-design/icons';
import { useAppDispatch, useAppSelector } from '@/hooks/redux';
import {
  fetchAvailableModels,
  setSelectedModel,
  startTraining,
  clearTrainingMessage,
} from '@/store/slices/modelSlice';
import CurrentModelInfo from '@/components/model-management/CurrentModelInfo';
import TrainingHistory from '@/components/model-management/TrainingHistory';
import ExperimentHistory from '@/components/model-management/ExperimentHistory';
import ModelVersions from '@/components/model-management/ModelVersions';
import { useIsMobile } from '@/hooks/useBreakpoint';

const { Title, Text } = Typography;

const ModelManagement = () => {
  const isMobile = useIsMobile();
  const dispatch = useAppDispatch();
  const { notification } = App.useApp();
  const {
    availableModels,
    selectedModelName,
    loading,
    error,
    trainingLoading,
    trainingMessage,
  } = useAppSelector((state) => state.model);

  useEffect(() => {
    dispatch(fetchAvailableModels());
  }, [dispatch]);

  useEffect(() => {
    if (trainingMessage) {
      notification.success({
        message: 'Training Initiated',
        description: trainingMessage,
      });
      dispatch(clearTrainingMessage());
    }
  }, [trainingMessage, notification, dispatch]);

  const handleModelChange = (value) => {
    dispatch(setSelectedModel(value));
  };

  const handleStartTraining = () => {
    dispatch(startTraining());
  };

  return (
    <div>
      <Title level={2} style={{ marginBottom: '24px' }}>
        Model Management
      </Title>

      <Card style={{ marginBottom: '24px' }}>
        <Row align="middle" justify="space-between" style={{ gap: 16 }}>
          <Col>
            <Space align="center" wrap>
              <Text strong>Select Model:</Text>
              <Select
                value={selectedModelName}
                style={{ width: isMobile ? 200 : 250 }}
                onChange={handleModelChange}
                loading={loading}
                placeholder="Select a model"
                disabled={!availableModels.length}
              >
                {availableModels.map((model) => (
                  <Select.Option key={model.name} value={model.name}>
                    {model.name}
                  </Select.Option>
                ))}
              </Select>
            </Space>
          </Col>
          <Col>
            <Button
              type="primary"
              icon={<PlayCircleOutlined />}
              onClick={handleStartTraining}
              loading={trainingLoading}
            >
              {isMobile ? 'Train' : 'Start New Training'}
            </Button>
          </Col>
        </Row>
      </Card>

      {loading && !selectedModelName && (
        <div style={{ textAlign: 'center', padding: '50px 0' }}>
          <Spin size="large" />
        </div>
      )}

      {error && (
        <Alert
          message="Error fetching models"
          description={error}
          type="error"
          showIcon
        />
      )}

      {selectedModelName ? (
        <Row gutter={[24, 24]}>
          <Col xs={24} md={24} lg={24} xl={12}>
            <CurrentModelInfo modelName={selectedModelName} />
          </Col>
          <Col xs={24} md={24} lg={24} xl={12}>
            <ModelVersions modelName={selectedModelName} />
          </Col>
          <Col xs={24}>
            <TrainingHistory modelName={selectedModelName} />
          </Col>
        </Row>
      ) : (
        !loading && (
          <Empty description="No model selected. Please choose a model from the list above." />
        )
      )}
      <Row gutter={[24, 24]} style={{ marginTop: 24 }}>
        <Col xs={24}>
          <ExperimentHistory />
        </Col>
      </Row>
    </div>
  );
};

export default ModelManagement;
