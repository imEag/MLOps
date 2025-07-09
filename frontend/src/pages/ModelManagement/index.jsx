import React, { useEffect } from 'react';
import { Typography, Select, Row, Col, Spin, Alert, Empty, Card } from 'antd';
import { useAppDispatch, useAppSelector } from '@/hooks/redux';
import {
  fetchAvailableModels,
  setSelectedModel,
} from '@/store/slices/modelSlice';
import CurrentModelInfo from '@/components/model-management/CurrentModelInfo';
import TrainingHistory from '@/components/model-management/TrainingHistory';
import ModelVersions from '@/components/model-management/ModelVersions';
import { useIsMobile } from '@/hooks/useBreakpoint';

const { Title, Text } = Typography;

const ModelManagement = () => {
  const isMobile = useIsMobile();
  const dispatch = useAppDispatch();
  const { availableModels, selectedModelName, loading, error } = useAppSelector(
    (state) => state.model,
  );

  useEffect(() => {
    dispatch(fetchAvailableModels());
  }, [dispatch]);

  const handleModelChange = (value) => {
    dispatch(setSelectedModel(value));
  };

  return (
    <div>
      <Title level={2} style={{ marginBottom: '24px' }}>
        Model Management
      </Title>

      <Card style={{ marginBottom: '24px' }}>
        <Row align="middle" gutter={16}>
          <Col>
            <Text strong>Select a Model to Manage:</Text>
          </Col>
          <Col>
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
    </div>
  );
};

export default ModelManagement;
