import React, { useState, useEffect } from 'react';
import { Modal, Button, Select, Typography, Alert } from 'antd';
import { useDispatch, useSelector } from 'react-redux';
import { fetchAvailableModels } from '../../store/slices/modelSlice';

const { Text } = Typography;

const PredictionModal = ({ visible, onCancel, onConfirm, filePath }) => {
  const [selectedModel, setSelectedModel] = useState(null);
  const dispatch = useDispatch();
  const { availableModels, loading } = useSelector((state) => state.models);

  useEffect(() => {
    if (visible) {
      dispatch(fetchAvailableModels());
    }
  }, [dispatch, visible]);

  const handleConfirm = () => {
    if (selectedModel) {
      onConfirm(selectedModel, filePath);
    }
  };

  return (
    <Modal
      title="Confirm Prediction"
      visible={visible}
      onCancel={onCancel}
      footer={[
        <Button key="back" onClick={onCancel}>
          Cancel
        </Button>,
        <Button
          key="submit"
          type="primary"
          loading={loading}
          onClick={handleConfirm}
          disabled={!selectedModel}
        >
          Confirm
        </Button>,
      ]}
    >
      <Text>Are you sure you want to make a prediction with this dataset?</Text>
      <br />
      <br />
      <Select
        style={{ width: '100%' }}
        placeholder="Select a model"
        onChange={(value) => setSelectedModel(value)}
        loading={loading}
        value={selectedModel}
      >
        {availableModels.map((model) => (
          <Select.Option key={model.name} value={model.name}>
            {model.name}
          </Select.Option>
        ))}
      </Select>
      <br />
      <br />
      <Alert
        message="Note"
        description="The prediction can take several minutes. The results will be shown in the prediction history section."
        type="info"
        showIcon
      />
    </Modal>
  );
};

export default PredictionModal;
