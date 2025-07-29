import React, { useEffect, useState } from 'react';
import PropTypes from 'prop-types';
import { Card, Table, Spin, Alert, Button, Space } from 'antd';
import { HistoryOutlined } from '@ant-design/icons';
import { useAppDispatch, useAppSelector } from '@/hooks/redux';
import { fetchTrainingHistory } from '@/store/slices/modelSlice';
import { formatDate } from '@/utils/dateFormatter';
import ModelMetricsModal from '@/components/model-management/ModelMetricsModal';

const TrainingHistory = ({ modelName }) => {
  const dispatch = useAppDispatch();
  const { trainingHistory, loading, error } = useAppSelector(
    (state) => state.model,
  );

  const [isModalVisible, setIsModalVisible] = useState(false);
  const [selectedMetrics, setSelectedMetrics] = useState(null);

  useEffect(() => {
    if (modelName) {
      dispatch(fetchTrainingHistory({ modelName, limit: 10 }));
    }
  }, [modelName, dispatch]);

  const showMetricsModal = (metrics) => {
    setSelectedMetrics(metrics);
    setIsModalVisible(true);
  };

  const handleModalClose = () => {
    setIsModalVisible(false);
    setSelectedMetrics(null);
  };

  const columns = [
    {
      title: 'Run Name',
      dataIndex: 'run_name',
      key: 'run_name',
      render: (text) => text || 'N/A',
    },
    {
      title: 'End Time',
      dataIndex: 'end_time',
      key: 'end_time',
      render: (text) => formatDate(text),
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
    },
    {
      title: 'Version',
      dataIndex: 'model_version',
      key: 'model_version',
      render: (text) => text || 'N/A',
    },
    {
      title: 'Metrics',
      key: 'metrics',
      render: (_, record) => (
        <Button
          type="link"
          onClick={() => showMetricsModal(record.metrics)}
          disabled={!record.metrics}
        >
          View Metrics
        </Button>
      ),
    },
  ];

  return (
    <Card
      title={
        <Space>
          <HistoryOutlined />
          <span>Training History</span>
        </Space>
      }
      style={{ height: '100%' }}
    >
      {loading ? (
        <div style={{ textAlign: 'center', padding: '50px 0' }}>
          <Spin size="large" />
        </div>
      ) : error ? (
        <Alert
          message="Error fetching training history"
          description={error.detail || error.message || 'An unknown error occurred.'}
          type="error"
          showIcon
        />
      ) : (
        <Table
          columns={columns}
          dataSource={trainingHistory}
          rowKey="run_id"
          pagination={{ pageSize: 5 }}
          scroll={{ x: 'max-content' }}
        />
      )}

      {selectedMetrics && (
        <ModelMetricsModal
          visible={isModalVisible}
          onClose={handleModalClose}
          metrics={selectedMetrics}
        />
      )}
    </Card>
  );
};

TrainingHistory.propTypes = {
  modelName: PropTypes.string.isRequired,
};

export default TrainingHistory;
