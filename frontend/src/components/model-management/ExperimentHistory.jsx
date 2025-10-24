import React, { useEffect, useState } from 'react';
import { Card, Table, Spin, Alert, Button, Space, Tag, Modal, Input, App } from 'antd';
import { HistoryOutlined } from '@ant-design/icons';
import { useAppDispatch, useAppSelector } from '@/hooks/redux';
import {
  fetchExperimentHistory,
  registerModel,
  clearRegistrationStatus,
} from '@/store/slices/modelSlice';
import { formatDate } from '@/utils/dateFormatter';
import ModelMetricsModal from '@/components/model-management/ModelMetricsModal';
import externalLinkIcon from '@/assets/images/external-link.svg';

const ExperimentHistory = () => {
  const { notification } = App.useApp();
  const dispatch = useAppDispatch();
  const {
    experimentHistory,
    loading,
    error,
    registrationStatus,
    registrationMessage,
    registrationError,
  } = useAppSelector((state) => state.model);

  const [isModalVisible, setIsModalVisible] = useState(false);
  const [selectedMetrics, setSelectedMetrics] = useState(null);
  const [isRegisterModalVisible, setIsRegisterModalVisible] = useState(false);
  const [selectedRunId, setSelectedRunId] = useState(null);
  const [newModelName, setNewModelName] = useState('');

  useEffect(() => {
    dispatch(fetchExperimentHistory({ limit: 100 }));
  }, [dispatch]);

  useEffect(() => {
    if (registrationStatus === 'succeeded') {
      notification.success({
        message: 'Model Registered',
        description: registrationMessage,
      });
      dispatch(clearRegistrationStatus());
      setIsRegisterModalVisible(false);
      setNewModelName('');
    } else if (registrationStatus === 'failed') {
      notification.error({
        message: 'Registration Failed',
        description:
          registrationError?.detail ||
          registrationError?.message ||
          'An unknown error occurred.',
      });
      dispatch(clearRegistrationStatus());
    }
  }, [
    registrationStatus,
    registrationMessage,
    registrationError,
    dispatch,
    notification,
  ]);

  const showMetricsModal = (metrics) => {
    setSelectedMetrics(metrics);
    setIsModalVisible(true);
  };

  const handleModalClose = () => {
    setIsModalVisible(false);
    setSelectedMetrics(null);
  };

  const showRegisterModal = (runId) => {
    setSelectedRunId(runId);
    setIsRegisterModalVisible(true);
  };

  const handleRegisterModalClose = () => {
    setIsRegisterModalVisible(false);
    setNewModelName('');
    setSelectedRunId(null);
  };

  const handleRegisterModel = () => {
    if (selectedRunId && newModelName) {
      dispatch(registerModel({ runId: selectedRunId, modelName: newModelName }));
    }
  };

  const getStatusTag = (status) => {
    switch (status?.toUpperCase()) {
      case 'FINISHED':
        return <Tag color="success">Finished</Tag>;
      case 'FAILED':
        return <Tag color="error">Failed</Tag>;
      case 'RUNNING':
        return <Tag color="processing">Running</Tag>;
      default:
        return <Tag>{status}</Tag>;
    }
  };

  const formatDurationFromTimestamps = (startDate, endDate) => {
    if (!startDate || !endDate) return 'N/A';

    const start = new Date(startDate).getTime();
    const end = new Date(endDate).getTime();

    const duration = (end - start) / 1000; // seconds
    if (duration < 0) return 'N/A';
    if (duration < 60) return `${duration.toFixed(1)}s`;
    const minutes = Math.floor(duration / 60);
    const seconds = Math.round(duration % 60);
    return `${minutes}m ${seconds}s`;
  };

  const externalLinkStyle = {
    width: '12px',
    height: '12px',
    marginLeft: '4px',
    transition: 'filter 0.2s ease',
  };

  const expandedRowRender = (record) => {
    const childColumns = [
      {
        title: 'Task Name',
        dataIndex: 'run_name',
        key: 'run_name',
        render: (text) => text || 'N/A',
      },
      {
        title: 'Status',
        dataIndex: 'status',
        key: 'status',
        render: getStatusTag,
      },
      {
        title: 'Duration',
        key: 'duration',
        render: (_, childRecord) =>
          formatDurationFromTimestamps(
            childRecord.start_time,
            childRecord.end_time,
          ),
      },
      {
        title: 'Metrics',
        key: 'metrics',
        render: (_, childRecord) => (
          <Button
            type="link"
            onClick={() => showMetricsModal(childRecord.metrics)}
            disabled={
              !childRecord.metrics ||
              Object.values(childRecord.metrics).every((v) => v === null)
            }
          >
            View Metrics
          </Button>
        ),
      },
      {
        title: 'Actions',
        key: 'actions',
        render: (_, childRecord) => {
          const hasModelOutput =
            childRecord.tags?.['mlflow.log-model.history'];
          const isFinished = childRecord.status?.toUpperCase() === 'FINISHED';

          return (
            <Space>
              <Button
                type="link"
                className="external-button"
                style={{ display: 'flex', alignItems: 'center' }}
                onClick={() =>
                  window.open(
                    `http://localhost:5001/#/experiments/${childRecord.experiment_id}/runs/${childRecord.run_id}`,
                    '_blank',
                    'noopener,noreferrer',
                  )
                }
              >
                View in MLflow
                <img
                  src={externalLinkIcon}
                  alt="External Link"
                  className="external-icon"
                  style={externalLinkStyle}
                />
              </Button>
              {isFinished && hasModelOutput && (
                <Button
                  type="primary"
                  ghost
                  size="small"
                  onClick={() => showRegisterModal(childRecord.run_id)}
                >
                  Register Model
                </Button>
              )}
            </Space>
          );
        },
      },
    ];

    return (
      <Table
        columns={childColumns}
        dataSource={record.child_runs}
        rowKey="run_id"
        pagination={false}
      />
    );
  };

  const columns = [
    {
      key: 'expand',
      width: 50,
      render: () => null,
      fixed: 'left',
    },
    {
      title: 'Experiment Name',
      dataIndex: 'experiment_name',
      key: 'experiment_name',
      render: (text) => text || 'N/A',
    },
    {
      title: 'Flow Run Name',
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
      render: getStatusTag,
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record) => (
        <Button
          type="link"
          className="external-button"
          style={{ display: 'flex', alignItems: 'center' }}
          onClick={() =>
            window.open(
              `http://localhost:5001/#/experiments/${record.experiment_id}/runs/${record.run_id}`,
              '_blank',
              'noopener,noreferrer',
            )
          }
        >
          View in MLflow
          <img
            src={externalLinkIcon}
            alt="External Link"
            className="external-icon"
            style={externalLinkStyle}
          />
        </Button>
      ),
    },
  ];

  return (
    <>
      <style>
        {`
          .external-button:hover .external-icon {
            filter: invert(24%) sepia(100%) saturate(1352%) hue-rotate(204deg) brightness(95%) contrast(106%);
          }
        `}
      </style>
      <Card
        title={
          <Space>
            <HistoryOutlined />
            <span>Experiment History</span>
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
            message="Error fetching experiment history"
            description={
              error.detail || error.message || 'An unknown error occurred.'
            }
            type="error"
            showIcon
          />
        ) : (
          <Table
            columns={columns}
            dataSource={experimentHistory}
            rowKey="run_id"
            pagination={{ pageSize: 5 }}
            scroll={{ x: 'max-content' }}
            expandable={{
              expandedRowRender,
              rowExpandable: (record) =>
                record.child_runs && record.child_runs.length > 0,
              expandIconColumnIndex: 0,
            }}
          />
        )}

        {selectedMetrics && (
          <ModelMetricsModal
            visible={isModalVisible}
            onClose={handleModalClose}
            metrics={selectedMetrics}
          />
        )}

        <Modal
          title="Register New Model"
          visible={isRegisterModalVisible}
          onCancel={handleRegisterModalClose}
          footer={[
            <Button key="back" onClick={handleRegisterModalClose}>
              Cancel
            </Button>,
            <Button
              key="submit"
              type="primary"
              loading={registrationStatus === 'loading'}
              onClick={handleRegisterModel}
              disabled={!newModelName}
            >
              Register
            </Button>,
          ]}
        >
          <p>
            Enter a name for the new model. If the name already exists, a new
            version will be created.
          </p>
          <Input
            placeholder="e.g., Iris-Classifier-V1"
            value={newModelName}
            onChange={(e) => setNewModelName(e.target.value)}
          />
        </Modal>
      </Card>
    </>
  );
};

export default ExperimentHistory;
