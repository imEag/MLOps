import React, { useEffect, useState } from 'react';
import { Card, Table, Spin, Alert, Button, Space, Tag } from 'antd';
import { HistoryOutlined } from '@ant-design/icons';
import { useAppDispatch, useAppSelector } from '@/hooks/redux';
import { fetchExperimentHistory } from '@/store/slices/modelSlice';
import { formatDate } from '@/utils/dateFormatter';
import ModelMetricsModal from '@/components/model-management/ModelMetricsModal';
import externalLinkIcon from '@/assets/images/external-link.svg';

const ExperimentHistory = () => {
  const dispatch = useAppDispatch();
  const { experimentHistory, loading, error } = useAppSelector(
    (state) => state.model,
  );

  const [isModalVisible, setIsModalVisible] = useState(false);
  const [selectedMetrics, setSelectedMetrics] = useState(null);

  useEffect(() => {
    dispatch(fetchExperimentHistory({ limit: 10 }));
  }, [dispatch]);

  const showMetricsModal = (metrics) => {
    setSelectedMetrics(metrics);
    setIsModalVisible(true);
  };

  const handleModalClose = () => {
    setIsModalVisible(false);
    setSelectedMetrics(null);
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

  const formatDurationFromTimestamps = (start, end) => {
    if (!start || !end) return 'N/A';
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
        render: (_, childRecord) => (
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
        ),
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
      </Card>
    </>
  );
};

export default ExperimentHistory;
