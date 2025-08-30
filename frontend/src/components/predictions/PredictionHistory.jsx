import React, { useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { Table, Spin, Alert, Button, Tooltip, Modal, Descriptions } from 'antd';
import { EyeOutlined, DownOutlined } from '@ant-design/icons';
import { fetchPredictionHistory } from '../../store/slices/predictionSlice';
import { format } from 'date-fns';

const PredictionHistory = () => {
  const dispatch = useDispatch();
  const { predictionHistory, loading, error } = useSelector(
    (state) => state.prediction,
  );
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [selectedPrediction, setSelectedPrediction] = useState(null);

  useEffect(() => {
    dispatch(fetchPredictionHistory({}));
  }, [dispatch]);

  const showModal = (prediction) => {
    setSelectedPrediction(prediction);
    setIsModalVisible(true);
  };

  const handleCancel = () => {
    setIsModalVisible(false);
    setSelectedPrediction(null);
  };

  const expandedRowRender = (record) => {
    const nestedColumns = [
      {
        title: 'Run ID',
        dataIndex: 'run_id',
        key: 'run_id',
        render: (text) => (text ? <Tooltip title={text}>{text.substring(0, 8)}...</Tooltip> : 'N/A'),
      },
      {
        title: 'Start Time',
        dataIndex: 'start_time',
        key: 'start_time',
        render: (text) => (text ? format(new Date(text), 'yyyy-MM-dd HH:mm:ss') : 'N/A'),
      },
      {
        title: 'Status',
        dataIndex: 'status',
        key: 'status',
      },
      {
        title: 'Prediction',
        dataIndex: 'prediction',
        key: 'prediction',
        render: (text) => JSON.stringify(text),
      },
      {
        title: 'Actions',
        key: 'actions',
        render: (text, record) => (
          <Button
            icon={<EyeOutlined />}
            onClick={() => showModal(record)}
          >
            View Inputs
          </Button>
        ),
      },
    ];

    return <Table columns={nestedColumns} dataSource={record.predictions} pagination={false} rowKey="run_id" />;
  };

  const columns = [
    {
      title: 'Batch Run ID',
      dataIndex: 'run_id',
      key: 'run_id',
      render: (text) => (text ? <Tooltip title={text}>{text.substring(0, 8)}...</Tooltip> : 'N/A'),
    },
    {
      title: 'Model',
      dataIndex: 'model_name',
      key: 'model_name',
      render: (text, record) => `${text} (v${record.model_version})`,
    },
    {
      title: 'Batch Time',
      dataIndex: 'start_time',
      key: 'start_time',
      render: (text) => (text ? format(new Date(text), 'yyyy-MM-dd HH:mm:ss') : 'N/A'),
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
    },
    {
      title: '# Records',
      dataIndex: 'num_records',
      key: 'num_records',
      align: 'center',
    },
  ];

  if (loading) {
    return <Spin tip="Loading prediction history..." />;
  }

  if (error) {
    return (
      <Alert
        message="Error"
        description={error}
        type="error"
        showIcon
      />
    );
  }

  return (
    <>
      <Table
        columns={columns}
        dataSource={
          predictionHistory && predictionHistory.predictions
            ? predictionHistory.predictions
            : []
        }
        rowKey="run_id"
        pagination={{ pageSize: 10, total: predictionHistory?.total_count }}
        expandable={{
          expandedRowRender,
          expandIcon: ({ expanded, onExpand, record }) =>
            expanded ? (
              <DownOutlined onClick={e => onExpand(record, e)} />
            ) : (
              <EyeOutlined onClick={e => onExpand(record, e)} />
            ),
        }}
      />
      <Modal
        title="Prediction Input Details"
        visible={isModalVisible}
        onCancel={handleCancel}
        footer={[
          <Button key="back" onClick={handleCancel}>
            Close
          </Button>,
        ]}
      >
        {selectedPrediction && (
          <Descriptions bordered column={1}>
            {Object.entries(selectedPrediction.inputs).map(([key, value]) => (
              <Descriptions.Item label={key} key={key}>
                {String(value)}
              </Descriptions.Item>
            ))}
          </Descriptions>
        )}
      </Modal>
    </>
  );
};

export default PredictionHistory;
