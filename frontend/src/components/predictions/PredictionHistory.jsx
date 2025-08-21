import React, { useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { Table, Spin, Alert, Button, Tooltip } from 'antd';
import { EyeOutlined } from '@ant-design/icons';
import { fetchPredictionHistory } from '../../store/slices/predictionSlice';
import { format } from 'date-fns';

const PredictionHistory = () => {
  const dispatch = useDispatch();
  const { predictionHistory, loading, error } = useSelector(
    (state) => state.prediction,
  );

  useEffect(() => {
    dispatch(fetchPredictionHistory({}));
  }, [dispatch]);

  const columns = [
    {
      title: 'Run ID',
      dataIndex: 'run_id',
      key: 'run_id',
      render: (text) => (text ? text.substring(0, 8) + '...' : 'N/A'),
    },
    {
      title: 'Model Name',
      dataIndex: 'model_name',
      key: 'model_name',
    },
    {
      title: 'Model Version',
      dataIndex: 'model_version',
      key: 'model_version',
    },
    {
      title: 'Prediction Time',
      dataIndex: 'prediction_time',
      key: 'prediction_time',
      render: (text) => {
        if (!text) return 'N/A';
        const date = new Date(text);
        if (isNaN(date.getTime())) {
          return 'Invalid Date';
        }
        return format(date, 'yyyy-MM-dd HH:mm:ss');
      },
    },
    {
      title: 'Input Data',
      dataIndex: 'input_data_path',
      key: 'input_data_path',
      render: (text) =>
        text ? (
          <Tooltip title={text}>
            <span>{text.split('/').pop()}</span>
          </Tooltip>
        ) : (
          'N/A'
        ),
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (text, record) => (
        <Button
          icon={<EyeOutlined />}
          onClick={() => console.log('View details for', record.run_id)}
        >
          View Results
        </Button>
      ),
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
    <Table
      columns={columns}
      dataSource={
        predictionHistory && predictionHistory.predictions
          ? predictionHistory.predictions
          : []
      }
      rowKey="run_id"
      pagination={{ pageSize: 10 }}
    />
  );
};

export default PredictionHistory;
