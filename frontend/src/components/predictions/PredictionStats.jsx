import React, { useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { Card, Col, Row, Statistic, Spin, Alert } from 'antd';
import { fetchPredictionStats } from '../../store/slices/predictionSlice';

const PredictionStats = () => {
  const dispatch = useDispatch();
  const { predictionStats: stats, loading, error } = useSelector(
    (state) => state.prediction,
  );

  useEffect(() => {
    dispatch(fetchPredictionStats());
  }, [dispatch]);

  if (loading) {
    return <Spin tip="Loading stats..." />;
  }

  if (error) {
    return <Alert message="Error" description={error} type="error" showIcon />;
  }

  return (
    <Row gutter={16}>
      <Col span={8}>
        <Card>
          <Statistic
            title="Total Predictions"
            value={stats?.total_predictions || 0}
          />
        </Card>
      </Col>
      <Col span={8}>
        <Card>
          <Statistic
            title="Unique Models Used"
            value={stats?.unique_models_used || 0}
          />
        </Card>
      </Col>
      <Col span={8}>
        <Card>
          <Statistic
            title="Most Frequent Model"
            value={stats?.most_frequent_model || 'N/A'}
          />
        </Card>
      </Col>
    </Row>
  );
};

export default PredictionStats;
