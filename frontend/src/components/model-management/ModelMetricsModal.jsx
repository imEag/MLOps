import React from 'react';
import PropTypes from 'prop-types';
import { Modal, Row, Col, Statistic, Empty } from 'antd';

const ModelMetricsModal = ({ visible, onClose, metrics }) => {
  return (
    <Modal
      title="Training Metrics"
      visible={visible}
      onCancel={onClose}
      footer={null}
      width={600}
    >
      {metrics ? (
        <Row gutter={[16, 16]}>
          {Object.entries(metrics).map(([key, value]) => (
            <Col xs={12} sm={8} key={key}>
              <Statistic
                title={key.replace(/_/g, ' ').replace('avg', 'Avg.').replace(/\b\w/g, c => c.toUpperCase())}
                value={typeof value === 'number' ? (value * 100).toFixed(2) + '%' : 'N/A'}
                precision={2}
              />
            </Col>
          ))}
        </Row>
      ) : (
        <Empty description="No metrics available for this run." />
      )}
    </Modal>
  );
};

ModelMetricsModal.propTypes = {
  visible: PropTypes.bool.isRequired,
  onClose: PropTypes.func.isRequired,
  metrics: PropTypes.object,
};

export default ModelMetricsModal;
