import React, { useEffect } from 'react';
import PropTypes from 'prop-types';
import { Card, Table, Spin, Alert, Button, Space, Tag, App } from 'antd';
import { BranchesOutlined, TrophyOutlined } from '@ant-design/icons';
import { useAppDispatch, useAppSelector } from '@/hooks/redux';
import {
  fetchModelVersions,
  promoteModel,
  clearTrainingMessage,
} from '@/store/slices/modelSlice';
import { formatDate } from '@/utils/dateFormatter';
import { useIsMobile } from '@/hooks/useBreakpoint';

const ModelVersions = ({ modelName }) => {
  const isMobile = useIsMobile();
  const { modal, notification } = App.useApp();
  const dispatch = useAppDispatch();
  const { modelVersions, loading, error, trainingMessage } = useAppSelector(
    (state) => state.model,
  );

  useEffect(() => {
    if (modelName) {
      dispatch(fetchModelVersions(modelName));
    }
  }, [modelName, dispatch]);

  useEffect(() => {
    if (trainingMessage) {
      notification.success({
        message: 'Promotion Successful',
        description: trainingMessage,
      });
      dispatch(clearTrainingMessage());
    }
  }, [trainingMessage, notification, dispatch]);

  const handlePromote = (version) => {
    modal.confirm({
      title: 'Promote Model to Production',
      content: `Are you sure you want to promote version ${version} to production?`,
      okText: 'Promote',
      onOk: () => {
        dispatch(promoteModel({ modelName, version }));
      },
    });
  };

  const columns = [
    {
      title: 'Version',
      dataIndex: 'version',
      key: 'version',
      sorter: (a, b) => a.version - b.version,
    },
    {
      title: 'Last Updated',
      dataIndex: 'last_updated_timestamp',
      key: 'last_updated_timestamp',
      render: (text) => formatDate(text),
    },
    {
      title: 'Action',
      key: 'action',
      render: (_, record) => (
        <Button
          type="primary"
          icon={<TrophyOutlined />}
          onClick={() => handlePromote(record.version)}
          disabled={record.current_stage === 'Production'}
          style={
            isMobile
              ? {
                  fontSize: '12px',
                  height: 'auto',
                  padding: '4px 8px',
                  lineHeight: '1.4',
                }
              : {}
          }
        >
          {isMobile ? (
            <>
              Promote
              <br />
              to Production
            </>
          ) : (
            'Promote to Production'
          )}
        </Button>
      ),
    },
  ];

  return (
    <Card
      title={
        <Space>
          <BranchesOutlined />
          <span>Model Versions</span>
        </Space>
      }
    >
      {loading ? (
        <div style={{ textAlign: 'center', padding: '50px 0' }}>
          <Spin />
        </div>
      ) : error ? (
        <Alert
          message="Error fetching model versions"
          description={
            error.detail || error.message || 'An unknown error occurred.'
          }
          type="error"
          showIcon
        />
      ) : (
        <Table
          columns={columns}
          dataSource={modelVersions}
          rowKey="version"
          pagination={{ pageSize: 5 }}
          scroll={{ x: 'max-content' }}
        />
      )}
    </Card>
  );
};

ModelVersions.propTypes = {
  modelName: PropTypes.string.isRequired,
};

export default ModelVersions;
