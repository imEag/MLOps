import { InboxOutlined, DeleteOutlined } from '@ant-design/icons';
import {
  Button,
  Typography,
  Upload,
  message,
  Tree,
  Spin,
  Popconfirm,
  Divider,
} from 'antd';
import { useState, useEffect } from 'react';
import { fileService } from '../../services/fileService';
import { modelService } from '../../services/modelService';
import { useIsMobile } from '../../hooks/useBreakpoint';
import PredictionModal from '../../components/predictions/PredictionModal';
import PredictionHistory from '../../components/predictions/PredictionHistory';
import PredictionStats from '../../components/predictions/PredictionStats';

const { Title, Text } = Typography;
const { Dragger } = Upload;
const { DirectoryTree } = Tree;

const Predictions = () => {
  const [fileList, setFileList] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [treeData, setTreeData] = useState([]);
  const [loadingTree, setLoadingTree] = useState(false);
  const [selectedKey, setSelectedKey] = useState(null);
  const [selectedNodeTitle, setSelectedNodeTitle] = useState('');
  const [predicting, setPredicting] = useState(false);
  const [isModalVisible, setIsModalVisible] = useState(false);
  const isMobile = useIsMobile();

  const fetchFiles = async () => {
    setLoadingTree(true);
    try {
      const response = await fileService.getFiles();
      setTreeData(response.data);
    } catch (error) {
      message.error('Failed to fetch file list.');
    } finally {
      setLoadingTree(false);
    }
  };

  useEffect(() => {
    fetchFiles();
  }, []);

  const handleDelete = async () => {
    if (!selectedKey) return;
    try {
      await fileService.deleteFile(selectedKey);
      message.success('File or folder deleted successfully.');
      setSelectedKey(null); // Reset selection
      setSelectedNodeTitle('');
      fetchFiles(); // Refresh file list
    } catch (error) {
      message.error('Failed to delete file or folder.');
    }
  };

  const handleMakePrediction = () => {
    if (!selectedKey) return;
    setIsModalVisible(true);
  };

  const handleConfirmPrediction = async (modelName, dataPath) => {
    setPredicting(true);
    setIsModalVisible(false);
    try {
      await modelService.predict(modelName, dataPath);
      message.success(
        'Prediction started successfully. You can see the results in the prediction history section.',
      );
    } catch (error) {
      message.error('Failed to start prediction.');
    } finally {
      setPredicting(false);
    }
  };

  const handleUpload = async () => {
    if (fileList.length === 0) {
      return;
    }
    setUploading(true);
    try {
      await fileService.uploadFile(fileList[0]);
      setFileList([]);
      message.success('Upload successfully.');
      fetchFiles(); // Refresh file list after upload
    } catch (error) {
      message.error('Upload failed.');
    } finally {
      setUploading(false);
    }
  };

  const props = {
    onRemove: () => {
      setFileList([]);
    },
    beforeUpload: (file) => {
      const isZip =
        file.type === 'application/zip' || file.name.endsWith('.zip');
      if (!isZip) {
        message.error('You can only upload .zip files!');
      } else {
        setFileList([file]);
      }
      return false;
    },
    fileList,
    multiple: false,
    accept: '.zip',
  };

  const handleSelect = (keys, { node }) => {
    if (keys.length > 0) {
      setSelectedKey(keys[0]);
      setSelectedNodeTitle(node.title);
    } else {
      setSelectedKey(null);
      setSelectedNodeTitle('');
    }
  };

  return (
    <div>
      <Title level={2}>New prediction</Title>
      <Text>
        The selected file must be a compressed folder (.zip) in a valid BIDS
        format, it can contain multiple subjects.
      </Text>
      <Dragger {...props} style={{ marginTop: 20 }}>
        <p className="ant-upload-drag-icon">
          <InboxOutlined />
        </p>
        <p className="ant-upload-text">
          Click or drag a .zip file to this area to upload
        </p>
        <p className="ant-upload-hint">
          The compressed file can be quite heavy (even gigabytes).
        </p>
      </Dragger>
      <Button
        type="primary"
        onClick={handleUpload}
        disabled={fileList.length === 0}
        loading={uploading}
        style={{
          marginTop: 16,
        }}
      >
        {uploading ? 'Uploading' : 'Start Upload'}
      </Button>

      <div className="uploaded-files-container">
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '16px',
          }}
        >
          <Title level={3}>Uploaded Files</Title>
          <div
            style={{
              display: 'flex',
              flexDirection: isMobile ? 'column' : 'row',
              gap: '8px',
              alignItems: isMobile ? 'flex-end' : 'center',
            }}
          >
            <Button
              type="primary"
              onClick={handleMakePrediction}
              disabled={!selectedKey || predicting}
              loading={predicting}
              style={{ width: isMobile ? '100%' : 'auto' }}
            >
              Make Prediction
            </Button>
            <Popconfirm
              title={`Are you sure you want to delete "${selectedNodeTitle}"?`}
              onConfirm={handleDelete}
              okText="Yes"
              cancelText="No"
              disabled={!selectedKey}
            >
              <Button
                icon={<DeleteOutlined />}
                type="primary"
                danger
                disabled={!selectedKey}
                style={{ width: isMobile ? '100%' : 'auto' }}
              >
                Delete
              </Button>
            </Popconfirm>
          </div>
        </div>
        {loadingTree ? (
          <Spin />
        ) : (
          <DirectoryTree treeData={treeData} onSelect={handleSelect} />
        )}
      </div>
      <PredictionModal
        visible={isModalVisible}
        onCancel={() => setIsModalVisible(false)}
        onConfirm={handleConfirmPrediction}
        filePath={selectedKey}
      />
      <Divider />
      <Title level={3}>Prediction Statistics</Title>
      <PredictionStats />
      <Divider />
      <Title level={3}>Prediction History</Title>
      <PredictionHistory />
    </div>
  );
};

export default Predictions;
