import { InboxOutlined } from '@ant-design/icons';
import { Button, Typography, Upload, message } from 'antd';
import { useState } from 'react';
import { predictionService } from '../../services/predictionService';

const { Title, Text } = Typography;
const { Dragger } = Upload;

const Predictions = () => {
  const [fileList, setFileList] = useState([]);
  const [uploading, setUploading] = useState(false);
  const handleUpload = async () => {
    if (fileList.length === 0) {
      return;
    }
    setUploading(true);
    try {
      await predictionService.uploadFile(fileList[0]);
      setFileList([]);
      message.success('Upload successfully.');
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
      const isZip = file.type === 'application/zip' || file.name.endsWith('.zip');
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
    </div>
  );
};

export default Predictions;
