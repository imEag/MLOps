import ModelManagementCard from '../../components/home/ModelManagementCard';
import PredictionsCard from '../../components/home/PredictionsCard';
import { Space } from 'antd';

const Dashboard = () => {
  return (
    <Space direction="vertical" size="large" style={{ display: 'flex' }}>
      <ModelManagementCard />
      <PredictionsCard />
    </Space>
  );
};

export default Dashboard;
