#!/usr/bin/env python
import rospy
from sensor_msgs.msg import PointCloud2
from tensorflow.keras.models import load_model
from data import pre_processing_raw_data
from std_msgs.msg import Header
from numpy2cloud import PointCloudProcessor as pc
from rosmsg2npy import process_point_cloud
import sensor_msgs.point_cloud2 as pc2


class LidarSuperResolution:
    def __init__(self):
        rospy.init_node('lidar_super_resolution_node', anonymous=True)
        self.subscriber = rospy.Subscriber('/velodyne_points', PointCloud2, self.callback, queue_size=1)
        self.publisher = rospy.Publisher('/output_lidar_superres', PointCloud2, queue_size=1)
        
        # Load the trained model with its weights
        self.model = load_model('/root/Documents/SuperResolution/weights/weights.h5')
        self.pc_processor = pc()
    def callback(self, data):
        # In your callback function
        # Step 1: Convert PointCloud2 ROS message to numpy array
        input_data = process_point_cloud(data)

        # Check input data shape
        print(f"Shape of input_data: {input_data.shape}")

        # Reshape to match the model's expected shape
        input_data = input_data.reshape(1, 16, 1024, 1) 

        # Check the new shape
        print(f"New Shape of input_data: {input_data.shape}")

        processed_data = self.model.predict(input_data)

        self.pc_processor.publishPointCloud(
                thisImage=processed_data,
                pubHandle=self.publisher,
                timeStamp=rospy.Time.now(),
                height=0  # Assuming height is 0
            )
if __name__ == '__main__':
    try:
        processor = LidarSuperResolution()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
