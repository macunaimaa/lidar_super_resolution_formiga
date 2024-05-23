#!/usr/bin/env python
import rospy
from sensor_msgs.msg import PointCloud2
from tensorflow.keras.models import load_model
from data import load_data_from_ros, preprocess_lidar_data
from numpy2cloud import array_to_pointcloud2

class LidarSuperResolution:
    def __init__(self):
        rospy.init_node('lidar_super_resolution_node', anonymous=True)
        self.subscriber = rospy.Subscriber('/input_lidar_topic', PointCloud2, self.callback, queue_size=1)
        self.publisher = rospy.Publisher('/output_lidar_topic', PointCloud2, queue_size=1)
        
        # Load the trained model with its weights
        self.model = load_model('path_to_your_model.h5')

    def callback(self, data):
        try:
            # Step 1: Convert PointCloud2 ROS message to numpy array
            raw_data = load_data_from_ros(data)
            
            # Step 2: Preprocess the data for model input
            input_data = preprocess_lidar_data(raw_data)
            
            # Step 3: Perform inference using the loaded model
            processed_data = self.model.predict(input_data)
            
            # Step 4: Convert processed data back to PointCloud2 message
            output_data = array_to_pointcloud2(processed_data, frame_id="lidar_frame")
            
            # Step 5: Publish the processed data
            self.publisher.publish(output_data)
        except Exception as e:
            rospy.logerr(f"Error processing LiDAR data: {e}")

if __name__ == '__main__':
    try:
        processor = LidarSuperResolution()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
