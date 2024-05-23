#!/usr/bin/env python
import os
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2

# Global Variables
image_rows_full = 16
image_cols = 1024

# Ouster OS1-64 (gen1)
ang_res_x = 360.0 / float(image_cols)  # horizontal resolution
ang_res_y = 33.2 / float(image_rows_full - 1)  # vertical resolution
ang_start_y = 16.6  # bottom beam angle
max_range = 80.0
min_range = 2.0


def ros_msg_to_range_image(msg):
    """
    Convert ROS PointCloud2 message to a range image.
    """
    # Convert ROS PointCloud2 message to numpy array
    points = pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z"))
    points_array = np.array(list(points), dtype=np.float32)

    # Project points to range image
    range_image = np.zeros((image_rows_full, image_cols), dtype=np.float32)

    for point in points_array:
        x, y, z = point
        # Find row id
        vertical_angle = np.arctan2(z, np.sqrt(x * x + y * y)) * 180.0 / np.pi
        relative_vertical_angle = vertical_angle + ang_start_y
        row_id = np.int_(np.round_(relative_vertical_angle / ang_res_y))

        # Find column id
        horizontal_angle = np.arctan2(x, y) * 180.0 / np.pi
        col_id = np.int_((horizontal_angle + 90.0) / ang_res_x + image_cols / 2)
        col_id = col_id % image_cols  # Ensure col_id stays within [0, image_cols - 1]

        if col_id >= image_cols:
            col_id -= image_cols

        # Filter range
        this_range = np.sqrt(x * x + y * y + z * z)
        if this_range > max_range or this_range < min_range:
            continue

        # Save range info to range image
        if 0 <= row_id < image_rows_full and 0 <= col_id < image_cols:
            range_image[row_id, col_id] = this_range

    return range_image


def process_ros_messages(topic):
    """
    Process ROS PointCloud2 messages in real-time.
    """
    rospy.init_node('range_image_processor', anonymous=True)
    rospy.Subscriber(f"{topic}", pc2.PointCloud2, callback=process_point_cloud)
    rospy.spin()


def process_point_cloud(msg):
    """
    Process a single ROS PointCloud2 message and convert it to a range image.
    """
    range_image = ros_msg_to_range_image(msg)
    return range_image


if __name__ == '__main__':
    range_image = process_ros_messages()
