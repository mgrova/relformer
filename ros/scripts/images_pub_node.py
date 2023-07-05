#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Pose, PoseArray
from sensor_msgs.msg   import Image
from cv_bridge import CvBridge

import os
import cv2 as cv
import numpy as np

def read_image_from_file(file_path):
    cv_image = cv.imread(file_path)
    cv_image_rgb = cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)
    bridge = CvBridge()
    image_msg = bridge.cv2_to_imgmsg(cv_image_rgb)
    image_msg.encoding = "rgb8"
    return image_msg

def images_to_infer_publisher():
    rospy.init_node('images_to_infer_publisher', anonymous=False)
    
    query_image_path  = rospy.get_param('~query_image_path', default="")
    query_image_topic = rospy.get_param('~query_image_topic', default="query_image")

    ref_image_path  = rospy.get_param('~ref_image_path', default="")
    ref_image_topic = rospy.get_param('~ref_image_topic', default="ref_image")

    # Convert image from file to sensor_msgs/Image message
    query_image_msg = read_image_from_file(query_image_path)
    query_image_msg.header.stamp    = rospy.Time.now()
    query_image_msg.header.frame_id = "camera_frame_id"
    
    ref_image_msg = read_image_from_file(ref_image_path)
    ref_image_msg.header.stamp    = rospy.Time.now()
    ref_image_msg.header.frame_id = "camera_frame_id"

    query_image_pub = rospy.Publisher(query_image_topic, Image, queue_size=1, latch=True)
    ref_image_pub   = rospy.Publisher(ref_image_topic, Image, queue_size=1, latch=True)
    
    # Only publish data once
    query_image_pub.publish(query_image_msg)
    ref_image_pub.publish(ref_image_msg)

    while not rospy.is_shutdown():
        rospy.spin()

if __name__ == '__main__':
    try:
        images_to_infer_publisher()
    except rospy.ROSInterruptException:
        pass