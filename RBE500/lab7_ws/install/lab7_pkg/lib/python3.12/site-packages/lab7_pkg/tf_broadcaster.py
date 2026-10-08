import math

from geometry_msgs.msg import TransformStamped

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException

def quaternion_from_euler(ai, aj, ak):
    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    q = np.empty((4, ))
    q[0] = cj*sc - sj*cs
    q[1] = cj*ss + sj*cc
    q[2] = cj*cs - sj*sc
    q[3] = cj*cc + sj*ss

    return q


class LabTfBroadcaster(Node):

    def __init__(self):
        super().__init__('lab_tf_broadcaster')

        # Initialize the transform broadcasters
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # frames
        self.base = TransformStamped()
        self.base.header.stamp = self.get_clock().now().to_msg()
        self.base.header.frame_id = 'world' # world frame seems to be implicitly created
        self.base.child_frame_id = 'base'
        self.base.transform.translation.x = 1.0
        self.base.transform.translation.y = 1.0
        self.base.transform.translation.z = 1.0

        self.frame_a = TransformStamped()
        self.frame_a.header.stamp = self.get_clock().now().to_msg()
        self.frame_a.header.frame_id = 'base'
        self.frame_a.child_frame_id = 'frame_A'

        self.frame_b = TransformStamped()
        self.frame_b.header.stamp = self.get_clock().now().to_msg()
        self.frame_b.header.frame_id = 'base'
        self.frame_b.child_frame_id = 'frame_B'
        self.frame_b.transform.translation.x = 2.0

    def timer_callback(self):
        # Send the transformation
        tick_time = self.get_clock().now().to_msg()
        time_delta = float(tick_time.sec - self.base.header.stamp.sec) + (tick_time.nanosec - self.base.header.stamp.nanosec)/1000000000

        self.frame_a.header.stamp = tick_time
        self.frame_a.transform.translation.x = time_delta

        self.frame_b.header.stamp = tick_time
        self.frame_b.transform.translation.x = 1.0 + np.cos(time_delta)
        self.frame_b.transform.translation.y = np.sin(time_delta)
        q = quaternion_from_euler(0, 0, np.arctan2(np.sin(time_delta), np.cos(time_delta)))
        self.frame_b.transform.rotation.x = q[0]
        self.frame_b.transform.rotation.y = q[1]
        self.frame_b.transform.rotation.z = q[2]
        self.frame_b.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(self.frame_a)
        self.tf_broadcaster.sendTransform(self.frame_b)

        try:
            self.get_logger().info(f'A to B transform: {self.tf_buffer.lookup_transform('frame_B', 'frame_A', Time(seconds = 0), Duration(seconds=0.1)).transform}')
        except TransformException as e:
            self.get_logger().warn('failed to get transform')



def main():
    rclpy.init()
    node = LabTfBroadcaster()
    node.get_logger().info('Starting up lab_tf_broadcaster')
    node.static_tf_broadcaster.sendTransform(node.base)
    node.tf_broadcaster.sendTransform(node.frame_a)
    node.tf_broadcaster.sendTransform(node.frame_b)
    try:
        node.tf_buffer.lookup_transform('frame_B', 'frame_A', Time(seconds=0), Duration(seconds=5))
    except TransformException as e:
        node.get_logger().warn('tf_buffer warming timed out')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()