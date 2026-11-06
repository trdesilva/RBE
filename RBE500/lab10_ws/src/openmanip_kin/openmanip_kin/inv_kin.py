import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node

from sensor_msgs.msg import JointState

import numpy as np
# note: ROS tries to install out-of-date scipy with apt, so this needs a venv to run
import scipy.spatial.transform as sci_tf


def rot(xyz):
    return sci_tf.Rotation.from_euler("XYZ", [xyz[0], xyz[1], xyz[2]])

def transl(xyz):
    return sci_tf.RigidTransform.from_translation(xyz)

class InverseKinematics(Node):
    def __init__(self):
        super().__init__('inverse_kinematics')
        self.pose_sub = self.create_subscription(PoseStamped, '/end_effector_pose', self.pose_callback, 10)
        self.joint_pub = self.create_publisher(JointState, '/joint_state_ik', 10)
        self.get_logger().info('inverse_kinematics started')

    def pose_callback(self, msg):
        px = msg.pose.position.x
        py = msg.pose.position.y
        pz = msg.pose.position.z
        ex = [1, 0, 0]
        ey = [0, 1, 0]
        ez = [0, 0, 1]
        pquat = np.array((msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w))
        p_as_tf = sci_tf.RigidTransform.from_components((px, py, pz), sci_tf.Rotation.from_quat(pquat))
        p_davenport = p_as_tf.rotation.as_davenport([ez, ey, ex], 'i')
        self.get_logger().info(f'p_as_tf: {p_as_tf}')
        self.get_logger().info(f'p_as_tf davenport angles: {p_davenport}')
        # q1 is only joint that affects z angle
        q1 = p_davenport[0]
        # q2 + q3 + q4 == p_davenport[1]
        z23 = pz - (0.019 + 0.0595) - np.sin(p_davenport[1])*0.126
        x23 = np.sqrt(px**2 + py**2) - np.cos(q1)*0.012 - np.cos(p_davenport[1])*0.126
        xz23 = np.sqrt(x23**2 + z23**2) # xz distance covered by links 2 and 3
        xz2 = np.sqrt(0.024**2 + 0.128**2) # link 2 length
        self.get_logger().info(f'z23: {z23}, x23: {x23}, xz23: {xz23}, xz2: {xz2}')
        q3 = np.pi - np.acos((xz2**2 + 0.124**2 - xz23**2)/(2 * xz2 * 0.128)) # law of cosines
        q2 = (np.pi/2) - (np.asin((xz2 * np.sin(q3))/xz23)) - np.atan(0.024/0.128) # law of sines to get interior angle from previous triangle, subtract from pi/2
        q4 = p_davenport[1] - q3 - q2

        joint_state = JointState()
        joint_state.header.stamp = msg.header.stamp
        joint_state.name.append('q1')
        joint_state.name.append('q2')
        joint_state.name.append('q3')
        joint_state.name.append('q4')

        joint_state.position.append(q1)
        joint_state.position.append(q2)
        joint_state.position.append(q3)
        joint_state.position.append(q4)

        self.get_logger().info(f'q: {q1, q2, q3, q4}')
        self.joint_pub.publish(joint_state)

def main(args=None):
    rclpy.init(args=args)

    node = InverseKinematics()
    rclpy.spin(node)
    node.destroy_node()

    rclpy.shutdown()

if __name__ == '__main__':
    main()