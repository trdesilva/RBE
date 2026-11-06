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

class ForwardKinematics(Node):
    def __init__(self):
        super().__init__('forward_kinematics')
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/end_effector_pose', 10)
        self.get_logger().info('forward_kinematics started')
        self.q = np.zeros((4,))

    def joint_state_callback(self, msg):
        q1 = msg.position[2]
        q2 = msg.position[3]
        q3 = msg.position[4]
        q4 = msg.position[5]

        if np.linalg.norm(np.array((q1, q2, q3, q4)) - self.q) < 0.001:
            return

        self.q = np.array((q1, q2, q3, q4))
        # numbers taken from the URDF
        tf_mat = (transl([0.012, 0, 0.019]) * rot([0, 0, q1])
                  * transl([0,0,0.0595]) * rot([0, q2, 0])
                  * transl([0.024, 0, 0.128]) * rot([0, q3, 0])
                  * transl([0.124, 0, 0]) * rot([0, q4, 0])
                  * transl([0.126, 0, 0]))
        self.get_logger().info(f'q = {q1, q2, q3, q4}, tf_mat = {tf_mat.as_matrix()}')
        ex = [1, 0, 0]
        ey = [0, 1, 0]
        ez = [0, 0, 1]
        self.get_logger().info(f'tf_mat davenport angles (zyx): {tf_mat.rotation.as_davenport([ez, ey, ex], 'i', degrees=True)}')
        translation = tf_mat.translation
        quat = tf_mat.rotation.as_quat()
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = 'world'
        pose.pose.position.x = translation[0]
        pose.pose.position.y = translation[1]
        pose.pose.position.z = translation[2] # z is .019 higher than RViz says because joint1 is attached wrong in URDF
        pose.pose.orientation.x = quat[0]
        pose.pose.orientation.y = quat[1]
        pose.pose.orientation.z = quat[2]
        pose.pose.orientation.w = quat[3]
        self.get_logger().info(f'pose: {pose}')
        self.pose_pub.publish(msg=pose)

def main(args=None):
    rclpy.init(args=args)

    node = ForwardKinematics()
    rclpy.spin(node)
    node.destroy_node()

    rclpy.shutdown()

if __name__ == '__main__':
    main()