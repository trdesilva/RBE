import rclpy
from rclpy.node import Node
import math
import random

from control_msgs.msg import MultiDOFStateStamped, MultiDOFCommand

class SetpointSetter(Node):
    def __init__(self):
        super().__init__("setpoint_setter")
        self.controller_listener = self.create_subscription(MultiDOFStateStamped, '/rrbot_pid_controller/controller_state',
                                                            self.controller_callback, 10)
        self.setpoint_publisher = self.create_publisher(MultiDOFCommand, '/rrbot_pid_controller/reference', 10)
        self.hour_counter = 0
        self.on_target = 0
        self.get_logger().info('setpoint_setter started')

    def controller_callback(self, msg):
        err1 = msg.dof_states[0].error
        err2 = msg.dof_states[1].error
        if abs(err1) < 0.005 and abs(err2) < 0.005:
            self.on_target += 1
        else:
            self.on_target = 0

        if self.on_target >= 10:
            self.on_target = 0
            self.hour_counter += 1
            center_angle = (self.hour_counter % 12)*2*math.pi/12
            elbow_dir = random.getrandbits(1)
            # (most) poses have two valid configurations, so randomly pick one
            if elbow_dir > 0:
                joint1_angle = 0
                joint2_angle = center_angle
            else:
                joint1_angle = center_angle
                joint2_angle = 2*math.pi - center_angle
            cmd = MultiDOFCommand()
            cmd.dof_names = [msg.dof_states[0].name, msg.dof_states[1].name]
            cmd.values = [joint1_angle, joint2_angle]
            self.setpoint_publisher.publish(cmd)
            self.get_logger().info(f'Time is now {((self.hour_counter - 1) % 12) + 1}:00, setting joints {cmd.dof_names} to {cmd.values}')

def main(args=None):
    rclpy.init(args=args)
    node = SetpointSetter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()