import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class EdgeDetector(Node):
    def __init__(self):
        super().__init__('edge_detector')
        self.subscription = self.create_subscription(
            Image,
            '/model_camera/image_raw',
            self.listener_callback,
            10)
        self.publisher = self.create_publisher(Image, '/edges', 10)
        self.br = CvBridge()
        self.get_logger().info('edge_detector started')

    def listener_callback(self, data):
        #self.get_logger().info('Receiving video frame')
        current_frame = self.br.imgmsg_to_cv2(data, desired_encoding='bgr8')
        #cv2.imshow("camera", current_frame)
        cv2.waitKey(1)
        edges = cv2.Canny(current_frame, 100, 200)
        self.publisher.publish(self.br.cv2_to_imgmsg(edges, encoding='passthrough'))


def main(args=None):
    rclpy.init(args=args)
    node = EdgeDetector()
    rclpy.spin(node)
    node.get_logger().error('edge_detector stopped')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()