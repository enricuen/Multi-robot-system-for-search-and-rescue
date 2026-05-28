#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class RedDetectorWanderer(Node):
    def __init__(self):
        super().__init__('red_detector_wanderer')
        
        # Parámetro para saber cuántos robots instanciar
        self.declare_parameter('num_robots', 3) # Valor por defecto acorde a tu lista de topics (0, 1 y 2)
        self.num_robots = self.get_parameter('num_robots').value
        
        self.bridge = CvBridge()
        
        # Diccionarios para manejar múltiples robots
        self.cmd_pubs = {}
        self.image_subs = {}
        self.robot_stopped = {}
        
        # Configuración del movimiento en círculos
        self.linear_vel = 0.2  # m/s
        self.radius = 1.0      # metros (radio determinado)
        self.angular_vel = self.linear_vel / self.radius # rad/s
        
        for i in range(self.num_robots):
            robot_name = f'robot_{i}'
            self.robot_stopped[i] = False
            
            # Publicador de cmd_vel para cada robot
            self.cmd_pubs[i] = self.create_publisher(
                Twist, 
                f'/{robot_name}/cmd_vel', 
                10
            )
            
            # Suscriptor de la cámara de cada robot
            # Usamos una función lambda para identificar qué robot activó el callback
            self.image_subs[i] = self.create_subscription(
                Image,
                f'/{robot_name}/camera/image_raw',
                lambda msg, robot_id=i: self.image_callback(msg, robot_id),
                10
            )
            
        # Timer del bucle de control a 10 Hz
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info(f"Nodo wanderer iniciado controlando {self.num_robots} robots.")

    def image_callback(self, msg, robot_id):
        try:
            # Convertir formato de ROS a imagen de OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # para ver ventana con la camara
            # cv2.imshow(f'Vista del Robot {robot_id}', cv_image)
            
            # Si el robot ya está detenido, ignoramos el procesamiento de color
            if self.robot_stopped[robot_id]:
                # cv2.waitKey(1)
                return
            
            # Convertir la imagen a HSV
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            
            # Definir los rangos de color rojo en HSV
            lower_red1 = np.array([0, 120, 70])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 120, 70])
            upper_red2 = np.array([180, 255, 255])
            
            # Crear las máscaras y unirlas
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = mask1 + mask2
            
            # para ver ventana con la mascara
            # cv2.imshow(f'Mascara Robot {robot_id}', mask)
            # cv2.waitKey(1)
            
            # Contar píxeles
            red_pixels = cv2.countNonZero(mask)
            
            if red_pixels > 2000:
                self.get_logger().info(f'¡Objeto ROJO detectado por robot_{robot_id}! Deteniendo motores...')
                self.robot_stopped[robot_id] = True
                
        except Exception as e:
            self.get_logger().error(f"Error en visión artificial del robot_{robot_id}: {e}")
            
    def control_loop(self):
        # Bucle continuo que publica las velocidades de forma constante
        for i in range(self.num_robots):
            twist = Twist()
            if self.robot_stopped[i]:
                # Comando para frenar en seco
                twist.linear.x = 0.0
                twist.angular.z = 0.0
            else:
                # Comando para describir la trayectoria circular
                twist.linear.x = self.linear_vel
                twist.angular.z = self.angular_vel
                
            self.cmd_pubs[i].publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = RedDetectorWanderer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
