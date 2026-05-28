#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
import cv2
import numpy as np

class RedDetectorWanderer(Node):
    def __init__(self):
        super().__init__('red_detector_wanderer')
        
        self.declare_parameter('num_robots', 3)
        self.num_robots = self.get_parameter('num_robots').value
        
        self.bridge = CvBridge()
        
        # Diccionarios para manejar múltiples robots
        self.cmd_pubs = {}
        self.image_subs = {}
        self.scan_subs = {}
        
        # --- MÁQUINA DE ESTADOS Y CONTROL ---
        self.state = {}           # Estado actual de cada robot
        self.avoid_timer = {}     # Temporizador para la marcha atrás
        self.target_error = {}    # Error de giro para centrar el color rojo
        
        # Variables de enjambre (compartidas por todos)
        self.global_red_found = False
        self.red_finder_id = None
        
        # Parámetros de movimiento y umbrales
        self.linear_vel = 0.2
        self.angular_vel = self.linear_vel / 1.0  # Radio de 1 metro
        self.obstacle_threshold = 0.45            # Metros para detectar obstáculo
        self.stop_at_red_threshold = 0.3          # Metros para detenerse frente al rojo
        
        for i in range(self.num_robots):
            robot_name = f'robot_{i}'
            self.state[i] = 'WANDERING'
            self.avoid_timer[i] = 0
            self.target_error[i] = 0.0
            
            # Publicador de velocidad
            self.cmd_pubs[i] = self.create_publisher(Twist, f'/{robot_name}/cmd_vel', 10)
            
            # Suscriptor de cámara
            self.image_subs[i] = self.create_subscription(
                Image,
                f'/{robot_name}/camera/image_raw',
                lambda msg, robot_id=i: self.image_callback(msg, robot_id),
                10
            )
            
            # Suscriptor de LIDAR
            self.scan_subs[i] = self.create_subscription(
                LaserScan,
                f'/{robot_name}/scan',
                lambda msg, robot_id=i: self.scan_callback(msg, robot_id),
                10
            )
            
        # Timer del bucle de control a 10 Hz
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info(f"Enjambre iniciado con {self.num_robots} robots. Lidar y Visión activos.")

    def scan_callback(self, msg, robot_id):
        # Limpiar los datos del Lidar (quitar infinitos o fuera de rango)
        ranges = np.array(msg.ranges)
        valid_ranges = ranges[(ranges > msg.range_min) & (ranges < msg.range_max)]
        
        if len(valid_ranges) > 0:
            min_dist = np.min(valid_ranges)
        else:
            min_dist = 10.0 # Valor seguro si no ve nada
            
        # 1. Lógica de evasión de obstáculos (solo si está deambulando)
        if self.state[robot_id] == 'WANDERING' and min_dist < self.obstacle_threshold:
            self.get_logger().info(f'Robot {robot_id} detectó obstáculo a {min_dist:.2f}m. Esquivando...')
            self.state[robot_id] = 'AVOIDING'
            self.avoid_timer[robot_id] = 15 # 1.5 segundos (15 iteraciones a 10Hz) dando marcha atrás
            
        # 2. Lógica de parada ante el objeto rojo
        elif self.state[robot_id] == 'HOMING' and min_dist < self.stop_at_red_threshold:
            self.get_logger().info(f'¡Robot {robot_id} ha llegado al objeto rojo! Deteniendo misión.')
            self.state[robot_id] = 'STOPPED_AT_RED'

    def image_callback(self, msg, robot_id):
        # Si el robot está esquivando, parado frente al rojo, o bloqueado por otro, no procesamos visión
        if self.state[robot_id] in ['AVOIDING', 'STOPPED_AT_RED', 'HALTED']:
            return
            
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            
            lower_red1 = np.array([0, 120, 70])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 120, 70])
            upper_red2 = np.array([180, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = mask1 + mask2
            
            red_pixels = cv2.countNonZero(mask)
            
            if red_pixels > 2000:
                # Si nadie ha encontrado el rojo aún, este robot lo reclama
                if not self.global_red_found:
                    self.global_red_found = True
                    self.red_finder_id = robot_id
                    self.get_logger().info(f'¡ROBOT {robot_id} ENCONTRÓ EL OBJETIVO! Bloqueando a los demás...')
                
                # Si este robot es el que encontró el rojo, calcula hacia dónde girar
                if self.red_finder_id == robot_id:
                    self.state[robot_id] = 'HOMING'
                    
                    # Calcular el centroide de la mancha roja
                    M = cv2.moments(mask)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        image_width = cv_image.shape[1]
                        
                        # Error normalizado: -1 (izquierda) a 1 (derecha)
                        self.target_error[robot_id] = (cx - (image_width / 2)) / (image_width / 2)
                        
            else:
                # Si pierde el rojo y era el buscador, se rinde y vuelve a deambular
                if self.red_finder_id == robot_id and self.state[robot_id] == 'HOMING':
                    self.get_logger().warn(f'Robot {robot_id} perdió de vista el rojo. Abortando búsqueda.')
                    self.state[robot_id] = 'WANDERING'
                    self.global_red_found = False
                    self.red_finder_id = None
                
        except Exception as e:
            self.get_logger().error(f"Error visión robot_{robot_id}: {e}")

    def control_loop(self):
        for i in range(self.num_robots):
            twist = Twist()
            
            # Forzar estado HALTED si otro robot ha reclamado el hallazgo
            if self.global_red_found and self.red_finder_id != i:
                self.state[i] = 'HALTED'
                
            # Lógica de movimiento basada en la máquina de estados
            if self.state[i] == 'WANDERING':
                twist.linear.x = self.linear_vel
                twist.angular.z = self.angular_vel
                
            elif self.state[i] == 'AVOIDING':
                twist.linear.x = -0.15 # Marcha atrás
                twist.angular.z = 0.8  # Girando rápido
                self.avoid_timer[i] -= 1
                if self.avoid_timer[i] <= 0:
                    self.state[i] = 'WANDERING'
                    
            elif self.state[i] == 'HOMING':
                twist.linear.x = 0.4
                # Controlador P: Gira en dirección contraria al error para centrar la imagen
                twist.angular.z = -2.5 * self.target_error[i] 
                
            elif self.state[i] == 'STOPPED_AT_RED' or self.state[i] == 'HALTED':
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                
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
        rclpy.shutdown()

if __name__ == '__main__':
    main()
