#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
from nav2_msgs.action import NavigateToPose
from tf2_ros import TransformException, Buffer, TransformListener
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from ultralytics import YOLO

class SwarmSearchAndRescue(Node):
    def __init__(self):
        super().__init__('swarm_search_rescue_node')
        
        self.declare_parameter('num_robots', 3)
        self.num_robots = self.get_parameter('num_robots').value
        
        self.bridge = CvBridge()
        
        #Configuración YOLO 
        self.yolo_model = YOLO('yolov8n.pt') 
        self.objetivo_yolo = 'person'
        self.confianza_minima = 0.5
        
        # Configuración TF 
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Diccionarios de ROS
        self.cmd_pubs = {}
        self.image_subs = {}
        self.scan_subs = {}
        self.yolo_pubs = {}
        self.nav_clients = {}
        self.goal_handles = {}
        
        # Variables de Estado
        self.state = {}           
        self.avoid_timer = {}     
        self.target_error = {}    
        self.global_object_found = False
        self.finder_id = None
        
        # --- Coordenadas de las esquinas (Búsqueda inicial) ---
        self.targets = {
            0: {'x': -13.0, 'y': 13.0, 'w': 1.0}, # Superior Izquierda
            1: {'x': 13.0,  'y': 13.0, 'w': 1.0}, # Superior Derecha
            2: {'x': 13.0,  'y': -13.0, 'w': 1.0} # Inferior Derecha
        }
        
        # --- Parámetros de Movimiento Reactivo ---
        self.linear_vel = 0.12              
        self.angular_vel = 0.15             
        self.obstacle_threshold = 0.55      
        self.stop_at_target_threshold = 0.8 # Distancia a la que el líder se detiene ante la persona
        
        for i in range(self.num_robots):
            robot_name = f'robot_{i}'
            
            # Inician yendo a las esquinas
            self.state[i] = 'NAVIGATING_CORNERS'
            self.avoid_timer[i] = 0
            self.target_error[i] = 0.0
            
            self.cmd_pubs[i] = self.create_publisher(Twist, f'/{robot_name}/cmd_vel', 10)
            self.yolo_pubs[i] = self.create_publisher(Image, f'/{robot_name}/camera/yolo_image', 10)
            
            self.image_subs[i] = self.create_subscription(
                Image, f'/{robot_name}/camera/image_raw',
                lambda msg, robot_id=i: self.image_callback(msg, robot_id), 10
            )
            self.scan_subs[i] = self.create_subscription(
                LaserScan, f'/{robot_name}/scan',
                lambda msg, robot_id=i: self.scan_callback(msg, robot_id), 10
            )
            
            self.nav_clients[i] = ActionClient(self, NavigateToPose, f'/{robot_name}/navigate_to_pose')
            self.goal_handles[i] = None

        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("Enjambre iniciado. Enviando robots a las esquinas...")
        self.send_initial_nav_goals()


    # LÓGICA NAV2 (ENVIAR Y CANCELAR GOALS)
    def send_initial_nav_goals(self):
        for robot_id, coords in self.targets.items():
            self.send_nav_goal(robot_id, coords['x'], coords['y'], coords['w'])

    def send_nav_goal(self, robot_id, x, y, w=1.0):
        client = self.nav_clients[robot_id]
        if not client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error(f'Nav2 Server de robot_{robot_id} no disponible.')
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.orientation.w = float(w)

        send_goal_future = client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(
            lambda future, r_id=robot_id: self.goal_response_callback(future, r_id)
        )

    def goal_response_callback(self, future, robot_id):
        goal_handle = future.result()
        if not goal_handle.accepted:
            return
        self.goal_handles[robot_id] = goal_handle

    def cancel_all_nav_goals(self):
        self.get_logger().info("Cancelando rutas de Nav2 activas...")
        for r_id, handle in self.goal_handles.items():
            if handle is not None:
                handle.cancel_goal_async()
        for i in range(self.num_robots):
            self.goal_handles[i] = None


    # LLAMADA AL ENJAMBRE
    def call_swarm_to_leader(self, leader_id):
        try:
            # Obtenemos la posición global (en el mapa) de donde se ha detenido el líder
            t = self.tf_buffer.lookup_transform(
                'map', 
                f'robot_{leader_id}/base_footprint',
                rclpy.time.Time()
            )
            
            lx = t.transform.translation.x
            ly = t.transform.translation.y
            
            self.get_logger().info(f'¡LÍDER EN POSICIÓN [X:{lx:.2f}, Y:{ly:.2f}]! Llamando a los seguidores...')
            
            # Offsets para que los seguidores rodeen al líder y no choquen contra él
            offsets = [(1.5, 0.0), (-1.5, 0.0), (0.0, 1.5), (0.0, -1.5)]
            offset_idx = 0
            
            for i in range(self.num_robots):
                if i != leader_id:
                    self.state[i] = 'NAVIGATING_TO_LEADER'
                    goal_x = lx + offsets[offset_idx][0]
                    goal_y = ly + offsets[offset_idx][1]
                    offset_idx += 1
                    
                    self.get_logger().info(f'Enviando a robot_{i} de apoyo a [X:{goal_x:.2f}, Y:{goal_y:.2f}]')
                    self.send_nav_goal(i, goal_x, goal_y)
                    
        except TransformException as e:
            self.get_logger().error(f'Error al obtener la pose del líder: {e}')


    # LÓGICA DE SENSORES Y YOLO
    def scan_callback(self, msg, robot_id):
        # Si Nav2 está manejando el robot, ignoramos la evasión manual de obstáculos
        if self.state[robot_id] in ['NAVIGATING_CORNERS', 'NAVIGATING_TO_LEADER']:
            return

        ranges = np.array(msg.ranges)
        # Creamos un array con el ángulo al que corresponde cada rayo del láser
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))
        
        # Máscara para filtrar lecturas infinitas o erróneas del sensor
        valid_mask = (ranges > msg.range_min) & (ranges < msg.range_max)
        
        # 1. Distancia MÍNIMA GLOBAL (360 grados) - Útil para no chocar al patrullar
        valid_ranges = ranges[valid_mask]
        min_dist_global = np.min(valid_ranges) if len(valid_ranges) > 0 else 10.0
        
        # 2. Distancia MÍNIMA FRONTAL (Un cono visual de +/- 25 grados hacia adelante)
        # Esto asume que 0 radianes es el frente del robot (estándar en ROS)
        front_mask = (np.abs(angles) < 0.45) | (np.abs(angles) > 2 * math.pi - 0.45)
        front_ranges = ranges[valid_mask & front_mask]
        min_dist_front = np.min(front_ranges) if len(front_ranges) > 0 else 10.0
            
        #  LÓGICA DE LLEGADA
        if self.state[robot_id] == 'HOMING':
            # Solo frena si hay algo JUSTO DELANTE a menos de 0.8m Y la persona está centrada en la cámara
            if min_dist_front < self.stop_at_target_threshold and abs(self.target_error[robot_id]) < 0.5:
                self.get_logger().info(f'¡Líder {robot_id} está frente a la persona (a {min_dist_front:.2f}m)! Asegurando el área...')
                self.state[robot_id] = 'STOPPED_AT_TARGET'
                self.call_swarm_to_leader(robot_id)
                return
            
            # Mientras persigue, solo esquivamos si el obstáculo frontal es muy inminente
            # Ignoramos paredes laterales para no perder a la persona por asustarnos de una pared
            if min_dist_front < self.obstacle_threshold:
                self.state[robot_id] = 'AVOIDING'
                self.avoid_timer[robot_id] = 10
            return 

        # --- LÓGICA DE ESQUIVA GLOBAL (Patrullando) ---
        if self.state[robot_id] == 'WANDERING' and min_dist_global < self.obstacle_threshold:
            self.state[robot_id] = 'AVOIDING'
            self.avoid_timer[robot_id] = 15

    def image_callback(self, msg, robot_id):
        # El líder ya paró o está esquivando. Los seguidores en Nav2 no necesitan procesar imagen para moverse
        if self.state[robot_id] in ['AVOIDING', 'STOPPED_AT_TARGET', 'NAVIGATING_TO_LEADER']:
            return
            
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            resultados = self.yolo_model(cv_image, verbose=False)
            
            imagen_anotada = resultados[0].plot() 
            msg_anotado = self.bridge.cv2_to_imgmsg(imagen_anotada, encoding="bgr8")
            self.yolo_pubs[robot_id].publish(msg_anotado)
            
            objeto_encontrado = False
            
            for r in resultados:
                for box in r.boxes:
                    id_clase = int(box.cls[0])
                    nombre_clase = self.yolo_model.names[id_clase]
                    confianza = float(box.conf[0])
                    
                    if nombre_clase == self.objetivo_yolo and confianza > self.confianza_minima:
                        objeto_encontrado = True
                        
                        # ALERTA GLOBAL: Alguien vio a la persona
                        if not self.global_object_found:
                            self.global_object_found = True
                            self.finder_id = robot_id
                            self.get_logger().info(f'¡ROBOT {robot_id} VIO A LA PERSONA!')
                            
                            self.cancel_all_nav_goals()
                            
                            for j in range(self.num_robots):
                                if j == robot_id:
                                    self.state[j] = 'HOMING' # El líder persigue con cámara
                                else:
                                    self.state[j] = 'STANDBY' # Los demás frenan y esperan coordenadas
                        
                        if self.finder_id == robot_id and self.state[robot_id] != 'AVOIDING':
                            self.state[robot_id] = 'HOMING'
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            cx = (x1 + x2) / 2.0
                            image_width = cv_image.shape[1]
                            self.target_error[robot_id] = (cx - (image_width / 2)) / (image_width / 2)
                            
                        break 
                if objeto_encontrado:
                    break
            
            # Fallback por si el líder se despista antes de llegar
            if not objeto_encontrado and self.finder_id == robot_id and self.state[robot_id] == 'HOMING':
                self.get_logger().warn(f'Líder {robot_id} perdió a la persona. Patrullando...')
                self.state[robot_id] = 'WANDERING'
                self.global_object_found = False
                self.finder_id = None
                
        except Exception as e:
            pass 

    # BUCLE DE CONTROL (PUBLICADOR CMD_VEL)
    def control_loop(self):
        for i in range(self.num_robots):
            
            # Nav2 controla las ruedas en estos estados
            if self.state[i] in ['NAVIGATING_CORNERS', 'NAVIGATING_TO_LEADER']:
                continue

            twist = Twist()
            robot_name = f'robot_{i}'
            
            # Los seguidores esperan quietos hasta que el líder llegue y les mande la ubicación
            if self.state[i] in ['STANDBY', 'STOPPED_AT_TARGET']:
                twist.linear.x = 0.0
                twist.angular.z = 0.0

            elif self.state[i] == 'WANDERING':
                twist.linear.x = self.linear_vel
                twist.angular.z = self.angular_vel
                
            elif self.state[i] == 'AVOIDING':
                twist.linear.x = -0.10
                twist.angular.z = 0.5 
                self.avoid_timer[i] -= 1
                if self.avoid_timer[i] <= 0:
                    self.state[i] = 'WANDERING' 
                    
            elif self.state[i] == 'HOMING':
                twist.linear.x = 0.2   
                twist.angular.z = -1.0 * self.target_error[i] 
                
            self.cmd_pubs[i].publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = SwarmSearchAndRescue()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()