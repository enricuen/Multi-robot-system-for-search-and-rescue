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
import random
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
        self.patience = {} # NUEVO: Memoria a corto plazo para que YOLO no pierda el objetivo por parpadeos
        
        self.global_object_found = False
        self.finder_id = None
        
        # --- Coordenadas de las esquinas (Ajustadas para no chocar con paredes) ---
        self.targets = self.generate_search_targets(self.num_robots)
        
        # --- Parámetros de Movimiento Reactivo ---
        self.linear_vel = 0.5              
        self.angular_vel = 0.4              
        self.obstacle_threshold = 1.0 #originalmente a o.55 
        self.stop_at_target_threshold = 1.0 
        
        for i in range(self.num_robots):
            robot_name = f'robot_{i}'
            
            self.state[i] = 'NAVIGATING_CORNERS'
            self.avoid_timer[i] = 0
            self.target_error[i] = 0.0
            self.patience[i] = 0
            
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
        self.get_logger().info("Enjambre iniciado. Preparando salida escalonada (control de tráfico)...")
        self.robots_to_dispatch = list(self.targets.keys())
        # Creamos un "semáforo" que deja salir a un robot cada 1 segundo
        self.dispatch_timer = self.create_timer(1.0, self.dispatch_next_robot)

    def generate_search_targets(self, num_robots):
        targets = {}
        # Lista de coordenadas estratégicas ordenadas por prioridad para cubrir el mapa
        # Formato: (X, Y)
        base_coords = [
            (-12.0,  12.0), # 0: Esquina Superior Izquierda
            ( 12.0,  12.0), # 1: Esquina Superior Derecha
            ( 12.0, -12.0), # 2: Esquina Inferior Derecha
            (-12.0, -12.0), # 3: Esquina Inferior Izquierda
            (  0.0,   0.0), # 4: Centro del mapa
            (  0.0,  12.0), # 5: Centro pared Superior
            (  0.0, -12.0), # 6: Centro pared Inferior
            (-12.0,   0.0), # 7: Centro pared Izquierda
            ( 12.0,   0.0), # 8: Centro pared Derecha
            ( -6.5,   6.5), # 9: Centro del Cuadrante 1
            (  6.5,   6.5), # 10: Centro del Cuadrante 2
            (  6.5,  -6.5), # 11: Centro del Cuadrante 3
            ( -6.5,  -6.5), # 12: Centro del Cuadrante 4
        ]
        
        for i in range(num_robots):
            if i < len(base_coords):
                x, y = base_coords[i]
            else:
                # Si hay más robots que posiciones base, genera posiciones aleatorias dentro del límite
                x = random.uniform(-12.0, 12.0)
                y = random.uniform(-12.0, 12.0)
            
            # Hacemos que el robot mire hacia el centro del mapa (0,0) al llegar
            yaw = math.atan2(0.0 - y, 0.0 - x)
            # Conversión básica de Euler (Yaw) a Cuaternión (Z, W) para 2D
            q_z = math.sin(yaw / 2.0)
            q_w = math.cos(yaw / 2.0)
            
            targets[i] = {'x': x, 'y': y, 'z': q_z, 'w': q_w}
            
        return targets

    def dispatch_next_robot(self):
        # Si ya han salido todos los robots, apagamos el semáforo
        if not self.robots_to_dispatch:
            self.dispatch_timer.cancel()
            return
            
        # Sacamos al siguiente robot de la lista de espera
        robot_id = self.robots_to_dispatch.pop(0)
        coords = self.targets[robot_id]
        
        self.get_logger().info(f'Semáforo en verde: Desplegando Robot {robot_id} hacia su esquina...')
        self.send_nav_goal(robot_id, coords['x'], coords['y'], coords['w'], coords['z'])

    def send_nav_goal(self, robot_id, x, y, w=1.0, z=0.0):
        client = self.nav_clients[robot_id]
        if not client.wait_for_server(timeout_sec=15.0):
            self.get_logger().error(f'Nav2 Server de robot_{robot_id} no disponible.')
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.orientation.z = float(z)
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
        
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(
            lambda f, r_id=robot_id: self.get_result_callback(f, r_id)
        )

    def get_result_callback(self, future, robot_id):
        if self.state[robot_id] == 'NAVIGATING_CORNERS':
            self.get_logger().info(f'¡Robot {robot_id} llegó a su esquina! Iniciando patrulla...')
            self.state[robot_id] = 'WANDERING'

    def cancel_all_nav_goals(self):
        for r_id, handle in self.goal_handles.items():
            if handle is not None:
                handle.cancel_goal_async()
        for i in range(self.num_robots):
            self.goal_handles[i] = None


# LÓGICA DE APOYO POR PROXIMIDAD
    def call_swarm_to_leader(self, leader_id):
        try:
            # 1. Obtener la posición del líder
            t_leader = self.tf_buffer.lookup_transform('map', f'robot_{leader_id}/base_link', rclpy.time.Time())
            lx = t_leader.transform.translation.x
            ly = t_leader.transform.translation.y
            q = t_leader.transform.rotation
            l_yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
            
            self.get_logger().info(f'¡Persona confirmada por robot_{leader_id}! Buscando al apoyo más cercano...')

            closest_robot_id = None
            min_dist = float('inf')
            best_rx = 0.0
            best_ry = 0.0

            # 2. Calcular qué robot está más cerca del líder
            for i in range(self.num_robots):
                if i == leader_id:
                    continue
                
                try:
                    t_robot = self.tf_buffer.lookup_transform('map', f'robot_{i}/base_link', rclpy.time.Time())
                    rx = t_robot.transform.translation.x
                    ry = t_robot.transform.translation.y
                    
                    dist = math.sqrt((lx - rx)**2 + (ly - ry)**2)
                    
                    if dist < min_dist:
                        min_dist = dist
                        closest_robot_id = i
                        best_rx = rx
                        best_ry = ry
                except TransformException:
                    pass

            # 3. Calcular los flancos AVANZADOS para quedar a la misma altura
            if closest_robot_id is not None:
                self.state[closest_robot_id] = 'NAVIGATING_TO_LEADER'
                
                side_offset = 1.0     # Distancia de separación LATERAL
                forward_offset = 1.5 # Avance hacia adelante para empatar altura con el líder
                
                # Primero, avanzamos la meta hacia adelante en la dirección a la que mira el líder
                base_x = lx + forward_offset * math.cos(l_yaw)
                base_y = ly + forward_offset * math.sin(l_yaw)
                
                # A partir de ese punto avanzado, calculamos la izquierda y la derecha
                # Flanco DERECHO
                right_x = base_x + side_offset * math.cos(l_yaw - math.pi/2)
                right_y = base_y + side_offset * math.sin(l_yaw - math.pi/2)
                
                # Flanco IZQUIERDO
                left_x = base_x + side_offset * math.cos(l_yaw + math.pi/2)
                left_y = base_y + side_offset * math.sin(l_yaw + math.pi/2)
                
                # Evaluamos qué flanco le pilla mejor al robot de apoyo
                dist_to_right = math.sqrt((best_rx - right_x)**2 + (best_ry - right_y)**2)
                dist_to_left  = math.sqrt((best_rx - left_x)**2 + (best_ry - left_y)**2)
                
                if dist_to_right < dist_to_left:
                    goal_x, goal_y = right_x, right_y
                    lado = "DERECHO"
                else:
                    goal_x, goal_y = left_x, left_y
                    lado = "IZQUIERDO"
                
                # Le pasamos la misma rotación (q.w, q.z) para que ambos miren a la persona igual
                self.get_logger().info(f'Robot_{closest_robot_id} acudiendo al flanco {lado}.')
                self.send_nav_goal(closest_robot_id, goal_x, goal_y, q.w, q.z)
                
                for i in range(self.num_robots):
                    if i != leader_id and i != closest_robot_id:
                        self.state[i] = 'STANDBY'
            
        except TransformException as e:
            self.get_logger().error(f'Error al obtener transformaciones: {e}')

    # LÓGICA DE SENSORES Y YOLO
    def scan_callback(self, msg, robot_id):
        if self.state[robot_id] in ['NAVIGATING_CORNERS', 'NAVIGATING_TO_LEADER']:
            return

        ranges = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))
        valid_mask = (ranges > msg.range_min) & (ranges < msg.range_max)
        
        valid_ranges = ranges[valid_mask]
        min_dist_global = np.min(valid_ranges) if len(valid_ranges) > 0 else 10.0
        
        front_mask = (np.abs(angles) < 0.15) | (np.abs(angles) > 2 * math.pi - 0.15)
        front_ranges = ranges[valid_mask & front_mask]
        min_dist_front = np.min(front_ranges) if len(front_ranges) > 0 else 10.0
            
        if self.state[robot_id] == 'HOMING':
            if min_dist_front < self.stop_at_target_threshold and abs(self.target_error[robot_id]) < 0.2:
                self.get_logger().info(f'¡Líder {robot_id} está frente a la persona (a {min_dist_front:.2f}m)! Asegurando el área...')
                self.state[robot_id] = 'STOPPED_AT_TARGET'
                self.call_swarm_to_leader(robot_id)
                return
            
            if min_dist_front < self.obstacle_threshold:
                self.state[robot_id] = 'AVOIDING'
                self.avoid_timer[robot_id] = 10
            return 

        if self.state[robot_id] == 'WANDERING' and min_dist_global < self.obstacle_threshold:
            self.state[robot_id] = 'AVOIDING'
            self.avoid_timer[robot_id] = 15


    def image_callback(self, msg, robot_id):
        if self.state[robot_id] in ['AVOIDING', 'STOPPED_AT_TARGET', 'NAVIGATING_TO_LEADER']:
            return
            
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            resultados = self.yolo_model(cv_image, verbose=False)
            
            imagen_anotada = resultados[0].plot() 
            msg_anotado = self.bridge.cv2_to_imgmsg(imagen_anotada, encoding="bgr8")
            self.yolo_pubs[robot_id].publish(msg_anotado)

            # Abre una ventana independiente para cada robot
            cv2.imshow(f"Vista YOLO - Robot {robot_id}", imagen_anotada)
            cv2.waitKey(1) # Actualiza la ventana (1 milisegundo)

            objeto_encontrado = False
            
            for r in resultados:
                for box in r.boxes:
                    id_clase = int(box.cls[0])
                    nombre_clase = self.yolo_model.names[id_clase]
                    confianza = float(box.conf[0])
                    
                    if nombre_clase == self.objetivo_yolo and confianza > self.confianza_minima:
                        objeto_encontrado = True
                        self.patience[robot_id] = 15 # Le damos 15 fotogramas de "memoria" si lo pierde
                        
                        if not self.global_object_found:
                            self.global_object_found = True
                            self.finder_id = robot_id
                            self.get_logger().info(f'¡ROBOT {robot_id} VIO A LA PERSONA DESDE LEJOS! Acercándose para confirmar...')
                            self.cancel_all_nav_goals()
                            
                            for j in range(self.num_robots):
                                if j == robot_id:
                                    self.state[j] = 'HOMING' 
                                else:
                                    self.state[j] = 'STANDBY' 
                        
                        if self.finder_id == robot_id and self.state[robot_id] != 'AVOIDING':
                            self.state[robot_id] = 'HOMING'
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            cx = (x1 + x2) / 2.0
                            image_width = cv_image.shape[1]
                            self.target_error[robot_id] = (cx - (image_width / 2)) / (image_width / 2)
                            
                        break 
                if objeto_encontrado:
                    break
            
            # MEMORIA A CORTO PLAZO: Si no lo ve, gastamos la paciencia antes de rendirnos
            if not objeto_encontrado and self.finder_id == robot_id and self.state[robot_id] == 'HOMING':
                if self.patience[robot_id] > 0:
                    self.patience[robot_id] -= 1
                else:
                    self.get_logger().warn(f'Líder {robot_id} perdió a la persona de vista definitivamente. Patrullando...')
                    self.state[robot_id] = 'WANDERING'
                    self.global_object_found = False
                    self.finder_id = None
                
        except Exception as e:
            pass 

    def control_loop(self):
        for i in range(self.num_robots):
            
            if self.state[i] in ['NAVIGATING_CORNERS', 'NAVIGATING_TO_LEADER']:
                continue

            twist = Twist()
            robot_name = f'robot_{i}'
            
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
                if not self.global_object_found:
                    twist.linear.x = 0.0
                    twist.angular.z = -0.2 * self.target_error[i] # Giro de búsqueda suave
                else:
                    twist.linear.x = 0.4   
                    
                    # reducido el multiplicador de 1.0 a 0.2
                    # zona muerta => si el error es menor al 10%, no gira
                    if abs(self.target_error[i]) > 0.10:
                        twist.angular.z = -0.2 * self.target_error[i]
                    else:
                        twist.angular.z = 0.0
                
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
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
