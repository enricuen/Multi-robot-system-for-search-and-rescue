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
import threading # <-- NUEVO: Para preguntar por terminal sin bloquear ROS
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
        self.patience = {} 
        
        self.active_nav_goals = {} # NUEVO: Guarda la coordenada de Nav2 actual
        self.previous_state = {}

        self.global_object_found = False
        self.finder_id = None
        
        # --- Coordenadas de las esquinas ---
        self.targets = self.generate_search_targets(self.num_robots)
        
        # --- Parámetros de Movimiento Reactivo ---
        self.linear_vel = 0.5             
        self.angular_vel = 0.4             
        self.obstacle_threshold = 1.0 
        self.stop_at_target_threshold = 1.2
        
        for i in range(self.num_robots):
            robot_name = f'robot_{i}'
            
            # NUEVO: robot_0 se queda como ambulancia, el resto navega
            if i == 0:
                self.state[i] = 'STATIONARY_AMBULANCE'
            else:
                self.state[i] = 'NAVIGATING_CORNERS'
                
            self.avoid_timer[i] = 0
            self.target_error[i] = 0.0
            self.patience[i] = 0
            self.active_nav_goals[i] = None   
            self.previous_state[i] = None
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
        self.get_logger().info("Enjambre iniciado. Preparando salida escalonada...")
        
        # NUEVO: Sacamos a robot_0 de la lista de despacho
        self.robots_to_dispatch = [r for r in list(self.targets.keys()) if r != 0]
        self.dispatch_timer = self.create_timer(1.0, self.dispatch_next_robot)

    def generate_search_targets(self, num_robots):
        targets = {}
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
        ]
        
        for i in range(num_robots):
            if i == 0:
                # El robot_0 (ambulancia) se queda donde aparece
                targets[i] = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}
                continue
                
            # NUEVO: Desplazamos el índice para que robot_1 vaya a la coordenada 0
            target_idx = i - 1
            if target_idx < len(base_coords):
                x, y = base_coords[target_idx]
            else:
                x = random.uniform(-12.0, 12.0)
                y = random.uniform(-12.0, 12.0)
            
            yaw = math.atan2(0.0 - y, 0.0 - x)
            q_z = math.sin(yaw / 2.0)
            q_w = math.cos(yaw / 2.0)
            
            targets[i] = {'x': x, 'y': y, 'z': q_z, 'w': q_w}
            
        return targets

    def dispatch_next_robot(self):
        if not self.robots_to_dispatch:
            self.dispatch_timer.cancel()
            return
            
        robot_id = self.robots_to_dispatch.pop(0)
        coords = self.targets[robot_id]
        
        self.get_logger().info(f'Semáforo en verde: Desplegando Robot {robot_id} hacia su esquina...')
        self.send_nav_goal(robot_id, coords['x'], coords['y'], coords['w'], coords['z'])

    def send_nav_goal(self, robot_id, x, y, w=1.0, z=0.0):
        self.active_nav_goals[robot_id] = {'x': x, 'y': y, 'w': w, 'z': z}
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
        # Si es la ambulancia llegando, no pasa a wandering
        if robot_id == 0:
            self.get_logger().info(f'¡Ambulancia (Robot 0) ha llegado a la persona!')
            self.state[robot_id] = 'STOPPED_AT_TARGET'
            return
            
        if self.state[robot_id] == 'NAVIGATING_CORNERS':
            self.get_logger().info(f'¡Robot {robot_id} llegó a su esquina! Iniciando patrulla...')
            self.state[robot_id] = 'WANDERING'

    def cancel_all_nav_goals(self):
        for r_id, handle in self.goal_handles.items():
            if handle is not None:
                handle.cancel_goal_async()
        for i in range(self.num_robots):
            self.goal_handles[i] = None

    # NUEVA FUNCIÓN: Lógica para despachar a la ambulancia (Robot 0)
    def dispatch_ambulance(self, leader_id):
        try:
            t = self.tf_buffer.lookup_transform('map', f'robot_{leader_id}/base_link', rclpy.time.Time())
            lx, ly = t.transform.translation.x, t.transform.translation.y
            q = t.transform.rotation
            
            # 1. MATAMOS cualquier orden de navegación que pudiera quedar viva en el sistema
            self.cancel_all_nav_goals()
            
            # 2. Mandamos SOLO al robot 0 a la posición de la persona
            self.state[0] = 'NAVIGATING_TO_LEADER'
            self.send_nav_goal(0, lx, ly, q.w, q.z)
            
            # 3. CONGELAMOS a todos los demás robots (del 1 en adelante, sin excepciones)
            for i in range(1, self.num_robots):
                self.state[i] = 'STANDBY'
                
        except Exception as e:
            self.get_logger().warn(f"Error al localizar líder para ambulancia: {e}")

    def call_swarm_to_leader(self, leader_id):
        try:
            t = self.tf_buffer.lookup_transform('map', f'robot_{leader_id}/base_link', rclpy.time.Time())
            lx, ly = t.transform.translation.x, t.transform.translation.y
            q = t.transform.rotation
            yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
            
            px, py = lx + 1.0 * math.cos(yaw), ly + 1.0 * math.sin(yaw)
            angs = [yaw + 2.09, yaw - 2.09] # Para formar el triángulo
            idx = 0
            
            self.rally_queue = []
            
            for i in range(self.num_robots):
                # NUEVO: Ignoramos al líder y a la ambulancia (robot 0)
                if i != leader_id and i != 0 and idx < 2:
                    gx, gy = px + 1.2 * math.cos(angs[idx]), py + 1.2 * math.sin(angs[idx])
                    gw, gz = math.cos((angs[idx]+3.14)/2), math.sin((angs[idx]+3.14)/2)
                    self.rally_queue.append({'id': i, 'x': gx, 'y': gy, 'w': gw, 'z': gz})
                    idx += 1
            
            if self.rally_queue:
                primer_robot = self.rally_queue.pop(0)
                self.state[primer_robot['id']] = 'NAVIGATING_TO_LEADER'
                self.get_logger().info(f"Formando Triángulo: Enviando Robot {primer_robot['id']}...")
                self.send_nav_goal(primer_robot['id'], primer_robot['x'], primer_robot['y'], primer_robot['w'], primer_robot['z'])
                
                if self.rally_queue:
                    self.rally_timer = self.create_timer(1.0, self.dispatch_next_rally_robot)
                    
        except Exception as e:
            self.get_logger().warn(f"Fallo temporal del GPS al avisar. ¡Reintentando llamada!")
            self.global_object_found = False 

    def dispatch_next_rally_robot(self):
        if hasattr(self, 'rally_queue') and self.rally_queue:
            sig_robot = self.rally_queue.pop(0)
            self.state[sig_robot['id']] = 'NAVIGATING_TO_LEADER'
            self.get_logger().info(f"Formando Triángulo: Enviando Robot {sig_robot['id']}...")
            self.send_nav_goal(sig_robot['id'], sig_robot['x'], sig_robot['y'], sig_robot['w'], sig_robot['z'])
            
        if hasattr(self, 'rally_timer') and self.rally_timer:
            self.rally_timer.cancel()

    def scan_callback(self, msg, robot_id):
        # Evitamos leer scan si es la ambulancia en la base
        if self.state[robot_id] in ['NAVIGATING_CORNERS', 'NAVIGATING_TO_LEADER', 'STATIONARY_AMBULANCE']:
            return

        ranges = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))
        valid_mask = (ranges > msg.range_min) & (ranges < msg.range_max)
        
        valid_ranges = ranges[valid_mask]
        min_dist_global = np.min(valid_ranges) if len(valid_ranges) > 0 else 10.0
        
        front_mask = (np.abs(angles) < 0.15) | (np.abs(angles) > 2 * math.pi - 0.15)
        front_ranges = ranges[valid_mask & front_mask]
        min_dist_front = np.min(front_ranges) if len(front_ranges) > 0 else 10.0

        # Si está navegando o parado en STANDBY, y algo se le acerca a menos de 60cm
        if self.state[robot_id] in ['NAVIGATING_CORNERS', 'NAVIGATING_TO_LEADER', 'STANDBY']:
            if min_dist_global < 0.6: 
                self.get_logger().warn(f'⚠️ [ANTI-CHOQUE] Robot {robot_id} iba a chocar. ¡Pausando y apartándose!')
                
                # Si estaba usando Nav2, le cortamos los frenos cancelando la meta
                if self.state[robot_id] in ['NAVIGATING_CORNERS', 'NAVIGATING_TO_LEADER']:
                    if self.goal_handles[robot_id] is not None:
                        self.goal_handles[robot_id].cancel_goal_async()
                        
                # Guardamos lo que estaba haciendo
                self.previous_state[robot_id] = self.state[robot_id]
                self.state[robot_id] = 'EMERGENCY_DODGE'
                self.avoid_timer[robot_id] = 15 # 1.5 segundos de maniobra de escape
                return # Salimos para que no procese nada más

        if self.state[robot_id] == 'HOMING':
            if min_dist_front < self.stop_at_target_threshold and abs(self.target_error[robot_id]) < 0.2:
                self.get_logger().info(f'¡Líder {robot_id} frente a la persona (a {min_dist_front:.2f}m)! Evaluando estado...')
                self.state[robot_id] = 'STOPPED_AT_TARGET'
                
                # NUEVO: Lógica interactiva con hilo para no bloquear ROS
                def ask_user_state():
                    print("\n" + "="*50)
                    respuesta = input(f"[ALERTA] Robot {robot_id} ha encontrado a la persona.\n¿La persona está 'herida' o 'a salvo'?: ").strip().lower()
                    print("="*50 + "\n")
                    
                    if respuesta == 'herida':
                        self.get_logger().info('¡Protocolo de Emergencia! Enviando Robot 0 (Ambulancia)...')
                        self.dispatch_ambulance(robot_id)
                    else:
                        self.get_logger().info('Persona a salvo. Iniciando protocolo de aseguramiento en Triángulo...')
                        self.call_swarm_to_leader(robot_id)

                threading.Thread(target=ask_user_state).start()
                return
            
            if min_dist_front < self.obstacle_threshold:
                self.state[robot_id] = 'AVOIDING'
                self.avoid_timer[robot_id] = 10
            return 

        if self.state[robot_id] == 'WANDERING' and min_dist_global < self.obstacle_threshold:
            self.state[robot_id] = 'AVOIDING'
            self.avoid_timer[robot_id] = 15


    def image_callback(self, msg, robot_id):
        if self.state[robot_id] in ['AVOIDING', 'STOPPED_AT_TARGET', 'NAVIGATING_TO_LEADER', 'STATIONARY_AMBULANCE']:
            return
            
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            resultados = self.yolo_model(cv_image, verbose=False)
            
            imagen_anotada = resultados[0].plot() 
            msg_anotado = self.bridge.cv2_to_imgmsg(imagen_anotada, encoding="bgr8")
            self.yolo_pubs[robot_id].publish(msg_anotado)

            cv2.imshow(f"Vista YOLO - Robot {robot_id}", imagen_anotada)
            cv2.waitKey(1)

            objeto_encontrado = False
            
            for r in resultados:
                for box in r.boxes:
                    id_clase = int(box.cls[0])
                    nombre_clase = self.yolo_model.names[id_clase]
                    confianza = float(box.conf[0])
                    
                    if nombre_clase == self.objetivo_yolo and confianza > self.confianza_minima:
                        objeto_encontrado = True
                        self.patience[robot_id] = 15
                        
                        if not self.global_object_found:
                            self.global_object_found = True
                            self.finder_id = robot_id
                            self.get_logger().info(f'¡ROBOT {robot_id} VIO A LA PERSONA DESDE LEJOS! Acercándose...')
                            self.cancel_all_nav_goals()
                            # Cancelar salida de robots rezagados
                            self.robots_to_dispatch = [] # Vaciamos la cola
                            if hasattr(self, 'dispatch_timer'):
                                self.dispatch_timer.cancel() # Apagamos el semáforo

                            for j in range(self.num_robots):
                                if j == 0:
                                    continue # La ambulancia se queda quieta
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
            
            if not objeto_encontrado and self.finder_id == robot_id and self.state[robot_id] == 'HOMING':
                if self.patience[robot_id] > 0:
                    self.patience[robot_id] -= 1
                else:
                    self.get_logger().warn(f'Líder {robot_id} perdió a la persona. Patrullando de nuevo...')
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
            
            # NUEVO: La ambulancia no publica movimiento a menos que sea despachada por NAV2
            if self.state[i] in ['STANDBY', 'STOPPED_AT_TARGET', 'STATIONARY_AMBULANCE']:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
            elif self.state[i] == 'EMERGENCY_DODGE':
                twist.linear.x = -0.20 # Da marcha atrás suave
                twist.angular.z = 0.8  # Gira sobre sí mismo para apartarse
                self.avoid_timer[i] -= 1
                
                if self.avoid_timer[i] <= 0:
                    prev = self.previous_state[i]
                    self.state[i] = prev
                    self.get_logger().info(f'Robot {i} se apartó con éxito. Retomando: {prev}')
                    
                    # Si antes iba a un sitio con Nav2, relanzamos el viaje exactamente a donde iba
                    if prev in ['NAVIGATING_CORNERS', 'NAVIGATING_TO_LEADER']:
                        goal = self.active_nav_goals[i]
                        if goal is not None:
                            self.send_nav_goal(i, goal['x'], goal['y'], goal['w'], goal['z'])
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
                    twist.angular.z = -0.2 * self.target_error[i]
                else:
                    twist.linear.x = 0.5 
                    
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