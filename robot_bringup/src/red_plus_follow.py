#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import Image, LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import random
import time # Para medir los 10 segundos
from tf2_ros import Buffer, TransformListener
from ultralytics import YOLO # Importación de YOLO

class SwarmAILeader(Node):
    def __init__(self):
        super().__init__('swarm_ai_leader')
        
        self.declare_parameter('num_robots', 3)
        self.num_robots = self.get_parameter('num_robots').value
        self.bridge = CvBridge()
        
        self.cmd_pubs = {}
        self.image_subs = {}
        self.scan_subs = {}
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Publicador de gráficos (Círculos azules y Flechas)
        self.marker_pub = self.create_publisher(MarkerArray, '/swarm_markers', 10)
        
        # Cargar YOLO Nano (compartido por todos)
        try:
            self.yolo_model = YOLO('yolov8n.pt', verbose=False)
            self.get_logger().info("Cerebro YOLO cargado correctamente (Modo CPU).")
        except Exception:
            self.get_logger().error("Error cargando YOLO. Asegúrate de tenerlo instalado.")
            self.destroy_node()
            return
            
        self.yolo_image_pubs = {} # Para ver las cajas en RViz/Ventanas
        
        # --- MÁQUINA DE ESTADOS ---
        self.state = {}           
        self.previous_state = {}  
        self.avoid_timer = {}     
        self.target_error = {}    
        
        # NUEVO: Timers y Variables para el retorno a base
        self.arrived_time = {} # Cuándo llegó al círculo objetivo
        self.encircle_duration = 10.0 # Segundos de espera
        self.return_tolerance = 0.5 # Tolerancia para considerar que está "en casa"
        # Definimos "Home" como un pequeño círculo alrededor del origen para que no colisionen
        self.home_poses = {} 
        angulo_home = (2 * math.pi) / self.num_robots
        radio_home = 1.0
        for idx in range(self.num_robots):
            self.home_poses[idx] = (radio_home * math.cos(idx * angulo_home), 
                                    radio_home * math.sin(idx * angulo_home))
        
        # --- DATOS DEL LIDAR ---
        self.closest_obs_dist = {}
        self.closest_obs_angle = {}
        
        self.global_object_found = False # Global para todos
        self.object_finder_id = None
        self.global_targets = {} # Objetivos matemáticos del cerco
        
        self.linear_vel = 0.2
        self.obstacle_threshold = 0.55            
        self.stop_at_obj_threshold = 0.3          
        
        self.radio_cerco = 0.8
        self.arrival_tolerance = 0.25 
        
        for i in range(self.num_robots):
            robot_name = f'robot_{i}'
            # Empiezan explorando
            self.state[i] = 'WANDERING'
            self.previous_state[i] = 'WANDERING'
            self.avoid_timer[i] = 0
            self.target_error[i] = 0.0
            self.arrived_time[i] = None # No ha llegado
            
            self.closest_obs_dist[i] = 10.0
            self.closest_obs_angle[i] = 0.0
            
            self.cmd_pubs[i] = self.create_publisher(Twist, f'/{robot_name}/cmd_vel', 10)
            
            # Suscriptor de imagen (pasando por YOLO en el callback)
            self.image_subs[i] = self.create_subscription(
                Image, f'/{robot_name}/camera/image_raw',
                lambda msg, robot_id=i: self.image_yolo_callback(msg, robot_id), 10)
            
            self.yolo_image_pubs[i] = self.create_publisher(
                Image, f'/{robot_name}/camera/yolo_visual', 10)
            
            self.scan_subs[i] = self.create_subscription(
                LaserScan, f'/{robot_name}/scan',
                lambda msg, robot_id=i: self.scan_callback(msg, robot_id), 10)
            
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info(f"Enjambre AI activado. Buscando botellas con retorno a base.")

    def scan_callback(self, msg, robot_id):
        ranges = np.array(msg.ranges)
        # Ignorar detrás (evitar miedo irracional)
        valid_indices = np.where((ranges > msg.range_min) & (ranges < msg.range_max))[0]
        
        min_dist = 10.0
        min_angle = 0.0
        
        for idx in valid_indices:
            angle = msg.angle_min + idx * msg.angle_increment
            angle = (angle + math.pi) % (2 * math.pi) - math.pi # Normalizar a [-pi, pi]
            
            # Solo consideramos el frente (180 grados frontales)
            if -math.pi/2 <= angle <= math.pi/2:
                if ranges[idx] < min_dist:
                    min_dist = ranges[idx]
                    min_angle = angle
            
        self.closest_obs_dist[robot_id] = min_dist
        self.closest_obs_angle[robot_id] = min_angle
            
        # Esquiva marchando atrás SOLO si deambula
        if self.state[robot_id] == 'WANDERING' and min_dist < self.obstacle_threshold:
            self.previous_state[robot_id] = self.state[robot_id] 
            self.state[robot_id] = 'AVOIDING'
            self.avoid_timer[robot_id] = 15 # Pasos hacia atrás
            
        elif self.state[robot_id] == 'HOMING' and min_dist < self.stop_at_obj_threshold:
            # ¡Llegó al objeto! Empezar a contar
            self.state[robot_id] = 'CALCULATING_PERIMETER'

    def image_yolo_callback(self, msg, robot_id):
        # Si ya ha llegado o está volviendo, ignorar YOLO para no resetear la misión
        if self.state[robot_id] in ['ARRIVED', 'RETURNING_TO_BASE', 'MISSION_COMPLETE', 'CALCULATING_PERIMETER']:
            return
            
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # 1. Ejecutar YOLO Nano
            results = self.yolo_model(cv_image) 
            result = results[0] 
            
            # 2. Visualización (Dibujar cajas para RViz/Ventanas)
            annotated_frame = result.plot() 
            # Publicar a ROS/RViz
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
            self.yolo_image_pubs[robot_id].publish(annotated_msg)
            
            # MOSTRAR VENTANA EMERGENTE (OPCIONAL, COMENTA SI VA LENTO)
            cv2.imshow(f"Vision YOLO - Robot {robot_id}", annotated_frame)
            cv2.waitKey(1) 
            
            # 3. Lógica de detección de objetivo
            object_detected = False
            object_cx = 0
            
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = self.yolo_model.names[class_id]
                
                # OBJETIVO: 'bottle' (botella) o cambia por 'person', 'cup'...
                if class_name == 'bottle': 
                    object_detected = True
                    # Centro de la caja delimitarora
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    object_cx = int((x1 + x2) / 2)
                    break 
            
            if object_detected:
                # El primero que lo ve se convierte en Líder/Buscador
                if not self.global_object_found:
                    self.global_object_found = True
                    self.object_finder_id = robot_id
                    self.get_logger().info(f'¡ROBOT {robot_id} CONFIRMA DETECCIÓN DE OBJETIVO (YOLO)!')
                
                # Si soy yo el que lo ve (Líder)
                if self.object_finder_id == robot_id:
                    self.state[robot_id] = 'HOMING'
                    image_width = cv_image.shape[1]
                    # Error para centrar el giro
                    self.target_error[robot_id] = (object_cx - (image_width / 2)) / (image_width / 2)
                        
            else:
                # Si el líder pierde el objeto en HOMING, resetea a WANDERING
                if self.object_finder_id == robot_id and self.state[robot_id] == 'HOMING':
                    self.state[robot_id] = 'WANDERING'
                    self.global_object_found = False
                    self.object_finder_id = None
                    self.global_targets.clear() 
                    self.get_logger().warn(f'Robot {robot_id} perdió el objeto. Reseteando búsqueda.')
                
        except Exception as e:
            pass

    def publish_swarm_visuals(self):
        marker_array = MarkerArray()
        
        for i in range(self.num_robots):
            try:
                t = self.tf_buffer.lookup_transform('world', f'robot_{i}/base_link', rclpy.time.Time())
                x = t.transform.translation.x
                y = t.transform.translation.y
                
                # Burbuja de Repulsión (Mantenemos por seguridad incluso en el retorno)
                bubble = Marker()
                bubble.header.frame_id = 'world'
                bubble.header.stamp = self.get_clock().now().to_msg()
                bubble.ns = 'repulsion_fields'
                bubble.id = i
                bubble.type = Marker.CYLINDER
                bubble.action = Marker.ADD
                bubble.pose.position.x = x
                bubble.pose.position.y = y
                bubble.pose.position.z = 0.05
                bubble.scale.x = 1.5
                bubble.scale.y = 1.5
                bubble.scale.z = 0.02
                bubble.color = ColorRGBA(r=0.0, g=0.5, b=1.0, a=0.2) # Azul suave
                
                marker_array.markers.append(bubble)
                
                # Flechas de Objetivo (Verde/Rojo para cerco, Blanco para base)
                target_x, target_y = None, None
                arrow_color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.8) # Por defecto blanco
                
                if self.state[i] in ['MOVING_TO_PERIMETER', 'ARRIVED'] and i in self.global_targets:
                    target_x, target_y = self.global_targets[i]
                    if self.state[i] == 'ARRIVED':
                        arrow_color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8) # Verde
                    else:
                        arrow_color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8) # Rojo
                
                elif self.state[i] == 'RETURNING_TO_BASE':
                    target_x, target_y = self.home_poses[i]
                    arrow_color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.6) # Blanco semi
                
                if target_x is not None:
                    arrow = Marker()
                    arrow.header.frame_id = 'world'
                    arrow.header.stamp = self.get_clock().now().to_msg()
                    arrow.ns = 'target_vectors'
                    arrow.id = i + 100 
                    arrow.type = Marker.ARROW
                    arrow.action = Marker.ADD
                    p_start = Point(x=x, y=y, z=0.1)
                    p_end = Point(x=target_x, y=target_y, z=0.1)
                    arrow.points = [p_start, p_end]
                    arrow.scale.x = 0.05 
                    arrow.scale.y = 0.15 
                    arrow.scale.z = 0.15 
                    arrow.color = arrow_color
                    marker_array.markers.append(arrow)
                    
            except Exception:
                pass
        self.marker_pub.publish(marker_array)

    def control_loop(self):
        for i in range(self.num_robots):
            twist = Twist()
            
            # --- DETENER SEGUIDORES SI LIDER ENCUENTRA ---
            if self.global_object_found and self.object_finder_id != i:
                if self.state[i] in ['WANDERING', 'HOMING']:
                    self.state[i] = 'HALTED'
            
            if self.state[i] == 'HALTED' and i in self.global_targets:
                self.state[i] = 'MOVING_TO_PERIMETER'

            # --- LÓGICA DE MOVIMIENTO ---
            
            elif self.state[i] == 'WANDERING':
                obs_dist = self.closest_obs_dist[i]
                obs_angle = self.closest_obs_angle[i]

                teammate_too_close = False
                try:
                    t_me = self.tf_buffer.lookup_transform('world', f'robot_{i}/base_link', rclpy.time.Time())
                    my_x, my_y = t_me.transform.translation.x, t_me.transform.translation.y
                    for j in range(self.num_robots):
                        if i != j:
                            try:
                                t_friend = self.tf_buffer.lookup_transform('world', f'robot_{j}/base_link', rclpy.time.Time())
                                fr_x, fr_y = t_friend.transform.translation.x, t_friend.transform.translation.y
                                if math.hypot(my_x - fr_x, my_y - fr_y) < 1.5:  
                                    teammate_too_close = True
                                    break 
                            except Exception: pass 
                except Exception: pass 

                if obs_dist < 0.75: # Paredes
                    twist.linear.x = 0.05
                    twist.angular.z = -1.2 if obs_angle > 0 else 1.2  
                elif teammate_too_close: # Compañeros
                    twist.linear.x = 0.1   
                    twist.angular.z = 1.8  
                else: # Exploración Heterogénea
                    twist.linear.x = 0.35
                    twist.angular.z = random.uniform(-0.15, 0.15)
                    if i == 0: twist.angular.z -= 0.4  
                    elif i == 1: twist.angular.z += 0.4   
                
            elif self.state[i] == 'AVOIDING': # Marcha atrás
                twist.linear.x = -0.15
                twist.angular.z = 0.8
                self.avoid_timer[i] -= 1
                if self.avoid_timer[i] <= 0:
                    self.state[i] = self.previous_state[i]
                    
            elif self.state[i] == 'HOMING': # Líder acorralando
                twist.linear.x = 0.4
                twist.angular.z = -2.5 * self.target_error[i] 
                
            elif self.state[i] == 'CALCULATING_PERIMETER':
                twist.linear.x, twist.angular.z = 0.0, 0.0
                if len(self.global_targets) == 0:
                    try:
                        t = self.tf_buffer.lookup_transform('world', f'robot_{i}/base_link', rclpy.time.Time())
                        lx, ly, lz = t.transform.translation.x, t.transform.translation.y, t.transform.translation.z
                        # Sacar Yaw del líder
                        q = t.transform.rotation
                        l_yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
                        # El objeto está justo delante del líder
                        ox = lx + 0.35 * math.cos(l_yaw)
                        oy = ly + 0.35 * math.sin(l_yaw)
                        # Calcular puntos del círculo
                        ang_sep = (2 * math.pi) / self.num_robots
                        for j in range(self.num_robots):
                            ang_act = j * ang_sep
                            self.global_targets[j] = (ox + (self.radio_cerco * math.cos(ang_act)),
                                                     oy + (self.radio_cerco * math.sin(ang_act)))
                        self.state[i] = 'MOVING_TO_PERIMETER'
                    except Exception: pass 

            elif self.state[i] == 'MOVING_TO_PERIMETER':
                # Sigue las flechas rojas
                self.move_to_point(i, self.global_targets[i], self.arrival_tolerance, 'ARRIVED', twist)
            
            elif self.state[i] == 'ARRIVED':
                twist.linear.x, twist.angular.z = 0.0, 0.0
                # NUEVO: Lógica del timer de espera
                now = time.time()
                if self.arrived_time[i] is None:
                    self.arrived_time[i] = now # Empezar a contar
                    self.get_logger().info(f'¡Robot {i} en formación. Esperando {self.encircle_duration}s...')
                
                # Comprobar si ha pasado el tiempo
                if (now - self.arrived_time[i]) > self.encircle_duration:
                    self.get_logger().warn(f'Robot {i}: Datos registrados. Volviendo al origen!')
                    self.state[i] = 'RETURNING_TO_BASE'
                    self.global_targets.clear() # Limpiar objetivos del cerco
            
            # NUEVOS ESTADOS DE RETORNO
            elif self.state[i] == 'RETURNING_TO_BASE':
                # Sigue las flechas blancas, manteniendo esquiva de obstáculos por seguridad
                self.move_to_point(i, self.home_poses[i], self.return_tolerance, 'MISSION_COMPLETE', twist)

            elif self.state[i] == 'MISSION_COMPLETE':
                twist.linear.x, twist.angular.z = 0.0, 0.0
                # Solo loguear una vez
                if self.arrived_time[i] is not None:
                    self.get_logger().info(f'¡Robot {i} está en casa! Misión finalizada.')
                    self.arrived_time[i] = None # Bandera para no loguear más
                
            self.cmd_pubs[i].publish(twist)
            
        self.publish_swarm_visuals()

    # Función genérica de navegación reactiva hacia un punto (con LIDAR)
    def move_to_point(self, robot_id, point, tolerance, next_state, twist):
        try:
            t = self.tf_buffer.lookup_transform('world', f'robot_{robot_id}/base_link', rclpy.time.Time())
            mx, my = t.transform.translation.x, t.transform.translation.y
            q = t.transform.rotation
            m_yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
            target_x, target_y = point
            
            dx, dy = target_x - mx, target_y - my
            dist = math.hypot(dx, dy)
            
            if dist < tolerance:
                self.state[robot_id] = next_state
            else:
                # 1. Movimiento normal hacia meta
                t_ang = math.atan2(dy, dx)
                a_diff = (t_ang - m_yaw + math.pi) % (2 * math.pi) - math.pi
                twist.angular.z = 1.5 * a_diff 
                twist.linear.x = 0.3 if abs(a_diff) < 0.5 else 0.0 
                    
                # 2. LIDAR (Repulsión mejorada anti-choque)
                obs_d = self.closest_obs_dist[robot_id]
                obs_a = self.closest_obs_angle[robot_id]
                if obs_d < 0.55: # Zona precaución
                    twist.angular.z = -1.5 if obs_a > 0 else 1.5 
                    if obs_d < 0.30: # Zona crítica: retroceder
                        twist.linear.x = -0.15
                        self.get_logger().warn(f'Robot {robot_id} demasiado cerca de pared! Retrocediendo...')
                    else: # Lentamente
                        twist.linear.x = 0.1
        except Exception: pass

def main(args=None):
    rclpy.init(args=args)
    node = SwarmAILeader()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
