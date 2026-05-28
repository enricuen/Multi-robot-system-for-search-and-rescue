#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from tf2_ros import TransformException, Buffer, TransformListener
import math

class LeaderFollower(Node):
    def __init__(self):
        super().__init__('leader_follower_node')
        
        # Parámetro dinámico
        self.declare_parameter('num_robots', 3)
        self.num_robots = self.get_parameter('num_robots').value
        
        # El líder real sigue siendo el último (el que no tiene a nadie delante)
        self.leader_index = self.num_robots - 1
        self.leader_name = f'robot_{self.leader_index}'
        
        # Publicadores para cada robot
        self.pubs = {f'robot_{i}': self.create_publisher(Twist, f'/robot_{i}/cmd_vel', 10) 
                     for i in range(self.num_robots)}
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.create_timer(0.1, self.control_loop)
        self.get_logger().info(f'Cadena iniciada: {self.num_robots} robots. El líder frontal es {self.leader_name}')

    def control_loop(self):
        for i in range(self.num_robots):
            robot_name = f'robot_{i}'
            msg = Twist()

            # LÓGICA DEL LÍDER (El último robot de la lista)
            if i == self.leader_index:
                msg.linear.x = 0.3  # Un poco más rápido para tirar de la cadena
                msg.angular.z = 0.2
            
            # LÓGICA DE LOS SEGUIDORES (Cada uno sigue al i + 1)
            else:
                target_name = f'robot_{i + 1}'
                try:
                    # Buscamos al robot que está justo delante
                    t = self.tf_buffer.lookup_transform(
                        f'{robot_name}/base_footprint', 
                        f'{target_name}/base_footprint',
                        rclpy.time.Time())

                    dx = t.transform.translation.x
                    dy = t.transform.translation.y
                    
                    # Cálculo de distancia y ángulo
                    # $dist = \sqrt{dx^2 + dy^2}$
                    dist = math.sqrt(dx**2 + dy**2)
                    # $angle = \operatorname{atan2}(dy, dx)$
                    angle = math.atan2(dy, dx)

                    # Control de seguimiento con distancia de seguridad
                    if dist > 0.7:
                        # Velocidad proporcional a la distancia
                        msg.linear.x = 0.6 * (dist - 0.5)
                        # Giro proporcional al ángulo
                        msg.angular.z = 2.0 * angle
                    
                except TransformException:
                    # Si no encuentra al  robto de delante se para
                    continue

            self.pubs[robot_name].publish(msg)

def main():
    rclpy.init()
    node = LeaderFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
