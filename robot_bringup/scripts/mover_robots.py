import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import argparse

class MultiRobotCircle(Node):
    def __init__(self, num_robots):
        super().__init__('multi_robot_circle')
        self.publishers_ = []
        
        # Crear un publicador dinámico para cada robot según la cantidad indicada
        for i in range(num_robots):
            topic_name = f'/robot_{i}/cmd_vel'
            pub = self.create_publisher(Twist, topic_name, 10)
            self.publishers_.append(pub)
            self.get_logger().info(f'Publicador configurado en: {topic_name}')
            
        # Timer para publicar los comandos de movimiento a 10 Hz
        timer_period = 0.1  # 0.1 segundos = 10 Hz
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.get_logger().info(f'Iniciando trayectoria circular para {num_robots} robots. Presiona Ctrl+C para detener.')

    def timer_callback(self):
        # Crear el mensaje de velocidad (Twist)
        msg = Twist()
        
        # Configurar la cinemática del círculo (R = V/W)
        msg.linear.x = 0.5  # Avanza a 0.5 m/s
        msg.angular.z = 0.5 # Gira a 0.5 rad/s (Radio resultante = 1 metro)
        
        # Enviar el comando a todos los publicadores registrados
        for pub in self.publishers_:
            pub.publish(msg)

def main(args=None):
    # 1. Configurar el lector de argumentos de la terminal
    parser = argparse.ArgumentParser(description='Mueve un enjambre de robots en circulo.')
    parser.add_argument('-n', '--num_robots', type=int, default=3, 
                        help='Numero total de robots en la simulacion (por defecto: 3)')
    
    # 2. Separar nuestros argumentos de los argumentos internos de ROS 2
    parsed_args, ros_args = parser.parse_known_args(sys.argv)
    
    # 3. Iniciar ROS 2 pasándole solo sus argumentos
    rclpy.init(args=ros_args)
    
    # 4. Instanciar el nodo inyectando el número de robots leído
    node = MultiRobotCircle(num_robots=parsed_args.num_robots) 
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('\nDeteniendo el enjambre de robots...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
