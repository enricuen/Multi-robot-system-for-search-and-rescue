#!/usr/bin/env python3
"""
map_relay_robusto.py - Republica /robot_0/map en /map ESPERANDO a que
el topic de origen exista.

El 'topic_tools relay' estandar no reintenta: si /robot_0/map aun no
existe cuando arranca (porque el map_server de robot_0 tarda en
activar), el relay se queda sin fuente y /map nunca se publica.

Este script espera a que /robot_0/map exista y entonces hace de puente
con QoS transient_local (igual que un map_server), para que RViz y
cualquier otro nodo reciban el mapa aunque se suscriban tarde.

Uso (lo invoca el launch):
  ros2 run robot_bringup map_relay_robusto.py
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from nav_msgs.msg import OccupancyGrid

ORIGEN = '/robot_0/map'
DESTINO = '/map'


class MapRelayRobusto(Node):
    def __init__(self):
        super().__init__('map_relay_robusto')

        # QoS transient_local: quien se suscriba despues recibe el
        # ultimo mapa publicado (igual que hace map_server).
        qos = QoSProfile(depth=1)
        qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        qos.reliability = QoSReliabilityPolicy.RELIABLE

        self.pub = self.create_publisher(OccupancyGrid, DESTINO, qos)
        self.sub = None
        self.ultimo_mapa = None

        # Cada 2 s comprueba si ya existe el topic de origen.
        self.timer = self.create_timer(2.0, self._intentar_conectar)
        self.get_logger().info(
            f'Esperando a que exista {ORIGEN}...')

    def _intentar_conectar(self):
        if self.sub is not None:
            return  # ya conectado
        topics = dict(self.get_topic_names_and_types())
        if ORIGEN in topics:
            qos = QoSProfile(depth=1)
            qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
            qos.reliability = QoSReliabilityPolicy.RELIABLE
            self.sub = self.create_subscription(
                OccupancyGrid, ORIGEN, self._cb_mapa, qos)
            self.get_logger().info(
                f'{ORIGEN} detectado. Relay {ORIGEN} -> {DESTINO} '
                f'activo.')

    def _cb_mapa(self, msg):
        self.ultimo_mapa = msg
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MapRelayRobusto()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
