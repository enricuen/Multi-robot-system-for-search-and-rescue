#!/usr/bin/env python3
"""
check_swarm.py - Diagnostico rapido del enjambre.

Verifica para cada robot_i (i en 0..N-1):
  1. TF       : existe transform map -> robot_i/base_link  (spawneado + AMCL)
  2. Scan     : se recibe /robot_i/scan                    (sensor laser ok)
  3. Cmd_vel  : el topic /robot_i/cmd_vel existe           (control ok)
  4. Nav2     : el action server /robot_i/navigate_to_pose responde

Uso:
  # En otra terminal, con el entorno sourceado y la simulacion arrancada:
  ros2 run robot_bringup check_swarm.py --ros-args -p num_robots:=5

  # O directamente:
  python3 check_swarm.py            (usa num_robots=5 por defecto)
  python3 check_swarm.py 3          (comprueba 3 robots)

Codigo de salida: 0 si TODOS los robots estan 100% OK, 1 si alguno falla.
Pensado para lanzarlo ~60 s despues del launch y ver de un vistazo
si algo fallo antes de que lo vea el corrector.
"""

import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import LaserScan
from nav2_msgs.action import NavigateToPose
from tf2_ros import Buffer, TransformListener

# Colores ANSI para que el resultado se lea de un golpe.
VERDE = '\033[92m'
ROJO = '\033[91m'
AMAR = '\033[93m'
GRIS = '\033[90m'
RST = '\033[0m'
BOLD = '\033[1m'

OK = f'{VERDE}OK{RST}'
FAIL = f'{ROJO}FALLO{RST}'


class SwarmChecker(Node):
    def __init__(self, num_robots):
        super().__init__('swarm_checker')
        self.num_robots = num_robots

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Estado de scan: marcamos True al recibir el primer mensaje.
        self.scan_ok = {i: False for i in range(num_robots)}
        self.scan_subs = {}
        for i in range(num_robots):
            self.scan_subs[i] = self.create_subscription(
                LaserScan, f'/robot_{i}/scan',
                lambda msg, rid=i: self._scan_cb(rid), 10)

    def _scan_cb(self, robot_id):
        self.scan_ok[robot_id] = True

    # ------------------------------------------------------------------
    def check_tf(self, robot_id):
        """True si existe map -> robot_i/base_link."""
        try:
            self.tf_buffer.lookup_transform(
                'map', f'robot_{robot_id}/base_link',
                rclpy.time.Time())
            return True
        except Exception:
            return False

    def check_topic_exists(self, topic):
        """True si el topic esta presente en el grafo."""
        topics = dict(self.get_topic_names_and_types())
        return topic in topics

    def check_nav2(self, robot_id, timeout=3.0):
        """True si el action server de Nav2 del robot responde."""
        client = ActionClient(
            self, NavigateToPose,
            f'/robot_{robot_id}/navigate_to_pose')
        ready = client.wait_for_server(timeout_sec=timeout)
        client.destroy()
        return ready


def spin_briefly(node, seconds):
    """Procesa callbacks (TF, scan) durante 'seconds' segundos."""
    end = time.time() + seconds
    while time.time() < end and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)


def main():
    num_robots = 5
    # Permite python3 check_swarm.py 3
    for a in sys.argv[1:]:
        if a.isdigit():
            num_robots = int(a)

    rclpy.init()
    node = SwarmChecker(num_robots)

    # Declara num_robots como parametro tambien (para 'ros2 run ... -p').
    node.declare_parameter('num_robots', num_robots)
    p = node.get_parameter('num_robots').value
    if p and p != num_robots:
        num_robots = p
        node.num_robots = p

    print(f'\n{BOLD}=== Diagnostico del enjambre '
          f'({num_robots} robots) ==={RST}')
    print(f'{GRIS}Recogiendo datos durante ~12 s '
          f'(TF y scan necesitan tiempo)...{RST}\n')

    # Dejamos que lleguen TF y scans.
    spin_briefly(node, 12.0)

    todo_ok = True
    cabecera = (f'{BOLD}{"Robot":<10}{"TF/Spawn":<14}'
                f'{"Scan":<12}{"Cmd_vel":<12}{"Nav2":<10}{RST}')
    print(cabecera)
    print('-' * 58)

    for i in range(num_robots):
        tf_ok = node.check_tf(i)
        sc_ok = node.scan_ok[i]
        cv_ok = node.check_topic_exists(f'/robot_{i}/cmd_vel')
        # 8 s en vez de 3: el Nav2 de robot_0 (primero en arrancar)
        # tarda mas en activar. Con 3 s daba falso negativo.
        nv_ok = node.check_nav2(i, timeout=8.0)

        if not (tf_ok and sc_ok and cv_ok and nv_ok):
            todo_ok = False

        nombre = f'robot_{i}'
        if i == 0:
            nombre += f' {AMAR}(amb){RST}'

        fila = (
            f'{nombre:<19}'
            f'{(OK if tf_ok else FAIL):<23}'
            f'{(OK if sc_ok else FAIL):<23}'
            f'{(OK if cv_ok else FAIL):<23}'
            f'{(OK if nv_ok else FAIL)}'
        )
        print(fila)

    print('-' * 58)
    if todo_ok:
        print(f'\n{VERDE}{BOLD}TODOS los robots estan OPERATIVOS. '
              f'Listo para la correccion.{RST}\n')
        code = 0
    else:
        print(f'\n{ROJO}{BOLD}Hay robots con fallos.{RST} '
              f'Sugerencias:')
        print(f'  - Si falla TF/Spawn de robot_0: sube '
              f'{BOLD}T_ARRANQUE_GAZEBO{RST} en el launch.')
        print(f'  - Si falla TF de robots altos (3,4): sube '
              f'{BOLD}T_ENTRE_ROBOTS{RST}.')
        print(f'  - Si falla solo Nav2: espera mas tiempo y '
              f'reejecuta este script (Nav2 tarda en activar).')
        print(f'  - Si falla Scan pero TF ok: revisa el bridge '
              f'de ese robot.\n')
        code = 1

    node.destroy_node()
    rclpy.shutdown()
    sys.exit(code)


if __name__ == '__main__':
    main()
