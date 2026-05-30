#!/usr/bin/env python3
"""
test_nav_goal.py - Diagnostico de UN goal de Nav2 para un robot.

Envia un goal a /robot_<id>/navigate_to_pose y reporta EXACTAMENTE
que responde Nav2: aceptado/rechazado, y el resultado (SUCCEEDED,
ABORTED, CANCELED) con el codigo de error.

Sirve para distinguir las dos causas de "el robot da vueltas en vez
de ir a su esquina":
  - Si un goal CERCANO y valido tiene exito  -> las esquinas (9,9)
    estan fuera del mapa / en pared (arreglar coordenadas).
  - Si NINGUN goal tiene exito                -> Nav2 no localizado /
    costmap no poblado (arreglar timing / AMCL).

Uso:
  # Goal por defecto: 2 m delante del spawn del robot (deberia ser valido)
  ros2 run robot_bringup test_nav_goal.py --id 1

  # Goal a una esquina concreta para ver si Nav2 la rechaza:
  ros2 run robot_bringup test_nav_goal.py --id 1 --x 9.0 --y 9.0

  # Goal a un punto que sabes libre:
  ros2 run robot_bringup test_nav_goal.py --id 2 --x -3.0 --y 0.0
"""

import argparse
import sys

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose


class NavGoalTester(Node):
    def __init__(self, robot_id, x, y):
        super().__init__(f'test_nav_goal_{robot_id}')
        self.robot_id = robot_id
        self.x = x
        self.y = y
        self.client = ActionClient(
            self, NavigateToPose,
            f'/robot_{robot_id}/navigate_to_pose')
        self.done = False
        self.exit_code = 1

    def run(self):
        print(f'\n[test] Esperando action server de robot_'
              f'{self.robot_id} (max 15 s)...')
        if not self.client.wait_for_server(timeout_sec=15.0):
            print(f'[test] FALLO: el action server '
                  f'/robot_{self.robot_id}/navigate_to_pose NO responde.')
            print('[test] -> Nav2 de este robot no esta activo. '
                  'Espera mas tiempo tras el launch.')
            self.exit_code = 1
            self.done = True
            return

        print(f'[test] Servidor OK. Enviando goal '
              f'(x={self.x}, y={self.y}) en frame "map"...')
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(self.x)
        goal.pose.pose.position.y = float(self.y)
        goal.pose.pose.orientation.w = 1.0

        self.client.send_goal_async(goal).add_done_callback(
            self._on_response)

    def _on_response(self, future):
        gh = future.result()
        if not gh.accepted:
            print('\n[test] >>> Nav2 RECHAZO el goal (not accepted).')
            print('[test] Causa tipica: costmap no listo o servidor '
                  'aun activando. El robot caeria a WANDERING.')
            self.exit_code = 1
            self.done = True
            return
        print('[test] Goal ACEPTADO por Nav2. Esperando resultado...')
        gh.get_result_async().add_done_callback(self._on_result)

    def _on_result(self, future):
        status = future.result().status
        # 4=SUCCEEDED, 5=CANCELED, 6=ABORTED (action_msgs GoalStatus)
        nombres = {
            1: 'ACCEPTED', 2: 'EXECUTING', 3: 'CANCELING',
            4: 'SUCCEEDED', 5: 'CANCELED', 6: 'ABORTED'}
        nombre = nombres.get(status, f'DESCONOCIDO({status})')
        print(f'\n[test] >>> RESULTADO: {nombre}')
        if status == 4:
            print('[test] EXITO: este goal es valido y Nav2 mueve el '
                  'robot.')
            print('[test] -> Si las esquinas (9,9) fallan pero este '
                  'punto NO, el problema son las COORDENADAS de las '
                  'esquinas (fuera de mapa / en pared).')
            self.exit_code = 0
        elif status == 6:
            print('[test] ABORTED: Nav2 acepto pero no pudo ejecutar.')
            print('[test] Causas: goal en celda ocupada/desconocida, '
                  'no hay plan posible, o robot mal localizado.')
            self.exit_code = 1
        elif status == 5:
            print('[test] CANCELED: el goal fue cancelado.')
            self.exit_code = 1
        else:
            print(f'[test] Estado inesperado: {nombre}')
            self.exit_code = 1
        self.done = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, required=True,
                        help='id del robot (1, 2, ...)')
    parser.add_argument('--x', type=float, default=None)
    parser.add_argument('--y', type=float, default=None)
    args, _ = parser.parse_known_args()

    # Goal por defecto: punto cercano y tipicamente valido.
    # El spawn es x = -5 + id*2.0, y = -1.35. Ponemos el goal
    # 2 m "hacia dentro" del almacen (y = +1.0) para que sea
    # alcanzable y claramente en zona libre.
    if args.x is None:
        args.x = -5.0 + args.id * 2.0
    if args.y is None:
        args.y = 1.0

    rclpy.init()
    node = NavGoalTester(args.id, args.x, args.y)
    node.run()
    while rclpy.ok() and not node.done:
        rclpy.spin_once(node, timeout_sec=0.2)
    code = node.exit_code
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(code)


if __name__ == '__main__':
    main()
