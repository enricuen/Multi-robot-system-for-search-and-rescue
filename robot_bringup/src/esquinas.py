#!/usr/bin/env python3
"""
esquinas.py  -  Busqueda y rescate en enjambre.

Flujo por robot (1-4):
  NAVIGATING_CORNERS -> SCANNING_CORNER (360 deg) -> WANDERING (patrulla
  local alrededor de la esquina) -> si detecta persona: HOMING ->
  STOPPED_AT_TARGET -> input terminal -> ambulancia o acordonamiento.

Robot 0: ambulancia, quieta en base hasta ser llamada.

Lanzamiento independiente (SIN run_swarm en el launch):
  ros2 run robot_bringup esquinas.py --ros-args -p num_robots:=5
"""

import math
import threading

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from tf2_ros import Buffer, TransformListener
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# CONFIGURACION CENTRAL  (un solo sitio para tocar valores)
# ---------------------------------------------------------------------------

# Esquinas a las que va cada robot al arrancar
ESQUINAS = {
    1: (-13.0,  13.0),   # NO
    2: ( 13.0,  13.0),   # NE
    3: ( 13.0, -13.0),   # SE
    4: (-13.0, -13.0),   # SO
}

# Patrulla LOCAL: 2 puntos cerca de la esquina asignada, a ~3-4 m hacia
# el interior. Cada robot solo usa SUS puntos -> sin cruces entre robots.
PATROL = {
    1: [(-10.0,  10.0), (-8.0,  8.0)],
    2: [( 10.0,  10.0), ( 8.0,  8.0)],
    3: [( 10.0, -10.0), ( 8.0, -8.0)],
    4: [(-10.0, -10.0), (-8.0, -8.0)],
}

SCAN_SECS     = 13.0   # duracion del barrido (360 deg a 0.5 rad/s = 12.6 s)
SCAN_W        = 0.5    # velocidad angular del barrido (rad/s)
DISPATCH_SECS = 10.0   # espera antes de mandar robots a esquinas.
                       # 10 s da margen para que Nav2 este listo cuando
                       # se lanza esquinas.py justo despues del launch.
OBSTACLE_D    = 1.0    # m - umbral obstaculo para activar AVOIDING
STOP_D        = 0.8    # m - umbral para parar en HOMING
YOLO_CONF     = 0.5
YOLO_OBJ      = 'person'
MAX_RETRIES   = 3      # reintentos si Nav2 aborta el goal de esquina


class SwarmNode(Node):

    def __init__(self):
        super().__init__('swarm_search_rescue_node')
        self.declare_parameter('num_robots', 5)
        self.num_robots = self.get_parameter('num_robots').value

        # --- Callback groups ---
        self.cbg_r = ReentrantCallbackGroup()         # acciones / timers
        self.cbg_s = MutuallyExclusiveCallbackGroup() # sensores
        self.cbg_t = MutuallyExclusiveCallbackGroup() # control loop

        # --- Utilidades ---
        self.bridge  = CvBridge()
        self.yolo    = YOLO('yolov8n.pt')
        self.tf_buf  = Buffer()
        self.tf_list = TransformListener(self.tf_buf, self)
        self._lock   = threading.Lock()

        # --- Estado por robot ---
        self.state        = {}
        self.avoid_cnt    = {}
        self.target_err   = {}
        self.patience     = {}
        self.patrol_idx   = {}
        self.gh           = {}   # goal handles activos
        self.corner_tries = {}   # reintentos de goal a esquina

        self.busy    = set()
        self.ui_lock = False

        # --- ROS interfaces ---
        self.cpub   = {}   # cmd_vel publishers
        self.ypub   = {}   # yolo image publishers
        self.isub   = {}   # image subscribers
        self.ssub   = {}   # scan subscribers
        self.client = {}   # nav2 action clients

        for i in range(self.num_robots):
            n = f'robot_{i}'
            self.state[i]        = ('STATIONARY_AMBULANCE'
                                    if i == 0 else 'NAVIGATING_CORNERS')
            self.avoid_cnt[i]    = 0
            self.target_err[i]   = 0.0
            self.patience[i]     = 0
            self.patrol_idx[i]   = 0
            self.gh[i]           = None
            self.corner_tries[i] = 0

            self.cpub[i] = self.create_publisher(Twist, f'/{n}/cmd_vel', 10)
            self.ypub[i] = self.create_publisher(
                Image, f'/{n}/camera/yolo_image', 10)
            self.isub[i] = self.create_subscription(
                Image, f'/{n}/camera/image_raw',
                lambda m, r=i: self._img(m, r), 10,
                callback_group=self.cbg_s)
            self.ssub[i] = self.create_subscription(
                LaserScan, f'/{n}/scan',
                lambda m, r=i: self._scan(m, r), 10,
                callback_group=self.cbg_s)
            self.client[i] = ActionClient(
                self, NavigateToPose, f'/{n}/navigate_to_pose',
                callback_group=self.cbg_r)

        self.create_timer(0.1, self._loop, callback_group=self.cbg_t)

        # Timer de despacho: espera DISPATCH_SECS antes de mandar a esquinas
        self._dt = self.create_timer(
            DISPATCH_SECS, self._go_corners, callback_group=self.cbg_r)

        self.get_logger().info(
            f'Swarm listo ({self.num_robots} robots). '
            f'Despachando a esquinas en {DISPATCH_SECS:.0f} s...')

    # =========================================================
    #  DESPACHO A ESQUINAS
    # =========================================================
    def _go_corners(self):
        """Manda todos los robots a su esquina en hilos separados.
        Cada hilo espera a que Nav2 este listo antes de enviar el goal,
        sin crear timers en cascada que saturan el sistema."""
        self._dt.cancel()
        for rid in range(1, self.num_robots):
            threading.Thread(
                target=self._wait_and_send_corner,
                args=(rid,), daemon=True).start()

    def _wait_and_send_corner(self, rid):
        """Espera a que Nav2 del robot este listo y manda el goal.
        Si el goal es rechazado, reintenta con pausa para no saturar Nav2."""
        if rid not in ESQUINAS:
            return
        if self.state[rid] != 'NAVIGATING_CORNERS':
            return
        client = self.client[rid]
        self.get_logger().info(f'Robot {rid}: esperando Nav2...')
        ready = client.wait_for_server(timeout_sec=60.0)
        if not ready:
            self.get_logger().error(
                f'Robot {rid}: Nav2 no responde tras 60 s. Abortando.')
            return
        # Pausa de 2 s tras el wait_for_server para dar margen a Nav2
        import time as _time
        _time.sleep(2.0)
        if self.state[rid] != 'NAVIGATING_CORNERS':
            return  # Estado cambio mientras esperabamos
        x, y = ESQUINAS[rid]
        yaw  = math.atan2(-y, -x)
        ok   = self._nav(rid, x, y, math.cos(yaw/2), math.sin(yaw/2))
        if ok:
            self.get_logger().info(f'Robot {rid} -> esquina ({x},{y})')

    def _retry_corner(self, rid):
        if self.state[rid] == 'NAVIGATING_CORNERS':
            threading.Thread(
                target=self._wait_and_send_corner,
                args=(rid,), daemon=True).start()

    # =========================================================
    #  NAV2 - envio de goal
    # =========================================================
    def _nav(self, rid, x, y, w=1.0, z=0.0):
        c = self.client[rid]
        if not c.server_is_ready():
            if not c.wait_for_server(timeout_sec=1.0):
                return False
        g = NavigateToPose.Goal()
        g.pose.header.frame_id    = 'map'
        g.pose.header.stamp       = self.get_clock().now().to_msg()
        g.pose.pose.position.x    = float(x)
        g.pose.pose.position.y    = float(y)
        g.pose.pose.orientation.z = float(z)
        g.pose.pose.orientation.w = float(w)
        f = c.send_goal_async(g)
        f.add_done_callback(lambda fut, r=rid: self._on_accept(fut, r))
        return True

    def _on_accept(self, future, rid):
        try:
            gh = future.result()
        except Exception as e:
            self.get_logger().warn(f'Robot {rid} accept err: {e}')
            return
        if not gh.accepted:
            self.get_logger().warn(f'Robot {rid}: goal rechazado.')
            self.gh[rid] = None
            if self.state[rid] == 'NAVIGATING_CORNERS':
                import time as _t
                def _retry_delayed(r=rid):
                    _t.sleep(3.0)  # pausa antes de reintentar
                    self._wait_and_send_corner(r)
                threading.Thread(target=_retry_delayed, daemon=True).start()
            return
        self.gh[rid] = gh
        gh.get_result_async().add_done_callback(
            lambda f, r=rid: self._on_result(f, r))

    def _on_result(self, future, rid):
        try:
            status = future.result().status  # 4=OK, 6=ABORT
        except Exception:
            status = 6
        self.gh[rid] = None

        # --- Ambulancia ---
        if rid == 0:
            self.state[0] = 'STATIONARY_AMBULANCE'
            with self._lock:
                self.busy.discard(0)
            if status == 4:
                self.get_logger().info('Ambulancia: objetivo alcanzado.')
            return

        st = self.state[rid]

        # --- Llego a la esquina ---
        if st == 'NAVIGATING_CORNERS':
            if status == 4:
                self.corner_tries[rid] = 0
                self.get_logger().info(
                    f'Robot {rid}: en esquina. Barrido 360deg ({SCAN_SECS}s).')
                self.state[rid] = 'SCANNING_CORNER'
                self.create_timer(
                    SCAN_SECS,
                    lambda r=rid: self._end_scan(r),
                    callback_group=self.cbg_r)
            else:
                # Aborto: reintentar hasta MAX_RETRIES
                self.corner_tries[rid] += 1
                if self.corner_tries[rid] <= MAX_RETRIES:
                    self.get_logger().warn(
                        f'Robot {rid}: esquina abortada '
                        f'({self.corner_tries[rid]}/{MAX_RETRIES}).')
                    threading.Thread(
                        target=self._wait_and_send_corner,
                        args=(rid,), daemon=True).start()
                else:
                    self.get_logger().warn(
                        f'Robot {rid}: {MAX_RETRIES} fallos. '
                        f'Patrullando zona de esquina.')
                    self.corner_tries[rid] = 0
                    self.state[rid] = 'WANDERING'

        # --- Llego a waypoint de patrulla ---
        elif st == 'WANDERING':
            pts = PATROL.get(rid, [])
            if pts:
                self.patrol_idx[rid] = (
                    (self.patrol_idx[rid] + 1) % len(pts))

        # --- Llego junto a la persona (acordonamiento) ---
        elif st == 'NAVIGATING_TO_LEADER':
            self.state[rid] = 'STOPPED_AT_TARGET'

    def _end_scan(self, rid):
        if self.state[rid] == 'SCANNING_CORNER':
            self.get_logger().info(
                f'Robot {rid}: barrido completo. Iniciando patrulla local.')
            self.state[rid] = 'WANDERING'

    # =========================================================
    #  RESCATE
    # =========================================================
    def _ambulance(self, leader_id):
        try:
            with self._lock:
                self.busy.add(0)
            t  = self.tf_buf.lookup_transform(
                'map', f'robot_{leader_id}/base_link', rclpy.time.Time())
            lx = t.transform.translation.x
            ly = t.transform.translation.y
            q  = t.transform.rotation
            self.state[0] = 'NAVIGATING_TO_LEADER'
            self._nav(0, lx, ly, q.w, q.z)
            self.get_logger().info(
                f'Ambulancia -> robot_{leader_id} en ({lx:.1f},{ly:.1f})')
        except Exception as e:
            self.get_logger().warn(f'ambulance err: {e}')

    def _cordon(self, leader_id):
        try:
            t   = self.tf_buf.lookup_transform(
                'map', f'robot_{leader_id}/base_link', rclpy.time.Time())
            lx  = t.transform.translation.x
            ly  = t.transform.translation.y
            q   = t.transform.rotation
            yaw = math.atan2(2*(q.w*q.z + q.x*q.y),
                             1 - 2*(q.y*q.y + q.z*q.z))
            px  = lx + 1.0*math.cos(yaw)
            py  = ly + 1.0*math.sin(yaw)
            with self._lock:
                libres = [i for i in range(1, self.num_robots)
                          if i not in self.busy and i != leader_id]
            for idx, r in enumerate(libres[:2]):
                with self._lock:
                    self.busy.add(r)
                ang = yaw + (2.09 if idx == 0 else -2.09)
                gx  = px + 1.2*math.cos(ang)
                gy  = py + 1.2*math.sin(ang)
                self.state[r] = 'NAVIGATING_TO_LEADER'
                self._nav(r, gx, gy,
                          math.cos((ang+3.14)/2), math.sin((ang+3.14)/2))
                self.get_logger().info(
                    f'Robot {r} -> cordon en ({gx:.1f},{gy:.1f})')
        except Exception as e:
            self.get_logger().warn(f'cordon err: {e}')

    # =========================================================
    #  SENSORES
    # =========================================================
    def _scan(self, msg, rid):
        st = self.state[rid]
        if st in ('NAVIGATING_CORNERS', 'NAVIGATING_TO_LEADER',
                  'STATIONARY_AMBULANCE', 'RETURNING_TO_BASE',
                  'SCANNING_CORNER'):
            return

        r  = np.array(msg.ranges)
        a  = np.linspace(msg.angle_min, msg.angle_max, len(r))
        ok = (r > msg.range_min) & (r < msg.range_max)
        vr = r[ok]
        mg = float(np.min(vr)) if len(vr) else 10.0

        front = (np.abs(a) < 0.15) | (np.abs(a) > 2*math.pi - 0.15)
        fr    = r[ok & front]
        mf    = float(np.min(fr)) if len(fr) else 10.0

        if st == 'HOMING' and mf < STOP_D:
            self.state[rid] = 'STOPPED_AT_TARGET'

            def ask():
                print('\n' + '='*50)
                resp = input(
                    f'[CENTRALITA] Robot {rid} en contacto. '
                    f"Victima 'herida' o 'a salvo'?: ").strip().lower()
                print('='*50 + '\n')
                if resp == 'herida':
                    self.get_logger().info('CODIGO ROJO. Ambulancia en camino.')
                    self._ambulance(rid)
                else:
                    self.get_logger().info('Acordonando zona.')
                    self._cordon(rid)
                self.ui_lock = False

            threading.Thread(target=ask, daemon=True).start()
            return

        if st == 'WANDERING' and mg < OBSTACLE_D:
            self.state[rid] = 'AVOIDING'
            self.avoid_cnt[rid] = 20

    def _img(self, msg, rid):
        st = self.state[rid]
        # YOLO solo activo en estados donde tiene sentido detectar
        if st not in ('WANDERING', 'SCANNING_CORNER', 'HOMING',
                      'NAVIGATING_CORNERS'):
            return
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            res = self.yolo(img, verbose=False)
            self.ypub[rid].publish(
                self.bridge.cv2_to_imgmsg(res[0].plot(), 'bgr8'))

            persona = None
            for det in res:
                for box in det.boxes:
                    if (self.yolo.names[int(box.cls[0])] == YOLO_OBJ and
                            float(box.conf[0]) > YOLO_CONF):
                        persona = box
                        break
                if persona:
                    break

            if persona:
                self.patience[rid] = 30

                if st in ('WANDERING', 'SCANNING_CORNER',
                          'NAVIGATING_CORNERS') and not self.ui_lock:
                    # Verificar que no hay otro robot busy cerca
                    cerca = False
                    try:
                        t1 = self.tf_buf.lookup_transform(
                            'map', f'robot_{rid}/base_link', rclpy.time.Time())
                        x1 = t1.transform.translation.x
                        y1 = t1.transform.translation.y
                        with self._lock:
                            blist = list(self.busy)
                        for bid in blist:
                            if bid in (rid, 0):
                                continue
                            t2 = self.tf_buf.lookup_transform(
                                'map', f'robot_{bid}/base_link',
                                rclpy.time.Time())
                            if math.hypot(
                                    x1 - t2.transform.translation.x,
                                    y1 - t2.transform.translation.y) < 3.5:
                                cerca = True
                                break
                    except Exception:
                        pass

                    if not cerca:
                        self.ui_lock = True
                        with self._lock:
                            self.busy.add(rid)
                        if self.gh[rid]:
                            self.gh[rid].cancel_goal_async()
                            self.gh[rid] = None
                        self.state[rid] = 'HOMING'
                        self.get_logger().info(
                            f'Robot {rid}: PERSONA DETECTADA. HOMING.')

                if self.state[rid] == 'HOMING':
                    x1, _, x2, _ = persona.xyxy[0].cpu().numpy()
                    c = (x1 + x2) / 2.0
                    self.target_err[rid] = float(
                        (c - img.shape[1]/2) / (img.shape[1]/2))

            else:
                # Sin persona visible
                if st == 'HOMING':
                    # NO cancelamos HOMING por falta de deteccion.
                    # Cuando el robot esta muy cerca, YOLO pierde la
                    # persona (ve solo piernas/suelo). El robot ya se
                    # comprometio: sigue avanzando recto hasta que el
                    # laser confirme proximidad (STOP_D) en _scan.
                    # Solo cancelamos si llevamos mucho tiempo sin verla
                    # Y el robot no ha avanzado (posible falsa alarma lejana).
                    if self.patience[rid] > 0:
                        self.patience[rid] -= 1
                    # Con patience=0 mantenemos HOMING igualmente:
                    # el laser parara al robot cuando llegue.

        except Exception as e:
            self.get_logger().debug(f'img {rid}: {e}')

    # =========================================================
    #  CONTROL LOOP  (10 Hz)
    # =========================================================
    def _loop(self):
        for i in range(self.num_robots):
            st    = self.state[i]
            twist = Twist()

            # Nav2 al mando: no tocamos cmd_vel
            if st in ('NAVIGATING_CORNERS', 'NAVIGATING_TO_LEADER',
                      'RETURNING_TO_BASE'):
                continue

            if st in ('STATIONARY_AMBULANCE', 'STOPPED_AT_TARGET'):
                pass   # twist cero

            elif st == 'SCANNING_CORNER':
                twist.angular.z = SCAN_W   # giro en sitio

            elif st == 'WANDERING':
                # Enviar a Nav2 el siguiente waypoint de la zona de esquina
                if i in PATROL and self.gh[i] is None:
                    pts = PATROL[i]
                    wx, wy = pts[self.patrol_idx[i] % len(pts)]
                    yaw = math.atan2(-wy, -wx)
                    ok  = self._nav(i, wx, wy,
                                    math.cos(yaw/2), math.sin(yaw/2))
                    if ok:
                        self.get_logger().info(
                            f'Robot {i} patrulla zona esquina '
                            f'({wx:.0f},{wy:.0f})')
                continue   # Nav2 conduce, no publicamos cmd_vel

            elif st == 'AVOIDING':
                twist.linear.x  = -0.15
                twist.angular.z =  0.5
                self.avoid_cnt[i] -= 1
                if self.avoid_cnt[i] <= 0:
                    self.state[i] = 'WANDERING'

            elif st == 'HOMING':
                twist.linear.x  =  0.35
                twist.angular.z = -0.5 * self.target_err[i]

            self.cpub[i].publish(twist)


# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = SwarmNode()
    ex   = MultiThreadedExecutor(num_threads=8)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        ex.shutdown()
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
