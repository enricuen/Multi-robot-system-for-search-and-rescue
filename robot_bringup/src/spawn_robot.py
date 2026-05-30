#!/usr/bin/env python3
"""
spawn_robot.py - Spawn ROBUSTO de un robot en Gazebo Harmonic.

A diferencia del nodo 'create' de ros_gz_sim (un solo intento, falla en
silencio si Gazebo esta saturado), este script:

  1. Espera a que /{name}/robot_description este publicado.
  2. Llama a 'ros2 run ros_gz_sim create' (mismo mecanismo de siempre).
  3. VERIFICA via el servicio de Gazebo que el modelo realmente existe.
  4. Si no existe, REINTENTA hasta MAX_INTENTOS.

Esto convierte "casi siempre salen todos" en "siempre salen todos",
que es justo lo que necesitas para una correccion fiable, sobre todo
en una maquina de 16 GB que va justa con 5 stacks de Nav2.

Uso (lo invoca el launch, no manual normalmente):
  ros2 run robot_bringup spawn_robot.py \
      --name robot_0 --x -5.0 --y -1.35 --z 0.1 --yaw 1.5708 \
      --world my_world_person

Args:
  --name   nombre/namespace del robot (robot_0, robot_1, ...)
  --x --y --z --yaw   pose inicial
  --world  nombre del mundo Gazebo (sin .sdf). Para verificar existencia.
"""

import argparse
import subprocess
import sys
import time

import rclpy
from rclpy.node import Node

MAX_INTENTOS = 5
TIMEOUT_RSP = 30.0       # s esperando robot_description
PAUSA_TRAS_CREATE = 3.0  # s antes de verificar (Gazebo necesita procesar)
PAUSA_ENTRE_INTENTOS = 2.0


def log(msg):
    print(f'[spawn_robot] {msg}', flush=True)


def robot_description_listo(node, name, timeout):
    """Espera a que exista el topic /{name}/robot_description con datos."""
    fin = time.time() + timeout
    topic = f'/{name}/robot_description'
    while time.time() < fin and rclpy.ok():
        topics = dict(node.get_topic_names_and_types())
        if topic in topics:
            # El topic existe; damos un margen corto para que publique.
            time.sleep(1.0)
            return True
        time.sleep(0.5)
    return False


def modelo_existe_en_gazebo(name, world):
    """Comprueba via 'gz model' si el modelo esta en la simulacion.

    Usa la CLI de gz, que no depende de bridges ROS y es fiable para
    verificar el estado real de Gazebo Harmonic.
    """
    try:
        # 'gz model --list' lista los modelos del mundo activo.
        out = subprocess.run(
            ['gz', 'model', '--list'],
            capture_output=True, text=True, timeout=8.0)
        if name in out.stdout:
            return True
    except Exception as e:
        log(f'No se pudo consultar gz model (--list): {e}')
    # Fallback: intentar info del modelo concreto.
    try:
        out = subprocess.run(
            ['gz', 'model', '-m', name],
            capture_output=True, text=True, timeout=8.0)
        # Si el modelo no existe, gz devuelve error / texto vacio util.
        if name in out.stdout and 'Pose' in out.stdout:
            return True
    except Exception:
        pass
    return False


def hacer_create(name, x, y, z, yaw):
    """Lanza el 'create' estandar de ros_gz_sim (un intento)."""
    cmd = [
        'ros2', 'run', 'ros_gz_sim', 'create',
        '-topic', f'/{name}/robot_description',
        '-name', name,
        '-x', str(x), '-y', str(y), '-z', str(z),
        '-Y', str(yaw),
    ]
    log(f'Ejecutando create para {name}: {" ".join(cmd)}')
    try:
        subprocess.run(cmd, timeout=30.0)
    except subprocess.TimeoutExpired:
        log(f'create de {name} excedio timeout (puede haber entrado igual)')
    except Exception as e:
        log(f'create de {name} fallo: {e}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--x', type=float, default=0.0)
    parser.add_argument('--y', type=float, default=0.0)
    parser.add_argument('--z', type=float, default=0.1)
    parser.add_argument('--yaw', type=float, default=0.0)
    parser.add_argument('--world', default='')
    # ROS pasa args extra (--ros-args ...); los ignoramos.
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = rclpy.create_node(f'spawn_helper_{args.name}')

    log(f'Esperando robot_description de {args.name} '
        f'(max {TIMEOUT_RSP:.0f}s)...')
    if not robot_description_listo(node, args.name, TIMEOUT_RSP):
        log(f'ERROR: robot_description de {args.name} no aparecio. '
            f'Abortando spawn de este robot.')
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)
    log(f'robot_description de {args.name} OK.')

    exito = False
    for intento in range(1, MAX_INTENTOS + 1):
        log(f'--- {args.name}: intento de spawn {intento}/'
            f'{MAX_INTENTOS} ---')
        hacer_create(args.name, args.x, args.y, args.z, args.yaw)

        time.sleep(PAUSA_TRAS_CREATE)

        if modelo_existe_en_gazebo(args.name, args.world):
            log(f'EXITO: {args.name} verificado en Gazebo '
                f'(intento {intento}).')
            exito = True
            break
        else:
            log(f'{args.name} aun NO esta en Gazebo. '
                f'Reintentando en {PAUSA_ENTRE_INTENTOS:.0f}s...')
            time.sleep(PAUSA_ENTRE_INTENTOS)

    node.destroy_node()
    rclpy.shutdown()

    if not exito:
        log(f'FALLO DEFINITIVO: {args.name} no entro tras '
            f'{MAX_INTENTOS} intentos. Revisa carga de CPU/RAM.')
        sys.exit(1)
    sys.exit(0)


if __name__ == '__main__':
    main()
