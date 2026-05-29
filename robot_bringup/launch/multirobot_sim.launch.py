import os
import yaml
import tempfile
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
    IncludeLaunchDescription,
    TimerAction,
    ExecuteProcess,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, Command

# =============================================================================
#  CONSTANTES DE TIMING  (ajusta si tu ordenador es lento)
# =============================================================================
# Tiempo entre el arranque de un robot y el siguiente. Con 5 robots y Nav2
# por robot, 8 s es un valor seguro. Si tu maquina es potente puedes bajarlo
# a 6 s; si va justa, subelo a 10 s.
T_ENTRE_ROBOTS = 8.0

# Retardo desde que se publica robot_description hasta que se hace 'create'.
# Garantiza que el topic /{name}/robot_description ya existe cuando Gazebo
# intenta spawnear. ESTA es la correccion de "no aparecen todos los robots".
T_RSP_A_SPAWN = 2.5

# Retardo desde el spawn hasta arrancar bridge + TF de ese robot.
T_SPAWN_A_BRIDGE = 2.0

# Retardo desde el bridge hasta arrancar Nav2 de ese robot.
T_BRIDGE_A_NAV2 = 3.0

# RETARDO INICIAL GLOBAL antes de empezar a spawnear NADA.
# Da tiempo a que Gazebo Harmonic cargue el mundo (my_world_person.sdf)
# y levante el servicio de spawn. Sin esto, el PRIMER robot (robot_0)
# intenta spawnear antes de que Gazebo este listo y falla en silencio:
# por eso aparecian robot_1..4 pero NO robot_0.
# Si tu mundo es pesado o el PC va justo, sube este valor.
T_ARRANQUE_GAZEBO = 8.0


def generate_dynamic_rviz(num_robots, base_rviz_path):
    """Genera un .rviz dinamico con los displays de cada robot."""
    with open(base_rviz_path, 'r') as f:
        rviz_dict = yaml.safe_load(f)

    if 'Displays' not in rviz_dict['Visualization Manager']:
        rviz_dict['Visualization Manager']['Displays'] = []

    displays = rviz_dict['Visualization Manager']['Displays']

    displays.append({
        'Class': 'rviz_default_plugins/Map',
        'Name': 'Map Global',
        'Enabled': True,
        'Topic': {
            'Value': '/robot_0/map',
            'Durability Policy': 'Transient Local',
            'History Policy': 'Keep Last',
            'Reliability Policy': 'Reliable',
            'Depth': 1,
        },
        'Alpha': 0.7,
        'Color Scheme': 'map',
    })

    for i in range(num_robots):
        robot_name = f'robot_{i}'
        displays.append({
            'Class': 'rviz_default_plugins/RobotModel',
            'Name': f'Model_{robot_name}',
            'Description Source': 'Topic',
            'Description Topic': {'Value': f'/{robot_name}/robot_description'},
            'TF Prefix': '',
            'Enabled': True,
        })
        displays.append({
            'Class': 'rviz_default_plugins/LaserScan',
            'Name': f'Laser_{robot_name}',
            'Topic': {'Value': f'/{robot_name}/scan'},
            'Size (m)': 0.05,
            'Style': 'Flat Squares',
            'Color Transformer': 'Intensity',
            'Enabled': True,
        })
        displays.append({
            'Class': 'rviz_default_plugins/Image',
            'Name': f'Cam_{robot_name}',
            'Topic': {'Value': f'/{robot_name}/camera/image_raw'},
            'Enabled': True,
        })

    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='_dynamic.rviz', delete=False)
    yaml.dump(rviz_dict, tmp, default_flow_style=False)
    tmp.close()
    return tmp.name


def build_robot_nav2_params(nav2_params_file, name, x_pos, y_pos, yaw):
    """Crea un yaml de Nav2 especifico para un robot con sus frames."""
    with open(nav2_params_file, 'r') as f:
        params = yaml.safe_load(f)

    params['amcl']['ros__parameters']['odom_frame_id'] = f'{name}/odom'
    params['amcl']['ros__parameters']['base_frame_id'] = f'{name}/base_link'
    params['amcl']['ros__parameters']['scan_topic'] = f'/{name}/scan'
    params['amcl']['ros__parameters']['tf_broadcast'] = False
    params['amcl']['ros__parameters']['transform_tolerance'] = 2.0
    params['amcl']['ros__parameters']['set_initial_pose'] = True
    params['amcl']['ros__parameters']['initial_pose'] = {
        'x': x_pos, 'y': y_pos, 'z': 0.0, 'yaw': yaw,
    }

    params['local_costmap']['local_costmap']['ros__parameters'][
        'obstacle_layer']['lidar']['topic'] = f'/{name}/scan'
    params['global_costmap']['global_costmap']['ros__parameters'][
        'obstacle_layer']['lidar']['topic'] = f'/{name}/scan'
    params['local_costmap']['local_costmap']['ros__parameters'][
        'global_frame'] = f'{name}/odom'
    params['local_costmap']['local_costmap']['ros__parameters'][
        'robot_base_frame'] = f'{name}/base_link'
    params['global_costmap']['global_costmap']['ros__parameters'][
        'robot_base_frame'] = f'{name}/base_link'
    params['bt_navigator']['ros__parameters'][
        'robot_base_frame'] = f'{name}/base_link'
    params['bt_navigator']['ros__parameters']['odom_topic'] = f'/{name}/odom'
    params['collision_monitor']['ros__parameters'][
        'base_frame_id'] = f'{name}/base_link'
    params['collision_monitor']['ros__parameters'][
        'odom_frame_id'] = f'{name}/odom'
    params['velocity_smoother']['ros__parameters'][
        'odom_topic'] = f'/{name}/odom'

    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix=f'_nav2_{name}.yaml', delete=False)
    yaml.dump(params, tmp)
    tmp.close()
    return tmp.name


def spawn_robots(context, *args, **kwargs):
    """Construye las acciones ESCALONADAS para cada robot."""
    num_robots = int(LaunchConfiguration('num_robots').perform(context))

    pkg_bringup = get_package_share_directory('robot_bringup')
    pkg_description = get_package_share_directory('my_robot_description')
    pkg_nav2 = get_package_share_directory('nav2_bringup')

    map_file = os.path.join(pkg_bringup, 'config', 'mapa_almacen.yaml')
    nav2_params_file = os.path.join(pkg_bringup, 'config', 'nav2.yaml')
    xacro_file = os.path.join(
        pkg_description, 'urdf', 'my_robot.urdf.xacro')
    nav2_launch = os.path.join(pkg_nav2, 'launch', 'bringup_launch.py')

    # Separacion entre robots aumentada de 1.25 a 2.0 m para reducir
    # choques cuando todos salen de la fila inicial a la vez.
    start_x = -5.0
    separacion = 2.0
    y_desplazamiento = -1.35
    yaw = 1.5708

    timed_actions = []

    for i in range(num_robots):
        name = f'robot_{i}'
        x_pos = start_x + float(i * separacion)
        color = 'Red' if i == 0 else 'Blue'

        # Offset temporal base para ESTE robot.
        # Sumamos T_ARRANQUE_GAZEBO para que NINGUN robot (en especial
        # robot_0) intente spawnear antes de que Gazebo este listo.
        t0 = T_ARRANQUE_GAZEBO + i * T_ENTRE_ROBOTS

        # --- 1) Robot State Publisher (publica robot_description) ---
        rsp = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name=f'robot_state_publisher_{name}',
            namespace=name,
            parameters=[{
                'robot_description': Command([
                    'xacro ', xacro_file,
                    ' prefix:=', f'{name}/',
                    ' color:=', color]),
                'use_sim_time': True,
                'frame_prefix': '',
            }],
        )
        timed_actions.append(
            TimerAction(period=t0, actions=[rsp]))

        # --- 2) Spawn ROBUSTO en Gazebo (con reintento y verificacion) ---
        # En lugar del Node(create) de un solo intento (que fallaba en
        # silencio con la CPU saturada y por eso desaparecia un robot),
        # usamos spawn_robot.py que reintenta hasta verificar que el
        # robot existe de verdad en Gazebo.
        spawn = ExecuteProcess(
            cmd=[
                'ros2', 'run', 'robot_bringup', 'spawn_robot.py',
                '--name', name,
                '--x', str(x_pos),
                '--y', str(y_desplazamiento),
                '--z', '0.1',
                '--yaw', str(yaw),
            ],
            name=f'spawn_{name}',
            output='screen',
        )
        timed_actions.append(
            TimerAction(period=t0 + T_RSP_A_SPAWN, actions=[spawn]))

        # --- 3) Bridge + TF estatica + relays ---
        bridge = Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name=f'bridge_{name}',
            arguments=[
                f'/{name}/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
                f'/{name}/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
                f'/model/{name}/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
                f'/{name}/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
                f'/{name}/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
                f'/{name}/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
                f'/{name}/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            ],
            remappings=[(f'/model/{name}/tf', '/tf')],
            parameters=[{'use_sim_time': True}],
        )
        # static_transform_publisher: publica map -> {name}/odom como
        # ANDAMIO mientras AMCL arranca. Sin esto, RViz y Nav2 no tienen
        # ningun TF en los primeros segundos y los robots no aparecen.
        # AMCL tiene tf_broadcast: FALSE (nav2.yaml) para no competir
        # con este publicador: AMCL corrige la pose internamente pero
        # no publica TF. El estatico mantiene el arbol TF estable.
        static_tf = Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name=f'map_to_odom_{name}',
            arguments=[
                '--x', str(x_pos), '--y', str(y_desplazamiento), '--z', '0',
                '--roll', '0', '--pitch', '0', '--yaw', str(yaw),
                '--frame-id', 'map', '--child-frame-id', f'{name}/odom'],
            parameters=[{'use_sim_time': True}],
        )
        relay_tf = Node(
            package='topic_tools',
            executable='relay',
            name=f'tf_relay_{name}',
            arguments=['/tf', f'/{name}/tf'],
            parameters=[{'use_sim_time': True}],
        )
        relay_tf_static = Node(
            package='topic_tools',
            executable='relay',
            name=f'tf_static_relay_{name}',
            arguments=['/tf_static', f'/{name}/tf_static'],
            parameters=[{'use_sim_time': True}],
        )
        timed_actions.append(TimerAction(
            period=t0 + T_RSP_A_SPAWN + T_SPAWN_A_BRIDGE,
            actions=[bridge, static_tf, relay_tf, relay_tf_static]))

        # --- 4) Nav2 especifico de este robot ---
        robot_params = build_robot_nav2_params(
            nav2_params_file, name, x_pos, y_desplazamiento, yaw)
        print(f'[INFO] Nav2 params {name} (x={x_pos}): {robot_params}')

        nav2 = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(nav2_launch),
            launch_arguments={
                # Esta instalacion de Nav2 (anterior al PR #4715) SI
                # requiere use_namespace. Verificado con grep en
                # /opt/ros/jazzy/share/nav2_bringup/launch/bringup_launch.py
                'namespace': name,
                'use_namespace': 'true',
                'map': map_file,
                'params_file': robot_params,
                'use_sim_time': 'true',
                'autostart': 'true',
            }.items(),
        )
        timed_actions.append(TimerAction(
            period=t0 + T_RSP_A_SPAWN + T_SPAWN_A_BRIDGE + T_BRIDGE_A_NAV2,
            actions=[nav2]))

    return timed_actions


def generate_launch_description():
    # num_robots y run_swarm ahora son ARGUMENTOS, no input() bloqueante.
    # Uso:
    #   ros2 launch robot_bringup multi_robot.launch.py num_robots:=5
    #   ros2 launch robot_bringup multi_robot.launch.py num_robots:=5 run_swarm:=true
    default_num = '5'

    pkg_bringup = get_package_share_directory('robot_bringup')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    pkg_description = get_package_share_directory('my_robot_description')

    bridge_config = os.path.join(
        pkg_bringup, 'config', 'bridge_config.yaml')
    world_path = os.path.join(
        pkg_bringup, 'world', 'my_world_person.sdf')
    base_rviz_path = os.path.join(
        pkg_description, 'rviz', 'rviz_base.rviz')

    dynamic_rviz_file = generate_dynamic_rviz(
        int(default_num), base_rviz_path)
    print(f'[INFO] RViz dinamico: {dynamic_rviz_file}')

    launch_actions = [
        DeclareLaunchArgument('num_robots', default_value=default_num),
        DeclareLaunchArgument('run_swarm', default_value='false'),

        # Gazebo Harmonic
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')),
            launch_arguments={'gz_args': f'-r {world_path}'}.items(),
        ),

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='bridge_clock',
            parameters=[{'config_file': bridge_config}],
            output='screen',
        ),

        # Relay ROBUSTO /robot_0/map -> /map. A diferencia de
        # topic_tools relay, este espera a que /robot_0/map exista
        # (el map_server de robot_0 tarda en activar) y usa QoS
        # transient_local para que RViz reciba el mapa aunque se
        # suscriba tarde. Puede arrancar pronto: el espera solo.
        Node(
            package='robot_bringup',
            executable='map_relay_robusto.py',
            name='map_relay_robusto',
            parameters=[{'use_sim_time': True}],
            output='screen',
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', dynamic_rviz_file],
            parameters=[{'use_sim_time': True}],
            output='screen',
        ),

        OpaqueFunction(function=spawn_robots),
    ]

    # Nodo de enjambre: arranca al final, cuando ya hay robots y Nav2.
    # Solo se lanza si run_swarm:=true
    from launch.conditions import IfCondition
    swarm_node = TimerAction(
        # Arranca cuando el ULTIMO robot ya tiene Nav2 activo:
        # T_ARRANQUE_GAZEBO + (N-1)*T_ENTRE_ROBOTS + cadena completa + margen
        period=(T_ARRANQUE_GAZEBO +
                int(default_num) * T_ENTRE_ROBOTS + 15.0),
        actions=[Node(
            package='robot_bringup',
            # El CMakeLists instala con install(PROGRAMS ...), asi que el
            # ejecutable es el nombre de archivo CON .py. El nodo de
            # rescate/esquinas es esquinas.py (NO leader_follower.py).
            executable='esquinas.py',
            parameters=[{
                'num_robots': LaunchConfiguration('num_robots'),
                'use_sim_time': True,
            }],
            output='screen',
            condition=IfCondition(LaunchConfiguration('run_swarm')),
        )],
    )
    launch_actions.append(swarm_node)

    return LaunchDescription(launch_actions)
