import os
import yaml
import tempfile
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, Command

def generate_dynamic_rviz(num_robots, base_rviz_path):
    # Cargamos el archivo base
    with open(base_rviz_path, 'r') as f:
        rviz_dict = yaml.safe_load(f)

    # Si no existe la lista de Displays, la creamos
    if 'Displays' not in rviz_dict['Visualization Manager']:
        rviz_dict['Visualization Manager']['Displays'] = []
        
    displays = rviz_dict['Visualization Manager']['Displays']

    displays.append({
        'Class': 'rviz_default_plugins/Map',
        'Name': 'Map Global',
        'Enabled': True,
        'Topic': {
            'Value': '/robot_0/map',
            'Durability Policy': 'Transient Local',  # Mantiene el mapa en memoria para nuevos suscriptores
            'History Policy': 'Keep Last',
            'Reliability Policy': 'Reliable',
            'Depth': 1
        },
        'Alpha': 0.7,
        'Color Scheme': 'map'
    })
    # Añadimos los visores dinámicamente para cada robot
    for i in range(num_robots):
        robot_name = f'robot_{i}'
        
        # 1. RobotModel
        displays.append({
            'Class': 'rviz_default_plugins/RobotModel',
            'Name': f'Model_{robot_name}',
            'Description Source': 'Topic',
            'Description Topic': {'Value': f'/{robot_name}/robot_description'},
            'TF Prefix': '',
            'Enabled': True
        })
        
        # 2. LaserScan
        displays.append({
            'Class': 'rviz_default_plugins/LaserScan',
            'Name': f'Laser_{robot_name}',
            'Topic': {'Value': f'/{robot_name}/scan'},
            'Size (m)': 0.05,
            'Style': 'Flat Squares',
            'Color Transformer': 'Intensity',
            'Enabled': True
        })
        
        # 3. Camera Image
        displays.append({
            'Class': 'rviz_default_plugins/Image',
            'Name': f'Cam_{robot_name}',
            'Topic': {'Value': f'/{robot_name}/camera/image_raw'},
            'Enabled': True
        })

    # Guardamos el diccionario resultante en un archivo temporal
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='_dynamic.rviz', delete=False)
    yaml.dump(rviz_dict, tmp, default_flow_style=False)
    tmp.close()
    
    return tmp.name

def spawn_robots(context, *args, **kwargs):
    num_robots = int(LaunchConfiguration('num_robots').perform(context))

    pkg_bringup = get_package_share_directory('robot_bringup')
    pkg_description = get_package_share_directory('my_robot_description')
    pkg_nav2 = get_package_share_directory('nav2_bringup')

    map_file = os.path.join(pkg_bringup, 'config', 'mapa_almacen.yaml')
    nav2_params_file = os.path.join(pkg_bringup, 'config', 'nav2.yaml')
    
    xacro_file = os.path.join(pkg_description, 'urdf', 'my_robot.urdf.xacro')
    nav2_launch = os.path.join(pkg_nav2, 'launch', 'bringup_launch.py')

    nodes = []
    
    start_x = -5.0
    y_desplazamiento = -1.35

    for i in range(num_robots):
        name = f'robot_{i}'
        x_pos = start_x + float(i * 1.0)
        color = 'Red' if i == 0 else 'Blue'
        # 1. Robot State Publisher
        nodes.append(Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name=f'robot_state_publisher_{name}',
            namespace=name,
            parameters=[{
                'robot_description': Command(['xacro ', xacro_file, ' prefix:=', f'{name}/',' color:=', color]),
                'use_sim_time': True,
                'frame_prefix': '',
            }]
        ))

        # 2. Spawn en Gazebo
        nodes.append(Node(
            package='ros_gz_sim',
            executable='create',
            name=f'spawn_{name}',
            arguments=[
                '-topic', f'/{name}/robot_description',
                '-name', name,
                '-x', str(x_pos),
                '-y', str(y_desplazamiento),
                '-z', '0.1',
                '-Y', '1.5708'
            ],
            output='screen'
        ))

        # 3. Bridge por robot
        nodes.append(Node(
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
            remappings=[
                (f'/model/{name}/tf', '/tf'),
            ],
            parameters=[{'use_sim_time': True}]
        ))

        nodes.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name=f'map_to_odom_{name}',
            arguments=['--x', str(x_pos), '--y', str(y_desplazamiento), '--z', '0',
                       '--roll', '0', '--pitch', '0', '--yaw', '1.5708',
                       '--frame-id', 'map', '--child-frame-id', f'{name}/odom'],
            parameters=[{'use_sim_time': True}]
        ))
        
        nodes.append(Node(
            package='topic_tools',
            executable='relay',
            name=f'tf_relay_{name}',
            arguments=['/tf', f'/{name}/tf'],
            parameters=[{'use_sim_time': True}],
        ))
        nodes.append(Node(
            package='topic_tools',
            executable='relay',
            name=f'tf_static_relay_{name}',
            arguments=['/tf_static', f'/{name}/tf_static'],
            parameters=[{'use_sim_time': True}],
        ))

        # 5. Genera yaml especifico para este robot con sus frame IDs correctos
        with open(nav2_params_file, 'r') as f:
            params = yaml.safe_load(f)

        params['amcl']['ros__parameters']['odom_frame_id'] = f'{name}/odom'
        params['amcl']['ros__parameters']['base_frame_id'] = f'{name}/base_link'
        params['amcl']['ros__parameters']['scan_topic'] = f'/{name}/scan'
        params['amcl']['ros__parameters']['tf_broadcast'] = True
        params['amcl']['ros__parameters']['transform_tolerance'] = 2.0
        
        # iniital pose a amcl
        params['amcl']['ros__parameters']['set_initial_pose'] = True
        params['amcl']['ros__parameters']['initial_pose'] = {
            'x': x_pos,
            'y': y_desplazamiento,
            'z': 0.0,
            'yaw': 1.5708
        }
        # ------------------------------------------------------------------------
        params['local_costmap']['local_costmap']['ros__parameters']['obstacle_layer']['lidar']['topic'] = f'/{name}/scan'
        params['global_costmap']['global_costmap']['ros__parameters']['obstacle_layer']['lidar']['topic'] = f'/{name}/scan'
        params['local_costmap']['local_costmap']['ros__parameters']['global_frame'] = f'{name}/odom'
        params['local_costmap']['local_costmap']['ros__parameters']['robot_base_frame'] = f'{name}/base_link'
        params['global_costmap']['global_costmap']['ros__parameters']['robot_base_frame'] = f'{name}/base_link'
        params['bt_navigator']['ros__parameters']['robot_base_frame'] = f'{name}/base_link'
        params['bt_navigator']['ros__parameters']['odom_topic'] = f'/{name}/odom'
        params['collision_monitor']['ros__parameters']['base_frame_id'] = f'{name}/base_link'
        params['collision_monitor']['ros__parameters']['odom_frame_id'] = f'{name}/odom'
        params['velocity_smoother']['ros__parameters']['odom_topic'] = f'/{name}/odom'

        tmp = tempfile.NamedTemporaryFile(mode='w', suffix=f'_nav2_{name}.yaml', delete=False)
        yaml.dump(params, tmp)
        tmp.close()
        robot_params = tmp.name
        print(f'[INFO] Nav2 params for {name} with initial pose (x={x_pos}): {robot_params}')

        # 6. Nav2 con yaml especifico del robot
        nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(nav2_launch),
            launch_arguments={
                'namespace': name,
                'use_namespace': 'true',
                'map': map_file,
                'params_file': robot_params,
                'use_sim_time': 'true',
                'autostart': 'true',
            }.items()
        ))
    return nodes


def generate_launch_description():
    print("\n" + "="*50)
    user_val = input(" Cuantos robots deseas spawnear? ")
    if not user_val.strip():
        user_val = "2"
    #exec_swarm = input(" Ejecutar nodo leader_follower? (y/N): ").lower()
    print("="*50 + "\n")

    num_robots_int = int(user_val)

    pkg_bringup = get_package_share_directory('robot_bringup')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    pkg_description = get_package_share_directory('my_robot_description')

    bridge_config = os.path.join(pkg_bringup, 'config', 'bridge_config.yaml')
    world_path = os.path.join(pkg_bringup, 'world', 'my_world_person.sdf')
    
    base_rviz_path = os.path.join(pkg_description, 'rviz', 'rviz_base.rviz')

    dynamic_rviz_file = generate_dynamic_rviz(num_robots_int, base_rviz_path)
    print(f'[INFO] Archivo RViz dinámico generado en: {dynamic_rviz_file}')

    launch_actions = [
        DeclareLaunchArgument('num_robots', default_value=user_val),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
            ),
            launch_arguments={'gz_args': f'-r {world_path}'}.items(),
        ),

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='bridge_clock',
            parameters=[{'config_file': bridge_config}],
            output='screen'
        ),
        
        # Republica el mapa en /map global para RViz
        Node(
            package='topic_tools',
            executable='relay',
            name='map_relay',
            arguments=['/robot_0/map', '/map'],
            parameters=[{'use_sim_time': True}],
            output='screen'
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', dynamic_rviz_file],
            parameters=[{'use_sim_time': True}],
            output='screen'
        ),

        OpaqueFunction(function=spawn_robots)
    ]

    return LaunchDescription(launch_actions)

'''
    if exec_swarm == 'y':
        launch_actions.append(Node(
            package='robot_bringup',
            executable='leader_follower.py',
            parameters=[{
                'num_robots': LaunchConfiguration('num_robots'),
                'use_sim_time': True
            }],
            output='screen'
        ))
'''
