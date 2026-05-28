import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import Command

def generate_launch_description():
    # 1. Definir las rutas de los paquetes
    pkg_bringup = get_package_share_directory('robot_bringup')
    pkg_description = get_package_share_directory('my_robot_description')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    # 2. Definir las rutas de los archivos específicos
    world_path = os.path.join(pkg_bringup, 'world', 'my_world_person.sdf')
    xacro_file = os.path.join(pkg_description, 'urdf', 'my_robot.urdf.xacro')
    bridge_config = os.path.join(pkg_bringup, 'config', 'bridge_config.yaml')
    rviz_config = os.path.join(pkg_description, 'rviz', 'rviz_base.rviz')

    # Nombre fijo del robot para que coincida con tus comandos de consola
    name = 'robot_0'

    return LaunchDescription([
        # 3. Lanzar Gazebo con tu mundo
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
            ),
            launch_arguments={'gz_args': f'-r {world_path}'}.items(),
        ),

        # 4. Bridge para el reloj de simulación 
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='bridge_clock',
            parameters=[{'config_file': bridge_config}],
            output='screen'
        ),

        # 5. Publicar el estado del robot
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name=f'robot_state_publisher_{name}',
            namespace=name,
            parameters=[{
                'robot_description': Command(['xacro ', xacro_file, ' prefix:=', f'{name}/']),
                'use_sim_time': True,
                'frame_prefix': '',
            }]
        ),

        # 6. Spawm del robot en Gazebo EXACTAMENTE en x=0.0, y=0.0
        Node(
            package='ros_gz_sim',
            executable='create',
            name=f'spawn_{name}',
            arguments=[
                '-topic', f'/{name}/robot_description',
                '-name', name,
                '-x', '0.0',
                '-y', '0.0',
                '-z', '0.1' # Un poco elevado para que no colisione con el suelo al nacer
            ],
            output='screen'
        ),

        # 7. Bridge de tópicos para que ROS 2 y Gazebo se comuniquen
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name=f'bridge_{name}',
            arguments=[
                f'/{name}/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
                f'/{name}/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
                f'/model/{name}/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
                f'/{name}/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
                f'/{name}/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            ],
            remappings=[
                (f'/model/{name}/tf', '/tf'),
            ],
            parameters=[{'use_sim_time': True}]
        ),

        # 8. Lanzar RViz2 con tu configuración base
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            parameters=[{'use_sim_time': True}],
            output='screen'
        )
    ])
