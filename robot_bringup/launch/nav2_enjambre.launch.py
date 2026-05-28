import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import PushRosNamespace

def generate_launch_description():
    pkg_dir = get_package_share_directory('robot_bringup')
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    
    # El mapa es común para todos
    map_file = os.path.join(pkg_dir, 'config', 'mapa_almacen.yaml')

    ld = LaunchDescription()
    
    # ¡LOS 3 ROBOTS DESBLOQUEADOS!
    nombres_robots = ['robot_0', 'robot_1', 'robot_2']

    for robot in nombres_robots:
        # Aquí hacemos la magia para que cada robot pille su cerebro clonado
        if robot == 'robot_0':
            archivo_yaml = 'nav2.yaml'
        elif robot == 'robot_1':
            archivo_yaml = 'nav2_rob1.yaml'
        else:
            archivo_yaml = 'nav2_rob2.yaml'
            
        params_file = os.path.join(pkg_dir, 'config', archivo_yaml)

        nav2_group = GroupAction([
            PushRosNamespace(robot),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(nav2_bringup_dir, 'launch', 'bringup_launch.py')),
                launch_arguments={
                    'namespace': robot,
                    'use_namespace': 'true',
                    'map': map_file,
                    'use_sim_time': 'true',
                    'params_file': params_file,
                    'autostart': 'true'
                }.items()
            )
        ])
        ld.add_action(nav2_group)

    return ld
