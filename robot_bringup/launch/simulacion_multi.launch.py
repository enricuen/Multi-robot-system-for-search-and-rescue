import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node, PushRosNamespace, SetRemap

def launch_setup(context, *args, **kwargs):
    num_robots = int(LaunchConfiguration('robots').perform(context))
    pkg_descripcion = get_package_share_directory('robot_descripcion')
    nodes_to_start = []

    for i in range(num_robots):
        robot_name = f'robot_{i}'
        y_pos = float(i * -1)

        # Aquí empieza el GroupAction con su corchete de apertura: [
        robot_group = GroupAction([
            PushRosNamespace(robot_name),
            
            # ¡LA MAGIA PARA SALVAR EL TF TREE!
            # Esto evita que el namespace atrape los TFs, mandándolos al global
            SetRemap(src='/tf', dst='/tf'),
            SetRemap(src='/tf_static', dst='/tf_static'),

            Node(
                package='robot_state_publisher',
                executable='robot_state_publisher',
                name='robot_state_publisher',
                parameters=[{
                    'robot_description': Command([
                        'xacro ', os.path.join(pkg_descripcion, 'urdf', 'robot.urdf.xacro'),
                        ' prefix:=', robot_name
                    ]),
                    'use_sim_time': True,
                    'frame_prefix': f'{robot_name}/'
                }]
            ),

            Node(
                package='ros_gz_sim',
                executable='create',
                arguments=[
                    '-name', robot_name,
                    '-topic', 'robot_description',
                    '-x', '0.0', '-y', str(y_pos), '-z', '0.1'
                ],
            ),

            Node(
                package='ros_gz_bridge',
                executable='parameter_bridge',
                arguments=[
                    f'/{robot_name}/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
                    f'/{robot_name}/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
                    f'/{robot_name}/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
                    f'/model/{robot_name}/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
                    f'/{robot_name}/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
                    f'/{robot_name}/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
                ],
                # Puenteamos el TF local de Gazebo al TF global de ROS 2
                remappings=[
                    (f'/model/{robot_name}/tf', '/tf')
                ]
            ),
       
        
            # Unión 1: world -> robot_X/odom
            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                arguments=['0', str(y_pos), '0', '0', '0', '0', 'world', f'{robot_name}/odom'],
            ),
            
            # UNIÓN 2 (EL PARCHE SALVA-EXÁMENES): robot_X/odom -> robot_X/base_footprint
            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                arguments=['0', '0', '0', '0', '0', '0', f'{robot_name}/odom', f'{robot_name}/base_footprint'],
            )
        ]) # <--- ¡AQUÍ ESTABA EL ERROR! Faltaba el cierre de corchete y paréntesis
        
        nodes_to_start.append(robot_group)

    return nodes_to_start


def generate_launch_description():
    pkg_bringup = get_package_share_directory('robot_bringup')
    pkg_descripcion = get_package_share_directory('robot_descripcion')
    
    return LaunchDescription([
        DeclareLaunchArgument('robots', default_value='2', description='Número de robots'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
            ]),
            launch_arguments={'gz_args': f"-r {os.path.join(pkg_bringup, 'world', 'mundo.sdf')}"}.items(),
        ),

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        ),

        # ¡RVIZ FUERA DEL BUCLE PARA QUE SOLO HAYA UNO!
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', os.path.join(pkg_descripcion, 'rviz', 'urdf_config.rviz')],
            parameters=[{'use_sim_time': True}],
        ),

        OpaqueFunction(function=launch_setup)
    ])
