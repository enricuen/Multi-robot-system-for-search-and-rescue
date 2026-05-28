from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'odom_frame': 'robot_0/odom',
                'base_frame': 'robot_0/base_link',
                'map_frame': 'map',
                'scan_topic': '/robot_0/scan'
            }]
        )
    ])
