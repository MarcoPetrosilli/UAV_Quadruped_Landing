"""
launch: avvia il nodo di atterraggio.

Uso:
    ros2 launch drone_landing landing.launch.py

Prerequisiti (fuori da ROS):
  - CrazySim in esecuzione (Gazebo + firmware SITL) e URI raggiungibile
    (default udp://127.0.0.1:19850, come nel main).
  - env Python con cflib, cvxpy, numpy, scipy, matplotlib.
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="drone_landing",
            executable="landing_node",
            name="drone_landing_node",
            output="screen",
            emulate_tty=True,
        ),
    ])
