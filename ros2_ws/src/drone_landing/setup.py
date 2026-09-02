import os
from glob import glob
from setuptools import setup

package_name = "drone_landing"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # launch files
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="thesis",
    maintainer_email="you@example.com",
    description="Nodo ROS2 per il controllo di atterraggio autonomo (cflib/CrazySim).",
    license="MIT",
    entry_points={
        "console_scripts": [
            # ros2 run drone_landing landing_node
            "landing_node = drone_landing.main_alpha_landing:main",
        ],
    },
)
