import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'vp_husky_sim'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*')),
        (os.path.join('share', package_name, 'meshes'), glob('meshes/*.dae') + glob('meshes/*.stl')),
        (os.path.join('share', package_name, 'meshes', 'accessories'), glob('meshes/accessories/*')),
        (os.path.join('share', package_name, 'meshes', 'prius'), glob('meshes/prius/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shubham',
    maintainer_email='shubhampathneja123@gmail.com',
    description='Clearpath Husky ported to ROS2 + new Gazebo (gz sim), spawning into the VisionPilot sim worlds',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gps_navigator_node = vp_husky_sim.gps_navigator_node:main',
        ],
    },
)
