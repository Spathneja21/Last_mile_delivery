import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'vp_car_sim'

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
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shubham',
    maintainer_email='shubhampathneja123@gmail.com',
    description='Simple ROS2 car with a front camera, simulated in Gazebo, running the VisionPilot SceneSeg model',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'scene_seg_node = vp_car_sim.scene_seg_node:main',
            'scene_3d_node = vp_car_sim.scene_3d_node:main',
            'scene_seg_navigator = vp_car_sim.scene_seg_navigator_node:main',
            'webcam_navigator_node = vp_car_sim.webcam_navigator_node:main',
        ],
    },
)
