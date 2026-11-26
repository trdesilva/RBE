from setuptools import find_packages, setup
import glob

package_name = 'rrbot_clock_trajectory'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob.glob('launch/*.py')),
        ('share/' + package_name + '/config', glob.glob('config/*.yaml')),
        ('share/' + package_name + '/ros2_control', glob.glob('description/ros2_control/*.xacro')),
        ('share/' + package_name + '/urdf', glob.glob('description/urdf/*.xacro')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='thomas',
    maintainer_email='5818730+trdesilva@users.noreply.github.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'setpoints = rrbot_clock_trajectory.setpoints:main',
        ],
    },
)
