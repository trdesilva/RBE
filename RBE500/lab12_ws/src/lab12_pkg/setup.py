import glob

from setuptools import find_packages, setup

package_name = 'lab12_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob.glob('launch/*.py')),
        ('share/' + package_name + '/models', glob.glob('models/*.urdf.xacro')),
        ('share/' + package_name + '/worlds', glob.glob('worlds/*.sdf')),
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
            'edge_detect = lab12_pkg.edge_detect:main',
        ],
    },
)
