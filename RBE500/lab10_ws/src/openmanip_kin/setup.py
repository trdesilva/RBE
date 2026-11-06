from setuptools import find_packages, setup

package_name = 'openmanip_kin'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='thomas',
    maintainer_email='5818730+trdesilva@users.noreply.github.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'forward_kinematics = openmanip_kin.fwd_kin:main',
            'inverse_kinematics = openmanip_kin.inv_kin:main'
        ],
    },
)
