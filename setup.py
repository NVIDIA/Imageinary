from setuptools import find_packages, setup

with open('README.md', 'r') as f:
    long_description = f.read()

with open('VERSION', 'r') as f:
    version = f.read().strip()

extras = {
    'tfrecord': ['tensorflow >= 1.14.0,!=2.0.x,!=2.1.x,!=2.2.0,!=2.4.0'],
    'mxnet': ['mxnet >= 1.6.0,!=1.8.0']
}

extras['all'] = [item for group in extras.values() for item in group]

setup(
    name='nvidia-imageinary',
    author='NVIDIA Corporation',
    author_email='roclark@nvidia.com',
    version=version,
    description='A tool to randomly generate image datasets of various resolutions',
    long_description=long_description,
    packages=find_packages(include=['imagine'], exclude=['tests']),
    license='Apache 2.0',
    python_requires='>=3.7',
    entry_points={
        'console_scripts': ['imagine=imagine:_main']
    },
    install_requires=[
        'numpy >= 1.18.0',
        'Pillow >= 7.1.2'
    ],
    extras_require=extras
)
