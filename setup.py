from setuptools import setup, find_packages

setup(
    name='hydraville',
    version='0.1',
    packages=find_packages(),
    package_data={
        'hydraville': ['json/*.json', 'json/catchmod/*.json'],
    },
    entry_points={
        'console_scripts': ['hydraville=hydraville.cli:start_cli'],
    }
)