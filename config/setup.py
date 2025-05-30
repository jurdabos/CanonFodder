from setuptools import find_packages, setup
import os

# Get the directory of this setup.py file
current_dir = os.path.dirname(os.path.abspath(__file__))
requirements_path = os.path.join(current_dir, "requirements.txt")

with open(requirements_path, "r") as f:
    requirements = f.read().splitlines()

setup(
    name='CanonFodder',
    packages=find_packages(exclude=['examples']),
    version='1.3',
    license='MIT',
    description='Code base for a written assignment in the field of Data Quality and Data Wrangling at IU',
    author='Torda Balázs',
    author_email='balazs.torda@iu-study.org',
    url='https://github.com/jurdabos/canonfodder',
    install_requires=requirements,
)