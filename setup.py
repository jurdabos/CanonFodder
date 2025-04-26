from setuptools import find_packages, setup

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name='CanonFodder',
    packages=find_packages(exclude=['examples']),
    version='0.2',
    license='MIT',
    description='Code base for a written assignment in the field of Data Quality and Data Wrangling at IU',
    author='Torda Balázs',
    author_email='balazs.torda@iu-study.org',
    url='https://github.com/jurdabos/canonfodder',
    install_requires=requirements,
)