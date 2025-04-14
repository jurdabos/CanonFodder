from setuptools import find_packages, setup

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name='MyLifeInData',
    packages=find_packages(exclude=['examples']),
    version='0.0.1',
    license='MIT',
    description='Code base for country-enrichment of last.fm data',
    author='Torda Balázs',
    author_email='balazs.torda@iu-study.org',
    url='https://github.com/jurdabos/mylifeindata',
    install_requires=requirements,
)