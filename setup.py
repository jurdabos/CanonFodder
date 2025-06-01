from setuptools import find_packages, setup
import os

# Get the directory of this setup.py file
current_dir = os.path.dirname(os.path.abspath(__file__))
requirements_path = os.path.join(current_dir, "requirements.txt")

# Read requirements but don't use them directly as install_requires
# to prevent exact version pinning from requirements.txt being enforced
# during dependency resolution
with open(requirements_path, "r") as f:
    requirements = f.read().splitlines()

# Define core dependencies without strict version pinning
# to allow pip to resolve conflicts more flexibly
core_dependencies = [
    'pandas',
    'numpy',
    'matplotlib',
    'seaborn',
    'requests',
    'openai>=1.60.2',
    'h11>=0.16.0',  # Security requirement
    'httpcore>=1.0.7',
    'httpx>=0.28.1',
    'Jinja2>=3.1.6',
    'setuptools>=78.1.1,<81'
]

# Define database dependencies separately to allow for different environments
db_dependencies = [
    'sqlalchemy>=2.0.40',  # Required for core application
    'alembic>=1.15.2',
]

# Define Airflow dependencies
airflow_dependencies = [
    'apache-airflow~=3.0.1',
]

setup(
    name='CanonFodder',
    packages=find_packages(exclude=['examples']),
    version='1.3',  # Match the currently installed version
    license='MIT',
    description='Code base for a written assignment in the field of Data Quality and Data Wrangling at IU',
    author='Torda Balázs',
    author_email='balazs.torda@iu-study.org',
    url='https://github.com/jurdabos/canonfodder',
    install_requires=core_dependencies,
    extras_require={
        'db': db_dependencies,
        'airflow': airflow_dependencies,
        'all': db_dependencies + airflow_dependencies,
    },
)
