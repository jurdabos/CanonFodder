"""
Test script to verify Airflow compatibility with CanonFodder.

This script checks if the necessary packages are installed and if they have the correct versions.
It can be used to verify that the changes made to resolve dependency conflicts are working.

Usage:
    python tests/test_airflow_compatibility.py
"""
import importlib
import sys
from typing import Dict, List, Tuple

def check_package_version(package_name: str) -> Tuple[bool, str]:
    """
    Check if a package is installed and return its version.
    
    Args:
        package_name: Name of the package to check
        
    Returns:
        Tuple of (is_installed, version)
    """
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError:
        return False, 'not installed'

def main():
    """Check compatibility of installed packages."""
    packages_to_check = [
        'sqlalchemy',
        'airflow',
        'pydantic',
        'pygments',
        'h11',
        'httpcore',
        'httpx',
        'jinja2',
        'setuptools',
    ]
    
    results: Dict[str, str] = {}
    
    print("Checking package versions for Airflow compatibility...\n")
    
    for package in packages_to_check:
        is_installed, version = check_package_version(package)
        results[package] = version
    
    # Print results in a table format
    print(f"{'Package':<20} {'Version':<15}")
    print(f"{'-'*20} {'-'*15}")
    
    for package, version in results.items():
        print(f"{package:<20} {version:<15}")
    
    print("\nCompatibility Analysis:")
    
    # Check SQLAlchemy version
    if 'sqlalchemy' in results and results['sqlalchemy'].startswith('1.4'):
        print("✓ SQLAlchemy 1.4.x detected - compatible with Airflow but may limit CanonFodder ORM functionality")
    elif 'sqlalchemy' in results and results['sqlalchemy'].startswith('2.0'):
        print("✗ SQLAlchemy 2.0.x detected - may cause conflicts with Airflow")
    
    # Check if Airflow is installed
    if 'airflow' in results and results['airflow'] != 'not installed':
        print(f"✓ Apache Airflow {results['airflow']} is installed")
    else:
        print("✗ Apache Airflow is not installed")
    
    print("\nRecommendations:")
    if 'sqlalchemy' in results and results['sqlalchemy'].startswith('1.4') and 'airflow' in results and results['airflow'] != 'not installed':
        print("- Current setup is optimized for Airflow compatibility")
        print("- Some CanonFodder database functionality may be limited")
        print("- For full CanonFodder functionality without Airflow, reinstall using: pip install -r requirements.txt")
    elif 'sqlalchemy' in results and results['sqlalchemy'].startswith('2.0') and 'airflow' in results and results['airflow'] != 'not installed':
        print("- Current setup may have dependency conflicts")
        print("- For Airflow compatibility, reinstall using: pip install -r requirements-airflow.txt")
    elif 'airflow' in results and results['airflow'] == 'not installed':
        print("- To install Airflow with compatibility fixes: pip install -r requirements-airflow.txt")
    
if __name__ == "__main__":
    main()