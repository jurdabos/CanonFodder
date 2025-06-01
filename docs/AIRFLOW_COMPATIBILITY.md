# Airflow Compatibility with CanonFodder

This document explains the dependency conflicts between CanonFodder and Apache Airflow, and how they have been resolved.

## The Problem

CanonFodder and Apache Airflow 3.0.1 have conflicting dependencies, particularly:

1. **SQLAlchemy version conflict**: 
   - CanonFodder requires SQLAlchemy 2.0.40 for its ORM models
   - Apache Airflow 3.0.1 requires SQLAlchemy 1.4.54

2. **Other package conflicts**:
   - pydantic: CanonFodder uses 2.10.4, Airflow needs 2.11.5
   - pydantic_core: CanonFodder uses 2.27.2, Airflow needs 2.33.2
   - Pygments: CanonFodder uses 2.19.0, Airflow needs 2.19.1
   - Other minor version conflicts

These conflicts prevent installing both packages in the same environment using standard methods.

## The Solution

To resolve these conflicts, the following changes have been made:

1. **Separated dependencies in setup.py**:
   - Core dependencies (without database-specific packages)
   - Database dependencies (SQLAlchemy 2.0.40+ and Alembic)
   - Airflow dependencies

2. **Added extras_require to setup.py**:
   - `db`: For database-related dependencies
   - `airflow`: For Airflow-specific dependencies
   - `all`: Combines both db and airflow dependencies

3. **Created requirements-airflow.txt**:
   - Contains flexible version constraints for core dependencies
   - Includes Airflow with its required SQLAlchemy version
   - Omits strict version pinning to allow pip's dependency resolver to work

4. **Updated requirements.txt**:
   - Added comments to explain potential conflicts
   - Maintained strict version pinning for core functionality
   - Added warnings about Airflow compatibility

5. **Updated README.md**:
   - Added installation instructions for Airflow compatibility
   - Explained the dependency conflicts and limitations

6. **Created test_airflow_compatibility.py**:
   - A script to check installed package versions
   - Provides compatibility analysis and recommendations

## How to Use

### For Core CanonFodder Functionality (without Airflow)

```shell
pip install -r requirements.txt
# or
pip install -e .
```

### For Airflow Compatibility

```shell
pip install -r requirements-airflow.txt
# or
pip install -e ".[airflow]"
```

## Limitations

When using CanonFodder with Airflow compatibility:

1. Some database functionality may be limited due to SQLAlchemy version differences
2. The ORM models in CanonFodder use SQLAlchemy 2.0 features that are not available in SQLAlchemy 1.4
3. Direct database operations may need to be modified or wrapped in compatibility layers

## Testing Compatibility

Run the included test script to check your environment:

```shell
python tests/test_airflow_compatibility.py
```

This will analyze your installed packages and provide recommendations.

## Future Considerations

1. Consider creating a compatibility layer for database operations that works with both SQLAlchemy 1.4 and 2.0
2. Explore using separate environments for Airflow and CanonFodder core functionality
3. Monitor for updates to Airflow that might resolve these dependency conflicts