# Docker Setup for CanonFodder

This document provides instructions for setting up CanonFodder with Docker.

## Changes Made

The following changes have been made to fix issues with the Docker setup:

1. **Fixed module path in docker-compose.yml**:
   - Changed `python -m canonfodder.scripts.uc_populate` to `python /opt/airflow/scripts/uc_populate.py`
   - This fixes the issue where the script was being referenced as a module in a non-existent package

2. **Updated Dockerfile to install the package properly**:
   - Added `RUN pip install -e ".[airflow]"` to install the package in development mode with Airflow compatibility
   - This ensures that the package is installed with the appropriate dependencies for Airflow compatibility

## Known Limitations

When running CanonFodder with Airflow in Docker, there are some limitations:

1. **SQLAlchemy version conflict**:
   - CanonFodder requires SQLAlchemy 2.0.40 for its ORM models
   - Apache Airflow 3.0.1 requires SQLAlchemy 1.4.54
   - Some database functionality may be limited due to this version difference
   - The ORM models use SQLAlchemy 2.0 features not available in SQLAlchemy 1.4

2. **Other package conflicts**:
   - pydantic: CanonFodder uses 2.10.4, Airflow needs 2.11.5
   - pydantic_core: CanonFodder uses 2.27.2, Airflow needs 2.33.2
   - Pygments: CanonFodder uses 2.19.0, Airflow needs 2.19.1

## Building and Running

To build and run the Docker container:

1. **Build the Docker image**:
   ```
   docker-compose build
   ```

2. **Start the containers**:
   ```
   docker-compose up -d
   ```

3. **Check the logs**:
   ```
   docker-compose logs -f app
   ```

4. **Access the application**:
   - Airflow web interface: http://localhost:8080
   - Adminer (database management): http://localhost:8081

5. **Stop the containers**:
   ```
   docker-compose down
   ```

## Troubleshooting

If you encounter issues:

1. **Check the logs**:
   ```
   docker-compose logs -f app
   docker-compose logs -f db
   ```

2. **Restart a specific service**:
   ```
   docker-compose restart app
   ```

3. **Rebuild the container**:
   ```
   docker-compose build --no-cache app
   docker-compose up -d
   ```

4. **Check database connectivity**:
   - Use Adminer at http://localhost:8081
   - Log in with:
     - System: MySQL
     - Server: db
     - Username: canon
     - Password: canon
     - Database: canonfodder

5. **Run commands inside the container**:
   ```
   docker-compose exec app bash
   ```

## Future Improvements

1. Create a compatibility layer for database operations that works with both SQLAlchemy 1.4 and 2.0
2. Explore using separate environments for Airflow and CanonFodder core functionality
3. Monitor for updates to Airflow that might resolve these dependency conflicts