FROM python:3.12-slim

LABEL maintainer="CanonFodder Team"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV AIRFLOW_HOME=/opt/airflow

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    default-libmysqlclient-dev \
    git \
    supervisor \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create airflow user
RUN useradd -ms /bin/bash -d ${AIRFLOW_HOME} airflow

# Create necessary directories
WORKDIR ${AIRFLOW_HOME}

# Copy requirements files
COPY requirements.txt requirements-airflow.txt ./

# Install Python dependencies (using Airflow-compatible requirements for Docker)
RUN pip install --no-cache-dir -r requirements-airflow.txt

# Copy project files
COPY . .

# Set ownership to airflow user
RUN chown -R airflow:airflow ${AIRFLOW_HOME}

# Switch to airflow user
USER airflow

# Expose ports
EXPOSE 8080 8081

# The initialization and startup commands are defined in docker-compose.yml
# This allows for proper sequencing with database initialization
CMD ["bash", "-c", "sleep infinity"]
