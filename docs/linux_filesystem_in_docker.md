# Understanding Linux Filesystem Paths in Docker

## What is `/opt` in Linux?

In the Dockerfile, you might have noticed this line:
```
ENV AIRFLOW_HOME=/opt/airflow
```

### The `/opt` Directory Explained

The `/opt` directory is a standard part of the Linux Filesystem Hierarchy Standard (FHS). Here's what it means:

- **Purpose**: `/opt` is intended for the installation of "optional" software packages that are not part of the default operating system installation.
- **Name Origin**: "opt" is short for "optional software"
- **Usage**: It's commonly used for third-party applications that don't follow the standard Linux directory structure or that need to be self-contained.
- **Isolation**: Software in `/opt` typically keeps all its files in one directory tree, making it easier to manage.

### Why Airflow Uses `/opt/airflow`

Airflow is installed in `/opt/airflow` in the Docker container for several reasons:

1. **Convention**: Many containerized applications use `/opt` for application-specific files.
2. **Isolation**: Keeping Airflow in its own directory tree makes it easier to manage.
3. **Permissions**: The Dockerfile creates a dedicated `airflow` user with a home directory at `/opt/airflow`.
4. **Standard Practice**: This is a common practice for Airflow installations in Docker.

### Docker Container Filesystem

In a Docker container:

- The filesystem starts fresh with each container creation
- Paths like `/opt/airflow` exist only inside the container
- These paths are not directly accessible from a Windows host machine
- Docker volumes (defined in `docker-compose.yml`) map between container paths and host paths

## How Docker Maps Directories

In the `docker-compose.yml` file, you'll see mappings like:

```yaml
volumes:
  - ./dags:/opt/airflow/dags
  - ./PQ:/opt/airflow/PQ
  - airflow_data:/opt/airflow
```

These mappings mean:
- Files in local `./dags` directory appear in `/opt/airflow/dags` inside the container
- Files in local `./PQ` directory appear in `/opt/airflow/PQ` inside the container
- The named volume `airflow_data` is mounted at `/opt/airflow` for persistent storage

## Windows vs. Docker Paths

When running Airflow directly on Windows:
- `AIRFLOW_HOME` is set to a Windows path like `C:\Users\jurda\airflow`

When running Airflow in Docker:
- `AIRFLOW_HOME` is set to a Linux path: `/opt/airflow`

This difference explains why the configuration looks different between your local Windows setup and the Docker setup.

## Summary

The `/opt` directory in Linux is a standard location for optional software packages. In the Docker container, Airflow is installed in `/opt/airflow` following this convention. The Docker setup maps directories between your Windows host and the Linux container, allowing you to work with files in both environments.