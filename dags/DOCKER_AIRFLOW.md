# Running Airflow with Docker for Windows Users

## Why Docker for Windows?

Running Airflow on Windows directly can be challenging due to compatibility issues with POSIX-specific modules like `fcntl`. While we've provided patches in `fix_airflow_windows.py` to address these issues, using Docker is the recommended approach for Windows users because:

1. **Consistent Environment**: Docker provides a Linux-based container that avoids Windows compatibility issues entirely
2. **Simplified Setup**: All dependencies are pre-configured in the container
3. **Production-Like Environment**: The Docker setup closely mirrors how Airflow would run in production
4. **Integrated Services**: The Docker Compose setup includes MySQL and Adminer for a complete development environment

## Quick Start Guide

1. **Prerequisites**:
   - Install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop)
   - Ensure Docker Desktop is running

2. **Setup**:
   ```powershell
   # Navigate to the CanonFodder project directory in PowerShell
   cd C:\path\to\CanonFodder

   # Copy environment variables template
   copy .env.example .env

   # Edit .env file to add your Last.fm API key and username
   # At minimum, set LASTFM_API_KEY and LASTFM_USER
   ```

3. **Start the Docker containers**:
   ```powershell
   docker-compose up -d
   ```

   This command starts all services defined in `docker-compose.yml`:
   - MySQL database
   - CanonFodder application with Airflow
   - Adminer (database management interface)

4. **Access Airflow**:
   - Open your browser and go to http://localhost:8080
   - Log in with username `admin` and password `admin`
   - The DAGs directory is automatically mounted from your local `dags` folder

5. **Run the CanonFodder DAG**:
   - In the Airflow UI, enable the `cf_ingest` DAG
   - Trigger the DAG manually or wait for the scheduled run

## Stopping the Services

When you're done, you can stop the Docker containers:

```powershell
docker-compose down
```

To stop the containers and remove the volumes (this will delete all data):

```powershell
docker-compose down -v
```

## Troubleshooting

If you encounter issues:

1. **Docker Desktop not running error**:
   If you see an error like `open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified`, it means Docker Desktop is not running on your system.
   - Open Docker Desktop application
   - Wait for it to fully start (check the whale icon in your system tray)
   - Try the docker-compose command again

2. **Cannot access Airflow UI at http://localhost:8080**:
   If you can't access the Airflow UI after starting the containers:
   - Check if the container is running: `docker-compose ps`
   - If the container is in a "Restarting" state, there's an issue with the startup process
   - Check the logs for errors: `docker-compose logs app`
   - Look specifically for supervisor logs: `docker-compose exec app cat /var/log/supervisord.log`
   - Check Airflow webserver logs: `docker-compose exec app cat /var/log/airflow-webserver.log`
   - Ensure port 8080 is not being used by another application on your system
   - If you see "Connection reset" errors, try restarting Docker Desktop completely
   - Check your firewall settings to ensure it's not blocking connections to port 8080

3. **Check Docker logs**:
   ```powershell
   docker-compose logs
   ```

4. **Check specific service logs**:
   ```powershell
   docker-compose logs app
   ```

5. **Restart the services**:
   ```powershell
   docker-compose restart
   ```

6. **Rebuild the containers**:
   ```powershell
   docker-compose up -d --build
   ```

7. **If all else fails**:
   - Stop and remove all containers: `docker-compose down`
   - Remove all volumes: `docker-compose down -v`
   - Rebuild from scratch: `docker-compose up -d --build`

For more detailed information, refer to:
- The Docker section in the main [README.md](../README.md) file
- [Understanding Linux Filesystem Paths in Docker](../docs/linux_filesystem_in_docker.md) for an explanation of the `/opt/airflow` directory
