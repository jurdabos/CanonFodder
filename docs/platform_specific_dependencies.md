# Platform-Specific Dependencies in CanonFodder

## Overview

CanonFodder uses certain packages that are platform-specific, meaning they are only available or needed on specific operating systems. This document explains how these dependencies are handled in the project.

## Windows-Specific Dependencies

### windows-curses

The `windows-curses` package is a Windows-specific implementation of the curses library, which provides terminal handling for console applications. This package is only needed on Windows systems and is not available for Linux or macOS.

In the `requirements.txt` file, we use a platform-specific marker to ensure that `windows-curses` is only installed on Windows:
python
```
windows-curses==2.4.1; platform_system == "Windows"
```

This syntax tells pip to only install the package when the platform is Windows. When running in a Linux environment (like Docker), pip will skip this dependency.

## Docker Compatibility

When running CanonFodder in Docker, the container uses a Linux-based image. The platform-specific marker ensures that `windows-curses` is not installed in the Docker container, preventing build failures.

### Dependency Conflicts in Docker

In addition to platform-specific dependencies, CanonFodder also has to manage dependency conflicts between packages. For example:

- The core application uses `pydantic==2.10.4`
- Apache Airflow requires `pydantic>=2.11.0`

To resolve these conflicts in Docker, the Dockerfile uses `requirements-airflow.txt` instead of `requirements.txt`. The Airflow-compatible requirements file:

1. Uses flexible version constraints instead of pinned versions
2. Omits conflicting dependencies
3. Includes only the essential packages needed for the application to run with Airflow

This approach ensures that Docker builds complete successfully without dependency conflicts.

## Adding New Platform-Specific Dependencies

If you need to add more platform-specific dependencies, follow this pattern:
python
```
package-name==version; platform_system == "Windows"  # For Windows-only packages
package-name==version; platform_system == "Linux"    # For Linux-only packages
package-name==version; platform_system == "Darwin"   # For macOS-only packages
```

You can also use more complex conditions if needed:
python
```
package-name==version; platform_system != "Windows"  # For non-Windows platforms
```

## Testing Platform-Specific Installations

To test that your platform-specific dependencies and dependency conflict resolutions are correctly configured:

1. On Windows: `pip install -r requirements.txt` should install all dependencies including `windows-curses`
2. In Docker: Building the Docker image should complete without errors:
   - It should skip `windows-curses` due to the platform marker
   - It should resolve dependency conflicts by using `requirements-airflow.txt`

You can verify the Docker build with:
bash
```
docker-compose build
```

If the build completes successfully, your configuration is correct.
