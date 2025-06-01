# Running Airflow on Windows

## Issues

Apache Airflow is primarily designed for POSIX-compliant operating systems (Linux, macOS) and has limited support for Windows. When trying to run Airflow on Windows, you might encounter the following errors:

1. **`os.register_at_fork` Error**:
   ```
   AttributeError: module 'os' has no attribute 'register_at_fork'
   ```
   This error occurs because Airflow uses the `os.register_at_fork` function, which is only available on POSIX systems and not on Windows.

2. **`fcntl` Module Error**:
   ```
   ModuleNotFoundError: No module named 'fcntl'
   ```
   This error occurs because Airflow's authentication manager uses the `fcntl` module for file locking, which is a Unix-specific module not available on Windows.

## Solution

We've created a patch script that modifies the Airflow installation to work on Windows by:

1. Adding a compatibility check for the `register_at_fork` function
2. Providing an alternative implementation for file locking on Windows using the `msvcrt` module instead of `fcntl`

### Steps to Fix

1. Make sure your virtual environment is activated:
   ```
   .venv\Scripts\activate
   ```

2. Run the patch script:
   ```
   python dags\fix_airflow_windows.py
   ```

3. The script will:
   - Create backups of the original Airflow files that need patching
   - Patch `settings.py` to check if `register_at_fork` exists before calling it
   - Patch multiple files that use the `fcntl` module to use Windows-compatible alternatives:
     - `simple_auth_manager.py`: Uses `msvcrt` for file locking instead of `fcntl`
     - `dag_processing/bundles/base.py`: Uses `msvcrt` for file locking instead of `fcntl`
     - `providers/standard/operators/python.py`: Uses `msvcrt` for file locking in the PythonVirtualenvOperator
   - Display a success message with next steps

4. After the patch is applied, you can continue with the Airflow setup as described in `dags/README.md`:
   ```
   airflow db migrate
   airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com --password admin
   airflow webserver -p 8080
   airflow scheduler
   ```

   Note: In Airflow 3.x, the command `airflow db init` has been replaced with `airflow db migrate`.

## Remaining Limitations

Even with these patches, some Airflow features might still have limitations on Windows:

1. **Other Unix-Specific Dependencies**: There might be other Unix-specific dependencies that are not covered by our patches. If you encounter additional errors, please report them so we can update the patch script.

2. **Performance Considerations**: Some operations might be slower on Windows compared to Unix-based systems due to differences in the operating system architecture.

## Alternative Solutions

If you prefer not to patch the Airflow installation or need a more complete and production-ready Airflow experience on Windows, consider the following options:

1. **Use Docker**: Run Airflow in a Docker container, which isolates it from the Windows environment. This is the recommended approach for production use.

2. **Use WSL2 (Windows Subsystem for Linux)**: Install and run Airflow inside WSL2, which provides a Linux environment on Windows.

For production use, it's strongly recommended to use Airflow on a POSIX-compliant system or through Docker/WSL2.

## Reverting the Patch

If you need to revert the patch, you can restore the original file from the backup created by the script:

```
copy C:\path\to\settings.py.bak C:\path\to\settings.py
```

Replace `C:\path\to\` with the actual path to your Airflow installation.
