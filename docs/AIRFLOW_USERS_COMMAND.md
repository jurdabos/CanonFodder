# Airflow 'users' Command Fix

## Issue

When trying to use the `airflow users` command in Airflow 3.x, you might encounter the following error:

```
airflow command error: argument GROUP_OR_COMMAND: invalid choice: 'users' (choose from api-server, assets, backfill, cheat-sheet, config, connections, dag-processor, dags, db, info, jobs, kerberos, plugins, pools, providers, rotate-fernet-key, scheduler, standalone, tasks, triggerer, variables, version), see help above.
```

This error occurs because in Airflow 3.x, there is no standalone `users` command. Instead, you need to use specific subcommands like `airflow user create`, `airflow user list`, etc.

## Solution

We've created patch scripts that modify the Airflow CLI to add a helpful 'users' command that provides guidance on using the correct commands.

### Steps to Fix (Local Environment)

1. Make sure your virtual environment is activated:
   ```
   .venv\Scripts\activate
   ```

2. Run the patch script:
   ```
   python scripts\fix_airflow_users_command.py
   ```

3. The script will:
   - Find the Airflow CLI parser file in your virtual environment
   - Create a backup of the original file
   - Add a new 'users' command to the Airflow CLI parser
   - The new command will provide helpful guidance when you run `airflow users`

### Steps to Fix (Docker Environment)

If you're running Airflow in Docker, use the Docker-specific version of the script:

1. Run the Docker-specific patch script:
   ```
   docker-compose exec app python /opt/airflow/scripts/fix_airflow_users_command_docker_v3.py
   ```

2. The script will:
   - Find the Airflow CLI parser file in the Docker container
   - Create a backup of the original file
   - Add a new 'users' command to the Airflow CLI parser
   - The new command will provide helpful guidance when you run `airflow users`

3. After the patch is applied, when you run `airflow users`, you'll get a helpful message explaining the correct commands to use:
   ```
   The 'users' command is not available in Airflow 3.x.
   Please use one of the following commands instead:
   - 'airflow user list' - List users
   - 'airflow user create' - Create a user
   - 'airflow user delete' - Delete a user
   - 'airflow user add-role' - Add role to a user
   - 'airflow user remove-role' - Remove role from a user
   ```

## Correct Commands for User Management

Here are the correct commands for user management in Airflow 3.x:

1. **Create a user**:
   ```
   airflow user create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com --password admin
   ```

2. **List users**:
   ```
   airflow user list
   ```

3. **Delete a user**:
   ```
   airflow user delete --username admin
   ```

4. **Add a role to a user**:
   ```
   airflow user add-role --username admin --role Public
   ```

5. **Remove a role from a user**:
   ```
   airflow user remove-role --username admin --role Public
   ```

## Reverting the Patch

### Local Environment

If you need to revert the patch in your local environment, you can restore the original file from the backup created by the script:

```
copy C:\path\to\cli_parser.py.bak C:\path\to\cli_parser.py
```

Replace `C:\path\to\` with the actual path to your Airflow installation.

### Docker Environment

If you need to revert the patch in your Docker environment, you can restore the original file from the backup created by the script:

```
docker-compose exec app bash -c "cp /path/to/cli_parser.py.bak /path/to/cli_parser.py"
```

Replace `/path/to/` with the actual path to your Airflow installation in the Docker container. This is typically one of:
- `/usr/local/lib/python3.12/site-packages/airflow/cli/cli_parser.py`
- `/usr/lib/python3.12/site-packages/airflow/cli/cli_parser.py`
