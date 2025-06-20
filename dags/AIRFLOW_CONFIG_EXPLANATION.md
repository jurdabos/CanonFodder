# Airflow Configuration Explanation

## Explanation

### AIRFLOW_HOME

The `AIRFLOW_HOME` setting defines where Airflow will store its configuration files, logs, and database (if using SQLite). This is correctly set to `C:\Users\jurda\airflow`, which is a separate directory from your project.

### AIRFLOW__CORE__DAGS_FOLDER

The `AIRFLOW__CORE__DAGS_FOLDER` setting tells Airflow where to look for DAG files. This should point to the `dags` directory in your CanonFodder project, not to a subdirectory of `AIRFLOW_HOME`.

According to the `dags/README.md` file (line 57):
> "Set the `AIRFLOW__CORE__DAGS_FOLDER` environment variable to the `dags` directory in your CanonFodder project"
This means that Airflow should look for DAGs in `\CanonFodder\dags`, not in `\airflow\dags`.

### Other Settings

The other Airflow settings in `.env` should be:

- `AIRFLOW__CORE__LOAD_EXAMPLES=False`: Disables the example DAGs to keep the UI clean
- `AIRFLOW__CORE__EXECUTOR=LocalExecutor`: Uses the LocalExecutor for a single machine setup
- `AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=sqlite:////%AIRFLOW_HOME%\airflow.db`: Uses SQLite for the database, stored in the Airflow home directory
- `AIRFLOW__WEBSERVER__BASE_URL=http://localhost:8080`: Sets the base URL for the web UI
- `AIRFLOW__WEBSERVER__EXPOSE_CONFIG=True`: Allows viewing the configuration in the web UI

## Summary

This setup allows us to keep the Airflow installation separate from the project files while still using the DAGs defined in the project.