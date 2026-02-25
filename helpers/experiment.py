"""
Centralises MLflow experiment-tracking for c9r model training.

Keeps MLflow concerns out of canon.py so the training logic stays readable.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any
import mlflow

log = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACKING_DB = PROJECT_ROOT / "mlruns.db"
TRACKING_URI = f"sqlite:///{TRACKING_DB.as_posix()}"
DEFAULT_EXPERIMENT = "c9r-record-linkage"


def init_experiment(name: str = DEFAULT_EXPERIMENT) -> str:
    """Configures the local SQLite tracking URI and returns the experiment ID."""
    mlflow.set_tracking_uri(TRACKING_URI)
    exp = mlflow.set_experiment(name)
    log.info("MLflow experiment '%s' (id=%s) at %s", name, exp.experiment_id, TRACKING_URI)
    return exp.experiment_id


def start_run(run_name: str | None = None, nested: bool = False) -> mlflow.ActiveRun:
    """Starts an MLflow run and returns the context manager."""
    return mlflow.start_run(run_name=run_name, nested=nested)


def log_params(params: dict[str, Any]) -> None:
    """Logs a flat dictionary of parameters to the active run."""
    mlflow.log_params(params)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Logs a dictionary of metrics to the active run."""
    mlflow.log_metrics(metrics, step=step)


def log_artifact(path: str | Path) -> None:
    """Logs a local file as an artefact of the active run."""
    mlflow.log_artifact(str(path))


def log_model(model: Any, artifact_path: str = "model") -> None:
    """Logs an sklearn-compatible model to the active run."""
    mlflow.sklearn.log_model(model, artifact_path=artifact_path)


def log_cv_fold(
    fold_idx: int,
    metrics: dict[str, float],
    run_name_prefix: str = "fold",
) -> None:
    """Logs one CV fold's metrics as a nested run."""
    with mlflow.start_run(run_name=f"{run_name_prefix}_{fold_idx}", nested=True):
        mlflow.log_param("fold", fold_idx)
        mlflow.log_metrics(metrics)
