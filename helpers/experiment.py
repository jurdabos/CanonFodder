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
    mlflow.sklearn.log_model(model, name=artifact_path)


def log_cv_fold(
    fold_idx: int,
    metrics: dict[str, float],
    run_name_prefix: str = "fold",
) -> None:
    """Logs one CV fold's metrics as a nested run."""
    with mlflow.start_run(run_name=f"{run_name_prefix}_{fold_idx}", nested=True):
        mlflow.log_param("fold", fold_idx)
        mlflow.log_metrics(metrics)


def log_confusion_matrix(y_true, y_pred, labels: list[str] | None = None) -> None:
    """Saves a confusion matrix heatmap as an MLflow artefact."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels or ["no link", "link"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    path = PROJECT_ROOT / "ML" / "confusion_matrix.png"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    mlflow.log_artifact(str(path))
    log.debug("Logged confusion matrix to MLflow.")


def log_feature_importance(model, feature_names: list[str], top_n: int = 20) -> None:
    """Saves a horizontal bar chart of feature importances as an MLflow artefact."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    # Extracting importances from tree-based models
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "named_steps"):
        # sklearn Pipeline — digging into the last step
        last = list(model.named_steps.values())[-1]
        if hasattr(last, "feature_importances_"):
            importances = last.feature_importances_
        else:
            log.warning("Model has no feature_importances_; skipping plot.")
            return
    else:
        log.warning("Model has no feature_importances_; skipping plot.")
        return
    indices = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(7, max(4, len(indices) * 0.3)))
    ax.barh([feature_names[i] for i in indices], importances[indices])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {min(top_n, len(indices))} Feature Importances")
    fig.tight_layout()
    path = PROJECT_ROOT / "ML" / "feature_importance.png"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    mlflow.log_artifact(str(path))
    log.debug("Logged feature importance plot to MLflow.")


def log_shap_summary(model, X_sample, feature_names: list[str]) -> None:
    """Computes SHAP values and saves the summary plot as an MLflow artefact."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    try:
        import shap
    except ImportError:
        log.warning("shap not installed; skipping SHAP summary.")
        return
    # Extracting the underlying estimator from a Pipeline if needed
    estimator = model
    if hasattr(model, "named_steps"):
        estimator = list(model.named_steps.values())[-1]
    # Ensuring X_sample is a DataFrame for SHAP compatibility
    if not isinstance(X_sample, pd.DataFrame):
        X_sample = pd.DataFrame(X_sample, columns=feature_names)
    try:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_sample)
        # Handling list output from binary classifiers (e.g. LightGBM)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    except Exception:
        # Falling back to generic Explainer (handles XGBoost param parsing issues)
        try:
            explainer = shap.Explainer(estimator, X_sample)
            shap_values = explainer(X_sample).values
        except Exception as exc:
            log.warning("SHAP computation failed (%s); skipping.", exc)
            return
    fig = plt.figure(figsize=(8, max(4, len(feature_names) * 0.3)))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    fig.tight_layout()
    path = PROJECT_ROOT / "ML" / "shap_summary.png"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(str(path))
    log.debug("Logged SHAP summary plot to MLflow.")
