"""
Provides the canonisation package.

Submodules
----------
model    – XGBoost training pipeline (train_model, evaluate).
trainer  – Unified training pipeline (run_training).
tuner    – Optuna hyperparameter tuning (run_tuning).
workflow – CLI-facing business logic (avc_summary, propagate_avc, …).
"""

from corefunc.canon.augment import augment_gold_standard  # noqa: F401
from corefunc.canon.model import evaluate, train_model  # noqa: F401
from corefunc.canon.tcn_trainer import run_tcn_training  # noqa: F401
from corefunc.canon.trainer import run_training  # noqa: F401
from corefunc.canon.tuner import run_tuning  # noqa: F401
from corefunc.canon.workflow import (  # noqa: F401
    PREDICTIONS_LOG_PQ,
    avc_summary,
    discover_candidates,
    list_mlflow_runs,
    load_run_model,
    propagate_avc,
    undecided_rows,
    update_avc_decision,
    write_new_candidates,
)
