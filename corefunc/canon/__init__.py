"""
Provides the canonisation package.

Submodules
----------
model    – XGBoost training pipeline (train_model, evaluate).
workflow – CLI-facing business logic (avc_summary, propagate_avc, …).
"""
from corefunc.canon.model import train_model, evaluate  # noqa: F401
from corefunc.canon.workflow import (  # noqa: F401
    avc_summary,
    propagate_avc,
    undecided_rows,
    update_avc_decision,
    list_mlflow_runs,
    load_run_model,
    discover_candidates,
    write_new_candidates,
)
from corefunc.canon.augment import augment_gold_standard  # noqa: F401
