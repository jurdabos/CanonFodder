"""
Probes GPU availability for XGBoost and caches the result.

Other models (RF, LightGBM pip, Extra Trees, sklearn composites) are
CPU-only by design — this module only concerns XGBoost's ``device`` param.
"""
from __future__ import annotations
import logging

log = logging.getLogger(__name__)
_CACHED_DEVICE: str | None = None


def get_device() -> str:
    """Returns ``"cuda"`` when an XGBoost GPU probe succeeds, else ``"cpu"``.

    The result is cached after the first call so subsequent imports are free.
    """
    global _CACHED_DEVICE
    if _CACHED_DEVICE is not None:
        return _CACHED_DEVICE
    try:
        import numpy as np
        import xgboost as xgb
        dtrain = xgb.DMatrix(np.array([[1, 2], [3, 4]], dtype=np.float32), label=[0, 1])
        bst = xgb.train({"device": "cuda", "max_depth": 1, "verbosity": 0}, dtrain, num_boost_round=1)
        del bst, dtrain
        _CACHED_DEVICE = "cuda"
        log.info("GPU probe succeeded — using device='cuda'.")
    except Exception as exc:
        _CACHED_DEVICE = "cpu"
        log.warning("GPU probe failed (%s) — falling back to device='cpu'.", exc)
    return _CACHED_DEVICE


def reset_cache() -> None:
    """Clears the cached device selection (useful in tests)."""
    global _CACHED_DEVICE
    _CACHED_DEVICE = None
