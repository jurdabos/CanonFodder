"""Auto-marks every test under tests/integration/ as integration."""

import pytest


def pytest_collection_modifyitems(items):
    """Adds the integration marker to all items in this directory tree."""
    for item in items:
        if "/integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
