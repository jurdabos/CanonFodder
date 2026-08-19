"""Path helpers for cross-platform CLI inputs (WSL ↔ Windows mounts)."""

from __future__ import annotations

import os
import re
from pathlib import Path

# Matches Windows drive paths with a separator after the colon: D:\x or D:/x
_DRIVE_RE = re.compile(r"^([A-Za-z]):[\\/](.*)$")

WSL_MOUNT_ROOT = "/mnt"


def is_wsl() -> bool:
    """Returns True when running inside Windows Subsystem for Linux."""
    if "WSL_DISTRO_NAME" in os.environ:
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        return False


def to_wsl_mounted(raw: str) -> Path:
    """Converts a Windows drive path to its WSL mount when running under WSL.

    `D:\\data\\f.txt` and `D:/data/f.txt` both become `/mnt/d/data/f.txt`.
    POSIX paths, strings without a drive-letter-plus-separator prefix (e.g. the
    backslash-stripped `D:dataf.txt` a shell may produce), and non-WSL platforms
    pass through unchanged.
    """
    m = _DRIVE_RE.match(raw)
    if not m or not is_wsl():
        return Path(raw)
    drive = m.group(1).lower()
    rest = m.group(2).replace("\\", "/")
    return Path(WSL_MOUNT_ROOT) / drive / rest
