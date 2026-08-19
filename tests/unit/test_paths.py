"""Tests for helpers/paths.py — WSL drive-path conversion."""

from pathlib import Path


class TestToWslMounted:
    """Tests to_wsl_mounted under forced WSL/non-WSL detection."""

    @staticmethod
    def _wsl(monkeypatch, value=True):
        """Forces is_wsl() to the given value."""
        import helpers.paths as paths

        monkeypatch.setattr(paths, "is_wsl", lambda: value)
        return paths

    def test_backslash_windows_path(self, monkeypatch):
        """Converts D:\\dir\\file.txt to /mnt/d/dir/file.txt under WSL."""
        paths = self._wsl(monkeypatch)
        assert paths.to_wsl_mounted("D:\\adat\\mylifeindata\\mikormelyikcountry.txt") == Path(
            "/mnt/d/adat/mylifeindata/mikormelyikcountry.txt"
        )

    def test_forward_slash_windows_path(self, monkeypatch):
        """Converts D:/dir/file.txt to the same mount under WSL."""
        paths = self._wsl(monkeypatch)
        assert paths.to_wsl_mounted("D:/adat/file.txt") == Path("/mnt/d/adat/file.txt")

    def test_drive_letter_lowercased(self, monkeypatch):
        """Lowercases the drive letter for the mount point."""
        paths = self._wsl(monkeypatch)
        assert paths.to_wsl_mounted("C:\\Users\\x") == Path("/mnt/c/Users/x")

    def test_posix_path_unchanged(self, monkeypatch):
        """Passes POSIX paths through."""
        paths = self._wsl(monkeypatch)
        assert paths.to_wsl_mounted("/home/blai/f.txt") == Path("/home/blai/f.txt")

    def test_mangled_drive_path_unchanged(self, monkeypatch):
        """Passes the backslash-stripped D:dirfile.txt form through (unresolvable)."""
        paths = self._wsl(monkeypatch)
        assert paths.to_wsl_mounted("D:adatfile.txt") == Path("D:adatfile.txt")

    def test_non_wsl_unchanged(self, monkeypatch):
        """Never converts on non-WSL platforms."""
        paths = self._wsl(monkeypatch, value=False)
        assert paths.to_wsl_mounted("D:\\adat\\f.txt") == Path("D:\\adat\\f.txt")

    def test_custom_mount_root(self, monkeypatch):
        """Honours the WSL_MOUNT_ROOT constant when overridden."""
        paths = self._wsl(monkeypatch)
        monkeypatch.setattr(paths, "WSL_MOUNT_ROOT", "/fake/mnt")
        assert paths.to_wsl_mounted("D:\\adat\\f.txt") == Path("/fake/mnt/d/adat/f.txt")


class TestIsWsl:
    """Tests is_wsl detection sources."""

    def test_env_var_detected(self, monkeypatch):
        """Detects WSL via WSL_DISTRO_NAME."""
        import helpers.paths as paths

        monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
        assert paths.is_wsl() is True

    def test_no_env_falls_back_to_proc(self, monkeypatch):
        """Falls back to /proc/version when the env var is absent."""
        import helpers.paths as paths

        monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
        # Whatever the host answer is, it must be a bool without raising
        assert isinstance(paths.is_wsl(), bool)
