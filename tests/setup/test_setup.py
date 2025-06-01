"""
Test script to verify that the project setup works correctly.
This script checks if the setup.py file in the project root works correctly
and that the setup.py file in the config directory no longer exists.
"""
import subprocess
import sys
import os
from pathlib import Path


def test_root_setup_py():
    """Test that the setup.py file in the project root works correctly."""
    print("Testing setup.py in project root...")

    # Get the project root directory
    project_root = Path(__file__).resolve().parents[2]

    # Save the current directory
    original_dir = os.getcwd()

    try:
        # Change to the project root directory
        os.chdir(project_root)

        # Try to run the setup.py file in the project root in development mode
        # Using --dry-run to avoid actually installing anything
        result = subprocess.run(
            [sys.executable, "setup.py", "develop", "--dry-run"],
            capture_output=True,
            text=True,
            check=False
        )

        # Check if the command succeeded
        if result.returncode == 0:
            print("✅ Test passed: setup.py in project root works correctly")
        else:
            print("❌ Test failed: setup.py in project root failed")
            print(f"Return code: {result.returncode}")
            print(f"Stdout: {result.stdout}")
            print(f"Stderr: {result.stderr}")
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
    finally:
        # Change back to the original directory
        os.chdir(original_dir)


def test_config_setup_py():
    """Test that the setup.py file in the config directory no longer exists."""
    print("\nTesting setup.py in config directory...")

    # Get the project root directory
    project_root = Path(__file__).resolve().parents[2]
    config_dir = project_root / "config"

    # Save the current directory
    original_dir = os.getcwd()

    if not config_dir.exists():
        print("✅ Test passed: config directory no longer exists")
        return

    # Check if setup.py exists in the config directory
    setup_py_path = config_dir / "setup.py"
    if not setup_py_path.exists():
        print("✅ Test passed: setup.py in config directory no longer exists")
        return

    try:
        # Change to the config directory
        os.chdir(config_dir)

        # Try to run the setup.py file in the config directory
        result = subprocess.run(
            [sys.executable, "setup.py", "install"],
            capture_output=True,
            text=True,
            check=False
        )

        # Check if the command failed (it should, since the file should be removed)
        if result.returncode != 0:
            print("✅ Test passed: setup.py in config directory correctly fails")
        else:
            print("❌ Test failed: setup.py in config directory should not work")
            print(f"Return code: {result.returncode}")
            print(f"Stdout: {result.stdout}")
            print(f"Stderr: {result.stderr}")
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
    finally:
        # Change back to the original directory
        os.chdir(original_dir)


if __name__ == "__main__":
    test_root_setup_py()
    test_config_setup_py()
