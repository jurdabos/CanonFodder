"""
Test script to verify that the project setup works correctly.
This script checks if the setup.py file in the project root raises an error
when someone tries to install the package from the project root.
"""
import subprocess
import sys
import os
from pathlib import Path

def test_root_setup_py():
    """Test that the setup.py file in the project root raises an error."""
    print("Testing setup.py in project root...")
    
    # Get the project root directory
    project_root = Path(__file__).resolve().parents[2]
    
    # Save the current directory
    original_dir = os.getcwd()
    
    try:
        # Change to the project root directory
        os.chdir(project_root)
        
        # Try to run the setup.py file in the project root
        result = subprocess.run(
            [sys.executable, "setup.py", "install"],
            capture_output=True,
            text=True,
            check=False
        )
        
        # Check if the command failed with the expected error message
        if result.returncode != 0 and "Installation from the project root is not supported" in result.stderr:
            print("✅ Test passed: setup.py in project root correctly raises an error")
        else:
            print("❌ Test failed: setup.py in project root did not raise the expected error")
            print(f"Return code: {result.returncode}")
            print(f"Stdout: {result.stdout}")
            print(f"Stderr: {result.stderr}")
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
    finally:
        # Change back to the original directory
        os.chdir(original_dir)

def test_config_setup_py():
    """Test that the setup.py file in the config directory works correctly."""
    print("\nTesting setup.py in config directory...")
    
    # Get the project root directory
    project_root = Path(__file__).resolve().parents[2]
    config_dir = project_root / "config"
    
    # Save the current directory
    original_dir = os.getcwd()
    
    if not config_dir.exists():
        print(f"❌ Test failed: config directory not found at {config_dir.absolute()}")
        return
    
    try:
        # Change to the config directory
        os.chdir(config_dir)
        
        # Try to run the setup.py file in the config directory in development mode
        # Using --dry-run to avoid actually installing anything
        result = subprocess.run(
            [sys.executable, "setup.py", "develop", "--dry-run"],
            capture_output=True,
            text=True,
            check=False
        )
        
        # Check if the command succeeded
        if result.returncode == 0:
            print("✅ Test passed: setup.py in config directory works correctly")
        else:
            print("❌ Test failed: setup.py in config directory failed")
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