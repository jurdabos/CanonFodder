import os
import re
import shutil
import sys
import platform

def backup_file(file_path):
    """Create a backup of the original file"""
    backup_path = file_path + '.bak'
    shutil.copy2(file_path, backup_path)
    print(f"Created backup at {backup_path}")
    return backup_path

def patch_settings_file():
    """Patch the Airflow settings.py file to work on Windows"""
    # Path to the Airflow settings.py file
    # Get the virtual environment directory (parent of sys.executable)
    venv_dir = os.path.dirname(os.path.dirname(sys.executable))
    airflow_dir = os.path.join(venv_dir, 'Lib', 'site-packages', 'airflow')
    settings_path = os.path.join(airflow_dir, 'settings.py')
    print(f"Looking for Airflow settings.py at {settings_path}")

    if not os.path.exists(settings_path):
        print(f"Error: Could not find Airflow settings.py at {settings_path}")
        return False

    # Create a backup of the original file
    backup_path = backup_file(settings_path)

    try:
        # Read the original file with explicit utf-8 encoding
        try:
            with open(settings_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # If utf-8 fails, try with binary mode and then decode with error handling
            with open(settings_path, 'rb') as f:
                content = f.read().decode('utf-8', errors='replace')

        # Find the line with os.register_at_fork
        register_at_fork_pattern = r'(\s+)os\.register_at_fork\(after_in_child=clean_in_fork\)'
        match = re.search(register_at_fork_pattern, content)

        if not match:
            print("Error: Could not find the os.register_at_fork line in settings.py")
            return False

        # Get the indentation
        indentation = match.group(1)

        # Replace the line with a conditional check
        patched_code = f"{indentation}if hasattr(os, 'register_at_fork'):\n{indentation}    os.register_at_fork(after_in_child=clean_in_fork)\n{indentation}else:\n{indentation}    print(\"Warning: os.register_at_fork is not available on this platform (Windows). This is expected and safe to ignore.\")"

        # Apply the patch
        patched_content = re.sub(register_at_fork_pattern, patched_code, content)

        # Write the patched content back to the file with the same encoding
        with open(settings_path, 'w', encoding='utf-8') as f:
            f.write(patched_content)

        print(f"Successfully patched {settings_path}")
        print("You can now run 'airflow db migrate' without errors on Windows.")
        return True

    except Exception as e:
        print(f"Error patching settings.py: {e}")
        print(f"Restoring from backup {backup_path}")
        shutil.copy2(backup_path, settings_path)
        return False

def patch_simple_auth_manager():
    """Patch the Airflow simple_auth_manager.py file to work on Windows"""
    # Path to the Airflow simple_auth_manager.py file
    venv_dir = os.path.dirname(os.path.dirname(sys.executable))
    airflow_dir = os.path.join(venv_dir, 'Lib', 'site-packages', 'airflow')
    auth_manager_path = os.path.join(airflow_dir, 'api_fastapi', 'auth', 'managers', 'simple', 'simple_auth_manager.py')
    print(f"Looking for Airflow simple_auth_manager.py at {auth_manager_path}")

    if not os.path.exists(auth_manager_path):
        print(f"Error: Could not find Airflow simple_auth_manager.py at {auth_manager_path}")
        return False

    # Create a backup of the original file
    backup_path = backup_file(auth_manager_path)

    try:
        # Read the original file to extract the license header and imports
        try:
            with open(auth_manager_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # If utf-8 fails, try with binary mode and then decode with error handling
            with open(auth_manager_path, 'rb') as f:
                content = f.read().decode('utf-8', errors='replace')

        # Find the line with from __future__ import annotations
        future_import_pattern = r'from __future__ import annotations\s*\n'
        match = re.search(future_import_pattern, content)

        if not match:
            print("Error: Could not find the 'from __future__ import annotations' line in simple_auth_manager.py")
            return False

        # Find the next import after the future import
        next_import_pattern = r'import\s+[a-zA-Z_]+'
        next_import_match = re.search(next_import_pattern, content[match.end():])

        if not next_import_match:
            print("Error: Could not find the next import after 'from __future__ import annotations' in simple_auth_manager.py")
            return False

        # Calculate the position of the next import in the original content
        next_import_pos = match.end() + next_import_match.start()

        # Create the fcntl block as specified in the issue description
        fcntl_block = """
# -----------------------------------------------------------------------------
# Optional POSIX-only file-locking. Windows will just skip it.
try:
    import fcntl                     # noqa: WPS433  (Unix only)
except ModuleNotFoundError:
    fcntl = None                     # type: ignore
# -----------------------------------------------------------------------------

"""
        # Create the new content with the fcntl block
        new_content = content[:match.end()] + fcntl_block + content[next_import_pos:]

        # Find and replace the fcntl.flock usage in the init method
        flock_pattern = r'(\s+)fcntl\.flock\(file, fcntl\.LOCK_EX \| fcntl\.LOCK_NB\)'
        match = re.search(flock_pattern, new_content)

        if match:
            indentation = match.group(1)
            flock_replacement = f"{indentation}if fcntl:\n{indentation}    fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)\n{indentation}else:\n{indentation}    import msvcrt\n{indentation}    msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)"
            new_content = re.sub(flock_pattern, flock_replacement, new_content)
        else:
            print("Warning: Could not find the fcntl.flock line in simple_auth_manager.py")

        # Find and replace the fcntl.flock unlock in the finally block
        unlock_pattern = r'(\s+)fcntl\.flock\(file, fcntl\.LOCK_UN\)'
        match = re.search(unlock_pattern, new_content)

        if match:
            indentation = match.group(1)
            unlock_replacement = f"{indentation}if fcntl:\n{indentation}    fcntl.flock(file, fcntl.LOCK_UN)\n{indentation}else:\n{indentation}    import msvcrt\n{indentation}    msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)"
            new_content = re.sub(unlock_pattern, unlock_replacement, new_content)
        else:
            print("Warning: Could not find the fcntl.flock unlock line in simple_auth_manager.py")

        # Write the new content back to the file
        with open(auth_manager_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"Successfully patched {auth_manager_path}")
        return True

    except Exception as e:
        print(f"Error patching simple_auth_manager.py: {e}")
        print(f"Restoring from backup {backup_path}")
        shutil.copy2(backup_path, auth_manager_path)
        return False

def patch_dag_bundles_base():
    """Patch the Airflow dag_processing/bundles/base.py file to work on Windows"""
    # Path to the Airflow base.py file
    venv_dir = os.path.dirname(os.path.dirname(sys.executable))
    airflow_dir = os.path.join(venv_dir, 'Lib', 'site-packages', 'airflow')
    base_path = os.path.join(airflow_dir, 'dag_processing', 'bundles', 'base.py')
    print(f"Looking for Airflow dag_processing/bundles/base.py at {base_path}")

    if not os.path.exists(base_path):
        print(f"Error: Could not find Airflow dag_processing/bundles/base.py at {base_path}")
        return False

    # Create a backup of the original file
    backup_path = backup_file(base_path)

    try:
        # Read the original file with explicit utf-8 encoding
        try:
            with open(base_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # If utf-8 fails, try with binary mode and then decode with error handling
            with open(base_path, 'rb') as f:
                content = f.read().decode('utf-8', errors='replace')

        # Replace the fcntl import with a conditional import
        fcntl_import_pattern = r'import fcntl'
        # Get the indentation from the original line
        match = re.search(r'(\s*)import fcntl', content)
        if match:
            indent = match.group(1)
        else:
            indent = ''

        fcntl_import_replacement = f"""{indent}try:
{indent}    import fcntl
{indent}except ImportError:
{indent}    # fcntl is not available on Windows
{indent}    fcntl = None
{indent}    import msvcrt  # Windows-specific module for file locking"""

        # Apply the patch for import
        patched_content = content.replace(fcntl_import_pattern, fcntl_import_replacement)

        # Replace the fcntl import in the from statement
        from_fcntl_pattern = r'from fcntl import LOCK_SH, LOCK_UN, flock'
        # Get the indentation from the original line
        match = re.search(r'(\s*)from fcntl import LOCK_SH, LOCK_UN, flock', patched_content)
        if match:
            indent = match.group(1)
        else:
            indent = ''

        from_fcntl_replacement = f"""{indent}try:
{indent}    from fcntl import LOCK_SH, LOCK_UN, flock
{indent}except ImportError:
{indent}    # Define constants for Windows
{indent}    LOCK_SH = 0  # Shared lock
{indent}    LOCK_UN = 0  # Unlock
{indent}    # Define flock function for Windows
{indent}    def flock(file, operation):
{indent}        if operation == LOCK_SH:
{indent}            msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
{indent}        elif operation == LOCK_UN:
{indent}            msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)"""

        # Apply the patch for from import
        patched_content = patched_content.replace(from_fcntl_pattern, from_fcntl_replacement)

        # Find and replace the fcntl.flock usage in the _remove_stale_bundle method
        flock_pattern = r'(\s+)flock\(f, fcntl\.LOCK_EX \| fcntl\.LOCK_NB\)'
        match = re.search(flock_pattern, patched_content)

        if match:
            indentation = match.group(1)
            flock_replacement = f"{indentation}if fcntl:\n{indentation}    flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)\n{indentation}else:\n{indentation}    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)"
            patched_content = re.sub(flock_pattern, flock_replacement, patched_content)
        else:
            print("Warning: Could not find the flock line in _remove_stale_bundle method")

        # Find and replace the fcntl.flock usage in the lock method
        flock_pattern = r'(\s+)fcntl\.flock\(lock_file, fcntl\.LOCK_EX\)'
        match = re.search(flock_pattern, patched_content)

        if match:
            indentation = match.group(1)
            flock_replacement = f"{indentation}if fcntl:\n{indentation}    fcntl.flock(lock_file, fcntl.LOCK_EX)\n{indentation}else:\n{indentation}    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)"
            patched_content = re.sub(flock_pattern, flock_replacement, patched_content)
        else:
            print("Warning: Could not find the fcntl.flock line in lock method")

        # Find and replace the fcntl.flock unlock in the finally block of lock method
        unlock_pattern = r'(\s+)fcntl\.flock\(lock_file, LOCK_UN\)'
        match = re.search(unlock_pattern, patched_content)

        if match:
            indentation = match.group(1)
            unlock_replacement = f"{indentation}if fcntl:\n{indentation}    fcntl.flock(lock_file, LOCK_UN)\n{indentation}else:\n{indentation}    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)"
            patched_content = re.sub(unlock_pattern, unlock_replacement, patched_content)
        else:
            print("Warning: Could not find the fcntl.flock unlock line in lock method")

        # Write the patched content back to the file with the same encoding
        with open(base_path, 'w', encoding='utf-8') as f:
            f.write(patched_content)

        print(f"Successfully patched {base_path}")
        return True

    except Exception as e:
        print(f"Error patching dag_processing/bundles/base.py: {e}")
        print(f"Restoring from backup {backup_path}")
        shutil.copy2(backup_path, base_path)
        return False

def patch_python_virtualenv_operator():
    """Patch the Airflow providers/standard/operators/python.py file to work on Windows"""
    # Path to the Airflow python.py file
    venv_dir = os.path.dirname(os.path.dirname(sys.executable))
    airflow_dir = os.path.join(venv_dir, 'Lib', 'site-packages', 'airflow')
    python_path = os.path.join(airflow_dir, 'providers', 'standard', 'operators', 'python.py')
    print(f"Looking for Airflow providers/standard/operators/python.py at {python_path}")

    if not os.path.exists(python_path):
        print(f"Error: Could not find Airflow providers/standard/operators/python.py at {python_path}")
        return False

    # Create a backup of the original file
    backup_path = backup_file(python_path)

    try:
        # Read the original file with explicit utf-8 encoding
        try:
            with open(python_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # If utf-8 fails, try with binary mode and then decode with error handling
            with open(python_path, 'rb') as f:
                content = f.read().decode('utf-8', errors='replace')

        # Find and replace the fcntl import and usage in the _ensure_venv_cache_exists method
        fcntl_pattern = r'(\s+)import fcntl\n\n(\s+)fcntl\.flock\(f, fcntl\.LOCK_EX\)'
        match = re.search(fcntl_pattern, content)

        if match:
            indentation1 = match.group(1)
            indentation2 = match.group(2)
            # Ensure proper indentation for the nested code blocks
            fcntl_replacement = f"{indentation1}try:\n{indentation1}    import fcntl\n{indentation1}    fcntl.flock(f, fcntl.LOCK_EX)\n{indentation1}except ImportError:\n{indentation1}    # fcntl is not available on Windows\n{indentation1}    import msvcrt\n{indentation1}    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)"
            patched_content = re.sub(fcntl_pattern, fcntl_replacement, content)
        else:
            print("Warning: Could not find the fcntl import and usage in _ensure_venv_cache_exists method")
            return False

        # Write the patched content back to the file with the same encoding
        with open(python_path, 'w', encoding='utf-8') as f:
            f.write(patched_content)

        print(f"Successfully patched {python_path}")
        return True

    except Exception as e:
        print(f"Error patching providers/standard/operators/python.py: {e}")
        print(f"Restoring from backup {backup_path}")
        shutil.copy2(backup_path, python_path)
        return False

def fix_simple_auth_manager_indentation():
    """Fix indentation issues in the patched simple_auth_manager.py file"""
    # Path to the Airflow simple_auth_manager.py file
    venv_dir = os.path.dirname(os.path.dirname(sys.executable))
    airflow_dir = os.path.join(venv_dir, 'Lib', 'site-packages', 'airflow')
    auth_manager_path = os.path.join(airflow_dir, 'api_fastapi', 'auth', 'managers', 'simple', 'simple_auth_manager.py')
    print(f"Checking indentation in patched {auth_manager_path}")

    if not os.path.exists(auth_manager_path):
        print(f"Error: Could not find Airflow simple_auth_manager.py at {auth_manager_path}")
        return False

    try:
        # Read the original file with explicit utf-8 encoding
        try:
            with open(auth_manager_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # If utf-8 fails, try with binary mode and then decode with error handling
            with open(auth_manager_path, 'rb') as f:
                content = f.read().decode('utf-8', errors='replace')

        # Find the line with from __future__ import annotations
        future_import_pattern = r'from __future__ import annotations\s*\n'
        match = re.search(future_import_pattern, content)

        if not match:
            print("Error: Could not find the 'from __future__ import annotations' line in simple_auth_manager.py")
            return False

        # Replace everything between the future import and the next import with the correct fcntl block
        fcntl_block = """
# -----------------------------------------------------------------------------
# Optional POSIX-only file-locking. Windows will just skip it.
try:
    import fcntl                     # noqa: WPS433  (Unix only)
except ModuleNotFoundError:
    fcntl = None                     # type: ignore
# -----------------------------------------------------------------------------

"""
        # Find the next import after the future import
        next_import_pattern = r'import\s+[a-zA-Z_]+'
        next_import_match = re.search(next_import_pattern, content[match.end():])

        if not next_import_match:
            print("Error: Could not find the next import after 'from __future__ import annotations' in simple_auth_manager.py")
            return False

        # Calculate the position of the next import in the original content
        next_import_pos = match.end() + next_import_match.start()

        # Replace the content between the future import and the next import
        fixed_content = content[:match.end()] + fcntl_block + content[next_import_pos:]

        # Write the fixed content back to the file
        with open(auth_manager_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)

        print(f"Fixed fcntl import block in {auth_manager_path}")
        return True

    except Exception as e:
        print(f"Error fixing indentation in simple_auth_manager.py: {e}")
        return False

if __name__ == "__main__":
    print("Patching Airflow to work on Windows...")

    # Patch settings.py for os.register_at_fork issue
    settings_success = patch_settings_file()

    # Patch simple_auth_manager.py for fcntl module issue
    auth_manager_success = patch_simple_auth_manager()

    # Fix indentation issues in the patched simple_auth_manager.py file
    indentation_fix_success = fix_simple_auth_manager_indentation()

    # Patch dag_processing/bundles/base.py for fcntl module issue
    dag_bundles_success = patch_dag_bundles_base()

    # Patch providers/standard/operators/python.py for fcntl module issue
    python_operator_success = patch_python_virtualenv_operator()

    # Check if all patches were successful
    all_success = settings_success and auth_manager_success and indentation_fix_success and dag_bundles_success and python_operator_success

    if all_success:
        print("\nAll patches applied successfully!")
        print("\nInstructions:")
        print("1. Run 'airflow db migrate' to initialize the Airflow database")
        print("   Note: In Airflow 3.x, the command 'airflow db init' has been replaced with 'airflow db migrate'")
        print("2. Continue with the Airflow setup as described in dags/README.md")
    else:
        print("\nSome patches failed to apply:")
        if not settings_success:
            print("- Settings patch failed")
        if not auth_manager_success:
            print("- Auth manager patch failed")
        if not indentation_fix_success:
            print("- Auth manager indentation fix failed")
        if not dag_bundles_success:
            print("- DAG bundles patch failed")
        if not python_operator_success:
            print("- Python VirtualEnv Operator patch failed")
        print("\nPlease check the error messages above and try running the script again.")
