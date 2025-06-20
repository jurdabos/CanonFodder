import os
import sys
import shutil
import re
from pathlib import Path

def find_airflow_cli_py():
    """Find the airflow cli.py file in the Docker container."""
    # In Docker, Airflow is installed in the system Python
    possible_paths = [
        "/usr/local/lib/python3.12/site-packages/airflow/cli/cli_parser.py",
        "/usr/lib/python3.12/site-packages/airflow/cli/cli_parser.py",
    ]

    for path in possible_paths:
        print(f"Looking for Airflow cli_parser.py at {path}")
        if os.path.exists(path):
            return path

    # If not found in standard locations, try to find it using pip
    try:
        import airflow
        airflow_path = os.path.dirname(airflow.__file__)
        cli_path = os.path.join(airflow_path, 'cli', 'cli_parser.py')
        print(f"Looking for Airflow cli_parser.py at {cli_path}")
        if os.path.exists(cli_path):
            return cli_path
    except ImportError:
        pass

    print("Error: Could not find Airflow cli_parser.py")
    return None

def backup_file(file_path):
    """Create a backup of the original file"""
    backup_path = file_path + '.bak'
    shutil.copy2(file_path, backup_path)
    print(f"Created backup at {backup_path}")
    return backup_path

def patch_cli_parser(cli_path):
    """Patch the Airflow cli_parser.py file to add 'users' command with helpful message."""
    if not cli_path:
        return False

    # Create a backup of the original file
    backup_path = backup_file(cli_path)

    try:
        # Read the original file with explicit utf-8 encoding
        try:
            with open(cli_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # If utf-8 fails, try with binary mode and then decode with error handling
            with open(cli_path, 'rb') as f:
                content = f.read().decode('utf-8', errors='replace')

        # In Airflow 3.0.1, the commands are defined differently
        # Look for the get_parser function
        parser_pattern = r'def get_parser\(.*?\):'
        parser_match = re.search(parser_pattern, content, re.DOTALL)
        
        if not parser_match:
            print("Error: Could not find the get_parser function in cli_parser.py")
            return False
        
        # Find the subparsers section in the get_parser function
        subparsers_pattern = r'subparsers\s*=\s*parser\.add_subparsers\('
        subparsers_match = re.search(subparsers_pattern, content, re.DOTALL)
        
        if not subparsers_match:
            print("Error: Could not find the subparsers section in the get_parser function")
            return False
        
        # Find the position to insert our new command
        # We'll look for the end of the get_parser function
        function_end_pattern = r'def get_parser\(.*?\):.*?return parser'
        function_end_match = re.search(function_end_pattern, content, re.DOTALL)
        
        if not function_end_match:
            print("Error: Could not find the end of the get_parser function")
            return False
        
        # Get the function content
        function_content = function_end_match.group(0)
        
        # Check if 'users' is already in the function
        if "add_subparsers(dest='users'" in function_content:
            print("The 'users' command is already defined in cli_parser.py")
            return False
        
        # Find the position to insert our new command
        # We'll insert it just before the 'return parser' statement
        return_pattern = r'(\s+)return parser'
        return_match = re.search(return_pattern, function_content)
        
        if not return_match:
            print("Error: Could not find the 'return parser' statement in the get_parser function")
            return False
        
        # Get the indentation
        indentation = return_match.group(1)
        
        # Create the new command code
        new_command = f"\n{indentation}# Add 'users' command with helpful message\n"
        new_command += f"{indentation}users_subparser = subparsers.add_parser(\n"
        new_command += f"{indentation}    'users',\n"
        new_command += f"{indentation}    help='User management commands (deprecated in Airflow 3.x)'\n"
        new_command += f"{indentation})\n"
        new_command += f"{indentation}users_subparser.add_argument(\n"
        new_command += f"{indentation}    '--help', action='store_true', help='Show this help message'\n"
        new_command += f"{indentation})\n"
        new_command += f"{indentation}users_subparser.set_defaults(func=lambda args: print(\"\"\"\n"
        new_command += f"The 'users' command is not available in Airflow 3.x.\n"
        new_command += f"Please use one of the following commands instead:\n"
        new_command += f"- 'airflow user list' - List users\n"
        new_command += f"- 'airflow user create' - Create a user\n"
        new_command += f"- 'airflow user delete' - Delete a user\n"
        new_command += f"- 'airflow user add-role' - Add role to a user\n"
        new_command += f"- 'airflow user remove-role' - Remove role from a user\n"
        new_command += f"\"\"\"))\n"
        
        # Insert the new command before the 'return parser' statement
        modified_function = function_content.replace(
            return_match.group(0),
            f"{new_command}{return_match.group(0)}"
        )
        
        # Replace the original function with the modified one
        modified_content = content.replace(function_end_match.group(0), modified_function)
        
        # Write the modified content back to the file
        with open(cli_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        print(f"Successfully patched {cli_path}")
        print("The 'airflow users' command now provides helpful guidance on using the correct commands.")
        return True
    
    except Exception as e:
        print(f"Error patching cli_parser.py: {e}")
        print(f"Restoring from backup {backup_path}")
        shutil.copy2(backup_path, cli_path)
        return False

if __name__ == "__main__":
    print("Patching Airflow CLI to add helpful 'users' command...")
    
    # Find the Airflow cli_parser.py file
    cli_path = find_airflow_cli_py()
    
    # Patch the cli_parser.py file
    success = patch_cli_parser(cli_path)
    
    if success:
        print("\nPatch applied successfully!")
        print("\nInstructions:")
        print("1. Now when you run 'airflow users', you'll get helpful guidance on using the correct commands.")
        print("2. For example, to create a user, use: 'airflow user create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com --password admin'")
    else:
        print("\nPatch failed to apply. Please check the error messages above.")