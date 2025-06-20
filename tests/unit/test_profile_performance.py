import subprocess
import time
import sys
from pathlib import Path

def run_command_with_timeout(cmd, timeout=60):
    """Run a command with a timeout and return the result."""
    print(f"Running command: {cmd}")
    start_time = time.time()
    
    try:
        # Run the command with a timeout
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout
        )
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"Command completed in {elapsed:.2f} seconds")
        print(f"Return code: {result.returncode}")
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout[:500])  # Print first 500 chars of stdout
            if len(result.stdout) > 500:
                print("... (output truncated)")
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr[:500])  # Print first 500 chars of stderr
            if len(result.stderr) > 500:
                print("... (output truncated)")
        
        return True, elapsed
    
    except subprocess.TimeoutExpired:
        print(f"Command timed out after {timeout} seconds")
        return False, timeout

# Test running profile.py with --no-interactive flag
print("Testing profile.py with --no-interactive flag")
success, elapsed = run_command_with_timeout(
    ["python", "dev\\profile.py", "--no-interactive"],
    timeout=120  # 2 minutes timeout
)

if success:
    print(f"SUCCESS: profile.py completed in {elapsed:.2f} seconds with --no-interactive flag")
    
    # Check if text files were created in the pics directory
    pics_dir = Path.cwd() / "pics"
    txt_files = list(pics_dir.glob("*.txt"))
    
    if txt_files:
        print(f"Found {len(txt_files)} text files in the pics directory:")
        for txt_file in txt_files:
            print(f"  - {txt_file.name}")
            # Print the content of the first text file
            if txt_file == txt_files[0]:
                print("Content of first text file:")
                print(txt_file.read_text())
    else:
        print("No text files found in the pics directory")
else:
    print("FAILURE: profile.py did not complete within the timeout period")
    sys.exit(1)