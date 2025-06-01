"""
Simple test script to verify that INTERACTIVE_MODE is accessible and can be modified.
"""
import sys
import os

# Add the parent directory to sys.path to allow importing the profile module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the profile module
import dev.profile as profile

# Print the initial value of INTERACTIVE_MODE
print(f"Initial INTERACTIVE_MODE: {profile.INTERACTIVE_MODE}")

# Modify INTERACTIVE_MODE
profile.INTERACTIVE_MODE = False

# Print the modified value
print(f"Modified INTERACTIVE_MODE: {profile.INTERACTIVE_MODE}")


# Verify that the variable is accessible in functions
def test_access():
    print(f"INTERACTIVE_MODE in function: {profile.INTERACTIVE_MODE}")


test_access()

print("Test completed successfully!")
