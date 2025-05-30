# This file is deprecated and will be removed in a future release.
# Please use the setup.py file in the config directory instead.
# For development installation, run: pip install -e config/

print("Warning: This setup.py file is deprecated and should not be used.")
print("Please use the setup.py file in the config directory instead.")
print("For development installation, run: pip install -e config/")
print("The CanonFodder.egg-info directory in the project root is not needed and can be safely deleted.")

# Raise an error to prevent installation from the project root
raise RuntimeError(
    "Installation from the project root is not supported. "
    "Please use 'pip install -e config/' instead."
)
