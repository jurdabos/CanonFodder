import matplotlib.pyplot as plt
import os
from pathlib import Path

# Create a simple plot
plt.figure(figsize=(8, 6))
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.title("Test Plot")
plt.xlabel("X")
plt.ylabel("Y")

# Save to pics folder
pics_dir = Path.cwd() / "pics"
pics_dir.mkdir(exist_ok=True)  # Ensure the directory exists
filepath = pics_dir / "test_plot.png"

try:
    plt.savefig(filepath)
    print(f"Plot saved to {filepath}")
except Exception as e:
    print(f"Error saving plot: {e}")

plt.close()  # Close the figure to free memory

# Verify the file was created
if os.path.exists(filepath):
    print(f"File exists at {filepath}")
    print(f"File size: {os.path.getsize(filepath)} bytes")
else:
    print(f"File does not exist at {filepath}")