"""
Teacher Assessment Demo for CanonFodder

This script demonstrates how to create and save Plotly visualizations
for teacher assessment using the updated show_or_save_plotly function.
"""
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import required modules
import plotly.graph_objects as go
from dev.profile import show_or_save_plotly

# Create a simple Plotly figure
fig = go.Figure()
fig.add_trace(go.Scatter(x=[1, 2, 3, 4], y=[10, 11, 12, 13],
                         mode='lines+markers',
                         name='Example Data'))
fig.update_layout(title='Example Visualization for Teacher Assessment',
                  xaxis_title='X Axis',
                  yaxis_title='Y Axis')

# Add a Markdown description
description = """
# Example Visualization

This is a demonstration of how to create a visualization for teacher assessment.

## Key Features
- Automatically saves to the public_visualizations directory
- Includes this Markdown description
- Adds a header indicating it's for teacher assessment
- Includes timestamp and filename information

## How to Use
Simply call `show_or_save_plotly` with your figure, filename, description, and 
make sure the `public` parameter is set to `True` (which is the default).
"""

# Save the visualization for teacher assessment
show_or_save_plotly(fig, "teacher_assessment_demo.html", description=description)

print("\nDemo completed!")
print(f"Check the following locations for the saved visualizations:")
print(f"1. {project_root / 'pics' / 'teacher_assessment_demo.html'}")
print(f"2. {project_root / 'public_visualizations' / 'teacher_assessment_demo.html'}")
print("\nThe second file is specifically formatted for teacher assessment.")