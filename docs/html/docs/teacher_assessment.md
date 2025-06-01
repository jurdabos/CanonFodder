# Teacher Assessment Guide for CanonFodder

This guide explains how to save Plotly visualizations in a way that makes them accessible for teacher assessment.

## Overview

The CanonFodder project now includes functionality to automatically save Plotly visualizations in a dedicated directory for teacher assessment. These visualizations include special formatting and metadata that make it clear they are intended for assessment purposes.

## How It Works

When you create a Plotly visualization using the `show_or_save_plotly` function, it now:

1. Saves the visualization to the regular `pics` directory (as before)
2. Also saves a copy to the `public_visualizations` directory with:
   - A header indicating it's for teacher assessment
   - A timestamp showing when it was generated
   - The filename for reference
   - Any Markdown description you provide, rendered as HTML

## Using the Feature

### Basic Usage

The `show_or_save_plotly` function now has a `public` parameter that defaults to `True`:

```python
from dev.profile import show_or_save_plotly
import plotly.graph_objects as go

# Create a Plotly figure
fig = go.Figure()
fig.add_trace(go.Scatter(x=[1, 2, 3, 4], y=[10, 11, 12, 13]))
fig.update_layout(title='My Visualization')

# Save for teacher assessment (default behavior)
show_or_save_plotly(fig, "my_visualization.html")
```

This will save the visualization to both:
- `pics/my_visualization.html`
- `public_visualizations/my_visualization.html`

### Adding a Description

You can add a Markdown description to your visualization:

```python
description = """
# My Data Analysis

This visualization shows the relationship between X and Y.

## Key Findings
- Finding 1
- Finding 2
"""

show_or_save_plotly(fig, "my_visualization.html", description=description)
```

The description will be rendered as HTML in both the regular and public versions.

### Disabling Public Saving

If you don't want to save a public copy for teacher assessment, you can set `public=False`:

```python
# Only save to the regular pics directory
show_or_save_plotly(fig, "my_visualization.html", public=False)
```

## Finding the Files

The public visualizations are saved in the `public_visualizations` directory at the root of the project. You can:

1. Navigate to this directory in your file explorer
2. Share the HTML files with your teacher via email, cloud storage, or other means
3. The files are self-contained and will work in any modern web browser

## Demo Script

A demo script is provided to show how to use this feature:

```
python examples/teacher_assessment_demo.py
```

This will create example visualizations in both the `pics` and `public_visualizations` directories.

## Best Practices

1. **Add Descriptive Filenames**: Use clear, descriptive filenames that indicate what the visualization shows.
2. **Include Markdown Descriptions**: Add detailed descriptions to explain your analysis and findings.
3. **Organize by Assignment**: Consider creating subdirectories within `public_visualizations` for different assignments.
4. **Test Before Submitting**: Open the HTML files in a browser to ensure they display correctly before submitting.