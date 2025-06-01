# Showdown Integration for CanonFodder

## Overview
This document outlines a recommended pathway for integrating Showdown, a JavaScript Markdown-to-HTML converter, into the CanonFodder project. Showdown can enhance the project by providing rich text rendering capabilities for documentation, reports, and interactive visualizations.

## Integration Points

### 1. Documentation Rendering
CanonFodder contains several Markdown files that could benefit from HTML rendering:
- `docs/CanonFodder.md` - Main project documentation
- `README.md` - Project README
- `scripts/README.md` - Documentation for scripts
- `docs/lfAPI_improvements.md` - Documentation for Last.fm API improvements

Showdown can be used to render these files as HTML for improved readability in a web interface.

### 2. Enhanced Visualizations
The `dev/profile.py` file generates HTML visualizations using libraries like Plotly and Folium. These visualizations could be enhanced with Markdown-based descriptions and annotations rendered through Showdown.

### 3. Data Profiling Reports
The data profiling functionality in `corefunc/dataprofiler.py` could be extended to generate Markdown reports that are then converted to HTML using Showdown, providing rich, interactive data profiling reports.

## Implementation Steps

### 1. Add Showdown as a Dependency
Add Showdown to the project's dependencies:

```bash
npm install showdown
```

For Python integration, consider using a Python wrapper like `py-showdown` or using a JavaScript runtime like `PyExecJS`:

```bash
pip install py-showdown
# or
pip install PyExecJS
```

### 2. Create a Markdown Rendering Module
Create a new module in the `helpers` directory to handle Markdown rendering:

```python
# helpers/markdown.py
"""
Helper functions for rendering Markdown content using Showdown.
"""
import os
from pathlib import Path
try:
    import showdown
    SHOWDOWN_AVAILABLE = True
except ImportError:
    try:
        import execjs
        SHOWDOWN_AVAILABLE = True
    except ImportError:
        SHOWDOWN_AVAILABLE = False

def render_markdown(markdown_text, output_file=None):
    """
    Renders Markdown text to HTML using Showdown.
    
    Parameters
    ----------
    markdown_text : str
        The Markdown text to render
    output_file : str or Path, optional
        If provided, the HTML will be written to this file
        
    Returns
    -------
    str
        The rendered HTML
    """
    if not SHOWDOWN_AVAILABLE:
        raise ImportError(
            "Showdown is not available. Install py-showdown or PyExecJS."
        )
    
    try:
        # Try using py-showdown if available
        converter = showdown.Converter()
        html = converter.convert(markdown_text)
    except NameError:
        # Fall back to PyExecJS
        showdown_js = """
        var showdown = require('showdown');
        var converter = new showdown.Converter();
        function convert(text) {
            return converter.makeHtml(text);
        }
        """
        ctx = execjs.compile(showdown_js)
        html = ctx.call("convert", markdown_text)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
    
    return html

def render_markdown_file(input_file, output_file=None):
    """
    Renders a Markdown file to HTML using Showdown.
    
    Parameters
    ----------
    input_file : str or Path
        The Markdown file to render
    output_file : str or Path, optional
        If provided, the HTML will be written to this file
        If not provided, the output file will have the same name as the input file
        but with an .html extension
        
    Returns
    -------
    str
        The rendered HTML
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {input_path}")
    
    if output_file is None:
        output_file = input_path.with_suffix('.html')
    
    with open(input_path, 'r', encoding='utf-8') as f:
        markdown_text = f.read()
    
    return render_markdown(markdown_text, output_file)
```

### 3. Integrate with Visualization Code
Modify the visualization code in `dev/profile.py` to include Markdown descriptions rendered through Showdown:

```python
# Example integration in dev/profile.py
from helpers.markdown import render_markdown

# When generating HTML visualizations
def create_visualization_with_description(figura, description_md, filepath):
    """
    Creates a visualization with a Markdown description.
    
    Parameters
    ----------
    figura : plotly.graph_objects.Figure
        The Plotly figure to render
    description_md : str
        Markdown text describing the visualization
    filepath : str or Path
        Path to save the visualization
    """
    # Convert Markdown to HTML
    description_html = render_markdown(description_md)
    
    # Create HTML with visualization and description
    html_path = Path(filepath).with_suffix('.html')
    
    # Generate HTML with both the figure and description
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>CanonFodder Visualization</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .description {{ margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <div class="description">
            {description_html}
        </div>
        <div id="plotly-figure">
            {figura.to_html(full_html=False, include_plotlyjs='cdn')}
        </div>
    </body>
    </html>
    """
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Visualization with description saved to {html_path}")
    return html_path
```

### 4. Add Documentation Rendering Command
Add a command to `main.py` to render project documentation:

```python
# In main.py, add to the argument parser
parser.add_argument(
    "--render-docs",
    action="store_true",
    help="Render Markdown documentation to HTML using Showdown",
)

# In the _cli_entry function, add handling for the new argument
if args.render_docs:
    from helpers.markdown import render_markdown_file
    import glob
    from pathlib import Path
    
    print("Rendering Markdown documentation to HTML...")
    
    # Create docs/html directory if it doesn't exist
    html_dir = Path("docs/html")
    html_dir.mkdir(exist_ok=True, parents=True)
    
    # Render all Markdown files in the project
    md_files = glob.glob("**/*.md", recursive=True)
    for md_file in md_files:
        md_path = Path(md_file)
        # Skip files in the .venv directory
        if ".venv" in md_path.parts:
            continue
        
        # Create output path in docs/html
        rel_path = md_path.relative_to(Path.cwd())
        output_path = html_dir / rel_path.with_suffix('.html')
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        try:
            render_markdown_file(md_path, output_path)
            print(f"Rendered {md_path} to {output_path}")
        except Exception as e:
            print(f"Error rendering {md_path}: {e}")
    
    print("Documentation rendering complete.")
    return 0
```

### 5. Extend Data Profiling with Markdown Reports
Enhance the data profiling functionality to generate Markdown reports:

```python
# In corefunc/dataprofiler.py, add a function to generate Markdown reports
def generate_markdown_report(profile: ProfileResult) -> str:
    """
    Generates a Markdown report from a ProfileResult.
    
    Parameters
    ----------
    profile : ProfileResult
        The profile result to generate a report for
        
    Returns
    -------
    str
        The Markdown report
    """
    # Get the top 20 artists
    top_artists = profile.artist_counts.head(20)
    
    # Generate Markdown
    md = f"""# Data Profiling Report
    
## Overview
- Total scrobbles: {len(profile.df)}
- Unique artists: {len(profile.artist_counts)}

## Top 20 Artists
| Artist | Scrobbles |
|--------|-----------|
"""
    
    for artist, count in top_artists.items():
        md += f"| {artist} | {count} |\n"
    
    # Add more sections as needed
    
    return md

# Add a function to generate and render an HTML report
def generate_html_report(profile: ProfileResult, output_file=None) -> str:
    """
    Generates an HTML report from a ProfileResult using Showdown.
    
    Parameters
    ----------
    profile : ProfileResult
        The profile result to generate a report for
    output_file : str or Path, optional
        If provided, the HTML will be written to this file
        
    Returns
    -------
    str
        The HTML report
    """
    from helpers.markdown import render_markdown
    
    # Generate Markdown report
    md_report = generate_markdown_report(profile)
    
    # Render to HTML
    html_report = render_markdown(md_report, output_file)
    
    return html_report
```

## Benefits of Integration

1. **Improved Documentation**: Showdown enables rich, interactive documentation with syntax highlighting, tables, and other Markdown features.

2. **Enhanced Visualizations**: Adding Markdown descriptions to visualizations improves their interpretability and usefulness.

3. **Rich Data Profiling Reports**: Generating HTML reports from Markdown provides a more user-friendly way to explore data profiling results.

4. **Consistent Styling**: Using Markdown as an intermediate format ensures consistent styling across all documentation and reports.

## Conclusion

Integrating Showdown into CanonFodder would enhance the project's documentation and reporting capabilities, making it more user-friendly and accessible. The recommended approach leverages existing project structures and adds new functionality without disrupting the current workflow.