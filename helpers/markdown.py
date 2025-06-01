"""
Helper functions for rendering Markdown content using Showdown.

This module provides functions to convert Markdown to HTML using the Showdown
JavaScript library. It supports both direct text conversion and file-based operations.
"""
import os
from pathlib import Path
from typing import Optional, Union

# Flag to track if Showdown is available
SHOWDOWN_AVAILABLE = False

# Try to import showdown or execjs
try:
    import showdown
    SHOWDOWN_AVAILABLE = True
except ImportError:
    try:
        import execjs
        SHOWDOWN_AVAILABLE = True
    except ImportError:
        pass


def render_markdown(markdown_text: str, output_file: Optional[Union[str, Path]] = None) -> str:
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
    
    Raises
    ------
    ImportError
        If neither py-showdown nor PyExecJS is available
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


def render_markdown_file(input_file: Union[str, Path], output_file: Optional[Union[str, Path]] = None) -> str:
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
    
    Raises
    ------
    FileNotFoundError
        If the input file does not exist
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {input_path}")
    
    if output_file is None:
        output_file = input_path.with_suffix('.html')
    
    with open(input_path, 'r', encoding='utf-8') as f:
        markdown_text = f.read()
    
    return render_markdown(markdown_text, output_file)


def create_visualization_with_description(figura, description_md: str, filepath: Union[str, Path]) -> Path:
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
        
    Returns
    -------
    Path
        The path to the saved HTML file
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