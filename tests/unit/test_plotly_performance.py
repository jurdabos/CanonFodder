import time
import plotly.graph_objects as go
from pathlib import Path

# Create a simple Plotly figure
fig = go.Figure()
fig.add_trace(go.Scatter(x=[1, 2, 3, 4], y=[10, 11, 12, 13], mode='lines+markers', name='Test'))
fig.update_layout(title='Performance Test Figure', height=500, width=500)

# Save to pics folder
pics_dir = Path.cwd() / "pics"
pics_dir.mkdir(exist_ok=True)  # Ensure the directory exists
filepath = pics_dir / "performance_test.png"

# Test with different scale factors
scale_factors = [2.0, 1.0, 0.5, 0.3]

for scale in scale_factors:
    print(f"\nTesting with scale factor: {scale}")
    
    # Time the write_image operation
    start_time = time.time()
    try:
        fig.write_image(str(filepath), scale=scale)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Image saved successfully in {elapsed:.2f} seconds")
        print(f"File size: {filepath.stat().st_size / 1024:.2f} KB")
    except Exception as e:
        print(f"Error saving image: {e}")