import matplotlib.pyplot as plt
import pandas as pd


def test_function():
    """
    This is a test function to verify the PEP formatter.
    """
    # Loadingdata from CSV
    df = pd.read_csv("data.csv")
    
    # Processingthe data
    df = df.dropna()
    
    # Creatinga plot

    plt.figure(figsize=(10, 6))
    plt.plot(df['x'], df['y'])
    
    # Addinglabels to the plot
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.title("Test Plot")
    
    # Save the plot
    plt.savefig("test_plot.png", dpi=300)  # to high resolution for better quality
    
    return df


def another_function(x, y):
    """Another test function."""
    result = x + y  # to add the two numbers
    
    return result