"""
Flask server for serving the XGBoost model for artist name canonization.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Global variables
app = Flask(__name__)
model: Optional[XGBClassifier] = None
columns: Optional[List[str]] = None
scaler: Optional[RobustScaler] = None
server_thread: Optional[threading.Thread] = None
is_running = False
port = 5000


def load_model() -> Tuple[XGBClassifier, List[str]]:
    """
    Load the XGBoost model and column names from the saved files.
    Returns:
        Tuple[XGBClassifier, List[str]]: The loaded model and column names
    """
    try:
        # Getting the project root directory
        project_root = Path(__file__).resolve().parent.parent
        # Loading the model
        model_path = project_root / "ML" / "xgb.json"
        columns_path = project_root / "ML" / "xgb_columns.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not columns_path.exists():
            raise FileNotFoundError(f"Columns file not found: {columns_path}")
        xgb_model = XGBClassifier()
        xgb_model.load_model(model_path)
        # Loading the column names
        with open(columns_path, "r") as f:
            cols = json.load(f)
        logger.info(f"Model loaded successfully from {model_path}")
        logger.info(f"Columns loaded successfully from {columns_path}")
        return xgb_model, cols
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def preprocess_data(data: Dict[str, Union[float, int, str]]) -> pd.DataFrame:
    """
    Preprocess the input data for prediction.
    Args:
        data (Dict[str, Union[float, int, str]]): The input data
    Returns:
        pd.DataFrame: The preprocessed data
    """
    # Creating a DataFrame with the required columns
    df = pd.DataFrame([data])
    # Ensuring all required columns are present
    for col in columns:
        if col not in df.columns:
            df[col] = 0.0
    # Selecting only the columns used by the model
    df = df[columns]
    return df


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint for making predictions.
    Expected JSON format:
    {
        "data": {
            "feature1": value1,
            "feature2": value2,
            ...
        }
    }
    Returns:
        JSON response with prediction result
    """
    try:
        # Getting the input data
        input_data = request.json
        if not input_data or "data" not in input_data:
            return jsonify({"error": "Invalid input format. Expected JSON with 'data' field."}), 400
        # Preprocessing the data
        data = preprocess_data(input_data["data"])
        # Make prediction
        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0, 1]
        # Returning the prediction
        return jsonify({
            "prediction": int(prediction),
            "probability": float(probability),
            "should_link": bool(prediction == 1)
        })
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """
    Endpoint for making batch predictions.
    Expected JSON format:
    {
        "data": [
            {
                "feature1": value1,
                "feature2": value2,
                ...
            },
            ...
        ]
    }
    Returns:
        JSON response with prediction results
    """
    try:
        # Getting the input data
        input_data = request.json
        if not input_data or "data" not in input_data or not isinstance(input_data["data"], list):
            return jsonify({"error": "Invalid input format. Expected JSON with 'data' field containing a list."}), 400
        results = []
        for item in input_data["data"]:
            # Preprocessing the data
            data = preprocess_data(item)
            # Making prediction
            prediction = model.predict(data)[0]
            probability = model.predict_proba(data)[0, 1]
            # Adding to results
            results.append({
                "prediction": int(prediction),
                "probability": float(probability),
                "should_link": bool(prediction == 1)
            })
        # Returning the predictions
        return jsonify({"results": results})
    except Exception as e:
        logger.error(f"Error making batch prediction: {e}")
        return jsonify({"error": str(e)}), 500


def start_server(server_port: int = 5000) -> bool:
    """
    Start the Flask server in a separate thread.
    Args:
        server_port (int, optional): The port to run the server on. Defaults to 5000.
    Returns:
        bool: True if the server was started successfully, False otherwise
    """
    global model, columns, is_running, server_thread, port
    if is_running:
        logger.warning("Server is already running")
        return True
    try:
        # Loading the model and columns
        model, columns = load_model()
        # Initializing the global scaler
        global scaler
        scaler = RobustScaler()
        # Setting the port
        port = server_port
        # Define the thread function

        def run_server():
            app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
        # Start the server in a separate thread
        server_thread = threading.Thread(target=run_server)
        server_thread.daemon = True
        server_thread.start()
        # Wait for the server to start
        time.sleep(1)
        is_running = True
        logger.info(f"Server started on port {port}")
        return True
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        return False


def stop_server() -> bool:
    """
    Stop the Flask server.
    Returns:
        bool: True if the server was stopped successfully, False otherwise
    """
    global is_running, server_thread
    if not is_running:
        logger.warning("Server is not running")
        return True
    try:
        # Shutdown the server
        import requests
        requests.get(f"http://localhost:{port}/shutdown")
        # Wait for the thread to terminate
        if server_thread:
            server_thread.join(timeout=5)
        is_running = False
        server_thread = None
        logger.info("Server stopped")
        return True
    except Exception as e:
        logger.error(f"Error stopping server: {e}")
        return False


@app.route("/shutdown", methods=["GET"])
def shutdown():
    """Shutdown the server."""
    func = request.environ.get("werkzeug.server.shutdown")
    if func is None:
        raise RuntimeError("Not running with the Werkzeug Server")
    func()
    return "Server shutting down..."


def get_server_status() -> Dict[str, Union[bool, int]]:
    """
    Get the status of the server.
    Returns:
        Dict[str, Union[bool, int]]: The server status
    """
    return {
        "is_running": is_running,
        "port": port
    }


if __name__ == "__main__":
    # Start the server when the module is run directly
    start_server()
