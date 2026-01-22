# filename: sgd_model_utils.py

import pickle
import os
import numpy as np
import time
from sklearn.linear_model import SGDClassifier
import sys # Added for error logging

# --- Core Function: Training and Saving ---

def train_and_save_sgd(X: np.ndarray, y: np.ndarray, models_dir: str, model_id: int, model_type_str: str) -> tuple:
    """
    Trains a Scikit-learn SGDClassifier model using provided NumPy arrays, 
    saves it using pickle, and returns the training time wrapped in a tuple.
    
    Args:
        X: Scaled feature data (NumPy array).
        y: Label data (NumPy array).
        models_dir: The directory where the model should be saved.
        model_id: The ID of the model being trained.
        model_type_str: (IGNORED) Argument added for compatibility with the parallel worker script.

    Returns:
        A tuple of two elements: (0.0, train_time).
        0.0 is returned as a placeholder to satisfy the worker script's 
        expectation of two numeric return values.
    """
    
    # 1. Start Timing
    start_time = time.time()
    
    # Construct the save path using the models_dir and model_id
    model_save_path = os.path.join(models_dir, f"model_{model_id}.pkl")
    
    # Ensure labels are in the correct integer format for Scikit-learn
    y_int = y.astype(np.int64)

    # 2. Define and Train SGD Model
    model = SGDClassifier(
        loss='log_loss', 
        penalty='l2', 
        max_iter=1000, 
        tol=1e-3, 
        # Use PID and model_id offset for ensemble diversity
        random_state=42 + os.getpid() + model_id 
    )
    
    # Check for class representation
    if len(np.unique(y_int)) < 2:
        print(f"[SGD-TRAIN-WARN] Model {model_id}: Only one class found in training data. Skipping training.", file=sys.stderr)
        train_time = 0.0
        # Save None model to signify skipped/failed training
        with open(model_save_path, 'wb') as f:
            pickle.dump(None, f) 
        
        # CHANGE 1/2: Return (train_time, 0.0) instead of (train_time, None).
        # This prevents the float() conversion error in evaluate_ensemble.py.
        return (0.0, train_time)
            
    model.fit(X, y_int)
    
    # 3. Calculate Training Time
    train_time = time.time() - start_time
    
    # 4. Save the Trained Model using Pickle
    try:
        with open(model_save_path, 'wb') as f:
            pickle.dump(model, f)
    except Exception as e:
        print(f"[SGD-TRAIN-FATAL] Failed to save SGD model to {model_save_path}: {e}", file=sys.stderr)
        raise
        
    # CHANGE 2/2: Return (train_time, 0.0) instead of (train_time, None).
    # This prevents the float() conversion error in evaluate_ensemble.py.
    return (0.0, train_time)

# --- Utility Function: Loading ---

def load_sgd_model(model_path: str):
    """
    Loads a saved SGDClassifier model using pickle.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[SGD-LOAD-FATAL] Model file not found at: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    # Check if the model saved was None (due to failed training)
    if model is None:
        raise ValueError(f"[SGD-LOAD-FATAL] Model file {model_path} contains a None object, indicating training failure.")
        
    return model

# --- Utility Function: Prediction (for evaluate_ensemble.py) ---

def predict_sgd(model, X_test_scaled: np.ndarray) -> np.ndarray:
    """
    Performs prediction using the loaded Scikit-learn SGD model.
    """
    # Scikit-learn's predict method returns a numpy array of class labels (0, 1, 2...)
    return model.predict(X_test_scaled)