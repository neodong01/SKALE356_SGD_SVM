# filename: svm_model_utils.py

import time
import os
import pickle
import sys
import numpy as np
# Switching to LinearSVC for high-throughput high-dimensional feature vectors (better performance)
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning
import warnings

# Suppress ConvergenceWarning from Scikit-learn LinearSVC
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def train_and_save_svm(X_train_np, y_train_np, models_dir, model_id, config):
    """
    Trains a Linear Support Vector Machine (LinearSVC) model using the provided data
    and saves the trained model using pickle.

    Args:
        X_train_np (np.ndarray): The feature data for training.
        y_train_np (np.ndarray): The label data for training (must be 1D integer array).
        models_dir (str): The directory where the trained model should be saved.
        model_id (int): A unique identifier for the model.
        config (dict): Configuration dictionary containing model hyperparameters (e.g., 'C').

    Returns:
        tuple: (model_path: str, train_time_s: float)
    """
    if not os.path.isdir(models_dir):
        os.makedirs(models_dir, exist_ok=True)
    
    # 1. Label Conversion
    # Labels loaded from shared memory are np.float32 (0.0, 1.0, 2.0, etc.)
    # We convert them explicitly to integers for SVM classification.
    y_train_int = y_train_np.astype(np.int64)
    
    # 2. Model Initialization (using configurable parameters)
    C_param = config.get('C', 1.0)
    max_iter = config.get('max_iter', 1000)
    
    print(f"[SVM] Initializing LinearSVC (C={C_param}, max_iter={max_iter}).", file=sys.stderr)
    try:
        model = LinearSVC(
            C=C_param, 
            loss='squared_hinge', # Standard for LinearSVC
            random_state=42 + model_id, 
            max_iter=max_iter,
            dual='auto', # 'auto' selects the best algorithm based on data size
            tol=1e-3 # Looser tolerance for speed
        )
    except Exception as e:
        print(f"[SVM-FATAL] Failed to initialize LinearSVC: {e}", file=sys.stderr)
        return None, 0.0

    # 3. Training the SVM model
    start_time = time.time()
    try:
        # Note: ConvergenceWarning is suppressed globally as it's common in high-throughput.
        model.fit(X_train_np, y_train_int)
        train_time_s = time.time() - start_time
    except Exception as e:
        print(f"[SVM-FATAL] LinearSVC training failed for model {model_id}: {e}", file=sys.stderr)
        return None, 0.0
    
    # 4. Save the trained model
    model_path = os.path.join(models_dir, f"model_{model_id}.pkl")
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    except Exception as e:
        print(f"[SVM-FATAL] Failed to save LinearSVC model {model_id}: {e}", file=sys.stderr)
        return None, 0.0
        
    # CRITICAL CHANGE: Return model_path and time
    return model_path, train_time_s

# ----------------------------------------------------------------------
# Functions required by evaluate_ensemble.py
# ----------------------------------------------------------------------

def load_svm_model(model_path):
    """
    Loads a trained LinearSVC model from a pickle file.
    (Used by evaluate_ensemble.py)
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        print(f"[SVM-LOAD-ERROR] Model file not found: {model_path}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[SVM-LOAD-ERROR] Failed to load model {model_path}: {e}", file=sys.stderr)
        return None

def predict_svm(model, X_test):
    """
    Performs prediction using a loaded LinearSVC model.
    (Used by evaluate_ensemble.py)
    """
    if model is None:
        return np.array([])
    # Scikit-learn models expose a .predict() method on NumPy arrays
    return model.predict(X_test)