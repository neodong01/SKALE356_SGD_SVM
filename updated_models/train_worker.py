# filename: train_worker.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import argparse, os, pickle, sys
from multiprocessing import shared_memory
import math
import time 
import json 

# >>> START OF CHANGE 1: Import SGD utility functions conditionally <<<
try:
    from sgd_model_utils import train_and_save_sgd
    SGD_ENABLED = True
except ImportError:
    # This warning helps debug the pipeline setup
    print("[WARN] sgd_model_utils.py not found. SGD training will not work.", file=sys.stderr)
    SGD_ENABLED = False
# >>> END OF CHANGE 1 <<<

# >>> START OF NEW CHANGE 1: Import SVM utility functions conditionally <<<
try:
    from svm_model_utils import train_and_save_svm
    SVM_ENABLED = True
except ImportError:
    # This warning helps debug the pipeline setup
    print("[WARN] svm_model_utils.py not found. SVM training will not work.", file=sys.stderr)
    SVM_ENABLED = False
# >>> END OF NEW CHANGE 1 <<<

def create_adaptive_model(input_size, num_classes, config):
    """Builds a model dynamically based on the configuration."""
    layers = []
    current_size = input_size
    
    for layer_idx, neurons in enumerate(config['neuron_counts']):
        layers.append(nn.Linear(current_size, neurons))
        if config['use_batchnorm']:
            layers.append(nn.BatchNorm1d(neurons))
        layers.append(nn.LeakyReLU())
        if config['dropout_rate'] > 0:
            layers.append(nn.Dropout(config['dropout_rate']))
        current_size = neurons
        
    layers.append(nn.Linear(current_size, num_classes))
    model = nn.Sequential(*layers)
    print(f"--- [Model Config] Created a model with {len(config['neuron_counts'])} hidden layers. Neuron counts: {config['neuron_counts']}. Dropout: {config['dropout_rate']:.2f}. BatchNorm: {config['use_batchnorm']}.", flush=True, file=sys.stderr)
    return model

def check_class_representation(y_data, num_expected_classes=11):
    """Checks if there is at least one sample for each class."""
    # This function must handle the NumPy array passed from train_model_loop
    # The labels are expected to be integers (0, 1, 2, ...)
    unique, counts = np.unique(y_data.astype(np.int64), return_counts=True)
    
    if len(unique) < 2:
        return False, len(unique)

    # Check if all classes are present
    all_classes_present = (len(unique) == num_expected_classes)
    if not all_classes_present:
        print(f"[WARN] Incomplete class representation! Found {len(unique)} out of {num_expected_classes} classes.", flush=True, file=sys.stderr)
        
    # Simple check for minimum class size (optional but safe)
    if np.min(counts) < 1: 
        return False, len(unique)

    return all_classes_present, len(unique)


def get_training_config(num_samples, all_classes_present):
    """
    Returns a dictionary of hyperparameters adapted to the number of training samples
    and class representation.
    """
    config = {}

    # --- Batch Size Configuration ---
    if num_samples < 64:
        config['batch_size'] = num_samples
    else:
        # Log-scaled batch size, capped at 512
        log_scaled_batch = int(32 * math.log10(num_samples))
        config['batch_size'] = min(512, max(32, log_scaled_batch))

    # --- Dropout, Regularization, and Normalization Configuration (for MLP) ---
    config['use_scheduler'] = False
    
    # Tiny dataset policy
    if num_samples < 30 and all_classes_present:
        print("[Config] Applying 'no dropout' policy for tiny, complete dataset.", flush=True, file=sys.stderr)
        config['dropout_rate'] = 0.0
        config['l1_lambda'] = 0.0
        config['use_batchnorm'] = False
    # Small dataset policy
    elif num_samples < 200:
        print("[Config] Applying 'low dropout' policy for small dataset.", flush=True, file=sys.stderr)
        config['dropout_rate'] = 0.15
        config['l1_lambda'] = 1e-7
        config['use_batchnorm'] = True
    # Large dataset policy
    else:
        print("[Config] Applying 'high dropout' policy for large dataset.", flush=True, file=sys.stderr)
        config['dropout_rate'] = 0.5
        config['l1_lambda'] = 1e-5
        config['use_batchnorm'] = True
        config['use_scheduler'] = True
        config['scheduler_step_size'] = 7
        config['scheduler_gamma'] = 0.1

    # --- Neural Network Architecture Configuration (for MLP) ---
    if num_samples < 100:
        config['neuron_counts'] = [256, 128]
    elif num_samples < 2000:
        config['neuron_counts'] = [1024, 512, 256]
    else:
        config['neuron_counts'] = [2048, 1024, 1024, 512]

    # --- Jitter for Learning Rate and Dropout (for Ensemble Diversity) ---
    if config['dropout_rate'] > 0:
        jitter = (np.random.rand() - 0.5) * 0.1
        config['dropout_rate'] = max(0, min(0.7, config['dropout_rate'] + jitter))
    
    # Base learning rate with jitter
    config['learning_rate'] = 0.0001 * (1 + (np.random.rand() - 0.5) * 0.2)
    
    # --- SGD/SVM Configuration Overrides (Used by Sklearn models) ---
    # These are common hyperparams passed to SGDClassifier or LinearSVC
    config['penalty'] = 'elasticnet' # Good general choice for SGD
    config['alpha'] = 1e-4 * (1 + (np.random.rand() - 0.5) * 0.5) # Regularization strength
    config['l1_ratio'] = 0.15 + (np.random.rand() - 0.5) * 0.1 # Elastic Net mix (0.0=L2, 1.0=L1)
    
    # SVM (LinearSVC) Configuration
    config['C'] = 1.0 + (np.random.rand() - 0.5) * 0.5 # Regularization parameter for SVM
    config['max_iter'] = 1000 # Max iterations for iterative solvers
    
    return config


def train_model_loop(args):
    """
    Trains a 'batch' of models from a start_id to an end_id within a single
    persistent worker process, dispatching based on model type.
    """
    shm = None
    try:
        shm = shared_memory.SharedMemory(name=args.shm_name)
        full_data = np.ndarray((args.num_rows, args.num_cols), dtype=np.float32, buffer=shm.buf)
        X_train_np = full_data[:, :-1]
        y_train_np = full_data[:, -1]

        with open(args.scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        X_train_np = scaler.transform(X_train_np)
        
        # --- Training Loop Dispatch ---
        for model_id in range(args.start_id, args.end_id):
            print(f"--- Worker (PID:{os.getpid()}) starting Model ID: {model_id}, Type: {args.model_type.upper()} ---", flush=True, file=sys.stderr)
            
            # >>> START OF NEW CHANGE 3: Move common config generation outside the if/elif blocks <<<
            num_classes = 11
            # Check for class representation consistency
            all_classes_present, num_unique_classes = check_class_representation(y_train_np, num_classes)
            if num_unique_classes < 2:
                 print(f"[FATAL] Insufficient class representation (only {num_unique_classes} unique classes). Skipping model {model_id}.", file=sys.stderr)
                 continue # Skip this model/fasta file
                 
            # Generate adaptive configuration
            config = get_training_config(len(y_train_np), all_classes_present)
            # Initialize model_path and time variables
            model_path = None
            model_train_time = 0.0
            # >>> END OF NEW CHANGE 3 <<<

            # --- MLP Training (PyTorch) ---
            if args.model_type == 'mlp':
                # Data must be PyTorch Tensors
                X_train = torch.tensor(X_train_np, dtype=torch.float32)
                y_train = torch.tensor(y_train_np, dtype=torch.long)
                dataset = TensorDataset(X_train, y_train)
                input_size = X_train.shape[1]
                
                # Setup
                model = create_adaptive_model(input_size, num_classes, config)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
                train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0) 

                scheduler = None
                if config['use_scheduler']:
                    print(f"[Config] Using StepLR scheduler: step_size={config['scheduler_step_size']}, gamma={config['scheduler_gamma']}", flush=True, file=sys.stderr)
                    scheduler = StepLR(optimizer, step_size=config['scheduler_step_size'], gamma=config['scheduler_gamma'])

                # Start timing the model training
                model_start_time = time.time()
                num_epochs = 20 # Defined for MLP
                
                # Training Loop
                model.train()
                for epoch in range(num_epochs):
                    epoch_loss = 0.0
                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        # L1 Regularization
                        if config['l1_lambda'] > 0:
                            l1_penalty = sum(p.abs().sum() for p in model.parameters())
                            loss = criterion(outputs, batch_y) + config['l1_lambda'] * l1_penalty
                        else:
                            loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                        
                    if scheduler:
                        scheduler.step()

                    avg_loss = epoch_loss / len(train_loader)
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Model {model_id}, Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.1e}", flush=True, file=sys.stderr)

                # End timing and save the model
                model_train_time = time.time() - model_start_time

                os.makedirs(args.models_dir, exist_ok=True)
                model_path = os.path.join(args.models_dir, f"model_{model_id}.pt")
                torch.save(model, model_path)
                print(f"--- Worker (PID:{os.getpid()}) finished and saved MLP Model ID: {model_id} ---", flush=True, file=sys.stderr)

            # --- SGD Training (Sklearn) ---
            elif args.model_type == 'sgd':
                if not SGD_ENABLED:
                    print(f"[FATAL] Cannot train SGD model {model_id}: sgd_model_utils is missing.", file=sys.stderr)
                    sys.exit(1)
                
                try:
                    # >>> START OF NEW CHANGE 4: Pass the config dictionary to SGD utility <<<
                    # NOTE: Assuming train_and_save_sgd is updated to return (model_path, time)
                    model_path, model_train_time = train_and_save_sgd( 
                        X_train_np, 
                        y_train_np, 
                        args.models_dir, 
                        model_id,
                        config # <--- CONFIG ADDED
                    )
                    # >>> END OF NEW CHANGE 4 <<<
                    
                    if model_path is None: raise Exception("SGD training utility returned a failure status.")

                    print(f"--- Worker (PID:{os.getpid()}) finished and saved SGD Model ID: {model_id} ---", flush=True, file=sys.stderr)
                except Exception as e:
                    print(f"[FATAL] SGD training failed for model {model_id}: {e}", file=sys.stderr)
                    sys.exit(1)

            # >>> START OF NEW CHANGE 5: SVM Training Logic (Corrected) <<<
            elif args.model_type == 'svm':
                if not SVM_ENABLED:
                    print(f"[FATAL] Cannot train SVM model {model_id}: svm_model_utils is missing.", file=sys.stderr)
                    sys.exit(1)
                
                try:
                    # CRITICAL FIX: The utility function returns (model_path, time)
                    model_path, model_train_time = train_and_save_svm(
                        X_train_np, 
                        y_train_np, 
                        args.models_dir, 
                        model_id,
                        config # <--- CONFIG ADDED
                    )
                    
                    if model_path is None: raise Exception("SVM training utility returned a failure status.")

                    print(f"--- Worker (PID:{os.getpid()}) finished and saved SVM Model ID: {model_id} ---", flush=True, file=sys.stderr)
                except Exception as e:
                    print(f"[FATAL] SVM training failed for model {model_id}: {e}", file=sys.stderr)
                    sys.exit(1)
            # >>> END OF NEW CHANGE 5 <<<

            # --- Reporting (Common to all models) ---
            result = {
                "model_id": model_id,
                "train_time_s": model_train_time
            }
            # Print to stdout so the parent process (preprocess_data.py) can capture this JSON output.
            print(json.dumps(result), flush=True, file=sys.stdout)

    finally:
        if shm:
            shm.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a batch of models in a loop.")
    parser.add_argument('--shm-name', required=True)
    parser.add_argument('--scaler-path', required=True)
    parser.add_argument('--models-dir', required=True)
    parser.add_argument('--start-id', type=int, required=True)
    parser.add_argument('--end-id', type=int, required=True)
    parser.add_argument('--num-rows', type=int, required=True)
    parser.add_argument('--num-cols', type=int, required=True)
    # >>> START OF CHANGE 3: Accept model-type argument <<<
    parser.add_argument('--model-type', required=True, choices=['mlp', 'sgd', 'svm'], help="The type of model to train.")
    # >>> END OF CHANGE 3 <<<
    
    args = parser.parse_args()
    train_model_loop(args)