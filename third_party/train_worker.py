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

def check_class_representation(labels_tensor, num_expected_classes=11):
    """Checks if there is at least one sample for each class."""
    unique_labels = torch.unique(labels_tensor)
    has_all_classes = (len(unique_labels) == num_expected_classes)
    if not has_all_classes:
        print(f"  [WARN] Incomplete class representation! Found {len(unique_labels)} out of {num_expected_classes} classes.", flush=True, file=sys.stderr)
    return has_all_classes

def get_training_config(num_samples, all_classes_present):
    """
    Returns a dictionary of hyperparameters adapted to the number of training samples
    and class representation.
    """
    config = {}

    if num_samples < 64:
        config['batch_size'] = num_samples
    else:
        log_scaled_batch = int(32 * math.log10(num_samples))
        config['batch_size'] = min(512, max(32, log_scaled_batch))

    config['use_scheduler'] = False
    if num_samples < 30 and all_classes_present:
        print("  [Config] Applying 'no dropout' policy for tiny, complete dataset.", flush=True, file=sys.stderr)
        config['dropout_rate'] = 0.0
        config['l1_lambda'] = 0.0
        config['use_batchnorm'] = False
    elif num_samples < 200:
        print("  [Config] Applying 'low dropout' policy for small dataset.", flush=True, file=sys.stderr)
        config['dropout_rate'] = 0.15
        config['l1_lambda'] = 1e-7
        config['use_batchnorm'] = True
    else:
        print("  [Config] Applying 'high dropout' policy for large dataset.", flush=True, file=sys.stderr)
        config['dropout_rate'] = 0.5
        config['l1_lambda'] = 1e-5
        config['use_batchnorm'] = True
        config['use_scheduler'] = True
        config['scheduler_step_size'] = 7
        config['scheduler_gamma'] = 0.1

    if num_samples < 100:
        config['neuron_counts'] = [256, 128]
    elif num_samples < 2000:
        config['neuron_counts'] = [1024, 512, 256]
    else:
        config['neuron_counts'] = [2048, 1024, 1024, 512]

    if config['dropout_rate'] > 0:
        jitter = (np.random.rand() - 0.5) * 0.1
        config['dropout_rate'] = max(0, min(0.7, config['dropout_rate'] + jitter))
    config['learning_rate'] = 0.0001 * (1 + (np.random.rand() - 0.5) * 0.2)

    return config

def train_model_loop(args):
    """
    Trains a 'batch' of models from a start_id to an end_id within a single
    persistent worker process.
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
        
        X_train = torch.tensor(X_train_np, dtype=torch.float32)
        y_train = torch.tensor(y_train_np, dtype=torch.long)
        
        dataset = TensorDataset(X_train, y_train)
        input_size = X_train.shape[1]
        num_classes = 11
        num_epochs = 20

        for model_id in range(args.start_id, args.end_id):
            print(f"--- Worker (PID:{os.getpid()}) starting Model ID: {model_id} ---", flush=True, file=sys.stderr)
            
            all_classes_present = check_class_representation(y_train, num_classes)
            config = get_training_config(len(y_train), all_classes_present)
            
            model = create_adaptive_model(input_size, num_classes, config)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
            train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0) 

            scheduler = None
            if config['use_scheduler']:
                print(f"  [Config] Using StepLR scheduler: step_size={config['scheduler_step_size']}, gamma={config['scheduler_gamma']}", flush=True, file=sys.stderr)
                scheduler = StepLR(optimizer, step_size=config['scheduler_step_size'], gamma=config['scheduler_gamma'])

            model.train()
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
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

            os.makedirs(args.models_dir, exist_ok=True)
            model_path = os.path.join(args.models_dir, f"model_{model_id}.pt")
            torch.save(model, model_path)
            print(f"--- Worker (PID:{os.getpid()}) finished and saved Model ID: {model_id} ---", flush=True, file=sys.stderr)

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
    
    args = parser.parse_args()
    train_model_loop(args)