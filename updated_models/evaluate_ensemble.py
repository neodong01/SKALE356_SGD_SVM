import torch
import torch.nn as nn
import numpy as np
import time
import os
import csv
import re
import argparse
import pickle
import itertools
import json
import subprocess
import sys
from multiprocessing import shared_memory
from filelock import FileLock
from Bio import SeqIO
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

try:
    from sgd_model_utils import load_sgd_model, predict_sgd
    SGD_UTILS_AVAILABLE = True
except ImportError:
    SGD_UTILS_AVAILABLE = False
    print("[EVAL-WARN] sgd_model_utils.py not found. SGD evaluation will fail if selected.", file=sys.stderr)

try:
    from svm_model_utils import load_svm_model, predict_svm
    SVM_UTILS_AVAILABLE = True
except ImportError:
    SVM_UTILS_AVAILABLE = False
    print("[EVAL-WARN] svm_model_utils.py not found. SVM evaluation will fail if selected.", file=sys.stderr)

try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False

VARIANT_TO_FLOAT = {'Alpha':0.0,'Beta':1.0,'Gamma':2.0,'Delta':3.0,'Epsilon':4.0,'Zeta':5.0,'Eta':6.0,'Iota':7.0,'Lambda':8.0,'Mu':9.0,'Omicron':10.0}
ALL_3MERS = sorted(["".join(p) for p in itertools.product('ATGC', repeat=3)])
ALL_5MERS = sorted(["".join(p) for p in itertools.product('ATGC', repeat=5)])
ALL_6MERS = sorted(["".join(p) for p in itertools.product('ATGC', repeat=6)])
NUM_KMER_FEATURES = len(ALL_3MERS) + len(ALL_5MERS) + len(ALL_6MERS)
MODELS_TO_TRAIN = None
TOTAL_SEQUENCES = 11000

CONFIG = {
    "MODELS_SAVE_DIR": "saved_models",
    "FINAL_LOG_FILE": "PIPELINE_RESULTS_HPC.csv",
    "KEY_FILE_NAME": "key.fasta"
}

def extract_features_on_demand(fasta_path, variants_dict_arg, variant_map_arg):
    executable_path = "./feature_extractor"
    if not os.path.exists(executable_path):
        raise FileNotFoundError("Error: C++ executable './feature_extractor' not found.")
    temp_shm_name = f"temp_eval_{os.getpid()}_{time.time_ns()}"
    try:
        num_records = sum(1 for _ in SeqIO.parse(fasta_path, "fasta"))
    except FileNotFoundError:
        print(f"[EVAL-ERROR] Test file not found: {fasta_path}", file=sys.stderr)
        return None, 0
    if num_records == 0: return None, 0
    num_cols = 1 + NUM_KMER_FEATURES + 1
    buffer_size = num_records * num_cols * np.dtype(np.float32).itemsize
    shm = None
    try:
        shm = shared_memory.SharedMemory(name=temp_shm_name, create=True, size=buffer_size)
        subprocess.run([executable_path, fasta_path, temp_shm_name, str(buffer_size)], check=True, capture_output=True)
        data_array_view = np.ndarray((num_records, num_cols), dtype=np.float32, buffer=shm.buf)
        sequence_ids = [rec.id for rec in SeqIO.parse(fasta_path, "fasta")]
        for i, seq_id in enumerate(sequence_ids):
            variant_name = variants_dict_arg.get(seq_id, 'UNKNOWN')
            data_array_view[i, -1] = variant_map_arg.get(variant_name, -1.0)
        result_array = np.copy(data_array_view)
        return result_array, num_records
    finally:
        if shm:
            shm.close()
            shm.unlink()

def get_performance_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    return {'accuracy': f"{accuracy:.4f}", 'precision_macro': f"{precision:.4f}", 'recall_macro': f"{recall:.4f}", 'f1_macro': f"{f1:.4f}"}

def main():
    parser = argparse.ArgumentParser(description="Perform granular evaluation and log all metadata.")
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--train-extract-time', type=float, required=True)
    parser.add_argument('--total-train-time', type=float, required=True)
    parser.add_argument('--num-train-seqs', type=int, required=True)
    parser.add_argument('--num-models', type=int, required=True, help="The number of models in the ensemble to evaluate.")
    parser.add_argument('--individual-train-times', type=str, required=True, help="JSON string mapping model_id to training time.")
    parser.add_argument('--model-type', type=str, required=True, choices=['mlp', 'sgd', 'svm'], help="The type of model to evaluate ('mlp', 'sgd', or 'svm').")
    args = parser.parse_args()

    # Load the JSON string into a Python dictionary
    individual_train_times_map = json.loads(args.individual_train_times)
    
    # Convert map values to float and store as an ordered list based on model_id
    model_train_times = [
        float(individual_train_times_map.get(str(i), 0.0))
        for i in range(args.num_models)
    ]

    # --- Dynamically determine file paths ---
    training_fastas_dir = os.path.dirname(args.train_file)
    fasta_root_dir = os.path.dirname(training_fastas_dir)
    
    train_filename = os.path.basename(args.train_file)
    base_name = os.path.splitext(train_filename)[0]
    
    test_dir = os.path.join(fasta_root_dir, "testingFastas")
    
    version_num = -1
    test_filename = ""
    match_by_count = re.match(r'^train_(\d+)_(\d+)\.fasta$', train_filename)
    match_by_percent = re.match(r'^train(\d+)_(\d+)_(\d+)\.fasta$', train_filename)
    
    if match_by_count:
        count, version = int(match_by_count.group(1)), int(match_by_count.group(2))
        version_num = version
        test_seq_count = TOTAL_SEQUENCES - count
        test_filename = f'test_{test_seq_count}_{version}.fasta'
    elif match_by_percent:
        percent, total, version = int(match_by_percent.group(1)), int(match_by_percent.group(2)), int(match_by_percent.group(3))
        version_num = version
        name_by_percent = f'test{100 - percent}_{total}_{version}.fasta'
        test_seq_count = total - (total * percent // 100)
        name_by_count = f'test_{test_seq_count}_{version}.fasta'
        
        if os.path.exists(os.path.join(test_dir, name_by_percent)):
            test_filename = name_by_percent
        elif os.path.exists(os.path.join(test_dir, name_by_count)):
            test_filename = name_by_count
        else:
            print(f"[EVAL-FATAL] Could not find a matching test file for '{train_filename}'", file=sys.stderr)
            sys.exit(1)
    
    if not test_filename:
        print(f"[EVAL-FATAL] Could not determine test filename for {train_filename}", file=sys.stderr)
        sys.exit(1)
        
    test_filename_full_path = os.path.join(test_dir, test_filename)
    job_models_save_dir = os.path.join(CONFIG["MODELS_SAVE_DIR"], base_name)
    
    final_ensemble_accuracy = "N/A"

    try:
        extract_start_time = time.time()
        print(f"  [EVAL] Extracting features on-demand from {test_filename}...", file=sys.stderr)
        
        key_file_path = CONFIG['KEY_FILE_NAME']
        if not os.path.exists(key_file_path):
             print(f"[EVAL-FATAL] Key file not found at expected path: {key_file_path}", file=sys.stderr)
             sys.exit(1)
        variants_main = {rec.id: str(rec.seq) for rec in SeqIO.parse(key_file_path, "fasta")}
        
        test_data, num_test_seqs = extract_features_on_demand(test_filename_full_path, variants_main, VARIANT_TO_FLOAT)
        if test_data is None: raise ValueError(f"Test data extraction failed. Check if '{test_filename_full_path}' exists and is not empty.")
        
        scaler_path = os.path.join(CONFIG["MODELS_SAVE_DIR"], f"{base_name}_scaler.pkl")
        with open(scaler_path, 'rb') as f: scaler_main = pickle.load(f)
        
        # --- Prepare data for both model types ---
        X_test_scaled_np = scaler_main.transform(test_data[:, :-1]) # NumPy Array (for Sklearn models)
        X_test_tensor = torch.FloatTensor(X_test_scaled_np)          # PyTorch Tensor (for MLP)
        y_true_numpy = test_data[:, -1].astype(np.int64)
        # ----------------------------------------
        
        test_extract_time = time.time() - extract_start_time
        print(f"[EVAL] Test feature extraction took {test_extract_time:.4f}s.", file=sys.stderr)

        # >>> Conditional Model Loading and Prediction MLP, SGD, SVM Model <<<
        model_ext = '.pt' if args.model_type == 'mlp' else '.pkl'

        all_preds_tensors, individual_timings = [], []
        print(f"[EVAL] Loading all {args.model_type.upper()} models and timing individual predictions...", file=sys.stderr)
        
        for model_idx in range(args.num_models):
            model_path = os.path.join(job_models_save_dir, f'model_{model_idx}{model_ext}')
            
            if not os.path.exists(model_path):
                print(f"[EVAL-WARN] Model file not found: {model_path}. Skipping model {model_idx}.", file=sys.stderr)
                continue
            
            single_pred_start_time = time.time()
            y_pred_tensor = None
            
            try:
                if args.model_type == 'mlp':
                    # --- MLP Loading & Prediction (PyTorch) ---
                    model = torch.load(model_path, map_location='cpu', weights_only=False)
                    model.eval()
                    if IPEX_AVAILABLE: model = ipex.optimize(model, inplace=True)
                    
                    with torch.no_grad(): 
                        _, y_pred_tensor = torch.max(model(X_test_tensor).data, 1)
                        
                elif args.model_type == 'sgd':
                    # --- SGD Loading & Prediction (NumPy/Sklearn) ---
                    if not SGD_UTILS_AVAILABLE:
                            raise ImportError("SGD utilities not found. Cannot evaluate SGD model.")

                    model = load_sgd_model(model_path) 
                    y_pred_numpy = predict_sgd(model, X_test_scaled_np)
                    y_pred_tensor = torch.LongTensor(y_pred_numpy)
                    
                elif args.model_type == 'svm':
                    # --- SVM Loading & Prediction (NumPy/Sklearn) ---
                    if not SVM_UTILS_AVAILABLE:
                             raise ImportError("SVM utilities not found. Cannot evaluate SVM model.")

                    model = load_svm_model(model_path) 
                    y_pred_numpy = predict_svm(model, X_test_scaled_np)
                    y_pred_tensor = torch.LongTensor(y_pred_numpy)
                    

            except Exception as e:
                print(f"[EVAL-ERROR] Model {model_idx} failed (Type: {args.model_type}). Error: {e}. Skipping.", file=sys.stderr)
                continue

            # --- Append results for successful prediction ---
            single_pred_time = time.time() - single_pred_start_time
            all_preds_tensors.append(y_pred_tensor)
            individual_timings.append(single_pred_time)
            
        if not all_preds_tensors: raise ValueError("No models were successfully loaded/predicted.")

        lock_path = CONFIG["FINAL_LOG_FILE"] + ".lock"
        with FileLock(lock_path):
            file_exists = os.path.isfile(CONFIG["FINAL_LOG_FILE"])
            fieldnames = ['train_file', 'test_file', 'version', 'num_train_seqs', 'num_test_seqs',
                          'train_extract_time_s', 'test_extract_time_s', 'total_train_time_s',
                          'run_type', 'model_id_or_k', 'prediction_time_s',
                          'accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
                          'model_type']
                          
            with open(CONFIG["FINAL_LOG_FILE"], 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists: writer.writeheader()
                
                base_log_info = {
                    'train_file': train_filename, 'test_file': test_filename, 'version': version_num,
                    'num_train_seqs': args.num_train_seqs, 'num_test_seqs': num_test_seqs,
                    'train_extract_time_s': f"{args.train_extract_time:.4f}",
                    'test_extract_time_s': f"{test_extract_time:.4f}",
                    'total_train_time_s': f"{args.total_train_time:.4f}",
                    'model_type': args.model_type
                }

                print("[LOG] Writing individual model performance...", file=sys.stderr)
                
                successful_model_indices = [
                    i for i in range(args.num_models) 
                    if os.path.exists(os.path.join(job_models_save_dir, f'model_{i}{model_ext}')) and 
                       (args.model_type == 'mlp' or 
                        (args.model_type == 'sgd' and SGD_UTILS_AVAILABLE) or
                        (args.model_type == 'svm' and SVM_UTILS_AVAILABLE))
                ]
                
                # Reset cumulative train time for the next loop
                cumulative_train_time_s = 0 
                cumulative_pred_time = 0 
                
                # Iterate over successful predictions
                for i, y_pred_tensor in enumerate(all_preds_tensors):
                    original_model_idx = successful_model_indices[i]
                    metrics = get_performance_metrics(y_true_numpy, y_pred_tensor.numpy())
                    
                    individual_train_time = model_train_times[original_model_idx]
                    
                    log_entry = {
                        **base_log_info, 
                        'run_type': 'individual', 
                        'model_id_or_k': original_model_idx, # Use original index
                        'total_train_time_s': f"{individual_train_time:.4f}",
                        'prediction_time_s': f"{test_extract_time + individual_timings[i]:.4f}", 
                        **metrics
                    }
                    writer.writerow(log_entry)

                print("[LOG] Writing cumulative ensemble performance...", file=sys.stderr)
                stacked_preds = torch.stack(all_preds_tensors)
                
                # Recalculate cumulative sums using only successful models
                cumulative_train_time_s = 0
                cumulative_pred_time = 0 
                
                for k in range(1, len(all_preds_tensors) + 1):
                    # Sum times for the first k successful models
                    original_model_idx = successful_model_indices[k-1]
                    cumulative_train_time_s += model_train_times[original_model_idx]
                    cumulative_pred_time += individual_timings[k-1]
                    total_ensemble_time = test_extract_time + cumulative_pred_time
                    
                    ensemble_pred_k, _ = torch.mode(stacked_preds[:k, :], dim=0)
                    metrics = get_performance_metrics(y_true_numpy, ensemble_pred_k.numpy())
                    
                    log_entry = {
                        **base_log_info, 
                        'run_type': 'ensemble', 
                        'model_id_or_k': k, # 'k' here refers to the number of models used
                        'total_train_time_s': f"{cumulative_train_time_s:.4f}",
                        'prediction_time_s': f"{total_ensemble_time:.4f}", 
                        **metrics
                    }
                    writer.writerow(log_entry)
                    if k == len(all_preds_tensors):
                        final_ensemble_accuracy = metrics['accuracy']

    except Exception as e:
        import traceback
        print(f"[EVAL-FATAL] An error occurred during evaluation: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    
    print(f"-> Final Ensemble Accuracy: {final_ensemble_accuracy}")

if __name__ == '__main__':
    main()