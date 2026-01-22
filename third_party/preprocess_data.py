# filename: preprocess_data.py

import numpy as np
import argparse, os, sys, time, pickle, itertools, subprocess
from Bio import SeqIO
from multiprocessing import shared_memory
from sklearn.preprocessing import StandardScaler
import signal
import json
import warnings

warnings.filterwarnings("ignore", message="resource_tracker: There appear to be", category=UserWarning)
warnings.filterwarnings("ignore", message="resource_tracker: ", category=UserWarning)

# --- CONSTANTS ---
VARIANT_TO_FLOAT = {'Alpha':0.0,'Beta':1.0,'Gamma':2.0,'Delta':3.0,'Epsilon':4.0,'Zeta':5.0,'Eta':6.0,'Iota':7.0,'Lambda':8.0,'Mu':9.0,'Omicron':10.0}
ALL_3MERS = sorted(["".join(p) for p in itertools.product('ATGC', repeat=3)])
ALL_5MERS = sorted(["".join(p) for p in itertools.product('ATGC', repeat=5)])
ALL_6MERS = sorted(["".join(p) for p in itertools.product('ATGC', repeat=6)])
NUM_KMER_FEATURES = len(ALL_3MERS) + len(ALL_5MERS) + len(ALL_6MERS)
CONFIG = { "MODELS_SAVE_DIR": "saved_models", "KEY_FILE_NAME": "key.fasta" }
TRAIN_WORKER_SCRIPT="train_worker.py"

shm_name_for_cleanup = None

def cleanup_shared_memory(shm_name):
    if not shm_name: return
    try:
        shm = shared_memory.SharedMemory(name=shm_name)
        shm.unlink()
        shm.close()
        print(f"  [CLEANUP] Successfully unlinked shared memory '{shm_name}'.", file=sys.stderr)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"  [CLEANUP-WARN] Error cleaning up shared memory {shm_name}: {e}", file=sys.stderr)

def extract_features_shm(fasta_path, variants_dict_arg, variant_map_arg, shm_base_name):
    executable_path = "./feature_extractor"
    if not os.path.exists(executable_path):
        raise FileNotFoundError("Error: C++ executable './feature_extractor' not found.")
    try:
        num_records = sum(1 for _ in SeqIO.parse(fasta_path, "fasta"))
    except Exception as e:
        print(f"  [ERROR] Could not parse FASTA file {fasta_path}: {e}", file=sys.stderr)
        return 0, None, 0, 0
    if num_records == 0:
        print(f"  [WARN] Input file {fasta_path} is empty.", file=sys.stderr)
        return 0, None, 0, 0
    num_cols = 1 + NUM_KMER_FEATURES + 1
    buffer_size = num_records * num_cols * np.dtype(np.float32).itemsize
    shm = None
    try:
        shm = shared_memory.SharedMemory(name=shm_base_name, create=True, size=buffer_size)
        print(f"  [SHM] Created shared memory '{shm.name}' ({buffer_size/1024**2:.2f} MB)", file=sys.stderr)
        subprocess.run([executable_path, fasta_path, shm.name, str(buffer_size)],
                       check=True, capture_output=True, text=True, timeout=300)
        print(f"  [FEATURE] C++ feature extraction completed.", file=sys.stderr)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        print(f"  [ERROR] Feature extraction failed: {e.stderr}", file=sys.stderr)
        if shm: shm.close(); shm.unlink()
        return 0, None, 0, 0
    except Exception as e:
        print(f"  [ERROR] Failed to setup or run feature extraction: {e}", file=sys.stderr)
        if shm: shm.close(); shm.unlink()
        return 0, None, 0, 0
    data_array_view = np.ndarray((num_records, num_cols), dtype=np.float32, buffer=shm.buf)
    sequence_ids = [rec.id for rec in SeqIO.parse(fasta_path, "fasta")]
    for i, seq_id in enumerate(sequence_ids):
        variant_name = variants_dict_arg.get(seq_id, 'UNKNOWN')
        data_array_view[i, -1] = variant_map_arg.get(variant_name, -1.0)
    return num_records, shm, num_cols, buffer_size

def run_persistent_workers(base_command, num_total_models, num_concurrent_workers):
    """
    Launches N persistent workers and divides the total models among them.
    This version streams output in real-time.
    """
    worker_batches = np.array_split(range(num_total_models), num_concurrent_workers)
    procs = []
    
    for i, batch in enumerate(worker_batches):
        if len(batch) == 0: continue
        start_id = batch[0]
        end_id = batch[-1] + 1
        
        print(f"  [SYSTEM] Launching Worker {i} for models {start_id}-{end_id-1}", file=sys.stderr)
        
        cmd = base_command + ['--start-id', str(start_id), '--end-id', str(end_id)]
        
        worker_env = os.environ.copy()
        worker_env['OMP_NUM_THREADS'] = '1'
        worker_env['MKL_NUM_THREADS'] = '1'
        
        # Merge stdout and stderr so we only have one stream to read
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                             text=True, env=worker_env, close_fds=True, bufsize=1)
        procs.append((p, i))

    active_workers = len(procs)
    failed_jobs = 0
    
    while active_workers > 0:
        for p, worker_idx in list(procs):
            # Check if the process has finished
            if p.poll() is not None:
                # Process has terminated, read any remaining output
                for remaining_line in p.stdout:
                    if remaining_line.strip():
                        print(f"[Worker {worker_idx}] {remaining_line.strip()}", file=sys.stderr)

                if p.returncode != 0:
                    failed_jobs += 1
                    print(f"  [ERROR] Worker {worker_idx} (PID: {p.pid}) failed with exit code {p.returncode}.", file=sys.stderr)
                else:
                    print(f"  [SYSTEM] Worker {worker_idx} (PID: {p.pid}) finished successfully.", file=sys.stderr)
                
                procs.remove((p, worker_idx))
                active_workers -= 1
            else:
                # Process is still running, try to read a line
                try:
                    line = p.stdout.readline()
                    if line.strip():
                        print(f"[Worker {worker_idx}] {line.strip()}", file=sys.stderr)
                except:
                    pass # Continue if readline fails for some reason
        
        time.sleep(0.05)

    print(f"\n  [SUMMARY] All workers finished. Success: {num_concurrent_workers - failed_jobs}, Failed: {failed_jobs}", file=sys.stderr)
    return failed_jobs == 0

def main():
    parser = argparse.ArgumentParser(description="Pre-process data and launch parallel training.")
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--concurrent-jobs', type=int, required=True)
    parser.add_argument('--models-to-train', type=int, required=True)
    args = parser.parse_args()
    
    output_metadata = {}
    base_name = os.path.splitext(os.path.basename(args.train_file))[0]
    os.makedirs(CONFIG["MODELS_SAVE_DIR"], exist_ok=True)
    try:
        variants_main = {rec.id: str(rec.seq) for rec in SeqIO.parse(CONFIG['KEY_FILE_NAME'], "fasta")}
    except FileNotFoundError:
        print(f"[FATAL] Key file '{CONFIG['KEY_FILE_NAME']}' not found. Aborting.", file=sys.stderr)
        sys.exit(1)
    train_shm_name = f"hpc_job_{base_name}_train_{os.getpid()}"
    global shm_name_for_cleanup
    shm_name_for_cleanup = train_shm_name
    def signal_handler(signum, frame):
        print(f"\n  [SIGNAL] Interrupt detected. Cleaning up and exiting.", file=sys.stderr)
        cleanup_shared_memory(shm_name_for_cleanup)
        sys.exit(1)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    train_shm = None

    try:
        cleanup_shared_memory(train_shm_name)
        preprocess_start_time = time.time()
        print(f"  -> [1/3] Pre-processing {args.train_file}...", file=sys.stderr)
        num_seqs_train, train_shm, num_cols, shm_size = extract_features_shm(
            args.train_file, variants_main, VARIANT_TO_FLOAT, train_shm_name
        )
        if not train_shm:
            sys.exit(1)
        
        output_metadata['num_train_seqs'] = num_seqs_train
        train_extract_time = time.time() - preprocess_start_time
        output_metadata['train_extract_time'] = train_extract_time
        print(f"  -> Extracted {num_seqs_train} sequences in {train_extract_time:.2f}s.", file=sys.stderr)
        print("  [INFO] Pre-fitting the data scaler...", file=sys.stderr)
        train_data_view = np.ndarray((num_seqs_train, num_cols), dtype=np.float32, buffer=train_shm.buf)
        scaler_main = StandardScaler()
        if num_seqs_train > 0:
             scaler_main.fit(train_data_view[:, :-1])
        scaler_path = os.path.join(CONFIG["MODELS_SAVE_DIR"], f"{base_name}_scaler.pkl")
        with open(scaler_path, 'wb') as f: pickle.dump(scaler_main, f)
        print(f"  [SUCCESS] Scaler saved to {scaler_path}", file=sys.stderr)
        
        train_shm.close()
        train_shm = None

        train_start_time = time.time()
        print(f"\n  -> [2/3] Launching persistent workers...", file=sys.stderr)
        
        command_base = [
            "python3", TRAIN_WORKER_SCRIPT,
            "--shm-name", train_shm_name,
            "--scaler-path", scaler_path,
            "--models-dir", os.path.join(CONFIG["MODELS_SAVE_DIR"], base_name),
            "--num-rows", str(num_seqs_train),
            "--num-cols", str(num_cols),
        ]
        
        success = run_persistent_workers(command_base, args.models_to_train, args.concurrent_jobs)

        total_train_time = time.time() - train_start_time
        output_metadata['total_train_time'] = total_train_time
        print(f"  -> Training phase completed in {total_train_time:.2f}s.", file=sys.stderr)
        
        if not success:
             print("  [FATAL] One or more worker processes failed. Aborting after cleanup.", file=sys.stderr)
             sys.exit(1)

        print(f"\n  -> [3/3] Verifying model creation...", file=sys.stderr)
        job_models_save_dir = os.path.join(CONFIG["MODELS_SAVE_DIR"], base_name)
        created_models = []
        if os.path.isdir(job_models_save_dir):
            created_models = [f for f in os.listdir(job_models_save_dir) if f.endswith('.pt')]
        print(f"  [VERIFY] Found {len(created_models)}/{args.models_to_train} created models.", file=sys.stderr)
        if len(created_models) < args.models_to_train:
            print("  [ERROR] Not all models were created! Check training job failures in the log above.", file=sys.stderr)
            sys.exit(1)
        
        # This is now the ONLY print to stdout
        print(json.dumps(output_metadata))

    finally:
        if train_shm:
            train_shm.close()
        print(f"  [CLEANUP] Main script finished. Finalizing cleanup...", file=sys.stderr)
        cleanup_shared_memory(shm_name_for_cleanup)

if __name__ == '__main__':
    main()