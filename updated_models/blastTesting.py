# filename: blast_benchmark.py
import os
import re
import subprocess
import time
import csv
import argparse # <-- MODIFICATION: Import argparse
from collections import defaultdict

# --- ⬇️ MANDATORY CONFIGURATION ⬇️ ---
# Set the FULL path to your NCBI BLAST+ bin directory.
# Use forward slashes `/` even on Windows.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BLAST_BIN_PATH = os.path.join(SCRIPT_DIR, "ncbi-blast-2.16.0+", "bin")

# The name of your ground-truth key file.
KEY_FILE_NAME = "key.fasta"
# --- ⬆️ END OF MANDATORY CONFIGURATION ⬆️ ---

# --- Other Configuration ---
OUTPUT_DIR = "output"
SUMMARY_CSV_FILE = os.path.join(OUTPUT_DIR, "blast_performance_metrics.csv")
BLASTN_PARAMS = ["-outfmt", "10", "-max_target_seqs", "1"]
MAX_NEW_RUNS = 100 # Set high to process all new files


def load_key_file(key_filepath):
    """
    Loads the key.fasta file into a dictionary for quick lookup.
    Format: {sequence_id: class_name}
    """
    print(f"[*] Loading ground truth from '{key_filepath}'...")
    if not os.path.exists(key_filepath):
        print(f"[FATAL ERROR] Key file not found at: {key_filepath}")
        return None, None

    key_dict = {}
    class_set = set()
    with open(key_filepath, 'r') as f:
        current_seq_id = None
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith('>'):
                current_seq_id = line[1:].strip()
            elif current_seq_id:
                class_name = line
                key_dict[current_seq_id] = class_name
                class_set.add(class_name)
                current_seq_id = None

    print(f"[*] Successfully loaded {len(key_dict)} sequence-to-class mappings for {len(class_set)} classes.")
    return key_dict, sorted(list(class_set))

def calculate_performance_metrics(blast_result_path, key_dict, all_classes):
    """
    Reads a BLAST output file (outfmt 10) and calculates a suite of
    classification metrics.
    """
    true_labels = []
    pred_labels = []
    not_found_in_key = 0

    with open(blast_result_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            query_id, subject_id = row[0], row[1]
            query_class, subject_class = key_dict.get(query_id), key_dict.get(subject_id)

            if query_class is None or subject_class is None:
                not_found_in_key += 1
                continue

            true_labels.append(query_class)
            pred_labels.append(subject_class)

    if not_found_in_key > 0:
        print(f"      [Warning] Could not find {not_found_in_key} sequence IDs in the key file.")

    if not true_labels:
        return { "accuracy_percent": "0.00", "macro_precision": "0.0000", "macro_recall": "0.0000", "macro_f1_score": "0.0000", "weighted_precision": "0.0000", "weighted_recall": "0.0000", "weighted_f1_score": "0.0000" }

    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    accuracy = accuracy_score(true_labels, pred_labels) * 100
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='macro', zero_division=0, labels=all_classes)
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted', zero_division=0, labels=all_classes)

    return {
        "accuracy_percent": f"{accuracy:.2f}",
        "macro_precision": f"{macro_precision:.4f}",
        "macro_recall": f"{macro_recall:.4f}",
        "macro_f1_score": f"{macro_f1:.4f}",
        "weighted_precision": f"{weighted_precision:.4f}",
        "weighted_recall": f"{weighted_recall:.4f}",
        "weighted_f1_score": f"{weighted_f1:.4f}"
    }

# <-- MODIFICATION: The function now takes the root directory as an argument -->
def find_file_pairs(fasta_root_dir):
    """
    Finds all matching train/test FASTA file pairs within the provided root directory.
    """
    train_dir = os.path.join(fasta_root_dir, 'trainingFastas')
    test_dir = os.path.join(fasta_root_dir, 'testingFastas')

    print(f"[*] Searching for FASTA file pairs in:\n    Train: {train_dir}\n    Test:  {test_dir}")
    try:
        train_files = os.listdir(train_dir)
        test_files_set = set(os.listdir(test_dir))
    except FileNotFoundError as e:
        print(f"[FATAL ERROR] Directory not found: {e.filename}")
        print("        Please ensure the path you provided is correct.")
        return []

    print("\n--- Running CORRECTED Logic Check ---")
    file_pairs = []
    unmatched_files = []

    for filename in sorted(train_files):
        if not filename.endswith('.fasta'):
            continue
        
        match_new = re.match(r"train(\d+)_(\d+)_(\d+)\.fasta", filename)
        if match_new:
            train_percent, size, version = [int(g) for g in match_new.groups()]
            test_percent = 100 - train_percent
            test_filename_expected = f"test{test_percent}_{size}_{version}.fasta"
            if test_filename_expected in test_files_set:
                print(f"[OK]     Matched '{filename}' (Pattern 1) with '{test_filename_expected}'")
                file_pairs.append({"train_file": filename, "test_file": test_filename_expected})
            else:
                unmatched_files.append(f"[FAIL]   '{filename}' (Pattern 1) -> Expected '{test_filename_expected}', but not found.")
        
        elif re.match(r"train\((\d+)\)\((\d+)\)\.fasta", filename):
            match_count = re.match(r"train\((\d+)\)\((\d+)\)\.fasta", filename)
            count, version = [int(g) for g in match_count.groups()]
            test_seq_count = 11000 - count
            test_filename_expected = f"test({test_seq_count})({version}).fasta"
            if test_filename_expected in test_files_set:
                print(f"[OK]     Matched '{filename}' (Pattern 2) with '{test_filename_expected}'")
                file_pairs.append({"train_file": filename, "test_file": test_filename_expected})
            else:
                unmatched_files.append(f"[FAIL]   '{filename}' (Pattern 2) -> Expected '{test_filename_expected}', but not found.")
                
        else:
            unmatched_files.append(f"[NO MATCH] File '{filename}' did not match any known pattern.")

    print("--- End of Logic Check ---")

    if unmatched_files:
        print("\n--- DIAGNOSTIC REPORT ---")
        for line in unmatched_files:
            print(line)
        print("-------------------------\n")
        
    return file_pairs
    
def get_processed_files(summary_file):
    if not os.path.exists(summary_file): return set()
    processed = set()
    try:
        with open(summary_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'train_database_file' in row:
                    processed.add(row['train_database_file'])
    except Exception as e:
        print(f"[Warning] Could not read existing summary file: {e}. Assuming fresh start.")
        return set()
    print(f"[*] Found {len(processed)} previously processed files in summary report. They will be skipped.")
    return processed

def append_row_to_summary(filepath, data_dict, headers):
    file_exists = os.path.exists(filepath)
    try:
        with open(filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists: writer.writeheader()
            writer.writerow(data_dict)
    except IOError as e:
        print(f"[ERROR] Could not write to summary file: {e}")

# <-- MODIFICATION: The function now takes the root directory as an argument -->
def run_experiment(fasta_root_dir):
    """ Main function to orchestrate the BLAST benchmark. """
    print("--- Starting BLAST Classification Benchmark ---")
    print(f"[*] Using data from: '{os.path.abspath(fasta_root_dir)}'")

    makeblastdb_exe = os.path.join(BLAST_BIN_PATH, "makeblastdb.exe" if os.name == 'nt' else "makeblastdb")
    blastn_exe = os.path.join(BLAST_BIN_PATH, "blastn.exe" if os.name == 'nt' else "blastn")
    if not (os.path.exists(makeblastdb_exe) and os.path.exists(blastn_exe)):
        print(f"\n[FATAL ERROR] BLAST executables not found! I looked in: {os.path.abspath(BLAST_BIN_PATH)}")
        print("Please check the `BLAST_BIN_PATH` variable at the top of the script.")
        return

    key_dict, all_classes = load_key_file(KEY_FILE_NAME)
    if key_dict is None: return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # <-- MODIFICATION: Pass the argument to the function -->
    all_pairs = find_file_pairs(fasta_root_dir)
    if not all_pairs:
        print(f"\n[ERROR] No valid train/test file pairs were found. Please check the debug output above for mismatches.")
        return
    print(f"[*] Found {len(all_pairs)} total valid train/test pairs.")

    processed_files = get_processed_files(SUMMARY_CSV_FILE)
    pairs_to_run = [p for p in all_pairs if p['train_file'] not in processed_files]
    
    def sort_key(pair):
        match = re.search(r'(\d+)', pair['train_file'])
        return int(match.group(1)) if match else 0
        
    sorted_pairs = sorted(pairs_to_run, key=sort_key)

    if not sorted_pairs:
        print("\n[*] All available file pairs have already been processed. Nothing to do.")
        return

    print(f"[*] {len(sorted_pairs)} new pairs to process. Will run up to {MAX_NEW_RUNS} in this session.")
    
    headers = [
        "train_database_file", "test_query_file", "db_creation_time_sec",
        "blast_prediction_time_sec", "accuracy_percent",
        "macro_precision", "macro_recall", "macro_f1_score",
        "weighted_precision", "weighted_recall", "weighted_f1_score",
        "blast_result_file"
    ]
    processed_count_this_run = 0

    total_known_files = len(all_pairs) + len(processed_files)

    for i, pair in enumerate(sorted_pairs):
        if processed_count_this_run >= MAX_NEW_RUNS:
            print(f"\n[*] Reached run limit of {MAX_NEW_RUNS}. Stopping for now.")
            break

        train_file = pair['train_file']
        test_file = pair['test_file']
        print(f"\n--- Processing Pair {len(processed_files) + processed_count_this_run + 1}/{total_known_files} ---")
        print(f"  Database: {train_file}")
        print(f"  Query:    {test_file}")
        
        # <-- MODIFICATION: Use the argument to build paths -->
        train_fasta_path = os.path.join(fasta_root_dir, 'trainingFastas', train_file)
        test_fasta_path = os.path.join(fasta_root_dir, 'testingFastas', test_file)
        
        db_name = os.path.splitext(train_file)[0]
        db_path = os.path.join(OUTPUT_DIR, db_name)
        
        blast_result_filename = f"{db_name}_blast_results.csv"
        blast_output_path = os.path.join(OUTPUT_DIR, blast_result_filename)

        print("  [A] Creating BLAST database...")
        db_command = [makeblastdb_exe, "-in", train_fasta_path, "-dbtype", "nucl", "-out", db_path, "-title", db_name]
        start_time_db = time.time()
        try:
            subprocess.run(db_command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] makeblastdb failed for {train_file}.\n    Error: {e.stderr}")
            continue
        db_creation_time = time.time() - start_time_db
        print(f"      Done. Time taken: {db_creation_time:.4f} seconds.")

        print(f"  [B] Running blastn prediction...")
        blast_command = [blastn_exe, "-query", test_fasta_path, "-db", db_path, "-out", blast_output_path] + BLASTN_PARAMS
        start_time_blast = time.time()
        try:
            subprocess.run(blast_command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] blastn failed for query {test_file}.\n    Error: {e.stderr}")
            continue
        prediction_time = time.time() - start_time_blast
        print(f"      Done. Time taken: {prediction_time:.4f} seconds.")

        print("  [C] Calculating performance metrics...")
        metrics = calculate_performance_metrics(blast_output_path, key_dict, all_classes)
        print(f"      Accuracy: {metrics['accuracy_percent']}% | Macro F1: {metrics['macro_f1_score']}")

        row_data = {"train_database_file": train_file, "test_query_file": test_file, "db_creation_time_sec": f"{db_creation_time:.4f}", "blast_prediction_time_sec": f"{prediction_time:.4f}", "blast_result_file": blast_result_filename}
        row_data.update(metrics)
        append_row_to_summary(SUMMARY_CSV_FILE, row_data, headers)
        print(f"      [OK] Result appended to '{os.path.basename(SUMMARY_CSV_FILE)}'.")

        processed_count_this_run += 1

    if processed_count_this_run > 0:
        print(f"\n--- Experiment Session Complete ---")
        print(f"Processed {processed_count_this_run} new pair(s).")
        print(f"Find your updated summary report at: {os.path.abspath(SUMMARY_CSV_FILE)}")
    else:
        if not pairs_to_run and all_pairs:
             print("\n[*] All available file pairs have already been processed. Nothing to do.")
        elif not all_pairs:
            print("\n[INFO] No new results were generated because no valid pairs were found to begin with.")
        else:
            print("\n[INFO] No new results were generated in this run.")

# <-- MODIFICATION: This block now handles command-line arguments -->
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a BLAST benchmark on pairs of train/test FASTA files.",
        formatter_class=argparse.RawTextHelpFormatter # For better help text formatting
    )
    parser.add_argument(
        "fasta_root_dir",
        help="The path to the root directory that contains the 'trainingFastas' and 'testingFastas' subdirectories.\n"
             "Example: python blast_benchmark.py path/to/your/rawShuffledFastas"
    )
    args = parser.parse_args()
    
    # Pass the command-line argument to the main function
    run_experiment(args.fasta_root_dir)