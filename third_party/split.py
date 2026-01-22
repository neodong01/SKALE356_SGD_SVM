# filename: create_all_data_splits_unified.py
# A comprehensive script to generate all required dataset splits.
# It now accepts a SINGLE command-line argument for the output folder name.

import os
import random
import argparse
from Bio import SeqIO

# ==============================================================================
# --- GENERAL CONFIGURATION ---
# ==============================================================================
NUM_VERSIONS = 5

# --- Assumed location for sample-based source files ---
# MODIFICATION: Changed "Data Setup" to "rawFasta" to match your project structure.
SAMPLE_SOURCE_DIR = "rawFasta" 

# ==============================================================================
# --- TASK 1: CONFIG FOR PERCENTAGE-BASED SPLITS ---
# ==============================================================================
PERCENTAGE_MASTER_FASTA_FILE = "comb11000.fasta" 
TOTAL_SEQUENCES = 11000
TRAIN_PERCENTAGES_TO_GENERATE = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]

# ==============================================================================
# --- TASK 2: CONFIG FOR SAMPLE-BASED SPLITS ---
# ==============================================================================
VARIANT_FILES = [
    "alpha1000.fasta", "beta1000.fasta", "delta1000.fasta",
    "epsilon1000.fasta", "eta1000.fasta", "gamma1000.fasta",
    "iota1000.fasta", "lambda1000.fasta", "mu1000.fasta",
    "omicron1000.fasta", "zeta1000.fasta"
]
SAMPLE_SPLITS_TO_GENERATE = [
    {'samples_per_variant': 1, 'versions': NUM_VERSIONS},
    {'samples_per_variant': 5, 'versions': NUM_VERSIONS},
    {'samples_per_variant': 10, 'versions': NUM_VERSIONS},
]

# --- FUNCTION FOR PERCENTAGE-BASED SPLITS ---
def create_percentage_split(original_records, train_percent, version_num, train_dir, test_dir):
    records_to_shuffle = original_records[:]
    random.shuffle(records_to_shuffle)
    test_percent = 100 - train_percent
    train_filename = f"train{train_percent}_{TOTAL_SEQUENCES}_{version_num}.fasta"
    test_filename = f"test{test_percent}_{TOTAL_SEQUENCES}_{version_num}.fasta"
    train_filepath = os.path.join(train_dir, train_filename)
    test_filepath = os.path.join(test_dir, test_filename)
    num_train = round(TOTAL_SEQUENCES * (train_percent / 100.0))
    print(f"    Version {version_num}: Splitting into {num_train} train / {TOTAL_SEQUENCES - num_train} test...")
    with open(train_filepath, "w") as train_handle:
        SeqIO.write(records_to_shuffle[:num_train], train_handle, "fasta")
    with open(test_filepath, "w") as test_handle:
        SeqIO.write(records_to_shuffle[num_train:], test_handle, "fasta")
    print(f"      -> Created: {os.path.abspath(train_filepath)}")
    print(f"      -> Created: {os.path.abspath(test_filepath)}")

# --- FUNCTION FOR SAMPLE-BASED SPLITS ---
def create_sample_based_split(num_train_per_variant, version_num, train_dir, test_dir):
    print(f"\n--- Generating Split: {num_train_per_variant} Samples/Variant, Version {version_num} ---")
    total_variants = len(VARIANT_FILES)
    total_train_size = num_train_per_variant * total_variants
    total_sequences_per_file = 1000
    total_test_size = (total_sequences_per_file * total_variants) - total_train_size
    train_filename = f"train({total_train_size})({version_num}).fasta"
    test_filename = f"test({total_test_size})({version_num}).fasta"
    train_output_path = os.path.join(train_dir, train_filename)
    test_output_path = os.path.join(test_dir, test_filename)
    training_sequences = []
    testing_sequences = []
    for filename in VARIANT_FILES:
        # This will now correctly look inside the 'rawFasta' directory
        full_source_path = os.path.join(SAMPLE_SOURCE_DIR, filename)
        if not os.path.exists(full_source_path):
            print(f"  [!] SKIPPING: File not found at '{full_source_path}'")
            continue
        try:
            all_records_from_file = list(SeqIO.parse(full_source_path, "fasta"))
            random.shuffle(all_records_from_file)
            training_sequences.extend(all_records_from_file[:num_train_per_variant])
            testing_sequences.extend(all_records_from_file[num_train_per_variant:])
        except Exception as e:
            print(f"  [!] SKIPPING: Failed to parse '{filename}'. Error: {e}")
            continue
    random.shuffle(training_sequences)
    random.shuffle(testing_sequences)
    print(f"  -> Writing {len(training_sequences)} sequences to '{os.path.abspath(train_output_path)}'")
    SeqIO.write(training_sequences, train_output_path, "fasta")
    print(f"  -> Writing {len(testing_sequences)} sequences to '{os.path.abspath(test_output_path)}'")
    SeqIO.write(testing_sequences, test_output_path, "fasta")

# --- MAIN WORKFLOW ---
def main(output_folder_name):
    print("======================================================")
    print("=== Starting Generation of All FASTA Dataset Splits ===")
    print("======================================================")
    base_output_dir = os.path.abspath(output_folder_name)
    train_output_dir = os.path.join(base_output_dir, "trainingFastas")
    test_output_dir = os.path.join(base_output_dir, "testingFastas")
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)
    print(f"\n[*] All output will be saved in: '{base_output_dir}'")
    print("\n\n======================================================")
    print("=== TASK 1: Generating Percentage-Based Splits     ===")
    print("======================================================")
    if not os.path.exists(PERCENTAGE_MASTER_FASTA_FILE):
        print(f"\n[FATAL] Master FASTA not found: {PERCENTAGE_MASTER_FASTA_FILE}")
        print("        Please ensure it's in the same folder as the script. Skipping Task 1.")
    else:
        print(f"\nLoading master file: '{PERCENTAGE_MASTER_FASTA_FILE}'...")
        all_records = []
        with open(PERCENTAGE_MASTER_FASTA_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    try:
                        delimiter_pos = line.index('$')
                        header = line[:delimiter_pos]
                        sequence = line[delimiter_pos+1:]
                        record = SeqIO.read(f"{header}\n{sequence}\n", "fasta")
                        all_records.append(record)
                    except ValueError:
                        print(f"[Warning] Line without '$' delimiter found and skipped: {line}")
        print(f"Successfully loaded {len(all_records)} sequences.")
        for train_percent in TRAIN_PERCENTAGES_TO_GENERATE:
            print(f"\n--- Generating splits for {train_percent}% Training Data ---")
            for version in range(1, NUM_VERSIONS + 1):
                create_percentage_split(all_records, train_percent, version, train_output_dir, test_output_dir)
    print("\n\n======================================================")
    print("=== TASK 2: Generating Sample-Based Splits       ===")
    print("======================================================")
    # This print statement will now correctly show 'rawFasta'
    print(f"[*] Reading source files from: '{os.path.abspath(SAMPLE_SOURCE_DIR)}'")
    for split_config in SAMPLE_SPLITS_TO_GENERATE:
        n_samples = split_config['samples_per_variant']
        n_versions = split_config['versions']
        for v in range(1, n_versions + 1):
            create_sample_based_split(n_samples, v, train_output_dir, test_output_dir)
    print("\n\n--- ✅ All dataset generation tasks successfully completed! ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script to generate dataset splits. Assumes 'comb11000.fasta' and a 'rawFasta' folder exist in the current directory.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "output_folder_name",
        help="The name for the main output folder (e.g., 'rawShuffledFasta')."
    )
    args = parser.parse_args()
    main(args.output_folder_name)