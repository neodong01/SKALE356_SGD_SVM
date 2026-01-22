#!/bin/bash

# filename: SKALE356.sh
# A script to run a high-throughput machine learning pipeline.
# It accepts a mandatory data directory and an optional number of concurrent jobs.

# --- Argument Parsing ---

# Check for the mandatory first argument (data directory)
if [ -z "$1" ]; then
    echo "Usage: ./ensemble.sh <base_data_directory> [concurrent_jobs]"
    echo "  - <base_data_directory> : (Required) Path to the folder containing 'trainingFastas'."
    echo "  - [concurrent_jobs]     : (Optional) Number of parallel jobs to run."
    echo "                            Defaults to half the number of available CPU cores."
    echo ""
    echo "Example: ./run_batch.sh rawShuffledFasta"
    echo "Example: ./run_batch.sh rawShuffledFasta 8"
    exit 1
fi

# --- Configuration ---

# The base directory is the first argument.
BASE_DATA_DIR=$1
TRAINING_DATA_DIR="${BASE_DATA_DIR}/trainingFastas"

# Set concurrent jobs based on the optional second argument or a calculated default.
if [ -n "$2" ]; then
    # Use the provided second argument if it exists.
    CONCURRENT_JOBS=$2
    echo "[INFO] Using user-provided concurrent jobs: ${CONCURRENT_JOBS}"
else
    # Calculate the default value if the second argument is not provided.
    if command -v nproc &> /dev/null; then
        cpu_cores=$(nproc)
        CONCURRENT_JOBS=$(( cpu_cores / 2 ))
        # Ensure the number of jobs is at least 1, even on single/dual-core machines.
        if [ "$CONCURRENT_JOBS" -lt 1 ]; then
            CONCURRENT_JOBS=1
        fi
        echo "[INFO] No job count provided. Defaulting to half of CPU cores (${cpu_cores}): ${CONCURRENT_JOBS}"
    else
        # Fallback if 'nproc' is not available.
        CONCURRENT_JOBS=1
        echo "[WARN] 'nproc' command not found. Defaulting to a single concurrent job: ${CONCURRENT_JOBS}"
    fi
fi

MODELS_TO_TRAIN=25

# --- Script Start ---
echo "🚀 Starting High-Throughput Batch Processing... 🚀"
echo "  [*] Ensemble size per file: ${MODELS_TO_TRAIN} models"
echo "  [*] Parallel training jobs: ${CONCURRENT_JOBS}"
echo "  [*] Base Data Directory:    ${BASE_DATA_DIR}"
echo "  [*] Training Data Source:   ${TRAINING_DATA_DIR}"
echo ""

# Check if the specified training directory actually exists
if [ ! -d "${TRAINING_DATA_DIR}" ]; then
    echo "[FATAL] Training directory not found at: '${TRAINING_DATA_DIR}'"
    echo "        Please ensure the path is correct and try again."
    exit 1
fi

# Always re-compile to ensure changes take effect.
rm -f ./feature_extractor 

echo "  [COMPILE] Compiling dependency-free C++ feature_extractor..."
g++ -std=c++17 -O3 -o feature_extractor feature_extractor.cpp -lrt
if [ $? -ne 0 ]; then
    echo ""
    echo "  [FATAL] Compilation failed. Please check the g++ output for errors. Aborting."
    exit 1
fi
echo "  [COMPILE] Compilation successful."

if ! command -v jq &> /dev/null; then
    echo "  [FATAL] 'jq' is not installed, which is required for parsing script output. Aborting."
    exit 1
fi

files_to_process=$(find "${TRAINING_DATA_DIR}" -type f -name "*.fasta")
if [ -z "${files_to_process}" ]; then
    echo "[ERROR] No .fasta files found in '${TRAINING_DATA_DIR}'. Exiting."
    exit 1
fi

num_files=$(echo "${files_to_process}" | wc -l)
echo "[*] Found ${num_files} FASTA files to process."
echo ""

# Get the absolute directory where this script is located.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "  [INFO] Forcing execution of Python scripts from: ${SCRIPT_DIR}"

file_count=0
while IFS= read -r fasta_file; do
    file_count=$((file_count + 1))
    echo "========================================================================"
    echo "  Processing File ${file_count} of ${num_files}: ${fasta_file}"
    echo "========================================================================"

    # Pass the CONCURRENT_JOBS variable to the Python script
    json_output=$(python3 "${SCRIPT_DIR}/preprocess_data.py" \
        --train-file "${fasta_file}" \
        --models-to-train ${MODELS_TO_TRAIN} \
        --concurrent-jobs ${CONCURRENT_JOBS})

    py_exit_code=$?
    if [ ${py_exit_code} -ne 0 ]; then
        echo ""
        echo "[FATAL] The python script 'preprocess_data.py' failed for ${fasta_file}. Aborting."
        exit 1
    fi

    if ! echo "${json_output}" | jq . > /dev/null 2>&1; then
        echo ""
        echo "[FATAL] The python script did not return valid JSON. Aborting."
        echo "        Captured output: ${json_output}"
        exit 1
    fi

    echo "  -> [EVAL] Evaluating ensemble and generating final report..."
    train_extract_time=$(echo "${json_output}" | jq .train_extract_time)
    total_train_time=$(echo "${json_output}" | jq .total_train_time)
    num_train_seqs=$(echo "${json_output}" | jq .num_train_seqs)

    eval_summary=$(python3 "${SCRIPT_DIR}/evaluate_ensemble.py" \
        --train-file "${fasta_file}" \
        --train-extract-time "${train_extract_time}" \
        --total-train-time "${total_train_time}" \
        --num-train-seqs "${num_train_seqs}")
    
    if [ $? -ne 0 ]; then
        echo "  [WARN] Evaluation script 'evaluate_ensemble.py' failed for ${fasta_file}."
    else
        echo "${eval_summary}"
    fi
    
    echo "  [INFO] Successfully processed and logged performance for ${fasta_file}."
    echo ""
done < <(echo "${files_to_process}")

echo "✅ Batch processing complete. ✅"
