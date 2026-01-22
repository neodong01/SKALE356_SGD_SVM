#!/bin/bash

# filename: SKALE356.sh
# A script to run a high-throughput machine learning pipeline.
# It accepts a mandatory data directory and an optional number of concurrent jobs, models to train, and model type.

# --- Argument Parsing ---

# Check for the mandatory first argument (data directory)
if [ -z "$1" ]; then
    echo "Usage: ./ensemble.sh <base_data_directory> [concurrent_jobs] [models_to_train] [model_type]"
    echo "- <base_data_directory> : (Required) Path to the folder containing 'trainingFastas'."
    echo "- [concurrent_jobs] : (Optional) Number of parallel jobs to run."
    echo " Defaults to half the number of available CPU cores."
    echo "- [models_to_train] : (Optional) Number of models to train per FASTA file."
    echo " Defaults to 25."
    echo "- [model_type] : (Optional) The model type to train. Options: 'mlp', 'sgd', or 'svm'."
    echo " Defaults to 'mlp'."
    echo ""
    echo "Example: ./run_batch.sh rawShuffledFasta"
    echo "Example: ./run_batch.sh rawShuffledFasta 8"
    echo "Example: ./run_batch.sh rawShuffledFasta 8 50 sgd"
    echo "Example: ./run_batch.sh rawShuffledFasta 8 50 svm"
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

# Set the number of models based on the optional third argument or a default.
DEFAULT_MODELS=25
if [ -n "$3" ]; then
    MODELS_TO_TRAIN=$3
    echo "[INFO] Using user-provided models to train: ${MODELS_TO_TRAIN}"
else
    MODELS_TO_TRAIN=${DEFAULT_MODELS}
    echo "[INFO] No model count provided. Defaulting to: ${MODELS_TO_TRAIN}"
fi

# Set the model type based on the optional fourth argument or a default.
DEFAULT_MODEL_TYPE="mlp"
if [ -n "$4" ]; then
    MODEL_TYPE=$4
    # Ensure the model type is valid
    if [[ "$MODEL_TYPE" != "mlp" && "$MODEL_TYPE" != "sgd" && "$MODEL_TYPE" != "svm" ]]; then
        echo "[FATAL] Invalid model type specified: '${MODEL_TYPE}'. Must be 'mlp', 'sgd', or 'svm'. Aborting."
        exit 1
    fi
    echo "[INFO] Using user-provided model type: ${MODEL_TYPE}"
else
    MODEL_TYPE=${DEFAULT_MODEL_TYPE}
    echo "[INFO] No model type provided. Defaulting to: ${MODEL_TYPE}"
fi

# --- Script Start ---
echo "🚀 Starting High-Throughput Batch Processing... 🚀"
echo "[*] Ensemble size per file: ${MODELS_TO_TRAIN} models"
echo "[*] Parallel training jobs: ${CONCURRENT_JOBS}"
echo "[*] Model Type: ${MODEL_TYPE}"
echo "[*] Base Data Directory: ${BASE_DATA_DIR}"
echo "[*] Training Data Source: ${TRAINING_DATA_DIR}"
echo ""

# Check if the specified training directory actually exists
if [ ! -d "${TRAINING_DATA_DIR}" ]; then
    echo "[FATAL] Training directory not found at: '${TRAINING_DATA_DIR}'"
    echo "Please ensure the path is correct and try again."
    exit 1
fi

# Always re-compile to ensure changes take effect.
rm -f ./feature_extractor 

echo "[COMPILE] Compiling dependency-free C++ feature_extractor..."
g++ -std=c++17 -O3 -o feature_extractor feature_extractor.cpp -lrt
if [ $? -ne 0 ]; then
    echo ""
    echo "[FATAL] Compilation failed. Please check the g++ output for errors. Aborting."
    exit 1
fi
echo "[COMPILE] Compilation successful."

if ! command -v jq &> /dev/null; then
    echo "[FATAL] 'jq' is not installed, which is required for parsing script output. Aborting."
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
echo "[INFO] Forcing execution of Python scripts from: ${SCRIPT_DIR}"

file_count=0
while IFS= read -r fasta_file; do
    file_count=$((file_count + 1))
    echo "========================================================================"
    echo "Processing File ${file_count} of ${num_files}: ${fasta_file}"
    echo "========================================================================"

    # Pass the CONCURRENT_JOBS and MODEL_TYPE variables to the Python script
    json_output=$(python3 "${SCRIPT_DIR}/preprocess_data.py" \
        --train-file "${fasta_file}" \
        --models-to-train ${MODELS_TO_TRAIN} \
        --concurrent-jobs ${CONCURRENT_JOBS} \
        --model-type "${MODEL_TYPE}")

    py_exit_code=$?
    if [ ${py_exit_code} -ne 0 ]; then
        echo ""
        echo "[FATAL] The python script 'preprocess_data.py' failed for ${fasta_file}. Aborting."
        exit 1
    fi

    if ! echo "${json_output}" | jq . > /dev/null 2>&1; then
        echo ""
        echo "[FATAL] The python script did not return valid JSON. Aborting."
        echo "Captured output: ${json_output}"
        exit 1
    fi

    echo "-> [EVAL] Evaluating ensemble and generating final report..."
    train_extract_time=$(echo "${json_output}" | jq .train_extract_time)
    total_train_time=$(echo "${json_output}" | jq .total_train_time)
    num_train_seqs=$(echo "${json_output}" | jq .num_train_seqs)
    
    # Extract the individual train times JSON structure.
    individual_train_times=$(echo "${json_output}" | jq -c .individual_train_times)

    eval_summary=$(python3 "${SCRIPT_DIR}/evaluate_ensemble.py" \
        --train-file "${fasta_file}" \
        --train-extract-time "${train_extract_time}" \
        --num-models "${MODELS_TO_TRAIN}" \
        --total-train-time "${total_train_time}" \
        --num-train-seqs "${num_train_seqs}" \
        --individual-train-times "${individual_train_times}" \
        --model-type "${MODEL_TYPE}")
    
    if [ $? -ne 0 ]; then
        echo "[WARN] Evaluation script 'evaluate_ensemble.py' failed for ${fasta_file}."
    else
        echo "${eval_summary}"
    fi
    
    echo "[INFO] Successfully processed and logged performance for ${fasta_file}."
    echo ""
done < <(echo "${files_to_process}")

echo "✅ Batch processing complete. ✅"