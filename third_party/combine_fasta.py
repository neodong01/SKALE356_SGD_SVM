import os

def simple_fasta_parser(fasta_filepath):
    """
    A simple, dependency-free generator to parse FASTA files.
    It yields tuples of (sequence_id, sequence_string).
    Handles FileNotFoundError gracefully.
    """
    try:
        with open(fasta_filepath, 'r') as f:
            sequence_id = None
            sequence_lines = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if sequence_id:
                        yield (sequence_id, "".join(sequence_lines))
                    sequence_id = line[1:].strip()
                    sequence_lines = []
                else:
                    if sequence_id:
                        sequence_lines.append(line)
            if sequence_id:
                yield (sequence_id, "".join(sequence_lines))
    except FileNotFoundError:
        print(f"  [Info] Optional file not found, skipping: {fasta_filepath}")
        return

def create_combined_files():
    """
    Main function to find and read all variant files from their specific
    subdirectories inside the 'rawFasta' parent directory.
    Outputs to 'comb11000.fasta' and 'key.fasta'.
    """
    # List of the base names for your variants
    variants = [
        "alpha", "beta", "delta", "epsilon", "eta", "gamma", 
        "iota", "lambda", "mu", "omicron", "zeta"
    ]
    # The suffixes for the files within each directory
    sub_files_suffixes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    
    # --- Parent directory for all input data ---
    parent_input_dir = "rawFasta"

    output_sequences_file = "comb11000.fasta"
    output_key_file = "key.fasta"

    print("--- Starting File Combination Process ---")
    print(f"[*] Input data directory: '{parent_input_dir}/'")
    print(f"[*] Writing all sequences to: '{output_sequences_file}'")
    print(f"[*] Writing corresponding keys to: '{output_key_file}'")

    total_sequences_processed = 0

    # Open both output files once at the start
    with open(output_sequences_file, 'w') as seq_out, open(output_key_file, 'w') as key_out:
        # Loop through each variant (alpha, beta, etc.)
        for variant_name in variants:
            print(f"\nProcessing variant: {variant_name.capitalize()}")
            variant_sequences_count = 0
            
            # Construct the subdirectory name, e.g., "alphaComb"
            directory_name = f"{variant_name}Comb"

            # Loop through each sub-file (a, b, c, etc.)
            for suffix in sub_files_suffixes:
                # Construct the filename, e.g., "alpha100a.fasta"
                base_filename = f"{variant_name}100{suffix}.fasta"
                
                # --- MODIFIED PATH LOGIC ---
                # Create the full path: e.g., "rawFasta/alphaComb/alpha100a.fasta"
                full_filepath = os.path.join(parent_input_dir, directory_name, base_filename)
                
                # Use the parser to read sequences from the constructed path
                for seq_id, sequence in simple_fasta_parser(full_filepath):
                    # 1. Write the sequence to the combined output file
                    seq_out.write(f">{seq_id}\n")
                    seq_out.write(f"{sequence}\n")
                    
                    # 2. Write the corresponding entry to the key file
                    key_out.write(f">{seq_id}\n")
                    key_out.write(f"{variant_name.capitalize()}\n")
                    
                    variant_sequences_count += 1
            
            if variant_sequences_count > 0:
                print(f"  > Processed {variant_sequences_count} sequences for {variant_name.capitalize()}.")
                total_sequences_processed += variant_sequences_count

    print("\n--- Process Complete ---")
    print(f"Total sequences combined: {total_sequences_processed}")
    print(f"Your files '{output_sequences_file}' and '{output_key_file}' are ready.")

# --- Run the main function ---
if __name__ == "__main__":
    create_combined_files()