// filename: feature_extractor.cpp

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <memory>
#include <cstdint>
#include <fstream> // For std::ifstream

// POSIX headers for shared memory
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

// Your original table, with 5s as the sentinel value.
static const uint8_t nt4_table[256] = {
	5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, 5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,
	5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, 5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,
	5, 0, 5, 1,  5, 5, 5, 2,  5, 5, 5, 5,  5, 5, 5, 5, 5, 5, 5, 5,  3, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,
    5, 0, 5, 1,  5, 5, 5, 2,  5, 5, 5, 5,  5, 5, 5, 5, 5, 5, 5, 5,  3, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,
    5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, 5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,
    5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, 5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,
    5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, 5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,
    5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, 5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5
};

void count_kmers(const std::string& seq, std::vector<float>& features) {
    const size_t k3_size = 64;
    const size_t k5_size = 1024;
    const size_t k6_size = 4096;
    size_t kmer3_offset = 0;
    size_t kmer5_offset = k3_size;
    size_t kmer6_offset = k3_size + k5_size;

    auto process_kmer_set = [&](size_t k, size_t offset) {
        if (seq.length() < k) return;
        uint64_t kmer = 0;
        uint64_t mask = (1ULL << (k * 2)) - 1;
        size_t consecutive_valid = 0;
        for (char c : seq) {
            uint8_t nt_val = nt4_table[static_cast<uint8_t>(c)];
            if (nt_val >= 4) {
                consecutive_valid = 0;
                continue;
            }
            kmer = ((kmer << 2) | nt_val) & mask;
            if (++consecutive_valid >= k) {
                features[offset + kmer]++;
            }
        }
    };
    process_kmer_set(3, kmer3_offset);
    process_kmer_set(5, kmer5_offset);
    process_kmer_set(6, kmer6_offset);
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <fasta_file> <shm_name> <buffer_size>" << std::endl;
        return 1;
    }

    std::string fasta_path = argv[1];
    std::string shm_name_arg = argv[2];
    std::string shm_path = "/" + shm_name_arg;
    size_t buffer_size = std::stoull(argv[3]);

    const int num_kmer_features = 64 + 1024 + 4096;
    const int num_cols = 1 + num_kmer_features + 1; 

    int shm_fd = shm_open(shm_path.c_str(), O_RDWR, 0666);
    if (shm_fd == -1) {
        perror("shm_open failed");
        return 1;
    }
    float* shm_ptr = (float*)mmap(0, buffer_size, PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shm_ptr == MAP_FAILED) {
        perror("mmap failed");
        close(shm_fd);
        return 1;
    }

    try {
        // --- DEPENDENCY-FREE FASTA PARSER ---
        std::ifstream fasta_file(fasta_path);
        if (!fasta_file.is_open()) {
            throw std::runtime_error("Could not open FASTA file.");
        }
        std::vector<std::string> sequences;
        std::string current_sequence;
        std::string line;
        
        while (std::getline(fasta_file, line)) {
            if (line.empty()) continue;
            if (line[0] == '>') {
                if (!current_sequence.empty()) {
                    sequences.push_back(current_sequence);
                }
                current_sequence.clear();
            } else {
                current_sequence += line;
            }
        }
        if (!current_sequence.empty()) {
            sequences.push_back(current_sequence);
        }
        // --- END OF PARSER ---
        
        // --- SEQUENTIAL PROCESSING (NO THREAD POOL) ---
        for (uint32_t i = 0; i < sequences.size(); ++i) {
            float* feature_row = shm_ptr + (static_cast<size_t>(i) * num_cols);
            std::vector<float> features(num_kmer_features, 0.0f);
            
            count_kmers(sequences[i], features);
            
            feature_row[0] = static_cast<float>(i);
            std::copy(features.begin(), features.end(), feature_row + 1);
        }

    } catch (const std::exception& e) {
        std::cerr << "An error occurred during C++ processing: " << e.what() << std::endl;
        munmap(shm_ptr, buffer_size);
        close(shm_fd);
        return 1;
    }

    munmap(shm_ptr, buffer_size);
    close(shm_fd);
    return 0;
}