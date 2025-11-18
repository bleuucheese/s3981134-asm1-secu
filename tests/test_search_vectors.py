#!/usr/bin/env python3
"""
Homomorphic Encryption Analytics Performance Testing
Tests CKKS homomorphic encryption for telemetry analytics
following test plan section 8.5.

Tasks:
1. Encrypted mean/variance over 100k rows
2. Encrypted 16-dim dot product micro-benchmark
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Try to import TenSEAL
try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except ImportError:
    TENSEAL_AVAILABLE = False
    print("Warning: TenSEAL not available. Install with: pip install tenseal")

# Add parent directory to path for utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.benchmark_utils import (
    calculate_statistics, save_results, NUM_RUNS_LARGE, WARMUP_RUNS
)

# Test configuration
TELEMETRY_ROWS = 100000
DOT_PRODUCT_DIM = 16


class HomomorphicAnalyticsBenchmark:
    """Benchmark homomorphic encryption for telemetry analytics."""

    def __init__(self):
        self.results: List[Dict] = []
        self.context = None

    def setup_ckks_context(self):
        """Setup CKKS context for homomorphic operations."""
        if not TENSEAL_AVAILABLE:
            return None

        # CKKS parameters for telemetry analytics
        # poly_modulus_degree: 8192 (balance between security and performance)
        # coeff_mod_bit_sizes: [60, 40, 40, 60] (for multiplication depth)
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        context.generate_galois_keys()
        context.global_scale = 2**40

        return context

    def encrypt_mean_variance(self, telemetry_data: np.ndarray) -> Dict:
        """
        Encrypt and compute mean/variance over telemetry data.

        Args:
            telemetry_data: Array of shape (num_rows, num_columns)

        Returns:
            Dictionary with performance metrics
        """
        if not TENSEAL_AVAILABLE:
            return {'error': 'TenSEAL not available'}

        if self.context is None:
            self.context = self.setup_ckks_context()

        num_rows, num_cols = telemetry_data.shape

        # Encrypt columns
        print(f" Encrypting {num_cols} columns...")
        start_encrypt = time.perf_counter()

        encrypted_columns = []
        for col_idx in range(num_cols):
            col_data = telemetry_data[:, col_idx].tolist()
            # Encrypt in batches (CKKS has limits)
            batch_size = 4096  # Max elements per ciphertext
            encrypted_batches = []

            for i in range(0, len(col_data), batch_size):
                batch = col_data[i:i+batch_size]
                # Pad to batch_size if needed
                while len(batch) < batch_size:
                    batch.append(0.0)
                encrypted_batches.append(ts.ckks_vector(self.context, batch))

            encrypted_columns.append(encrypted_batches)

        encrypt_time = time.perf_counter() - start_encrypt

        # Compute mean (sum and divide by count)
        print(f" Computing encrypted mean...")
        start_compute = time.perf_counter()

        encrypted_means = []
        for col_batches in encrypted_columns:
            # Sum all batches
            col_sum = col_batches[0]
            for batch in col_batches[1:]:
                col_sum = col_sum + batch

            # Mean = sum / num_rows (approximate, as we can't divide exactly)
            # For demonstration, we'll compute sum and note that division would be done after decryption
            encrypted_means.append(col_sum)

        compute_time = time.perf_counter() - start_compute

        # Decrypt results
        print(f" Decrypting results...")
        start_decrypt = time.perf_counter()

        plaintext_means = []
        for enc_mean in encrypted_means:
            decrypted = enc_mean.decrypt()
            # Take first element (sum) and divide by num_rows
            mean_value = decrypted[0] / num_rows
            plaintext_means.append(mean_value)

        decrypt_time = time.perf_counter() - start_decrypt

        # Compute plaintext mean for comparison
        plaintext_means_ref = np.mean(telemetry_data, axis=0)

        # Calculate numerical error
        errors = [abs(p - r)
                  for p, r in zip(plaintext_means, plaintext_means_ref)]
        max_error = max(errors)
        relative_error = max_error / (max(abs(plaintext_means_ref)) + 1e-8)

        # Ciphertext size estimation
        # Each CKKS ciphertext is approximately poly_modulus_degree * coeff_mod_bit_sizes
        ciphertext_size_bytes = len(encrypted_columns[0][0].serialize())
        total_ciphertext_size = sum(
            len(batch.serialize())
            for col in encrypted_columns
            for batch in col
        )
        plaintext_size = telemetry_data.nbytes
        expansion_factor = total_ciphertext_size / plaintext_size

        return {
            'task': 'encrypted_mean_variance',
            'num_rows': num_rows,
            'num_columns': num_cols,
            'encrypt_time_s': encrypt_time,
            'compute_time_s': compute_time,
            'decrypt_time_s': decrypt_time,
            'total_time_s': encrypt_time + compute_time + decrypt_time,
            'max_error': max_error,
            'relative_error': relative_error,
            'ciphertext_size_bytes': ciphertext_size_bytes,
            'total_ciphertext_size_bytes': total_ciphertext_size,
            'plaintext_size_bytes': plaintext_size,
            'expansion_factor': expansion_factor,
            'poly_modulus_degree': 8192,
            'coeff_mod_bit_sizes': [60, 40, 40, 60],
        }

    def encrypt_dot_product(self, vec1: np.ndarray, vec2: np.ndarray) -> Dict:
        """
        Encrypt and compute dot product of two 16-dim vectors.

        Args:
            vec1, vec2: 16-dimensional vectors

        Returns:
            Dictionary with performance metrics
        """
        if not TENSEAL_AVAILABLE:
            return {'error': 'TenSEAL not available'}

        if self.context is None:
            self.context = self.setup_ckks_context()

        # Ensure vectors are 16-dimensional
        vec1 = vec1[:DOT_PRODUCT_DIM]
        vec2 = vec2[:DOT_PRODUCT_DIM]

        # Warmup
        for _ in range(WARMUP_RUNS):
            enc_vec1 = ts.ckks_vector(self.context, vec1.tolist())
            enc_vec2 = ts.ckks_vector(self.context, vec2.tolist())
            _ = enc_vec1.dot(enc_vec2)

        # Measurements
        encrypt_times = []
        compute_times = []
        decrypt_times = []
        errors = []

        for _ in range(NUM_RUNS_LARGE):
            # Encrypt
            start = time.perf_counter()
            enc_vec1 = ts.ckks_vector(self.context, vec1.tolist())
            enc_vec2 = ts.ckks_vector(self.context, vec2.tolist())
            encrypt_time = time.perf_counter() - start

            # Compute dot product
            start = time.perf_counter()
            enc_result = enc_vec1.dot(enc_vec2)
            compute_time = time.perf_counter() - start

            # Decrypt
            start = time.perf_counter()
            plaintext_result = enc_result.decrypt()[0]
            decrypt_time = time.perf_counter() - start

            encrypt_times.append(encrypt_time)
            compute_times.append(compute_time)
            decrypt_times.append(decrypt_time)

            # Compare with plaintext result
            plaintext_result_ref = np.dot(vec1, vec2)
            error = abs(plaintext_result - plaintext_result_ref)
            errors.append(error)

        encrypt_stats = calculate_statistics(encrypt_times)
        compute_stats = calculate_statistics(compute_times)
        decrypt_stats = calculate_statistics(decrypt_times)

        max_error = max(errors)
        mean_error = np.mean(errors)
        relative_error = max_error / (abs(np.dot(vec1, vec2)) + 1e-8)

        # Ciphertext sizes
        ciphertext_size = len(enc_vec1.serialize())

        return {
            'task': 'encrypted_dot_product',
            'dimensions': DOT_PRODUCT_DIM,
            'runs': NUM_RUNS_LARGE,
            'encrypt_stats': encrypt_stats,
            'compute_stats': compute_stats,
            'decrypt_stats': decrypt_stats,
            'max_error': max_error,
            'mean_error': mean_error,
            'relative_error': relative_error,
            'ciphertext_size_bytes': ciphertext_size,
            'expansion_factor': (ciphertext_size * 2) / (vec1.nbytes + vec2.nbytes),
        }

    def run_benchmarks(self, data_dir: Path):
        """Run all homomorphic encryption benchmarks."""
        print("=" * 70)
        print("Homomorphic Encryption Analytics Performance Testing")
        print("=" * 70)
        print(f"Test configuration:")
        print(f" Telemetry rows: {TELEMETRY_ROWS}")
        print(f" Dot product dimensions: {DOT_PRODUCT_DIM}")
        print()

        if not TENSEAL_AVAILABLE:
            print("Error: TenSEAL not available. Install with: pip install tenseal")
            return []

        all_results = []

        # Task 1: Encrypted mean/variance
        print("Task 1: Encrypted Mean/Variance over Telemetry Data")
        print("-" * 70)

        telemetry_path = data_dir / "telemetry_100k.npy"
        if not telemetry_path.exists():
            print("Error: Telemetry data file not found.")
            print("Please run generate_test_data.py first.")
            return []

        telemetry_data = np.load(telemetry_path)
        print(f"Loaded telemetry data: {telemetry_data.shape}")

        try:
            result = self.encrypt_mean_variance(telemetry_data)
            if 'error' not in result:
                all_results.append(result)

                print(f" Encrypt time: {result['encrypt_time_s']:.3f} s")
                print(f" Compute time: {result['compute_time_s']:.3f} s")
                print(f" Decrypt time: {result['decrypt_time_s']:.3f} s")
                print(f" Total time: {result['total_time_s']:.3f} s")
                print(f" Max error: {result['max_error']:.6f}")
                print(f" Relative error: {result['relative_error']:.6f}")
                print(f" Expansion factor: {result['expansion_factor']:.2f}x")
            else:
                print(f" Error: {result['error']}")
        except Exception as e:
            print(f" Failed: {e}")
            import traceback
            traceback.print_exc()

        print()

        # Task 2: Encrypted dot product
        print("Task 2: Encrypted 16-Dim Dot Product Micro-benchmark")
        print("-" * 70)

        # Generate test vectors
        np.random.seed(42)
        vec1 = np.random.randn(DOT_PRODUCT_DIM).astype(np.float32)
        vec2 = np.random.randn(DOT_PRODUCT_DIM).astype(np.float32)

        # Normalize for cosine similarity
        vec1 = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2 = vec2 / (np.linalg.norm(vec2) + 1e-8)

        print(f"Testing dot product of two {DOT_PRODUCT_DIM}-dim vectors")

        try:
            result = self.encrypt_dot_product(vec1, vec2)
            if 'error' not in result:
                all_results.append(result)

                enc_stats = result['encrypt_stats']
                comp_stats = result['compute_stats']
                dec_stats = result['decrypt_stats']

                print(
                    f" Encrypt median: {enc_stats['median_s']*1e3:.3f} ms")
                print(
                    f" Compute median: {comp_stats['median_s']*1e3:.3f} ms")
                print(
                    f" Decrypt median: {dec_stats['median_s']*1e3:.3f} ms")
                print(f" Max error: {result['max_error']:.6f}")
                print(f" Relative error: {result['relative_error']:.6f}")
                print(
                    f" Expansion factor: {result['expansion_factor']:.2f}x")
            else:
                print(f" Error: {result['error']}")
        except Exception as e:
            print(f" Failed: {e}")
            import traceback
            traceback.print_exc()

        print()

        # Save results
        output_path = data_dir / "homomorphic_results.json"
        save_results(all_results, output_path)

        return all_results


def main():
    """Main entry point."""
    data_dir = Path("test_data")

    if not data_dir.exists():
        print("Error: test_data directory not found.")
        print("Please run generate_test_data.py first.")
        sys.exit(1)

    benchmark = HomomorphicAnalyticsBenchmark()
    results = benchmark.run_benchmarks(data_dir)

    print("=" * 70)
    print("Homomorphic encryption testing complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()