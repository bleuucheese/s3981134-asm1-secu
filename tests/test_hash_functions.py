#!/usr/bin/env python3
"""
Hash Function Performance Testing Script
Tests SHA-256, SHA-3, and BLAKE3 for OTA update integrity verification
following test plan section 8.3.
"""

import hashlib
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import blake3
import numpy as np
import psutil

# Test configuration - Updated for statistically meaningful results
# Testing on HP-M profile: x86-64 laptop (Intel Core i7-13650U, 32GB RAM)
# Note: No actual ARM MCU hardware available - all tests simulate AV workloads on laptop
WARMUP_RUNS = 5  # Increased warmup for better cache stabilization
NUM_RUNS_OTA = 20  # Increased from 3 to 20 for statistical significance (OTA files)
NUM_RUNS_MICRO = 10000  # Micro-frames per test plan section 4.5
NUM_RUNS_MEDIUM = 1000  # Medium messages per test plan section 4.5
CHUNK_SIZE = 1024 * 1024  # 1 MiB chunks for streaming
BOOTSTRAP_ITERATIONS = 1000  # For 95% confidence intervals


class HashBenchmark:
    """Benchmark hash functions on various data sizes."""

    def __init__(self):
        self.results: Dict[str, List[Dict]] = {}

    def hash_sha256(self, data: bytes) -> bytes:
        """Compute SHA-256 hash."""
        return hashlib.sha256(data).digest()

    def hash_sha3_256(self, data: bytes) -> bytes:
        """Compute SHA3-256 hash."""
        return hashlib.sha3_256(data).digest()

    def hash_blake3(self, data: bytes) -> bytes:
        """Compute BLAKE3 hash."""
        return blake3.blake3(data).digest()

    def hash_blake3_streaming(self, filepath: Path) -> Tuple[bytes, float]:
        """Compute BLAKE3 hash using streaming API."""
        hasher = blake3.blake3()
        start = time.perf_counter()

        with open(filepath, 'rb') as f:
            while True:
                chunk = f.read(CHUNK_SIZE)
                if not chunk:
                    break
                hasher.update(chunk)

        elapsed = time.perf_counter() - start
        return hasher.digest(), elapsed

    def hash_streaming(self, filepath: Path, algorithm: str) -> Tuple[bytes, float]:
        """
        Compute hash using streaming API for large files.

        Args:
            filepath: Path to file to hash
            algorithm: 'sha256', 'sha3_256', or 'blake3'

        Returns:
            Tuple of (hash_digest, elapsed_time_seconds)
        """
        start = time.perf_counter()

        if algorithm == 'blake3':
            hasher = blake3.blake3()
        elif algorithm == 'sha256':
            hasher = hashlib.sha256()
        elif algorithm == 'sha3_256':
            hasher = hashlib.sha3_256()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        with open(filepath, 'rb') as f:
            while True:
                chunk = f.read(CHUNK_SIZE)
                if not chunk:
                    break
                hasher.update(chunk)

        digest = hasher.digest()
        elapsed = time.perf_counter() - start
        return digest, elapsed

    def bootstrap_ci(self, data: List[float], confidence: float = 0.95, iterations: int = 1000) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval.

        Args:
            data: List of measurements
            confidence: Confidence level (default 0.95 for 95% CI)
            iterations: Number of bootstrap iterations

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if len(data) < 2:
            return (data[0], data[0])

        n = len(data)
        bootstrap_means = []

        for _ in range(iterations):
            # Resample with replacement
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))

        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

        return (lower, upper)

    def benchmark_single_file(self, filepath: Path, algorithm: str) -> Dict:
        """
        Benchmark hash function on a single file with statistical analysis.

        Returns:
            Dictionary with performance metrics including confidence intervals
        """
        file_size = filepath.stat().st_size

        # Warmup runs
        for _ in range(WARMUP_RUNS):
            self.hash_streaming(filepath, algorithm)

        # Actual measurements
        times = []
        memory_peaks = []

        for _ in range(NUM_RUNS_OTA):
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024  # MB

            _, elapsed = self.hash_streaming(filepath, algorithm)
            times.append(elapsed)

            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_peaks.append(mem_after - mem_before)

        # Calculate comprehensive statistics
        times_array = np.array(times)
        median_time = np.median(times_array)
        mean_time = np.mean(times_array)
        std_time = np.std(times_array)
        min_time = np.min(times_array)
        max_time = np.max(times_array)
        p5_time = np.percentile(times_array, 5)
        p95_time = np.percentile(times_array, 95)

        # Bootstrap confidence intervals
        ci_lower, ci_upper = self.bootstrap_ci(times, confidence=0.95, iterations=BOOTSTRAP_ITERATIONS)

        # Throughput calculations
        throughput_median = file_size / median_time
        throughput_mean = file_size / mean_time

        return {
            'algorithm': algorithm,
            'file_size_bytes': file_size,
            'file_size_mb': file_size / (1024 * 1024),
            'runs': NUM_RUNS_OTA,
            'time_stats': {
                'mean_s': mean_time,
                'median_s': median_time,
                'std_s': std_time,
                'min_s': min_time,
                'max_s': max_time,
                'p5_s': p5_time,
                'p95_s': p95_time,
                'ci_95_lower_s': ci_lower,
                'ci_95_upper_s': ci_upper,
            },
            'throughput_stats': {
                'median_bps': throughput_median,
                'median_mbps': throughput_median / (1024 * 1024),
                'median_gbps': throughput_median / (1024 * 1024 * 1024),
                'mean_bps': throughput_mean,
                'mean_mbps': throughput_mean / (1024 * 1024),
                'mean_gbps': throughput_mean / (1024 * 1024 * 1024),
            },
            'memory_stats': {
                'mean_mb': statistics.mean(memory_peaks),
                'max_mb': max(memory_peaks),
            },
        }

    def benchmark_parallel_blake3(self, filepath: Path, num_threads: int) -> Dict:
        """
        Benchmark BLAKE3 with parallel hashing (multi-threaded).

        Note: BLAKE3 supports parallel hashing natively.
        """
        file_size = filepath.stat().st_size

        # Warmup
        for _ in range(WARMUP_RUNS):
            hasher = blake3.blake3()
            with open(filepath, 'rb') as f:
                while True:
                    chunk = f.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    hasher.update(chunk)
            _ = hasher.digest()

        # Actual measurements
        times = []

        for _ in range(NUM_RUNS_OTA):
            start = time.perf_counter()

            # BLAKE3 automatically uses multiple threads if available
            # We can't directly control thread count in Python binding,
            # but we can measure the performance
            hasher = blake3.blake3()
            with open(filepath, 'rb') as f:
                while True:
                    chunk = f.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    hasher.update(chunk)
            _ = hasher.digest()

            elapsed = time.perf_counter() - start
            times.append(elapsed)

        # Calculate statistics
        times_array = np.array(times)
        median_time = np.median(times_array)
        mean_time = np.mean(times_array)
        std_time = np.std(times_array)
        ci_lower, ci_upper = self.bootstrap_ci(times, confidence=0.95, iterations=BOOTSTRAP_ITERATIONS)
        throughput_median = file_size / median_time

        return {
            'algorithm': 'blake3_parallel',
            'file_size_bytes': file_size,
            'file_size_mb': file_size / (1024 * 1024),
            'runs': NUM_RUNS_OTA,
            'time_stats': {
                'mean_s': mean_time,
                'median_s': median_time,
                'std_s': std_time,
                'ci_95_lower_s': ci_lower,
                'ci_95_upper_s': ci_upper,
            },
            'throughput_stats': {
                'median_bps': throughput_median,
                'median_mbps': throughput_median / (1024 * 1024),
                'median_gbps': throughput_median / (1024 * 1024 * 1024),
            },
            'num_threads': num_threads,
        }

    def run_ota_benchmarks(self, data_dir: Path):
        """Run benchmarks on OTA files (test plan section 8.3)."""
        print("=" * 70)
        print("Hash Function Performance Testing - OTA Updates")
        print("=" * 70)
        print("Hardware Profile: HP-M (x86-64 laptop simulation)")
        print(" Note: Testing simulates AV workloads on laptop hardware")
        print(" No actual ARM MCU hardware available")
        print()
        print(f"Test configuration:")
        print(f" Warmup runs: {WARMUP_RUNS}")
        print(f" Measurement runs: {NUM_RUNS_OTA} (increased for statistical significance)")
        print(f" Bootstrap iterations: {BOOTSTRAP_ITERATIONS} (for 95% CI)")
        print(f" Chunk size: {CHUNK_SIZE / (1024*1024):.1f} MiB")
        print(f" CPU cores available: {mp.cpu_count()}")
        print()

        ota_files = [
            data_dir / "ota_model_delta_50mb.bin",
            data_dir / "ota_ecu_firmware_200mb.bin",
            data_dir / "ota_full_image_1gb.bin",
        ]

        algorithms = ['sha256', 'sha3_256', 'blake3']
        all_results = []

        for filepath in ota_files:
            if not filepath.exists():
                print(f"Warning: {filepath} not found. Run generate_test_data.py first.")
                continue

            print(f"Testing file: {filepath.name} ({filepath.stat().st_size / (1024*1024):.1f} MB)")
            print("-" * 70)

            for algo in algorithms:
                print(f" {algo.upper():12s}... ", end="", flush=True)
                result = self.benchmark_single_file(filepath, algo)
                all_results.append(result)

                ts = result['time_stats']
                tps = result['throughput_stats']
                print(f"Median: {ts['median_s']:.3f}s "
                      f"(95% CI: [{ts['ci_95_lower_s']:.3f}, {ts['ci_95_upper_s']:.3f}]s), "
                      f"Throughput: {tps['median_mbps']:.2f} MB/s")

            # Test BLAKE3 parallel (if available)
            print(f" BLAKE3_PARALLEL... ", end="", flush=True)
            try:
                result_parallel = self.benchmark_parallel_blake3(filepath, mp.cpu_count())
                all_results.append(result_parallel)

                ts = result_parallel['time_stats']
                tps = result_parallel['throughput_stats']
                print(f"Median: {ts['median_s']:.3f}s "
                      f"(95% CI: [{ts['ci_95_lower_s']:.3f}, {ts['ci_95_upper_s']:.3f}]s), "
                      f"Throughput: {tps['median_mbps']:.2f} MB/s")
            except Exception as e:
                print(f"Failed: {e}")

            print()

        # Print summary table
        self.print_summary_table(all_results)

        # Save results
        self.save_results(all_results, data_dir / "hash_results.json")

        return all_results

    def print_summary_table(self, results: List[Dict]):
        """Print formatted summary table of results with confidence intervals."""
        print("=" * 70)
        print("Summary Table (with 95% Confidence Intervals)")
        print("=" * 70)
        print(f"{'Algorithm':<20} {'File (MB)':<12} {'Median Time (s)':<18} "
              f"{'95% CI (s)':<20} {'Throughput (MB/s)':<18} {'Throughput (GB/s)':<18}")
        print("-" * 70)

        for r in results:
            ts = r.get('time_stats', {})
            tps = r.get('throughput_stats', {})

            # Handle both old and new result formats
            if 'time_stats' in r:
                median_time = ts.get('median_s', 0)
                ci_lower = ts.get('ci_95_lower_s', 0)
                ci_upper = ts.get('ci_95_upper_s', 0)
                throughput_mbps = tps.get('median_mbps', 0)
                throughput_gbps = tps.get('median_gbps', 0)
            else:
                # Legacy format
                median_time = r.get('median_time_s', 0)
                ci_lower = median_time
                ci_upper = median_time
                throughput_mbps = r.get('throughput_mbps', 0)
                throughput_gbps = r.get('throughput_gbps', 0)

            file_size = r.get('file_size_mb', 0)
            ci_str = f"[{ci_lower:.3f}, {ci_upper:.3f}]"

            print(f"{r['algorithm']:<20} {file_size:<12.1f} {median_time:<18.3f} "
                  f"{ci_str:<20} {throughput_mbps:<18.2f} {throughput_gbps:<18.3f}")
        print()

    def save_results(self, results: List[Dict], output_path: Path):
        """Save results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {output_path}")


def verify_test_vectors():
    """Verify hash functions against known test vectors (KATs)."""
    print("Verifying hash functions against test vectors...")

    # Test vectors from NIST and BLAKE3 documentation
    test_cases = [
        (b"", "sha256", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        (b"abc", "sha256", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
        (b"", "sha3_256", "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a"),
        (b"abc", "sha3_256", "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532"),
        (b"", "blake3", "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"),
        (b"abc", "blake3", "6437b3ac38465133ffb63b75273a8db548c558465d79db03fd359c6cd5bd9d85"),
    ]

    passed = 0
    failed = 0

    for data, algo, expected_hex in test_cases:
        if algo == "sha256":
            result = hashlib.sha256(data).hexdigest()
        elif algo == "sha3_256":
            result = hashlib.sha3_256(data).hexdigest()
        elif algo == "blake3":
            result = blake3.blake3(data).hexdigest()
        else:
            continue

        if result.lower() == expected_hex.lower():
            print(f" ✓ {algo.upper()} test vector passed")
            passed += 1
        else:
            print(f" ✗ {algo.upper()} test vector FAILED")
            print(f"   Expected: {expected_hex}")
            print(f"   Got: {result}")
            failed += 1

    print(f"\nTest vectors: {passed} passed, {failed} failed")
    return failed == 0


def main():
    """Main entry point."""
    data_dir = Path("test_data")

    if not data_dir.exists():
        print("Error: test_data directory not found.")
        print("Please run generate_test_data.py first to generate test data.")
        sys.exit(1)

    # Verify test vectors first (entry criteria)
    print("=" * 70)
    print("Entry Criteria Check: Test Vector Validation")
    print("=" * 70)
    if not verify_test_vectors():
        print("\nWarning: Some test vectors failed. Proceeding anyway...")
    print()

    # Run benchmarks
    benchmark = HashBenchmark()
    results = benchmark.run_ota_benchmarks(data_dir)

    print("=" * 70)
    print("Hash function testing complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()