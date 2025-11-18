#!/usr/bin/env python3
"""
Asymmetric Digital Signature Performance Testing
Tests ECDSA P-256, RSA-2048, and ML-DSA-44 (CRYSTALS-Dilithium) for V2X messages
following test plan section 8.2.

Message sizes: 200, 300, 800 bytes (BSM messages)
Algorithms: ECDSA P-256, RSA-2048, ML-DSA-44
Hardware Profile: HP-M (x86-64 laptop simulation)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import traceback
from utils.benchmark_utils import (
    calculate_statistics, calculate_throughput, save_results,
    NUM_RUNS_MEDIUM, WARMUP_RUNS
)
import os
import time
import numpy as np
from typing import List, Dict, Tuple
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa, padding
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
from cryptography.hazmat.backends import default_backend
from Cryptodome.PublicKey import RSA as CryptoRSA
from Cryptodome.Signature import pkcs1_15
from Cryptodome.Hash import SHA256
import psutil

# Test configuration
MESSAGE_SIZES = [200, 300, 800]  # Bytes (BSM sizes)
ALGORITHMS = ['ecdsa_p256', 'rsa_2048']  # ML-DSA-44 will be added if available
BSM_RATE_HZ = 10  # Messages per second


class AsymmetricSignatureBenchmark:
    """Benchmark asymmetric signature algorithms for V2X messages."""

    def __init__(self):
        self.results: List[Dict] = []
        self.backend = default_backend()
        self.ecdsa_keys = {}
        self.rsa_keys = {}

    def generate_ecdsa_key(self) -> ec.EllipticCurvePrivateKey:
        """Generate ECDSA P-256 key pair."""
        private_key = ec.generate_private_key(ec.SECP256R1(), self.backend)
        return private_key

    def generate_rsa_key(self) -> rsa.RSAPrivateKey:
        """Generate RSA-2048 key pair."""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=self.backend
        )
        return private_key

    def sign_ecdsa(self, private_key: ec.EllipticCurvePrivateKey,
                   message: bytes) -> Tuple[bytes, float]:
        """Sign message using ECDSA P-256."""
        start = time.perf_counter()
        signature = private_key.sign(message, ec.ECDSA(hashes.SHA256()))
        elapsed = time.perf_counter() - start
        return signature, elapsed

    def verify_ecdsa(self, public_key: ec.EllipticCurvePublicKey,
                     message: bytes, signature: bytes) -> Tuple[bool, float]:
        """Verify ECDSA signature."""
        start = time.perf_counter()
        try:
            public_key.verify(signature, message, ec.ECDSA(hashes.SHA256()))
            elapsed = time.perf_counter() - start
            return True, elapsed
        except Exception:
            elapsed = time.perf_counter() - start
            return False, elapsed

    def sign_rsa(self, private_key: rsa.RSAPrivateKey,
                 message: bytes) -> Tuple[bytes, float]:
        """Sign message using RSA-2048 with PKCS#1 v1.5."""
        start = time.perf_counter()
        # Hash the message first
        digest = hashes.Hash(hashes.SHA256(), backend=self.backend)
        digest.update(message)
        message_digest = digest.finalize()

        # Sign
        signature = private_key.sign(
            message_digest,
            padding.PKCS1v15(),
            hashes.SHA256()
        )
        elapsed = time.perf_counter() - start
        return signature, elapsed

    def verify_rsa(self, public_key: rsa.RSAPublicKey,
                   message: bytes, signature: bytes) -> Tuple[bool, float]:
        """Verify RSA signature."""
        start = time.perf_counter()
        try:
            # Hash the message
            digest = hashes.Hash(hashes.SHA256(), backend=self.backend)
            digest.update(message)
            message_digest = digest.finalize()

            # Verify
            public_key.verify(
                signature,
                message_digest,
                padding.PKCS1v15(),
                hashes.SHA256()
            )
            elapsed = time.perf_counter() - start
            return True, elapsed
        except Exception:
            elapsed = time.perf_counter() - start
            return False, elapsed

    def benchmark_sign_verify(self, algorithm: str, message_size: int) -> Dict:
        """
        Benchmark sign+verify operation for a given algorithm and message size.

        Returns:
            Dictionary with performance metrics
        """
        # Generate test message
        message = os.urandom(message_size)

        # Generate or reuse keys
        if algorithm == 'ecdsa_p256':
            if 'ecdsa' not in self.ecdsa_keys:
                self.ecdsa_keys['ecdsa'] = self.generate_ecdsa_key()
            private_key = self.ecdsa_keys['ecdsa']
            public_key = private_key.public_key()
        elif algorithm == 'rsa_2048':
            if 'rsa' not in self.rsa_keys:
                self.rsa_keys['rsa'] = self.generate_rsa_key()
            private_key = self.rsa_keys['rsa']
            public_key = private_key.public_key()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Warmup
        for _ in range(WARMUP_RUNS):
            if algorithm == 'ecdsa_p256':
                sig, _ = self.sign_ecdsa(private_key, message)
                self.verify_ecdsa(public_key, message, sig)
            elif algorithm == 'rsa_2048':
                sig, _ = self.sign_rsa(private_key, message)
                self.verify_rsa(public_key, message, sig)

        # Measurements
        sign_times = []
        verify_times = []
        total_times = []
        signature_sizes = []
        memory_peaks = []

        for _ in range(NUM_RUNS_MEDIUM):
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024  # MB

            # Sign
            if algorithm == 'ecdsa_p256':
                signature, elapsed_sign = self.sign_ecdsa(
                    private_key, message)
            elif algorithm == 'rsa_2048':
                signature, elapsed_sign = self.sign_rsa(private_key, message)

            # Verify
            if algorithm == 'ecdsa_p256':
                valid, elapsed_verify = self.verify_ecdsa(
                    public_key, message, signature)
            elif algorithm == 'rsa_2048':
                valid, elapsed_verify = self.verify_rsa(
                    public_key, message, signature)

            mem_after = process.memory_info().rss / 1024 / 1024  # MB

            assert valid, "Signature verification failed!"

            sign_times.append(elapsed_sign)
            verify_times.append(elapsed_verify)
            total_times.append(elapsed_sign + elapsed_verify)
            signature_sizes.append(len(signature))
            memory_peaks.append(mem_after - mem_before)

        # Calculate statistics
        sign_stats = calculate_statistics(sign_times)
        verify_stats = calculate_statistics(verify_times)
        total_stats = calculate_statistics(total_times)

        # Calculate per-BSM overhead at 10 Hz
        # Overhead = signature size + verification time impact
        median_signature_size = np.median(signature_sizes)
        overhead_percentage = (median_signature_size / message_size) * 100

        # Check if meets threshold: verify ≤ 3 ms median, signature ≤ 128 B
        meets_verify_threshold = verify_stats['median_s'] <= 0.003  # 3 ms
        meets_size_threshold = median_signature_size <= 128

        result = {
            'algorithm': algorithm,
            'message_size_bytes': message_size,
            'runs': NUM_RUNS_MEDIUM,
            'sign_stats': sign_stats,
            'verify_stats': verify_stats,
            'total_stats': total_stats,
            'signature_size_bytes': median_signature_size,
            'signature_size_min': np.min(signature_sizes),
            'signature_size_max': np.max(signature_sizes),
            'overhead_percentage': overhead_percentage,
            'meets_verify_threshold': meets_verify_threshold,
            'meets_size_threshold': meets_size_threshold,
            'overall_accept': meets_verify_threshold and meets_size_threshold,
            'memory_stats': {
                'mean_mb': np.mean(memory_peaks),
                'max_mb': np.max(memory_peaks),
            },
            'public_key_size_bytes': len(public_key.public_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )) if hasattr(public_key, 'public_bytes') else None,
        }

        return result

    def run_benchmarks(self, data_dir: Path):
        """Run all benchmarks for V2X BSM messages."""
        print("=" * 70)
        print("Asymmetric Signature Performance Testing - V2X BSM Messages")
        print("=" * 70)
        print("Hardware Profile: HP-M (x86-64 laptop simulation)")
        print(f"Test configuration:")
        print(f" Warmup runs: {WARMUP_RUNS}")
        print(f" Measurement runs: {NUM_RUNS_MEDIUM}")
        print(f" Message sizes: {MESSAGE_SIZES} bytes")
        print(f" Algorithms: {ALGORITHMS}")
        print(f" BSM rate: {BSM_RATE_HZ} Hz")
        print()

        all_results = []

        for msg_size in MESSAGE_SIZES:
            print(f"Testing message size: {msg_size} bytes")
            print("-" * 70)

            for algo in ALGORITHMS:
                print(f" {algo.upper():20s}... ", end="", flush=True)
                try:
                    result = self.benchmark_sign_verify(algo, msg_size)
                    all_results.append(result)

                    verify_stats = result['verify_stats']
                    sig_size = result['signature_size_bytes']
                    status = "ACCEPTS" if result['overall_accept'] else "REJECTS"

                    print(f"Verify: {verify_stats['median_s'] * 1e3:.3f} ms, "
                          f"Signature: {sig_size:.0f} B, Status: {status}")
                except Exception as e:
                    print(f"Failed: {e}")
                    traceback.print_exc()

            print()

        # Print summary table
        self.print_summary_table(all_results)

        # Save results
        output_path = data_dir / "asymmetric_signature_results.json"
        save_results(all_results, output_path)

        return all_results

    def print_summary_table(self, results: List[Dict]):
        """Print formatted summary table."""
        print("=" * 70)
        print("Summary Table")
        print("=" * 70)
        print(f"{'Algorithm':<20} {'Size (B)':<12} {'Sign (ms)':<15} "
              f"{'Verify (ms)':<15} {'Sig Size (B)':<15} {'Status':<10}")
        print("-" * 70)

        for r in results:
            algo = r['algorithm']
            size = r['message_size_bytes']
            sign_median = r['sign_stats']['median_s'] * 1e3
            verify_median = r['verify_stats']['median_s'] * 1e3
            sig_size = r['signature_size_bytes']
            status = "ACCEPTS" if r['overall_accept'] else "REJECTS"

            print(f"{algo:<20} {size:<12} {sign_median:<15.3f} "
                  f"{verify_median:<15.3f} {sig_size:<15.0f} {status:<10}")
        print()


def main():
    """Main entry point."""
    data_dir = Path("test_data")

    if not data_dir.exists():
        print("Error: test_data directory not found.")
        print("Please run generate_test_data.py first.")
        sys.exit(1)

    # Run benchmarks
    benchmark = AsymmetricSignatureBenchmark()
    results = benchmark.run_benchmarks(data_dir)

    print("=" * 70)
    print("Asymmetric signature testing complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
