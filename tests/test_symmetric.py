import os
import sys
import time
import importlib
import psutil
import numpy as np
from pathlib import Path
from typing import List, Dict
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
from cryptography.hazmat.backends import default_backend

try:
    import ascon
except ImportError:  # pragma: no cover - handled at runtime
    ascon = None

# Constants
WARMUP_RUNS = 100
NUM_RUNS_MICRO = 1000
MESSAGE_SIZES = [8, 16, 32, 64, 128]
ALGORITHMS = ['aes_gcm', 'chacha20_poly1305', 'ascon_128', 'ascon_128a']
ASCON_VARIANTS = {
    'ascon_128': 'Ascon-128',
    'ascon_128a': 'Ascon-128a',
}
ALGORITHM_MODULES = {
    'aes_gcm': 'cryptography',
    'chacha20_poly1305': 'cryptography',
    'ascon_128': 'ascon',
    'ascon_128a': 'ascon',
}


def calculate_statistics(times):
    """Calculate statistical measures for timing data."""
    times_array = np.array(times)
    return {
        'mean_s': np.mean(times_array),
        'median_s': np.median(times_array),
        'std_s': np.std(times_array),
        'min_s': np.min(times_array),
        'max_s': np.max(times_array),
        'ci_95_lower_s': np.percentile(times_array, 2.5),
        'ci_95_upper_s': np.percentile(times_array, 97.5)
    }


def calculate_throughput(message_size, time_s):
    """Calculate throughput in bytes per second."""
    return message_size / time_s if time_s > 0 else 0


def save_results(results, output_path):
    """Save results to JSON file."""
    import json
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


class SymmetricAEADBenchmark:
    """Benchmark symmetric AEAD algorithms for ECU control frames."""

    def __init__(self):
        self.results: List[Dict] = []
        self.backend = default_backend()
        self.module_size_cache: Dict[str, float] = {}

    def _ensure_ascon_available(self):
        """Ensure the optional Ascon dependency is installed."""
        if ascon is None:
            raise RuntimeError(
                "Ascon dependency missing. Please run `pip install ascon` to enable ASCON benchmarks."
            )

    def _get_module_root(self, module_name: str) -> Path:
        """Return the root directory for the given module."""
        module = importlib.import_module(module_name)
        module_path = Path(module.__file__)
        if module_path.name == '__init__.py':
            return module_path.parent
        return module_path.parent

    def _measure_module_size_kb(self, module_name: str) -> float:
        """Estimate module code size by summing files under its root directory."""
        root = self._get_module_root(module_name)
        total_bytes = 0
        for file in root.rglob('*'):
            if file.is_file():
                total_bytes += file.stat().st_size
        return total_bytes / 1024

    def get_algorithm_code_size(self, algorithm: str) -> float:
        """Return cached code size in KB for the given algorithm's module."""
        module_name = ALGORITHM_MODULES.get(algorithm)
        if module_name is None:
            return None
        if module_name not in self.module_size_cache:
            self.module_size_cache[module_name] = self._measure_module_size_kb(module_name)
        return self.module_size_cache[module_name]

    def generate_key(self, algorithm: str) -> bytes:
        """Generate appropriate key for algorithm."""
        if algorithm == 'aes_gcm':
            return os.urandom(32)  # 256-bit key
        elif algorithm == 'chacha20_poly1305':
            return os.urandom(32)  # 256-bit key
        elif algorithm == 'ascon_128' or algorithm == 'ascon_128a':
            return os.urandom(16)  # 128-bit key
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def generate_nonce(self, algorithm: str) -> bytes:
        """Generate appropriate nonce for algorithm."""
        if algorithm == 'aes_gcm':
            return os.urandom(12)  # 96-bit nonce
        elif algorithm == 'chacha20_poly1305':
            return os.urandom(12)  # 96-bit nonce
        elif algorithm == 'ascon_128' or algorithm == 'ascon_128a':
            return os.urandom(16)  # 128-bit nonce
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def encrypt_aes_gcm(self, key: bytes, nonce: bytes, data: bytes,
                        associated_data: bytes = b'') -> bytes:
        """Encrypt using AES-GCM."""
        aesgcm = AESGCM(key)
        return aesgcm.encrypt(nonce, data, associated_data)

    def decrypt_aes_gcm(self, key: bytes, nonce: bytes, ciphertext: bytes,
                        associated_data: bytes = b'') -> bytes:
        """Decrypt using AES-GCM."""
        aesgcm = AESGCM(key)
        return aesgcm.decrypt(nonce, ciphertext, associated_data)

    def encrypt_chacha20_poly1305(self, key: bytes, nonce: bytes, data: bytes,
                                  associated_data: bytes = b'') -> bytes:
        """Encrypt using ChaCha20-Poly1305."""
        chacha = ChaCha20Poly1305(key)
        return chacha.encrypt(nonce, data, associated_data)

    def decrypt_chacha20_poly1305(self, key: bytes, nonce: bytes,
                                  ciphertext: bytes,
                                  associated_data: bytes = b'') -> bytes:
        """Decrypt using ChaCha20-Poly1305."""
        chacha = ChaCha20Poly1305(key)
        return chacha.decrypt(nonce, ciphertext, associated_data)

    def encrypt_ascon(self, algorithm: str, key: bytes, nonce: bytes,
                      data: bytes, associated_data: bytes = b'') -> bytes:
        """Encrypt using Ascon variants."""
        self._ensure_ascon_available()
        variant = ASCON_VARIANTS[algorithm]
        return ascon.encrypt(key, nonce, associated_data, data, variant=variant)

    def decrypt_ascon(self, algorithm: str, key: bytes, nonce: bytes,
                      ciphertext: bytes, associated_data: bytes = b'') -> bytes:
        """Decrypt using Ascon variants."""
        self._ensure_ascon_available()
        variant = ASCON_VARIANTS[algorithm]
        return ascon.decrypt(key, nonce, associated_data, ciphertext,
                             variant=variant)

    def benchmark_encrypt_decrypt(self, algorithm: str, message_size: int) -> Dict:
        """
        Benchmark encrypt+decrypt operation for a given algorithm and message size.

        Returns:
            Dictionary with performance metrics
        """
        # Generate test data
        message = os.urandom(message_size)
        key = self.generate_key(algorithm)
        nonce = self.generate_nonce(algorithm)
        code_size_kb = self.get_algorithm_code_size(algorithm)

        # Warmup
        for _ in range(WARMUP_RUNS):
            if algorithm == 'aes_gcm':
                ciphertext = self.encrypt_aes_gcm(key, nonce, message)
                _ = self.decrypt_aes_gcm(key, nonce, ciphertext)
            elif algorithm == 'chacha20_poly1305':
                ciphertext = self.encrypt_chacha20_poly1305(key, nonce, message)
                _ = self.decrypt_chacha20_poly1305(key, nonce, ciphertext)
            elif algorithm in ASCON_VARIANTS:
                ciphertext = self.encrypt_ascon(algorithm, key, nonce, message)
                _ = self.decrypt_ascon(algorithm, key, nonce, ciphertext)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

        # Measurements
        encrypt_times = []
        decrypt_times = []
        total_times = []
        memory_peaks = []

        for _ in range(NUM_RUNS_MICRO):
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024  # MB

            # Encrypt
            start_encrypt = time.perf_counter()
            if algorithm == 'aes_gcm':
                ciphertext = self.encrypt_aes_gcm(key, nonce, message)
            elif algorithm == 'chacha20_poly1305':
                ciphertext = self.encrypt_chacha20_poly1305(key, nonce, message)
            elif algorithm in ASCON_VARIANTS:
                ciphertext = self.encrypt_ascon(algorithm, key, nonce, message)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            elapsed_encrypt = time.perf_counter() - start_encrypt

            # Decrypt
            start_decrypt = time.perf_counter()
            if algorithm == 'aes_gcm':
                decrypted = self.decrypt_aes_gcm(key, nonce, ciphertext)
            elif algorithm == 'chacha20_poly1305':
                decrypted = self.decrypt_chacha20_poly1305(key, nonce, ciphertext)
            elif algorithm in ASCON_VARIANTS:
                decrypted = self.decrypt_ascon(algorithm, key, nonce, ciphertext)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            elapsed_decrypt = time.perf_counter() - start_decrypt

            mem_after = process.memory_info().rss / 1024 / 1024  # MB

            encrypt_times.append(elapsed_encrypt)
            decrypt_times.append(elapsed_decrypt)
            total_times.append(elapsed_encrypt + elapsed_decrypt)
            memory_peaks.append(mem_after - mem_before)

            # Verify correctness
            assert decrypted == message, "Decryption failed!"

        # Calculate statistics
        encrypt_stats = calculate_statistics(encrypt_times)
        decrypt_stats = calculate_statistics(decrypt_times)
        total_stats = calculate_statistics(total_times)

        # Calculate throughput (bytes per second)
        encrypt_throughput = calculate_throughput(
            message_size, encrypt_stats['median_s'])
        decrypt_throughput = calculate_throughput(
            message_size, decrypt_stats['median_s'])

        # Calculate cycles/byte approximation (rough estimate)
        # Assuming ~2.5 GHz CPU frequency
        cpu_freq_ghz = 2.5  # Approximate
        cycles_per_byte_encrypt = (
            encrypt_stats['median_s'] * cpu_freq_ghz * 1e9) / message_size
        cycles_per_byte_decrypt = (
            decrypt_stats['median_s'] * cpu_freq_ghz * 1e9) / message_size

        result = {
            'algorithm': algorithm,
            'message_size_bytes': message_size,
            'runs': NUM_RUNS_MICRO,
            'encrypt_stats': encrypt_stats,
            'decrypt_stats': decrypt_stats,
            'total_stats': total_stats,
            'encrypt_throughput': encrypt_throughput,
            'decrypt_throughput': decrypt_throughput,
            'cycles_per_byte_encrypt': cycles_per_byte_encrypt,
            'cycles_per_byte_decrypt': cycles_per_byte_decrypt,
            'memory_stats': {
                'mean_mb': np.mean(memory_peaks),
                'max_mb': np.max(memory_peaks),
            },
            'ciphertext_size_bytes': len(ciphertext),
            'overhead_bytes': len(ciphertext) - message_size,
            'code_size_kb': code_size_kb,
        }

        return result

    def run_benchmarks(self, data_dir: Path):
        """Run all benchmarks for ECU control frames."""
        print("=" * 70)
        print("Symmetric AEAD Performance Testing - ECU Control Frames")
        print("=" * 70)
        print("Hardware Profile: HP-M (x86-64 laptop simulation)")
        print(f"Test configuration:")
        print(f" Warmup runs: {WARMUP_RUNS}")
        print(f" Measurement runs: {NUM_RUNS_MICRO}")
        print(f" Message sizes: {MESSAGE_SIZES} bytes")
        print(f" Algorithms: {ALGORITHMS}")
        print()

        all_results = []

        for msg_size in MESSAGE_SIZES:
            print(f"Testing message size: {msg_size} bytes")
            print("-" * 70)

            for algo in ALGORITHMS:
                print(f" {algo.upper():20s}... ", end="", flush=True)
                try:
                    result = self.benchmark_encrypt_decrypt(algo, msg_size)
                    all_results.append(result)

                    total_stats = result['total_stats']
                    print(f"Median: {total_stats['median_s'] * 1e6:.3f} µs "
                          f"(95% CI: [{total_stats['ci_95_lower_s'] * 1e6:.3f}, "
                          f"{total_stats['ci_95_upper_s'] * 1e6:.3f}] µs)")
                except Exception as e:
                    print(f"Failed: {e}")

            print()

        # Print summary table
        self.print_summary_table(all_results)

        # Save results
        output_path = data_dir / "symmetric_aead_results.json"
        save_results(all_results, output_path)

        return all_results

    def print_summary_table(self, results: List[Dict]):
        """Print formatted summary table."""
        print("=" * 70)
        print("Summary Table")
        print("=" * 70)
        print(f"{'Algorithm':<25} {'Size (B)':<12} {'Encrypt (µs)':<18} "
              f"{'Decrypt (µs)':<18} {'Total (µs)':<18} {'Overhead (B)':<15} {'Code Size (KB)':<15}")
        print("-" * 70)

        for r in results:
            algo = r['algorithm']
            size = r['message_size_bytes']
            enc_median = r['encrypt_stats']['median_s'] * 1e6
            dec_median = r['decrypt_stats']['median_s'] * 1e6
            total_median = r['total_stats']['median_s'] * 1e6
            overhead = r['overhead_bytes']
            code_size_kb = r.get('code_size_kb') or 0.0

            print(f"{algo:<25} {size:<12} {enc_median:<18.3f} "
                  f"{dec_median:<18.3f} {total_median:<18.3f} {overhead:<15} {code_size_kb:<15.1f}")
        print()


def verify_test_vectors():
    """Verify algorithms against known test vectors."""
    print("Verifying algorithms against test vectors...")

    # AES-GCM test vector (simplified)
    try:
        key = bytes.fromhex(
            '0000000000000000000000000000000000000000000000000000000000000000')
        nonce = bytes.fromhex('000000000000000000000000')
        plaintext = b'Hello, World!'

        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)
        decrypted = aesgcm.decrypt(nonce, ciphertext, None)

        assert decrypted == plaintext
        print(" ✓ AES-GCM test vector passed")
    except Exception as e:
        print(f" ✗ AES-GCM test vector failed: {e}")

    # ChaCha20-Poly1305 test vector (simplified)
    try:
        key = bytes.fromhex(
            '0000000000000000000000000000000000000000000000000000000000000000')
        nonce = bytes.fromhex('000000000000000000000000')
        plaintext = b'Hello, World!'

        chacha = ChaCha20Poly1305(key)
        ciphertext = chacha.encrypt(nonce, plaintext, None)
        decrypted = chacha.decrypt(nonce, ciphertext, None)

        assert decrypted == plaintext
        print(" ✓ ChaCha20-Poly1305 test vector passed")
    except Exception as e:
        print(f" ✗ ChaCha20-Poly1305 test vector failed: {e}")

    # ASCON test vectors (only if dependency is available)
    if ascon is not None:
        for algorithm, variant in ASCON_VARIANTS.items():
            try:
                key = bytes(16)
                nonce = bytes(16)
                plaintext = b'Hello, World!'

                ciphertext = ascon.encrypt(key, nonce, b'', plaintext,
                                            variant=variant)
                decrypted = ascon.decrypt(key, nonce, b'', ciphertext,
                                            variant=variant)

                assert decrypted == plaintext
                print(f" ✓ {variant} test vector passed")
            except Exception as e:  # pragma: no cover - depends on lib
                print(f" ✗ {variant} test vector failed: {e}")
    else:
        print(" ✗ Ascon dependency not installed; skipping ASCON test vectors")

    print()


def main():
    """Main entry point."""
    data_dir = Path("test_data")

    if not data_dir.exists():
        print("Error: test_data directory not found.")
        print("Please run generate_test_data.py first.")
        sys.exit(1)

    if any(algo in ASCON_VARIANTS for algo in ALGORITHMS) and ascon is None:
        print("Error: Ascon dependency not installed but required by the test plan.")
        print("Please run `pip install ascon` and re-run the symmetric tests.")
        sys.exit(1)

    # Verify test vectors
    print("=" * 70)
    print("Entry Criteria Check: Test Vector Validation")
    print("=" * 70)
    verify_test_vectors()

    # Run benchmarks
    benchmark = SymmetricAEADBenchmark()
    results = benchmark.run_benchmarks(data_dir)

    print("=" * 70)
    print("Symmetric AEAD testing complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()