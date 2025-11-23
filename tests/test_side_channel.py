import os
import time
import statistics
import random
from scipy import stats
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305

def measure_execution_time(func, *args):
    """Measure execution time of a function call."""
    start = time.perf_counter_ns()
    try:
        func(*args)
    except Exception:
        pass # Expect exceptions for invalid tags/ciphertexts
    end = time.perf_counter_ns()
    return end - start

def vulnerable_compare(a, b):
    """A vulnerable comparison function that exits early."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x != y:
            return False
    return True

def constant_time_compare(a, b):
    """A constant time comparison simulation."""
    if len(a) != len(b):
        return False
    result = 0
    for x, y in zip(a, b):
        result |= x ^ y
    return result == 0

def test_timing_leak(name, target_func, valid_input_gen, invalid_input_gen, iterations=10000):
    """
    Perform a statistical timing analysis.
    
    Args:
        name: Name of the test case.
        target_func: The function to test.
        valid_input_gen: Function that returns arguments for a valid case.
        invalid_input_gen: Function that returns arguments for an invalid case.
        iterations: Number of measurements to take.
    """
    print(f"Testing {name} for timing leaks with {iterations} iterations...")
    
    valid_times = []
    invalid_times = []
    
    # Interleave measurements to minimize environmental noise
    for _ in range(iterations):
        # Measure valid
        args_valid = valid_input_gen()
        t_valid = measure_execution_time(target_func, *args_valid)
        valid_times.append(t_valid)
        
        # Measure invalid
        args_invalid = invalid_input_gen()
        t_invalid = measure_execution_time(target_func, *args_invalid)
        invalid_times.append(t_invalid)
        
    # Statistical analysis
    mean_valid = statistics.mean(valid_times)
    mean_invalid = statistics.mean(invalid_times)
    
    # Welch's t-test
    t_stat, p_value = stats.ttest_ind(valid_times, invalid_times, equal_var=False)
    
    print(f"  Mean Valid Time:   {mean_valid:.2f} ns")
    print(f"  Mean Invalid Time: {mean_invalid:.2f} ns")
    print(f"  Difference:        {abs(mean_valid - mean_invalid):.2f} ns")
    print(f"  T-statistic:       {t_stat:.4f}")
    print(f"  P-value:           {p_value:.4e}")
    
    if p_value < 0.01: # 1% significance level
        print(f"  [FAIL] Significant timing difference detected! (p < 0.01)")
    else:
        print(f"  [PASS] No significant timing difference detected.")
    print("-" * 50)

def run_side_channel_tests():
    # 1. Control Test: Vulnerable Comparison
    # We expect this to FAIL (detect a leak)
    def vulnerable_setup():
        secret = b'A' * 32
        return secret
        
    def valid_input_vulnerable():
        secret = b'A' * 32
        return (secret, secret)
        
    def invalid_input_vulnerable():
        secret = b'A' * 32
        # Different first byte causes immediate return
        wrong = b'B' + b'A' * 31 
        return (secret, wrong)

    test_timing_leak(
        "Control: Vulnerable String Compare", 
        vulnerable_compare, 
        valid_input_vulnerable, 
        invalid_input_vulnerable,
        iterations=100000 # Needs more iterations to detect small python overhead differences
    )

    # 2. AES-GCM Decryption (Tag Check)
    key = AESGCM.generate_key(bit_length=256)
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    data = b"Secret Message"
    ciphertext = aesgcm.encrypt(nonce, data, None)
    
    def valid_input_aes():
        return (nonce, ciphertext, None)
        
    def invalid_input_aes():
        # Modify the last byte (part of the tag)
        bad_ciphertext = bytearray(ciphertext)
        bad_ciphertext[-1] ^= 0x01 
        return (nonce, bytes(bad_ciphertext), None)
        
    test_timing_leak(
        "AES-GCM Decryption (Tag Check)", 
        aesgcm.decrypt, 
        valid_input_aes, 
        invalid_input_aes
    )

    # 3. ChaCha20-Poly1305 Decryption (Tag Check)
    key_chacha = ChaCha20Poly1305.generate_key()
    chacha = ChaCha20Poly1305(key_chacha)
    nonce_chacha = os.urandom(12)
    ciphertext_chacha = chacha.encrypt(nonce_chacha, data, None)
    
    def valid_input_chacha():
        return (nonce_chacha, ciphertext_chacha, None)
        
    def invalid_input_chacha():
        bad_ciphertext = bytearray(ciphertext_chacha)
        bad_ciphertext[-1] ^= 0x01
        return (nonce_chacha, bytes(bad_ciphertext), None)

    test_timing_leak(
        "ChaCha20-Poly1305 Decryption", 
        chacha.decrypt, 
        valid_input_chacha, 
        invalid_input_chacha
    )

if __name__ == "__main__":
    run_side_channel_tests()
