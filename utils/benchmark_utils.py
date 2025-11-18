#!/usr/bin/env python3
"""
Common utilities for cryptographic algorithm benchmarking.
Provides shared functionality for timing, statistics, and result handling.
"""

import json
import time
import os
import numpy as np
import psutil
from typing import List, Dict, Tuple, Optional
from pathlib import Path


# Test configuration constants
WARMUP_RUNS = 5
NUM_RUNS_MICRO = 10000  # For micro-frames (8-64 B)
NUM_RUNS_MEDIUM = 1000  # For medium messages (200-800 B)
NUM_RUNS_LARGE = 20  # For large operations (OTA files, etc.)
BOOTSTRAP_ITERATIONS = 1000  # For 95% confidence intervals


def bootstrap_ci(data: List[float], confidence: float = 0.95,
                   iterations: int = BOOTSTRAP_ITERATIONS) -> Tuple[float, float]:
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


def calculate_statistics(times: List[float]) -> Dict:
    """
    Calculate comprehensive statistics from timing measurements.

    Args:
        times: List of timing measurements in seconds

    Returns:
        Dictionary with statistical metrics
    """
    times_array = np.array(times)

    median_time = np.median(times_array)
    mean_time = np.mean(times_array)
    std_time = np.std(times_array)
    min_time = np.min(times_array)
    max_time = np.max(times_array)
    p5_time = np.percentile(times_array, 5)
    p95_time = np.percentile(times_array, 95)

    # Bootstrap confidence intervals
    ci_lower, ci_upper = bootstrap_ci(times, confidence=0.95)

    return {
        'mean_s': mean_time,
        'median_s': median_time,
        'std_s': std_time,
        'min_s': min_time,
        'max_s': max_time,
        'p5_s': p5_time,
        'p95_s': p95_time,
        'ci_95_lower_s': ci_lower,
        'ci_95_upper_s': ci_upper,
        'runs': len(times),
    }


def measure_memory_usage(func, *args, **kwargs) -> Tuple[any, float]:
    """
    Measure memory usage during function execution.

    Args:
        func: Function to execute
        *args, **kwargs: Arguments to pass to function

    Returns:
        Tuple of (function_result, memory_mb)
    """
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    result = func(*args, **kwargs)

    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    memory_used = mem_after - mem_before

    return result, memory_used


def benchmark_operation(operation_func, *args, num_runs: int = NUM_RUNS_MICRO,
                        warmup_runs: int = WARMUP_RUNS, **kwargs) -> Dict:
    """
    Benchmark a cryptographic operation with statistical analysis.

    Args:
        operation_func: Function to benchmark (should return result, time)
        *args: Positional arguments for operation_func
        num_runs: Number of measurement runs
        warmup_runs: Number of warmup runs
        **kwargs: Keyword arguments for operation_func

    Returns:
        Dictionary with timing statistics and results
    """
    # Warmup phase
    for _ in range(warmup_runs):
        try:
            operation_func(*args, **kwargs)
        except Exception:
            pass

    # Measurement phase
    times = []
    memory_peaks = []
    results = []

    for _ in range(num_runs):
        start = time.perf_counter()
        result, mem_used = measure_memory_usage(
            operation_func, *args, **kwargs)
        elapsed = time.perf_counter() - start

        times.append(elapsed)
        memory_peaks.append(mem_used)
        results.append(result)

    # Calculate statistics
    stats = calculate_statistics(times)

    # Memory statistics
    stats['memory_stats'] = {
        'mean_mb': np.mean(memory_peaks),
        'max_mb': np.max(memory_peaks),
    }

    return stats


def calculate_throughput(data_size_bytes: int, time_seconds: float) -> Dict:
    """
    Calculate throughput metrics from data size and time.

    Args:
        data_size_bytes: Size of data processed in bytes
        time_seconds: Time taken in seconds

    Returns:
        Dictionary with throughput in various units
    """
    throughput_bps = data_size_bytes / time_seconds

    return {
        'throughput_bps': throughput_bps,
        'throughput_mbps': throughput_bps / (1024 * 1024),
        'throughput_gbps': throughput_bps / (1024 * 1024 * 1024),
    }


def convert_numpy_types(obj):
    """
    Recursively convert NumPy types to native Python types for JSON serialization.

    Args:
        obj: Object that may contain NumPy types

    Returns:
        Object with NumPy types converted to native Python types
    """

    # Check for None first
    if obj is None:
        return None

    # Check for NumPy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # Check for NumPy scalar types using .item() method (works for all NumPy scalars)
    if hasattr(obj, 'item') and not isinstance(obj, (str, bytes, dict, list, tuple)):
        try:
            return obj.item()
        except (AttributeError, ValueError):
            pass

    # Check for NumPy integer types
    if isinstance(obj, np.integer):
        return int(obj)

    # Check for NumPy floating point types
    if isinstance(obj, np.floating):
        return float(obj)

    # Check for NumPy boolean types (handle both np.bool_ and bool)
    if isinstance(obj, bool) or (hasattr(np, 'bool_') and isinstance(obj, np.bool_)):
        return bool(obj)

    # Recursively process dictionaries
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}

    # Recursively process lists and tuples
    if isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]

    # Return as-is for other types
    return obj


def save_results(results: List[Dict], output_path: Path):
    """
    Save benchmark results to JSON file.

    Args:
        results: List of result dictionaries
        output_path: Path to save JSON file
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert NumPy types to native Python types
    results_serializable = convert_numpy_types(results)

    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print(f"Results saved to: {output_path}")


def load_results(json_path: Path) -> List[Dict]:
    """
    Load benchmark results from JSON file.

    Args:
        json_path: Path to JSON file

    Returns:
        List of result dictionaries
    """

    with open(json_path, 'r') as f:
        results = json.load(f)

    return results
