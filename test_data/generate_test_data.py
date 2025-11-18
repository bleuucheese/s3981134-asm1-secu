#!/usr/bin/env python3
"""
Test Data Generation Script
Generates synthetic test data for cryptographic algorithm evaluation
following the test plan specifications for autonomous vehicle workloads.
"""

import os
import struct
import random
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

# Fixed seed for reproducibility (as per test plan section 5)
DATA_SEED = 0xDEADBEEF
random.seed(DATA_SEED)
np.random.seed(DATA_SEED)

# Output directory
DATA_DIR = Path("test_data")
DATA_DIR.mkdir(exist_ok=True)


def generate_ota_file(size_mb: int, filename: str) -> Path:
    """
    Generate OTA (Over-The-Air) update file using seeded PRNG.

    Args:
        size_mb: Size in megabytes
        filename: Output filename

    Returns:
        Path to generated file
    """
    size_bytes = size_mb * 1024 * 1024
    filepath = DATA_DIR / filename

    print(f"Generating {size_mb} MB OTA file: {filename}")

    # Use seeded random generator for reproducibility
    rng = random.Random(DATA_SEED)
    chunk_size = 1024 * 1024  # 1 MiB chunks

    with open(filepath, 'wb') as f:
        remaining = size_bytes
        while remaining > 0:
            chunk = min(chunk_size, remaining)
            # Generate random bytes
            data = bytes([rng.randint(0, 255) for _ in range(chunk)])
            f.write(data)
            remaining -= chunk

    actual_size = filepath.stat().st_size
    print(f" Generated: {actual_size / (1024*1024):.2f} MB")
    return filepath


def generate_control_frame(payload_size: int, frame_id: int = 0) -> bytes:
    """
    Generate synthetic ECU control frame with structured header.

    Args:
        payload_size: Payload size in bytes (8-64 B)
        frame_id: Frame identifier

    Returns:
        Complete frame as bytes
    """
    # Structured header: 4 bytes frame ID + 2 bytes length + 2 bytes checksum placeholder
    header_size = 8
    total_size = header_size + payload_size

    # Header: frame_id (4 bytes), length (2 bytes), reserved (2 bytes)
    header = struct.pack('>IHH', frame_id, payload_size, 0)

    # Random payload (seeded for reproducibility)
    rng = random.Random(DATA_SEED + frame_id)
    payload = bytes([rng.randint(0, 255) for _ in range(payload_size)])

    return header + payload


def generate_v2x_bsm(size_bytes: int, message_id: int = 0) -> bytes:
    """
    Generate V2X Basic Safety Message (BSM) with realistic fields.

    BSM structure (canonicalized encoding):
    - Timestamp (8 bytes, Unix epoch nanoseconds)
    - Position: latitude (4 bytes float), longitude (4 bytes float)
    - Speed (2 bytes, m/s * 100)
    - Heading (2 bytes, degrees * 100)
    - Remaining payload: random data

    Args:
        size_bytes: Total message size (200, 300, or 800 B)
        message_id: Message identifier

    Returns:
        BSM message as bytes
    """
    fixed_fields_size = 20  # timestamp + position + speed + heading

    # Generate realistic fields
    rng = random.Random(DATA_SEED + message_id)

    # Timestamp (current time + small random offset)
    timestamp_ns = int(datetime.now(timezone.utc).timestamp() * 1e9) + rng.randint(0, 1000000)

    # Position (realistic vehicle coordinates: Melbourne area)
    latitude = -37.8136 + rng.uniform(-0.1, 0.1)
    longitude = 144.9631 + rng.uniform(-0.1, 0.1)

    # Speed (0-120 km/h = 0-33.3 m/s)
    speed_cm_per_s = int(rng.uniform(0, 33.3) * 100)

    # Heading (0-360 degrees)
    heading_centidegrees = int(rng.uniform(0, 360) * 100)

    # Pack fixed fields
    message = struct.pack('>QffHH',
                         timestamp_ns,
                         latitude,
                         longitude,
                         speed_cm_per_s,
                         heading_centidegrees)

    # Fill remaining with random data
    remaining = size_bytes - fixed_fields_size
    if remaining > 0:
        message += bytes([rng.randint(0, 255) for _ in range(remaining)])

    return message


def generate_vector_embeddings(num_vectors: int = 10000, dim: int = 512,
                              quantized: bool = False) -> np.ndarray:
    """
    Generate synthetic vector embeddings for similarity search.

    Args:
        num_vectors: Number of vectors to generate
        dim: Dimensionality of each vector
        quantized: If True, return int8 quantized vectors

    Returns:
        Array of shape (num_vectors, dim)
    """
    print(f"Generating {num_vectors} vectors × {dim} dimensions (quantized={quantized})")

    # Fixed covariance matrix for reproducibility
    rng = np.random.RandomState(DATA_SEED)
    mean = np.zeros(dim)
    cov = np.eye(dim) * 0.5  # Fixed covariance

    # Generate Gaussian vectors
    vectors = rng.multivariate_normal(mean, cov, size=num_vectors)

    # Normalize to unit vectors (for cosine similarity)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / (norms + 1e-8)

    if quantized:
        # Quantize to int8 range [-128, 127]
        vectors = (vectors * 127).astype(np.int8)

    return vectors


def generate_telemetry_data(num_rows: int = 100000) -> np.ndarray:
    """
    Generate synthetic telemetry data for homomorphic encryption experiments.

    Columns: speed (m/s), acceleration (m/s²), state_of_charge (%), temperature (°C)

    Args:
        num_rows: Number of telemetry records

    Returns:
        Array of shape (num_rows, 4)
    """
    print(f"Generating {num_rows} telemetry records")

    rng = np.random.RandomState(DATA_SEED)

    # Realistic ranges for vehicle telemetry
    speed = rng.uniform(0, 120/3.6, num_rows)  # 0-120 km/h in m/s
    acceleration = rng.uniform(-5, 5, num_rows)  # m/s²
    state_of_charge = rng.uniform(0, 100, num_rows)  # %
    temperature = rng.uniform(-10, 50, num_rows)  # °C

    telemetry = np.column_stack([speed, acceleration, state_of_charge, temperature])
    return telemetry


def main():
    """Generate all test data according to test plan specifications."""
    print("=" * 70)
    print("Test Data Generation for Cryptographic Algorithm Evaluation")
    print("=" * 70)
    print(f"Seed: 0x{DATA_SEED:X}")
    print(f"Output directory: {DATA_DIR.absolute()}\n")

    # 1. Generate OTA files (Section 3: 50 MB, 200 MB, 1 GB)
    print("1. Generating OTA files...")
    ota_files = [
        (50, "ota_model_delta_50mb.bin"),
        (200, "ota_ecu_firmware_200mb.bin"),
        (1024, "ota_full_image_1gb.bin"),
    ]
    for size_mb, filename in ota_files:
        generate_ota_file(size_mb, filename)
    print()

    # 2. Generate control frames (Section 3: 8-64 B)
    print("2. Generating ECU control frames...")
    control_frames_dir = DATA_DIR / "control_frames"
    control_frames_dir.mkdir(exist_ok=True)

    frame_sizes = [8, 32, 64]
    num_frames_per_size = 10000  # N = 10,000 per test plan section 4.5

    for size in frame_sizes:
        frames = []
        for i in range(num_frames_per_size):
            frame = generate_control_frame(size, i)
            frames.append(frame)

        # Save as binary file
        frames_file = control_frames_dir / f"frames_{size}B.bin"
        with open(frames_file, 'wb') as f:
            for frame in frames:
                f.write(frame)
        print(f" Generated {num_frames_per_size} frames of {size} B → {frames_file.name}")
    print()

    # 3. Generate V2X BSM messages (Section 3: 200, 300, 800 B)
    print("3. Generating V2X Basic Safety Messages...")
    v2x_dir = DATA_DIR / "v2x_messages"
    v2x_dir.mkdir(exist_ok=True)

    bsm_sizes = [200, 300, 800]
    num_messages_per_size = 1000  # N = 1,000 per test plan section 4.5

    for size in bsm_sizes:
        messages = []
        for i in range(num_messages_per_size):
            bsm = generate_v2x_bsm(size, i)
            messages.append(bsm)

        # Save as binary file
        bsm_file = v2x_dir / f"bsm_{size}B.bin"
        with open(bsm_file, 'wb') as f:
            for msg in messages:
                f.write(msg)
        print(f" Generated {num_messages_per_size} BSM messages of {size} B → {bsm_file.name}")
    print()

    # 4. Generate vector embeddings (Section 3: 10k vectors × 512-dim)
    print("4. Generating vector embeddings...")
    vectors_dir = DATA_DIR / "vectors"
    vectors_dir.mkdir(exist_ok=True)

    # Float32 embeddings
    embeddings_f32 = generate_vector_embeddings(10000, 512, quantized=False)
    vectors_file_f32 = vectors_dir / "embeddings_10k_512d_f32.npy"
    np.save(vectors_file_f32, embeddings_f32)
    print(f" Saved: {vectors_file_f32.name} ({embeddings_f32.nbytes / (1024*1024):.2f} MB)")

    # Int8 quantized embeddings
    embeddings_int8 = generate_vector_embeddings(10000, 512, quantized=True)
    vectors_file_int8 = vectors_dir / "embeddings_10k_512d_int8.npy"
    np.save(vectors_file_int8, embeddings_int8)
    print(f" Saved: {vectors_file_int8.name} ({embeddings_int8.nbytes / (1024*1024):.2f} MB)")

    # Query set: 1k vectors
    query_vectors = generate_vector_embeddings(1000, 512, quantized=False)
    query_file = vectors_dir / "queries_1k_512d_f32.npy"
    np.save(query_file, query_vectors)
    print(f" Saved: {query_file.name} ({query_vectors.nbytes / (1024*1024):.2f} MB)")
    print()

    # 5. Generate telemetry data (Section 3: ≤100k rows)
    print("5. Generating telemetry data...")
    telemetry = generate_telemetry_data(100000)
    telemetry_file = DATA_DIR / "telemetry_100k.npy"
    np.save(telemetry_file, telemetry)
    print(f" Saved: {telemetry_file.name} ({telemetry.nbytes / (1024*1024):.2f} MB)")
    print(f" Shape: {telemetry.shape}, Columns: [speed, acceleration, soc, temperature]")
    print()

    print("=" * 70)
    print("Test data generation complete!")
    print("=" * 70)
    print(f"Total files generated in: {DATA_DIR.absolute()}")
    print("\nGenerated datasets:")
    print(" - OTA files: 50MB, 200MB, 1GB")
    print(" - Control frames: 8B, 32B, 64B (10k each)")
    print(" - V2X BSM messages: 200B, 300B, 800B (1k each)")
    print(" - Vector embeddings: 10k×512D (f32 + int8) + 1k queries")
    print(" - Telemetry data: 100k rows × 4 columns")


if __name__ == "__main__":
    main()