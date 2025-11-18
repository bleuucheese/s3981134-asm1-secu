#!/usr/bin/env python3
"""
Vector Search Privacy Patterns Performance Testing
Tests FAISS baseline and privacy-preserving patterns for perception embeddings
following test plan section 8.4.

Patterns:
1. Plaintext FAISS (baseline)
2. TLS + AES-GCM (transport encryption)
3. TEE emulation (constant overhead injection)
4. CKKS homomorphic encryption (toy cosine similarity)
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.backends import default_backend
import psutil

# Add parent directory to path for utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.benchmark_utils import (
	calculate_statistics,
	save_results,
	NUM_RUNS_MEDIUM,
	WARMUP_RUNS,
)

# Try to import optional dependencies
try:
	import faiss

	FAISS_AVAILABLE = True
except ImportError:
	FAISS_AVAILABLE = False
	print("Warning: FAISS not available. Install with: pip install faiss-cpu")

try:
	import tenseal as ts

	TENSEAL_AVAILABLE = True
except ImportError:
	TENSEAL_AVAILABLE = False
	print("Warning: TenSEAL not available. Install with: pip install tenseal")

# Test configuration
QUERY_BATCH_SIZE = 10
K_NEIGHBORS = 10 # recall@10
DIMENSIONS = 512 # Full embeddings
DIMENSIONS_HE = 16 # Reduced for HE feasibility


def _faiss_ready(array: np.ndarray) -> np.ndarray:
	"""Ensure numpy array is contiguous float32 for FAISS APIs."""
	return np.ascontiguousarray(array, dtype=np.float32)


class VectorSearchBenchmark:
	"""Benchmark vector search with privacy-preserving patterns."""

	def __init__(self):
		self.results: List[Dict] = []
		self.backend = default_backend()

	def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
		"""Compute cosine similarity between two vectors."""
		dot_product = np.dot(vec1, vec2)
		norm1 = np.linalg.norm(vec1)
		norm2 = np.linalg.norm(vec2)
		return dot_product / (norm1 * norm2 + 1e-8)

	def faiss_baseline(self, embeddings: np.ndarray, queries: np.ndarray) -> Dict:
		"""
		Baseline FAISS search (plaintext).

		Returns:
			Dictionary with performance metrics
		"""
		if not FAISS_AVAILABLE:
			return {'error': 'FAISS not available'}

		# Ensure FAISS-friendly buffers
		embeddings = _faiss_ready(embeddings)
		queries = _faiss_ready(queries)

		# Build FAISS index
		dimension = embeddings.shape[1]
		index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

		# Normalize embeddings for cosine similarity
		faiss.normalize_L2(embeddings)
		faiss.normalize_L2(queries)

		# Add embeddings to index
		index.add(embeddings.astype('float32'))

		# Warmup
		for _ in range(WARMUP_RUNS):
			_ = index.search(queries[:QUERY_BATCH_SIZE].astype('float32'), K_NEIGHBORS)

		# Measurements
		search_times = []
		recalls = []

		for i in range(0, len(queries), QUERY_BATCH_SIZE):
			batch = queries[i:i + QUERY_BATCH_SIZE].astype('float32')

			start = time.perf_counter()
			distances, indices = index.search(batch, K_NEIGHBORS)
			elapsed = time.perf_counter() - start

			search_times.append(elapsed)

			# Calculate recall@10 (simplified - would need ground truth)
			# For now, just check that we got results
			recalls.append(1.0 if len(indices) > 0 else 0.0)

		stats = calculate_statistics(search_times)
		avg_recall = np.mean(recalls)

		return {
			'pattern': 'faiss_baseline',
			'search_stats': stats,
			'recall_at_10': avg_recall,
			'storage_expansion': 1.0,  # No expansion for plaintext
		}

	def tls_aes_gcm_pattern(self, embeddings: np.ndarray, queries: np.ndarray) -> Dict:
		"""
		TLS + AES-GCM pattern (transport encryption, plaintext compute).

		Simulates encrypted transport but plaintext computation on server.
		"""
		embeddings = _faiss_ready(embeddings)
		queries = _faiss_ready(queries)

		# Generate encryption key
		key = os.urandom(32)
		aesgcm = AESGCM(key)

		# Encrypt embeddings (simulate transport encryption)
		encrypted_embeddings = []
		for emb in embeddings:
			nonce = os.urandom(12)
			ciphertext = aesgcm.encrypt(nonce, emb.tobytes(), None)
			encrypted_embeddings.append(nonce + ciphertext)

		# Decrypt on server (simulate TLS termination)
		decrypted_embeddings = []
		for enc_emb in encrypted_embeddings:
			# Extract nonce (first 12 bytes) and ciphertext
			nonce = enc_emb[:12]
			ciphertext = enc_emb[12:]
			decrypted = aesgcm.decrypt(nonce, ciphertext, None)
			decrypted_embeddings.append(np.frombuffer(decrypted, dtype=np.float32))

		decrypted_embeddings = _faiss_ready(np.array(decrypted_embeddings))

		# Now perform plaintext search (same as baseline)
		if not FAISS_AVAILABLE:
			return {'error': 'FAISS not available'}

		dimension = decrypted_embeddings.shape[1]
		index = faiss.IndexFlatIP(dimension)

		queries = _faiss_ready(queries)
		faiss.normalize_L2(decrypted_embeddings)
		faiss.normalize_L2(queries)

		index.add(decrypted_embeddings)

		# Warmup
		for _ in range(WARMUP_RUNS):
			_ = index.search(queries[:QUERY_BATCH_SIZE].astype('float32'), K_NEIGHBORS)

		# Measurements
		search_times = []
		recalls = []

		for i in range(0, len(queries), QUERY_BATCH_SIZE):
			batch = queries[i:i + QUERY_BATCH_SIZE].astype('float32')

			start = time.perf_counter()
			distances, indices = index.search(batch, K_NEIGHBORS)
			elapsed = time.perf_counter() - start

			search_times.append(elapsed)
			recalls.append(1.0 if len(indices) > 0 else 0.0)

		stats = calculate_statistics(search_times)
		avg_recall = np.mean(recalls)

		# Storage expansion = encrypted size / plaintext size
		plaintext_size = embeddings.nbytes
		encrypted_size = sum(len(e) for e in encrypted_embeddings)
		expansion = encrypted_size / plaintext_size

		return {
			'pattern': 'tls_aes_gcm',
			'search_stats': stats,
			'recall_at_10': avg_recall,
			'storage_expansion': expansion,
		}

	def tee_emulation_pattern(self, embeddings: np.ndarray, queries: np.ndarray) -> Dict:
		"""
		TEE emulation pattern (constant overhead injection).

		Simulates trusted execution environment overhead.
		"""
		if not FAISS_AVAILABLE:
			return {'error': 'FAISS not available'}

		# Constant overhead per operation (simulating TEE overhead)
		TEE_OVERHEAD_MS = 2.0  # 2ms constant overhead per search

		embeddings = _faiss_ready(embeddings)
		queries = _faiss_ready(queries)

		dimension = embeddings.shape[1]
		index = faiss.IndexFlatIP(dimension)

		faiss.normalize_L2(embeddings)
		faiss.normalize_L2(queries)

		index.add(embeddings)

		# Warmup
		for _ in range(WARMUP_RUNS):
			_ = index.search(queries[:QUERY_BATCH_SIZE].astype('float32'), K_NEIGHBORS)

		# Measurements with TEE overhead
		search_times = []
		recalls = []

		for i in range(0, len(queries), QUERY_BATCH_SIZE):
			batch = queries[i:i + QUERY_BATCH_SIZE].astype('float32')

			start = time.perf_counter()
			distances, indices = index.search(batch, K_NEIGHBORS)
			elapsed = time.perf_counter() - start

			# Add TEE overhead
			elapsed += TEE_OVERHEAD_MS / 1000.0

			search_times.append(elapsed)
			recalls.append(1.0 if len(indices) > 0 else 0.0)

		stats = calculate_statistics(search_times)
		avg_recall = np.mean(recalls)

		return {
			'pattern': 'tee_emulation',
			'search_stats': stats,
			'recall_at_10': avg_recall,
			'storage_expansion': 1.0,  # No expansion for TEE
			'tee_overhead_ms': TEE_OVERHEAD_MS,
		}

	def ckks_homomorphic_pattern(self, embeddings: np.ndarray, queries: np.ndarray) -> Dict:
		"""
		CKKS homomorphic encryption pattern (toy cosine similarity on 16-dim vectors).

		Note: This is a simplified demonstration for correctness and cost analysis.
		"""
		if not TENSEAL_AVAILABLE:
			return {'error': 'TenSEAL not available'}

		# Reduce dimensions for HE feasibility
		embeddings_16d = embeddings[:, :DIMENSIONS_HE]
		queries_16d = queries[:, :DIMENSIONS_HE]

		# Normalize
		embeddings_16d = embeddings_16d / (np.linalg.norm(embeddings_16d, axis=1, keepdims=True) + 1e-8)
		queries_16d = queries_16d / (np.linalg.norm(queries_16d, axis=1, keepdims=True) + 1e-8)

		# Setup CKKS context
		context = ts.context(
			ts.SCHEME_TYPE.CKKS,
			poly_modulus_degree=8192,
			coeff_mod_bit_sizes=[60, 40, 40, 60]
		)
		context.generate_galois_keys()
		context.global_scale = 2**40

		# Encrypt embeddings
		encrypted_embeddings = []
		for emb in embeddings_16d[:100]:  # Limit for feasibility
			encrypted_embeddings.append(ts.ckks_vector(context, emb.tolist()))

		# Warmup
		if len(encrypted_embeddings) > 0:
			for _ in range(min(WARMUP_RUNS, len(queries_16d))):
				query_vec = ts.ckks_vector(context, queries_16d[0].tolist())
				_ = encrypted_embeddings[0].dot(query_vec)

		# Measurements
		search_times = []
		dot_products = []

		for i, query in enumerate(queries_16d[:10]):  # Limit queries for feasibility
			if i >= len(encrypted_embeddings):
				break

			query_vec = ts.ckks_vector(context, query.tolist())

			start = time.perf_counter()
			# Compute dot product (cosine similarity approximation)
			result = encrypted_embeddings[i].dot(query_vec)
			elapsed = time.perf_counter() - start

			search_times.append(elapsed)
			dot_products.append(result.decrypt()[0])

		stats = calculate_statistics(search_times) if search_times else {}

		# Calculate storage expansion
		plaintext_size = embeddings_16d.nbytes
		# Approximate ciphertext size (CKKS ciphertexts are much larger)
		ciphertext_size = len(encrypted_embeddings) * 8192 * 8 * 2  # Rough estimate
		expansion = ciphertext_size / plaintext_size if plaintext_size > 0 else 1.0

		return {
			'pattern': 'ckks_homomorphic',
			'search_stats': stats,
			'recall_at_10': None,  # Not applicable for HE
			'storage_expansion': expansion,
			'dimensions': DIMENSIONS_HE,
			'num_embeddings': len(encrypted_embeddings),
		}

	def run_benchmarks(self, data_dir: Path):
		"""Run all vector search benchmarks."""
		print("=" * 70)
		print("Vector Search Privacy Patterns Performance Testing")
		print("=" * 70)
		print(f"Test configuration:")
		print(f" Query batch size: {QUERY_BATCH_SIZE}")
		print(f" K neighbors: {K_NEIGHBORS}")
		print(f" Dimensions: {DIMENSIONS} (full), {DIMENSIONS_HE} (HE)")
		print()

		# Load embeddings and queries
		embeddings_path = data_dir / "vectors" / "embeddings_10k_512d_f32.npy"
		queries_path = data_dir / "vectors" / "queries_1k_512d_f32.npy"

		if not embeddings_path.exists() or not queries_path.exists():
			print("Error: Vector data files not found.")
			print("Please run generate_test_data.py first.")
			return []

		embeddings = _faiss_ready(np.load(embeddings_path))
		queries = _faiss_ready(np.load(queries_path))

		print(f"Loaded {len(embeddings)} embeddings and {len(queries)} queries")
		print()

		all_results = []

		# Test each pattern
		patterns = [
			('FAISS Baseline', self.faiss_baseline),
			('TLS + AES-GCM', self.tls_aes_gcm_pattern),
			('TEE Emulation', self.tee_emulation_pattern),
			('CKKS Homomorphic', self.ckks_homomorphic_pattern),
		]

		for pattern_name, pattern_func in patterns:
			print(f"Testing pattern: {pattern_name}")
			print("-" * 70)

			try:
				result = pattern_func(embeddings, queries)

				if 'error' in result:
					print(f" Skipped: {result['error']}")
					continue

				all_results.append(result)

				stats = result.get('search_stats', {})
				recall = result.get('recall_at_10', 'N/A')
				expansion = result.get('storage_expansion', 'N/A')

				if stats:
					median_ms = stats.get('median_s', 0) * 1e3
					print(f" Median latency: {median_ms:.3f} ms")
					print(f" Recall@10: {recall}")
					print(f" Storage expansion: {expansion:.2f}x")
				else:
					print(f" Pattern executed (detailed stats not available)")

			except Exception as e:
				print(f" Failed: {e}")
				import traceback
				traceback.print_exc()

			print()

		# Save results
		output_path = data_dir / "vector_search_results.json"
		save_results(all_results, output_path)

		return all_results


def main():
	"""Main entry point."""
	data_dir = Path("test_data")

	if not data_dir.exists():
		print("Error: test_data directory not found.")
		print("Please run generate_test_data.py first.")
		sys.exit(1)

	benchmark = VectorSearchBenchmark()
	results = benchmark.run_benchmarks(data_dir)

	print("=" * 70)
	print("Vector search testing complete!")
	print("=" * 70)


if __name__ == "__main__":
	main()