## Cryptographic Schemes Under Test

| Category                                       | Algorithms compared                                                                                  | Why this comparison matters                                                                                                                                                                                                                                                                         |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Hashing**                                    | **SHA-256**, **SHA-3**, **BLAKE3**                                                                   | SHA-256 and SHA-3 are NIST standards offering strong security but limited throughput. BLAKE3 represents a new design philosophy (tree hashing and SIMD parallelism). Comparing them shows how modern parallel hashes can reduce OTA-update time and energy consumption while maintaining integrity. |
| **Symmetric Authenticated Encryption (AEAD)**  | **AES-GCM**, **ChaCha20-Poly1305**, **ASCON-128/128a**                                               | AES-GCM is the industry standard but heavy on processors without AES-NI. ChaCha20-Poly1305 is software-friendly, widely used in TLS. ASCON is NIST’s 2025 Lightweight Cryptography standard. This set reveals which cipher best balances latency and resource use on embedded ECUs.                 |
| **Asymmetric / Digital Signatures**            | **ECDSA P-256**, **RSA-2048**, **ML-DSA-44 (CRYSTALS-Dilithium)**                                    | ECDSA is today’s V2X standard; RSA serves as a legacy baseline; ML-DSA represents the post-quantum future. Comparing them quantifies the cost of quantum-safe migration and its impact on V2X bandwidth and latency.                                                                                |
| **Privacy-preserving analytics (exploratory)** | **CKKS homomorphic encryption**, **vector-search patterns** (plain FAISS vs. encrypted or TEE-style) | Demonstrates the feasibility and current limits of privacy-preserving data processing for fleet learning or perception-model sharing — an emerging concern for connected-vehicle ecosystems.                                                                                                        |

Test Plan: Evaluating Cryptographic Schemes for Autonomous-Vehicle Workloads
1. Test objectives

This evaluation measures security-fit vs. system-fit for cryptographic schemes under representative autonomous-vehicle (AV) constraints. The goals are to:

Quantify latency, throughput, memory/code footprint, and bandwidth overhead for in-vehicle control, V2X messaging, and OTA integrity.

Compare symmetric AEAD (AES-GCM, ChaCha20-Poly1305, ASCON-128/128a), hashing (SHA-256, SHA3-256, BLAKE3), and signatures (ECDSA-P256, RSA-2048 as a pedagogical baseline, ML-DSA-44/Dilithium).

Explore feasibility of privacy-preserving analytics relevant to AVs: (a) homomorphic encryption (HE) for simple fleet-telemetry statistics and small cosine-similarity kernels; (b) private vector search design patterns for perception embeddings (plaintext FAISS vs. encrypted-transport/at-rest vs. TEE-backed processing vs. toy HE dot-product).

Produce decision thresholds that map algorithms to AV subsystems (ECU↔ECU, gateway↔cloud, V2X).

2. Scope and systems under test

Use-case profiles.
A) ECU control frames on microcontrollers: 8–64 B payloads at 100–1000 Hz, hard real-time.
B) V2X Basic Safety Messages: 200–800 B signed at 10 Hz; bandwidth-limited channel.
C) OTA payloads: 50 MB–1 GB artifacts; integrity check on multi-core gateway ECUs.
D) Privacy analytics: 256–1024-dim embeddings (float32 or quantized) for nearest-neighbour search; fleet-level stats on kinematic logs.

Hardware profiles.
HP-M: x86-64 laptop/desktop, ≥4 cores, AES-NI present (gateway proxy).
HP-µ: ARM Cortex-M4/M7 development board or cycle-accurate emulator (ECU proxy).
Results are reported per profile to avoid misleading generalisation from AES-NI-rich to MCU-class devices.

Software toolchain (Python).
cryptography (AES-GCM, ChaCha20-Poly1305, ECDSA), pycryptodomex (RSA), blake3, a maintained ASCON binding aligned to NIST SP 800-232 test vectors, FAISS (CPU), and one HE stack (e.g., TenSEAL/SEAL for CKKS). Timing via time.perf_counter_ns(). Randomness via os.urandom(); test vectors cross-checked against standards.

3. Test data

Control and V2X. Synthetic frames with structured headers and random payloads drawn from a fixed seed; BSM-like payloads at 200, 300, and 800 B. For signatures, messages include realistic fields (timestamp, position, speed, heading) and a canonicalised encoding to avoid variability from serialisation.

OTA. Three files: 50 MB (model delta), 200 MB (ECU firmware), 1 GB (full image). Data generated from seeded PRNG; integrity checks compare whole-file hash and chunked streaming variants (1 MiB blocks).

Vector search. Embeddings: 10 k vectors × 512-dim, synthetic Gaussian with fixed covariance; a second set quantised to int8 to mimic product-quantisation effects. Query set: 1 k vectors. Ground truth computed offline with exact cosine similarity.

Telemetry for HE. Small tables (≤100 k rows) with speed, accel, SoC, temperature; encrypted computation tasks include mean/variance and a 16-dim cosine similarity mini-bench.

4. Methods and Metrics

The evaluation will be based on three primary categories of metrics: **Performance**, **Security & Correctness**, and **Implementation & Usability**. These general metrics are supplemented by category-specific metrics to highlight the unique trade-offs for each class of algorithm.

**4.1. Core Performance Metrics**
*   **Latency**: The wall-clock time required to complete a single cryptographic operation (e.g., encrypt, decrypt, sign, verify, hash). Measured as median and 95th percentile to capture both typical and worst-case behavior.
*   **Throughput**: The rate at which data can be processed, measured in bytes per second (B/s). This is crucial for bulk data operations like hashing large files or encrypting large data streams.
*   **Resource Utilization**:
    *   **CPU Cycles**: The number of processor cycles per operation (or per byte), providing a hardware-agnostic measure of computational efficiency.
    *   **Memory Footprint**: The peak RAM usage (RSS) during operation and the static code/binary size required by the library. This is critical for memory-constrained environments.
*   **Energy Consumption**: The energy consumed per operation (Joules/op) or power drawn over time. This is a key metric for battery-powered devices and thermally constrained data centers.

**4.2. Security & Correctness Metrics**
*   **Test Vector Validation**: Pass/Fail validation against known-answer tests (KATs) from official sources (NIST, IETF RFCs). This is a fundamental check for correctness.
*   **Security Claims & Provenance**:
    *   **Security Level**: The claimed security strength in "bits" (e.g., 128-bit, 192-bit, 256-bit), which defines the theoretical resistance to key recovery attacks.
    *   **Public Specification**: Verification that the algorithm is publicly specified and standardized (e.g., by NIST, IETF, ISO), adhering to Kerckhoffs's principle.
    *   **Acceptance by Authorities**: The algorithm's status with major security authorities (e.g., NIST, BSI, ANSSI).
*   **Implementation Robustness**:
    *   **Malformed Input Handling**: The implementation's ability to fail safely when given invalid inputs (e.g., corrupted ciphertext, invalid public keys). The metric is a pass/fail check for returning a clear error instead of crashing.
    *   **Fault Injection Resistance (Qualitative)**: An assessment of the library's claimed or designed resistance to fault attacks (e.g., bit-flipping in hardware).
*   **Side-Channel Resistance (Qualitative)**: An assessment based on library documentation and design regarding its claims of being "constant-time" or resistant to timing, cache, or other side-channel attacks.
*   **Security Posture**: A categorical classification of the algorithm's standing in the security community (e.g., "Legacy," "Modern Standard," "Quantum-Resistant").

**4.3. Implementation & Usability Metrics**
*   **API Safety & Ergonomics (Qualitative)**: An assessment of how the library's API design helps prevent common cryptographic mistakes (e.g., nonce reuse, unvalidated parameters, insecure random number generation). This includes the complexity and learning curve for a developer.
*   **Key & Parameter Handling**:
    *   **Key Generation Time**: The latency to generate a new key pair or session key.
    *   **Parameter Size**: The size in bytes of public keys, private keys, and any other required domain parameters.
*   **Portability & Dependencies**: An evaluation of the effort required to compile and run the implementation on different architectures (e.g., x86-64, ARMv8, RISC-V) and its reliance on external libraries or system features.
*   **Maturity, Licensing & Patents**: A qualitative assessment of the library's maintenance status, community activity, software license, and any known patent encumbrances.

**4.4. Category-Specific Metrics**
*   **For Hashing**:
    *   **Parallelism Efficiency**: The speedup factor achieved when using multiple CPU cores (`Throughput(N cores) / Throughput(1 core)`). This directly measures the benefit for multi-core systems.
*   **For AEADs**:
    *   **Hardware Acceleration Speedup**: The performance gain when hardware-specific instructions (e.g., AES-NI) are available versus a pure software implementation.
*   **For Digital Signatures**:
    *   **Signature & Key Size**: The size in bytes of the output signature and the public key. This directly impacts bandwidth usage and storage requirements for any system managing public keys.
*   **For Homomorphic Encryption (HE)**:
    *   **Ciphertext Expansion Factor**: The ratio of ciphertext size to plaintext size, a primary cost driver for storage and transmission.
    *   **Numerical Precision Error**: The computational error introduced by approximate HE schemes, measured relative to the plaintext result.

**4.5. Statistical Design**
Warm-up runs discarded. Micro-frames: N = 10,000. Medium messages: N = 1,000. OTA files: 3 runs per file. Vector search: 100 query batches of size 10. Report 95 % CIs via bootstrap.

5. Entry criteria

Toolchain versions frozen and recorded (Python, OpenSSL backend, FAISS, HE library).

AES-NI detection status printed for HP-M; MCU clock/compilation flags recorded for HP-µ.

All KATs pass for each primitive; nonce-reuse tests in AEAD disabled except explicit negative checks.

Dataset seeds fixed and published; file IO uses buffered streams with stable chunk size.

6. Exit criteria

For each use-case profile, at least one scheme meets or beats the threshold below.

All metrics reported with CIs; plots and tables reproducible from a single make report.

Threat-model notes and limitations documented for each result (e.g., side-channel assumptions, key-management).

Code and raw results archived; an ablation note explains any surprising deltas.

7. Decision thresholds (accept/reject by use-case)

ECU control (HP-µ, 64 B frame).
Accept if AEAD encrypt+auth ≤ 100 µs median and ≤ 200 µs p95; code+RAM budget ≤ 64 KB; per-frame energy proxy ≤ baseline ChaCha20-Poly1305. Rationale: leaves headroom in 1 kHz loops.

V2X signing (HP-M, 300 B BSM @ 10 Hz).
Accept if verify ≤ 3 ms median and signature ≤ 128 B (overhead ≤ ~40 %). ECDSA-P256 expected to pass; ML-DSA-44 likely fails size/overhead in today’s channel, but may pass as “research-only” if latency is within 3–5 ms and you demonstrate a viable aggregation workaround.

OTA hashing (HP-M, 1 GB image).
Accept if whole-file verification ≥ 1.5 GB/s on 4 cores (BLAKE3 target) and ≤ 12 min wall-clock for a 1 GB image including disk IO. SHA3-256 is the standards baseline; BLAKE3 is the performance goal with a compliance caveat.

Private vector search.
Accept if the chosen privacy pattern delivers ≥ 95 % recall@10 vs. plaintext FAISS and median query latency ≤ 50 ms for batch size 10. Three patterns evaluated:
(a) Transport-and-at-rest encryption (TLS + AES-GCM; plaintext compute) — performance reference; privacy relies on server trust.
(b) TEE enclave (SGX conceptual design) — plaintext inside enclave; record enclave overhead from literature; mark as “architecturally feasible.”
(c) Toy HE dot-product (CKKS) on 16-dim vectors — demonstrate functional encrypted cosine similarity, report latency and ciphertext expansion; classify as not yet real-time but acceptable for offline analytics.

Homomorphic telemetry analytics.
Accept if encrypted mean/variance on 100 k rows completes in ≤ 5 s (CKKS) with numerical error ≤ 1e-3 relative to plaintext double; classify as cloud-side feasible and in-vehicle not recommended.

8. Experiments

8.1 Symmetric AEAD (ECU control).
Message sizes 8/32/64 B. Compare AES-GCM, ChaCha20-Poly1305, ASCON-128/128a. Run on HP-M (with and without AES-NI toggled if possible) and HP-µ. Report latency histograms and cycles/byte. Expectation: ASCON dominates on HP-µ; ChaCha competitive on HP-M when AES-NI is absent.

8.2 Digital signatures (V2X).
Sign/verify 200/300/800 B messages. Compare ECDSA-P256, RSA-2048 (pedagogical), ML-DSA-44. Report sign/verify time, signature size, and per-BSM overhead at 10 Hz. Expectation: ECDSA passes; RSA fails latency; ML-DSA passes latency on HP-M but fails size threshold, motivating hybrid or aggregation notes.

8.3 Hashing (OTA).
Hash 50 MB, 200 MB, 1 GB files with SHA-256, SHA3-256, BLAKE3. Measure single-thread and 4-thread streaming. Expectation: BLAKE3 significantly outperforms SHA-256 and SHA-3, likely achieving multi-GB/s throughput on multi-core systems. SHA-256, leveraging hardware acceleration (e.g., SHA-NI), is expected to be faster than SHA-3, which is known for its conservative design and lower performance in software. The performance of all algorithms should scale with file size.

8.4 Vector search privacy patterns.
Implement FAISS IVF-Flat baseline. Repeat under: (i) TLS + AES-GCM; (ii) “TEE emulation” by injecting constant overhead; (iii) CKKS toy cosine on 16-dim vectors for correctness and cost. Report recall@10, latency, and storage expansion.

8.5 Homomorphic analytics.
CKKS encrypted mean/variance over 100 k rows; micro-benchmark encrypted 16-dim dot product. Record parameter sets (poly modulus degree, coeff modulus), ciphertext sizes, latency, and numerical error.

9. Risks and controls

– AES-NI bias. Separate HP-M vs. HP-µ results; do not generalise AES-GCM wins on desktop to ECU class.
– HE library variability. Fix versions and parameters; emphasise that HE results are functional feasibility, not production latency.
– Side channels. Note that constant-time claims depend on library and platform; treat side-channel resistance as out-of-scope for timing, but document assumptions.
– Compliance. Flag where fast algorithms (e.g., BLAKE3) are not yet in FIPS evaluation; pair with a standards-compliant baseline (SHA-3).

10. Reporting format

A single “Methods & Results” bundle: tables for latency/throughput/overhead, violin plots for distributions, a suitability matrix mapping each scheme to AV subsystems with Accept / Caution / Reject labels based on thresholds above. An appendix contains the Python harness, pinned versions, and scripts to regenerate figures.