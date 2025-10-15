"""
Benchmark script for encoder latency.

Measures encoding latency for various text lengths on the development machine.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import numpy as np
from encoder import TextEncoder


def benchmark_encoder(encoder: TextEncoder, num_runs: int = 10):
    """
    Benchmark encoder latency.

    Args:
        encoder: TextEncoder instance
        num_runs: Number of runs per test
    """
    print("="*70)
    print("ENCODER LATENCY BENCHMARK")
    print("="*70)
    print(f"Model: {encoder.model_name}")
    print(f"Device: {encoder.device}")
    print(f"Target dimension: {encoder.target_dim}")
    print(f"Quantization: {encoder.use_quantization}")
    print()

    # Test cases with different text lengths
    test_cases = [
        ("Short (10 words)", "The quick brown fox jumps over the lazy dog again."),
        ("Medium (50 words)", " ".join([
            "This is a medium length text sample that contains approximately fifty words",
            "to test the encoding performance of our biometric text authentication system.",
            "We want to measure how long it takes to encode this text into a vector",
            "representation that can be used for user verification and authentication purposes."
        ])),
        ("Long (150 words)", " ".join([
            "This is a longer text sample that contains approximately one hundred and fifty words",
            "to test the encoding performance of our biometric text authentication system.",
            "In a real-world scenario, users might write messages of various lengths,",
            "and we need to ensure that our system can handle them efficiently.",
            "The encoding process involves several steps including tokenization,",
            "running the text through a transformer model, mean-pooling the hidden states,",
            "projecting to our target dimensionality, and finally L2 normalizing the result.",
            "All of these steps need to be performed quickly enough to provide a good user experience.",
            "Our target is to keep the latency under 200 milliseconds on typical hardware",
            "for texts up to 150 words in length, which should cover most common use cases",
            "in chat applications, email clients, and other text-based communication platforms.",
            "This benchmark will help us verify that we meet this performance target",
            "and identify any potential bottlenecks in our implementation that might need optimization."
        ])),
        ("Very Long (300 words)", " ".join([f"word{i}" for i in range(300)])),
    ]

    # Warm-up run
    print("Warming up...")
    encoder.encode(["Warm-up text"])
    print("Warm-up complete.\n")

    results = []

    for name, text in test_cases:
        word_count = len(text.split())
        latencies = []

        print(f"Test: {name} ({word_count} words)")
        print("-" * 70)

        # Run benchmark
        for i in range(num_runs):
            start = time.perf_counter()
            embeddings = encoder.encode([text])
            end = time.perf_counter()

            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)

            # Verify output
            assert embeddings.shape == (1, encoder.target_dim)
            norm = np.linalg.norm(embeddings[0])
            assert 0.99 <= norm <= 1.01, f"Norm check failed: {norm}"

        # Calculate statistics
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        median_latency = np.median(latencies)

        results.append({
            'name': name,
            'words': word_count,
            'mean': mean_latency,
            'std': std_latency,
            'min': min_latency,
            'max': max_latency,
            'median': median_latency,
        })

        print(f"  Mean latency:   {mean_latency:7.2f} ms")
        print(f"  Std deviation:  {std_latency:7.2f} ms")
        print(f"  Min latency:    {min_latency:7.2f} ms")
        print(f"  Max latency:    {max_latency:7.2f} ms")
        print(f"  Median latency: {median_latency:7.2f} ms")
        print()

    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Test':<25} {'Words':>8} {'Mean (ms)':>12} {'Target':>12}")
    print("-"*70)

    for result in results:
        target = "< 200 ms" if result['words'] <= 150 else "N/A"
        status = "✓" if result['mean'] < 200 or result['words'] > 150 else "✗"
        print(f"{result['name']:<25} {result['words']:>8} {result['mean']:>11.2f}  {target:>10} {status}")

    print()

    # Check acceptance criteria
    medium_result = [r for r in results if r['words'] >= 150 and r['words'] <= 160]
    if medium_result:
        latency_150 = medium_result[0]['mean']
        if latency_150 < 200:
            print(f"✅ PASS: 150-word sample latency ({latency_150:.2f} ms) < 200 ms")
        else:
            print(f"⚠️  WARNING: 150-word sample latency ({latency_150:.2f} ms) >= 200 ms")
            print("   This exceeds the target but may be acceptable depending on hardware.")

    print()
    print("Benchmark complete!")
    print("="*70)


def benchmark_batch_sizes(encoder: TextEncoder):
    """
    Benchmark different batch sizes.

    Args:
        encoder: TextEncoder instance
    """
    print("\n")
    print("="*70)
    print("BATCH SIZE BENCHMARK")
    print("="*70)
    print()

    text = "This is a test message for batch processing."
    batch_sizes = [1, 2, 4, 8, 16, 32]

    print(f"{'Batch Size':>12} {'Latency (ms)':>15} {'Per-item (ms)':>15}")
    print("-"*70)

    for batch_size in batch_sizes:
        texts = [text] * batch_size

        # Warm-up
        encoder.encode(texts)

        # Benchmark
        latencies = []
        for _ in range(5):
            start = time.perf_counter()
            encoder.encode(texts)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        mean_latency = np.mean(latencies)
        per_item = mean_latency / batch_size

        print(f"{batch_size:>12} {mean_latency:>14.2f} {per_item:>14.2f}")

    print()


def main():
    """Main benchmark function."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark encoder latency")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for benchmarking"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Use dynamic int8 quantization (CPU only)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of runs per test"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Also run batch size benchmark"
    )

    args = parser.parse_args()

    # Create encoder
    print(f"Initializing encoder (device={args.device}, quantize={args.quantize})...")
    encoder = TextEncoder(
        device=args.device,
        use_quantization=args.quantize,
    )
    print()

    # Run main benchmark
    benchmark_encoder(encoder, num_runs=args.runs)

    # Run batch benchmark if requested
    if args.batch:
        benchmark_batch_sizes(encoder)


if __name__ == "__main__":
    main()
