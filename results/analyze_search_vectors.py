#!/usr/bin/env python3
"""
Statistical Analysis of Vector Search Privacy Pattern Test Results
Performs analysis and visualizations for privacy-preserving vector search patterns.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Decision thresholds from test plan section 7
VECTOR_SEARCH_RECALL_THRESHOLD = 0.95  # >= 95% recall@10
VECTOR_SEARCH_LATENCY_THRESHOLD_MS = 50  # <= 50 ms median query latency


def load_results(json_path: Path) -> pd.DataFrame:
    """Load vector search results from JSON and convert to DataFrame."""
    with open(json_path, 'r') as f:
        results = json.load(f)

    records = []
    for r in results:
        search_stats = r.get('search_stats', {})

        record = {
            'pattern': r['pattern'],
            'median_latency_ms': search_stats.get('median_s', 0) * 1e3 if search_stats else None,
            'mean_latency_ms': search_stats.get('mean_s', 0) * 1e3 if search_stats else None,
            'recall_at_10': r.get('recall_at_10', None),
            'storage_expansion': r.get('storage_expansion', 1.0),
        }

        # Add pattern-specific fields
        if 'tee_overhead_ms' in r:
            record['tee_overhead_ms'] = r['tee_overhead_ms']
        if 'dimensions' in r:
            record['dimensions'] = r['dimensions']
        if 'num_embeddings' in r:
            record['num_embeddings'] = r['num_embeddings']

        records.append(record)

    df = pd.DataFrame(records)
    return df


def check_hypothesis_satisfaction(df: pd.DataFrame) -> Dict:
    """Check if results satisfy vector search thresholds."""
    results = {}

    for _, row in df.iterrows():
        pattern = row['pattern']
        recall = row.get('recall_at_10')
        latency_ms = row.get('median_latency_ms')

        # Handle NaN/None values safely
        is_recall_nan = pd.isna(recall)
        is_latency_nan = pd.isna(latency_ms)

        meets_recall = (not is_recall_nan and recall >= VECTOR_SEARCH_RECALL_THRESHOLD) or is_recall_nan
        meets_latency = (not is_latency_nan and latency_ms <= VECTOR_SEARCH_LATENCY_THRESHOLD_MS) or is_latency_nan

        # For patterns without recall (like HE), mark as N/A
        if is_recall_nan:
            overall_accept = "N/A (exploratory)"
        else:
            overall_accept = meets_recall and meets_latency

        results[pattern] = {
            'recall_at_10': recall,
            'median_latency_ms': latency_ms,
            'meets_recall_threshold': meets_recall if not is_recall_nan else None,
            'meets_latency_threshold': meets_latency if not is_latency_nan else None,
            'overall_accept': overall_accept,
        }

    return results


def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """Create comprehensive visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter out patterns with missing data
    df_plot = df[df['median_latency_ms'].notna()].copy()

    if len(df_plot) == 0:
        print("Warning: No data available for visualization")
        return

    # 1. Latency Comparison
    fig, ax = plt.subplots(figsize=(14, 8))

    patterns = df_plot['pattern'].values
    latencies = df_plot['median_latency_ms'].values

    x = np.arange(len(patterns))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#6A994E']

    bars = ax.bar(x, latencies, width=0.6, color=colors[:len(patterns)], alpha=0.8)

    ax.axhline(y=VECTOR_SEARCH_LATENCY_THRESHOLD_MS, color='r', linestyle='--',
               linewidth=2, label=f'Threshold ({VECTOR_SEARCH_LATENCY_THRESHOLD_MS} ms)')

    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Privacy Pattern', fontsize=12, fontweight='bold')
    ax.set_ylabel('Median Query Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Vector Search Privacy Pattern Latency Comparison',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace('_', ' ').title() for p in patterns], rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'vector_search_latency_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Recall@10 Comparison (if available)
    df_recall = df[df['recall_at_10'].notna()].copy()

    if len(df_recall) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))

        patterns_recall = df_recall['pattern'].values
        recalls = df_recall['recall_at_10'].values

        x = np.arange(len(patterns_recall))

        bars = ax.bar(x, recalls, width=0.6, color='#2E86AB', alpha=0.8)

        ax.axhline(y=VECTOR_SEARCH_RECALL_THRESHOLD, color='r', linestyle='--',
                   linewidth=2, label=f'Threshold ({VECTOR_SEARCH_RECALL_THRESHOLD * 100}%)')

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_xlabel('Privacy Pattern', fontsize=12, fontweight='bold')
        ax.set_ylabel('Recall@10', fontsize=12, fontweight='bold')
        ax.set_title('Vector Search Privacy Pattern Recall@10 Comparison',
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([p.replace('_', ' ').title() for p in patterns_recall], rotation=45, ha='right')
        ax.set_ylim([0, 1.1])
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'vector_search_recall_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 3. Storage Expansion Comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    patterns = df['pattern'].values
    expansions = df['storage_expansion'].values

    x = np.arange(len(patterns))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#6A994E']

    bars = ax.bar(x, expansions, width=0.6, color=colors[:len(patterns)], alpha=0.8)

    ax.axhline(y=1.0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='No expansion (1.0x)')

    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.2f}x',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Privacy Pattern', fontsize=12, fontweight='bold')
    ax.set_ylabel('Storage Expansion Factor', fontsize=12, fontweight='bold')
    ax.set_title('Vector Search Privacy Pattern Storage Expansion',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace('_', ' ').title() for p in patterns], rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'vector_search_storage_expansion.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Visualizations saved to: {output_dir}")


def generate_report(df: pd.DataFrame, hypothesis_results: Dict, output_path: Path):
    """Generate comprehensive analysis report."""
    report = []
    report.append("=" * 80)
    report.append("VECTOR SEARCH PRIVACY PATTERN ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")

    report.append("EXECUTIVE SUMMARY")
    report.append("-" * 80)
    report.append("This analysis evaluates privacy-preserving vector search patterns")
    report.append("for perception model embeddings in autonomous vehicle systems.")
    report.append("")

    report.append("KEY FINDINGS")
    report.append("-" * 80)
    report.append("")

    report.append("HYPOTHESIS SATISFACTION")
    report.append(f"Threshold: >= {VECTOR_SEARCH_RECALL_THRESHOLD * 100}% recall@10, <= {VECTOR_SEARCH_LATENCY_THRESHOLD_MS} ms latency")
    report.append("")

    for pattern, result in hypothesis_results.items():
        status = result['overall_accept']
        if status == "N/A (exploratory)":
            report.append(f"{pattern.replace('_', ' ').title()}: {status}")
        else:
            status_str = "✓ ACCEPTS" if status else "✗ REJECTS"
            report.append(f"{pattern.replace('_', ' ').title()}: {status_str}")
            
            if result['recall_at_10'] is not None and not pd.isna(result['recall_at_10']):
                report.append(f"  - Recall@10: {result['recall_at_10']:.3f} "
                             f"({'meets' if result['meets_recall_threshold'] else 'fails'} threshold)")
            if result['median_latency_ms'] is not None and not pd.isna(result['median_latency_ms']):
                report.append(f"  - Latency: {result['median_latency_ms']:.2f} ms "
                             f"({'meets' if result['meets_latency_threshold'] else 'fails'} threshold)")
        report.append("")

    report.append("=" * 80)

    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"Analysis report saved to: {output_path}")


def main():
    """Main analysis function."""
    print("=" * 80)
    print("Vector Search Privacy Pattern Analysis")
    print("=" * 80)
    print()

    # Load data
    results_path = Path("test_data/vector_search_results.json")
    if not results_path.exists():
        print(f"Error: {results_path} not found.")
        print("Please run vector search tests first.")
        return

    print("Loading results...")
    df = load_results(results_path)
    print(f"Loaded {len(df)} test results")
    print()

    # Check hypothesis satisfaction
    print("Checking hypothesis satisfaction...")
    hypothesis_results = check_hypothesis_satisfaction(df)
    for pattern, result in hypothesis_results.items():
        status = result['overall_accept']
        print(f"  {pattern.replace('_', ' ').title()}: {status}")
        if result['recall_at_10'] is not None and not pd.isna(result['recall_at_10']):
            print(f"    Recall@10: {result['recall_at_10']:.3f} "
                  f"(threshold: >={VECTOR_SEARCH_RECALL_THRESHOLD})")
        if result['median_latency_ms'] is not None and not pd.isna(result['median_latency_ms']):
            print(f"    Latency: {result['median_latency_ms']:.2f} ms "
                  f"(threshold: <={VECTOR_SEARCH_LATENCY_THRESHOLD_MS} ms)")
    print()

    # Create visualizations
    print("Creating visualizations...")
    viz_dir = Path("analysis_output")
    create_visualizations(df, viz_dir)
    print()

    # Generate report
    print("Generating analysis report...")
    report_path = viz_dir / "vector_search_analysis_report.txt"
    generate_report(df, hypothesis_results, report_path)
    print()

    print("=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  - Visualizations: {viz_dir}/")
    print(f"  - Report: {report_path}")


if __name__ == "__main__":
    main()