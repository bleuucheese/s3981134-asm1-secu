#!/usr/bin/env python3
"""Analyze homomorphic encryption telemetry outputs and generate reports."""

import json
import warnings
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# Plot styling
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10

# Decision thresholds from test plan section 7
HE_TOTAL_TIME_THRESHOLD_S = 5  # <= 5 seconds total time for mean/variance
HE_ERROR_THRESHOLD = 1e-3  # <= 1e-3 relative error


def load_results(json_path: Path) -> pd.DataFrame:
    """Load homomorphic encryption results and return a DataFrame."""
    with open(json_path, "r", encoding="utf-8") as handle:
        results = json.load(handle)

    rows: List[Dict] = []
    for sample in results:
        record = {
            "task": sample["task"],
            "encrypt_time_s": sample.get("encrypt_time_s"),
            "compute_time_s": sample.get("compute_time_s"),
            "decrypt_time_s": sample.get("decrypt_time_s"),
            "total_time_s": sample.get("total_time_s"),
            "max_error": sample.get("max_error"),
            "relative_error": sample.get("relative_error"),
            "expansion_factor": sample.get("expansion_factor"),
            "ciphertext_size_bytes": sample.get("ciphertext_size_bytes"),
        }

        if "num_rows" in sample:
            record["num_rows"] = sample["num_rows"]
        if "num_columns" in sample:
            record["num_columns"] = sample["num_columns"]
        if "dimensions" in sample:
            record["dimensions"] = sample["dimensions"]
        if "runs" in sample:
            record["runs"] = sample["runs"]

        rows.append(record)

    return pd.DataFrame(rows)


def check_hypothesis_satisfaction(df: pd.DataFrame) -> Dict:
    """Evaluate each task against timing and accuracy requirements."""
    results: Dict[str, Dict] = {}

    for _, row in df.iterrows():
        task = row["task"]
        total_time = row.get("total_time_s")
        rel_error = row.get("relative_error")

        meets_time = (
            (total_time is not None and total_time <= HE_TOTAL_TIME_THRESHOLD_S)
            or total_time is None
        )
        meets_error = (
            (rel_error is not None and rel_error <= HE_ERROR_THRESHOLD)
            or rel_error is None
        )

        if total_time is not None and rel_error is not None:
            overall = meets_time and meets_error
        else:
            overall = "N/A"

        results[task] = {
            "total_time_s": total_time,
            "relative_error": rel_error,
            "meets_time_threshold": meets_time if total_time is not None else None,
            "meets_error_threshold": meets_error if rel_error is not None else None,
            "overall_accept": overall,
        }

    return results


def _bar_label(ax, bars, fmt):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )


def create_visualizations(df: pd.DataFrame, output_dir: Path) -> None:
    """Build timing, error, and expansion plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    mean_var = df[df["task"] == "encrypted_mean_variance"].copy()
    if len(mean_var) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        times = [
            mean_var.iloc[0].get("encrypt_time_s", 0),
            mean_var.iloc[0].get("compute_time_s", 0),
            mean_var.iloc[0].get("decrypt_time_s", 0),
        ]
        labels = ["Encrypt", "Compute", "Decrypt"]
        colors = ["#2E86AB", "#A23B72", "#F18F01"]
        bars = ax.bar(labels, times, color=colors, alpha=0.8)

        total_time = sum(times)
        ax.axhline(
            y=HE_TOTAL_TIME_THRESHOLD_S,
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Threshold ({HE_TOTAL_TIME_THRESHOLD_S} s)"
        )
        _bar_label(ax, bars, "{:.3f}s")

        ax.text(
            0.5,
            0.95,
            f"Total: {total_time:.3f} s",
            transform=ax.transAxes,
            ha="center",
            fontsize=12,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        ax.set_ylabel("Time (seconds)", fontsize=12, fontweight="bold")
        ax.set_title(
            "Homomorphic Encryption Time Breakdown\n(Encrypted Mean/Variance over 100k Rows)",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "homomorphic_time_breakdown.png", dpi=300, bbox_inches="tight")
        plt.close()

    error_df = df[df["relative_error"].notna()].copy()
    if len(error_df) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        tasks = error_df["task"].values
        errors = error_df["relative_error"].values
        bars = ax.bar(
            np.arange(len(tasks)),
            errors,
            width=0.6,
            color=["#2E86AB", "#A23B72"][: len(tasks)],
            alpha=0.8,
        )

        ax.axhline(
            y=HE_ERROR_THRESHOLD,
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Threshold ({HE_ERROR_THRESHOLD})"
        )
        ax.set_yscale("log")
        _bar_label(ax, bars, "{:.2e}")

        ax.set_xlabel("Task", fontsize=12, fontweight="bold")
        ax.set_ylabel("Relative Error (log scale)", fontsize=12, fontweight="bold")
        ax.set_title("Homomorphic Encryption Numerical Error", fontsize=14, fontweight="bold")
        ax.set_xticks(np.arange(len(tasks)))
        ax.set_xticklabels([task.replace("_", " " ).title() for task in tasks])
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "homomorphic_error_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

    expansion_df = df[df["expansion_factor"].notna()].copy()
    if len(expansion_df) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        tasks = expansion_df["task"].values
        expansions = expansion_df["expansion_factor"].values
        bars = ax.bar(
            np.arange(len(tasks)),
            expansions,
            width=0.6,
            color=["#2E86AB", "#A23B72"][: len(tasks)],
            alpha=0.8,
        )
        _bar_label(ax, bars, "{:.1f}x")

        ax.set_xlabel("Task", fontsize=12, fontweight="bold")
        ax.set_ylabel("Ciphertext Expansion Factor", fontsize=12, fontweight="bold")
        ax.set_title("Homomorphic Encryption Ciphertext Expansion", fontsize=14, fontweight="bold")
        ax.set_xticks(np.arange(len(tasks)))
        ax.set_xticklabels([task.replace("_", " " ).title() for task in tasks])
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "homomorphic_expansion.png", dpi=300, bbox_inches="tight")
        plt.close()

    print(f"Visualizations saved to: {output_dir}")


def generate_report(df: pd.DataFrame, hypothesis_results: Dict, output_path: Path) -> None:
    """Write a plain-text report summarizing KPIs per task."""
    report: List[str] = []
    report.append("=" * 80)
    report.append("HOMOMORPHIC ENCRYPTION ANALYTICS ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")

    report.append("EXECUTIVE SUMMARY")
    report.append("-" * 80)
    report.append("This analysis evaluates CKKS homomorphic encryption for")
    report.append("encrypted telemetry analytics in autonomous vehicle systems.")
    report.append("")

    report.append("KEY FINDINGS")
    report.append("-" * 80)
    report.append("")

    report.append("HYPOTHESIS SATISFACTION")
    report.append(
        f"Threshold: <= {HE_TOTAL_TIME_THRESHOLD_S} s total time, <= {HE_ERROR_THRESHOLD} relative error"
    )
    report.append("")

    for task, result in hypothesis_results.items():
        status = result["overall_accept"]
        if status == "N/A":
            summary = "N/A (insufficient data)"
        else:
            summary = "ACCEPTS" if status else "REJECTS"
        report.append(f"{task.replace('_', ' ').title()}: {summary}")

        if result["total_time_s"] is not None:
            report.append(f" - Total time: {result['total_time_s']:.3f} s")
        if result["relative_error"] is not None:
            report.append(f" - Relative error: {result['relative_error']:.2e}")
        report.append("")

    report.append("PERFORMANCE COMPARISON")
    report.append("-" * 80)
    report.append("")

    for _, row in df.iterrows():
        report.append(f"{row['task'].replace('_', ' ').title()}:")
        if row["encrypt_time_s"] is not None:
            report.append(f" - Encrypt: {row['encrypt_time_s']:.3f} s")
        if row["compute_time_s"] is not None:
            report.append(f" - Compute: {row['compute_time_s']:.3f} s")
        if row["decrypt_time_s"] is not None:
            report.append(f" - Decrypt: {row['decrypt_time_s']:.3f} s")
        if row["total_time_s"] is not None:
            report.append(f" - Total: {row['total_time_s']:.3f} s")
        if row["relative_error"] is not None:
            report.append(f" - Relative error: {row['relative_error']:.2e}")
        if row["expansion_factor"] is not None:
            report.append(f" - Expansion: {row['expansion_factor']:.1f}x")
        report.append("")

    report.append("=" * 80)

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report))
        handle.write("\n")

    print(f"Analysis report saved to: {output_path}")


def main() -> None:
    """Entry point for processing test outputs and generating artifacts."""
    print("=" * 80)
    print("Homomorphic Encryption Results Analysis")
    print("=" * 80)
    print()

    results_path = Path("test_data/homomorphic_results.json")
    if not results_path.exists():
        print(f"Error: {results_path} not found.")
        print("Please run tests/homomorphic/test_homomorphic.py first.")
        return

    print("Loading results...")
    df = load_results(results_path)
    print(f"Loaded {len(df)} test results")
    print()

    print("Checking hypothesis satisfaction...")
    hypothesis_results = check_hypothesis_satisfaction(df)
    for task, result in hypothesis_results.items():
        print(f" {task.replace('_', ' ').title()}: {result['overall_accept']}")
    print()

    print("Creating visualizations...")
    viz_dir = Path("analysis_output/homomorphic")
    create_visualizations(df, viz_dir)
    print()

    print("Generating analysis report...")
    report_path = viz_dir / "homomorphic_analysis_report.txt"
    generate_report(df, hypothesis_results, report_path)
    print()

    print("=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
