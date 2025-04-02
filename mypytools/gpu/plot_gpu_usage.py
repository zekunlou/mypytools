#!/usr/bin/env python3
"""
GPU Usage Visualizer

This script visualizes GPU usage data collected by the monitoring script.
It creates plots for GPU utilization, temperature, and memory usage over time.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


def moving_average(x, window_size=1):
    """
    Apply a moving average filter to smooth data.

    Args:
        x (numpy.ndarray): Input array to be smoothed
        window_size (int): Size of the moving average window

    Returns:
        numpy.ndarray: Smoothed array
    """
    if window_size <= 1:
        return x

    return np.convolve(x, np.ones(window_size) / window_size, "same")


def visualize_gpu_usage(input_fpath, output_fpath, moving_avg_window=1, time_unit="seconds", figsize=(8, 8)):
    """
    Generate visualizations of GPU usage based on monitoring logs.

    Args:
        input_fpath (str): Path to the GPU monitoring log file
        output_fpath (str): Fpath to save the output figures
        moving_avg_window (int): Window size for moving average smoothing
        time_unit (str): Unit for time axis ('seconds', 'minutes', or 'hours')
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_fpath), exist_ok=True)

    # Load data from log file
    records = []
    with open(input_fpath, "r") as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line: {line.strip()}")

    # Create pandas DataFrame
    df = pd.DataFrame(records)

    # Convert time based on the selected unit
    time_divisor = 1.0  # seconds
    if time_unit == "minutes":
        time_divisor = 60.0
        x_label = "Time (minutes)"
    elif time_unit == "hours":
        time_divisor = 3600.0
        x_label = "Time (hours)"
    else:  # default: seconds
        x_label = "Time (seconds)"

    df["time"] = df["timestamp"] / time_divisor

    # Calculate max memory across all GPUs (for consistent y-axis)
    max_memory_gb = df["memory_total"].max() / (1024**3)

    # Group by GPU ID
    gpu_groups = df.groupby("gpu_id")
    num_gpus = len(gpu_groups)

    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Plot GPU utilization for each GPU
    for i, (gpu_id, group) in enumerate(gpu_groups):
        color = f"C{i}"

        # Plot utilization with moving average
        utilization = group["gpu_utilization"].values
        time_values = group["time"].values
        smooth_util = moving_average(utilization, moving_avg_window)

        axes[0].plot(time_values, smooth_util, color=color, linestyle="-", label=f"GPU {gpu_id}")

    axes[0].set_title("GPU Utilization (%)")
    axes[0].set_ylabel("Utilization (%)")
    axes[0].grid(True, linestyle="--", alpha=0.7)
    axes[0].legend(loc="upper right")
    axes[0].set_ylim(0, 105)  # A bit of headroom above 100%
    axes[0].set_yticks(np.arange(0, 105, 10))  # 10% ticks

    # Plot memory usage for each GPU
    for i, (gpu_id, group) in enumerate(gpu_groups):
        color = f"C{i}"

        # Memory used in GB
        memory_used_gb = group["memory_used"].values / (1024**3)
        time_values = group["time"].values

        axes[1].plot(time_values, memory_used_gb, color=color, linestyle="-", label=f"GPU {gpu_id}")

        # Add horizontal line for total memory
        memory_total_gb = group["memory_total"].iloc[0] / (1024**3)
        axes[1].axhline(memory_total_gb, color=color, linestyle=":", alpha=0.7, label=f"GPU {gpu_id} Total")

    axes[1].set_title("GPU Memory Usage")
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel("Memory (GB)")
    axes[1].grid(True, linestyle="--", alpha=0.7)
    axes[1].legend(loc="upper right")
    axes[1].set_ylim(0, max_memory_gb * 1.05)  # 5% headroom

    # Add x-axis label only to bottom subplot
    axes[1].set_xlabel(x_label)

    # Ensure integer ticks for GPU IDs
    for ax in axes:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_fpath, dpi=150)
    print(f"Visualization saved to: {output_fpath}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize GPU usage data from logs")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to the GPU monitoring log file")
    parser.add_argument("--output", "-o", type=str, default="./gpu_usage.png", help="Fpath to save the output figures")
    parser.add_argument("--moving-avg", "-m", type=int, default=1, help="Window size for moving average smoothing")
    parser.add_argument(
        "--time-unit",
        "-t",
        type=str,
        default="seconds",
        choices=["seconds", "minutes", "hours"],
        help="Unit for time axis",
    )
    parser.add_argument("--figsize", type=str, default="8,8", help="Figure size in inches, format: width,height")

    args = parser.parse_args()

    # Parse figsize as tuple
    width, height = map(float, args.figsize.split(","))
    figsize = (width, height)

    df = visualize_gpu_usage(args.input, args.output, args.moving_avg, args.time_unit, figsize)
