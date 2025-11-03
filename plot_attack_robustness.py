#!/usr/bin/env python3
"""
Plot TPR@FPR=1% for different attacks and strengths.

This script reads attack evaluation results from JSON files and creates plots showing
how different attack types and strengths affect watermark detection performance.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Attack parameter mappings
ATTACK_PARAMS = {
    'none': {'param_name': 'strength', 'values': [0]},
    'jpeg': {'param_name': 'quality', 'values': [90, 80, 70, 60, 50, 40, 30, 20, 10, 0]},
    'blurring': {'param_name': 'kernel_size', 'values': [4, 7, 10, 13, 16, 19]},
    'noise': {'param_name': 'std_fraction', 'values': [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]},
    'color_jitter': {'param_name': 'brightness', 'values': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]},
    'cropping': {'param_name': 'scale', 'values': [0.90, 0.80, 0.70, 0.60, 0.50]},
}

# Attack display names
ATTACK_DISPLAY_NAMES = {
    'none': 'No Attack',
    'jpeg': 'JPEG Compression',
    'blurring': 'Gaussian Blur',
    'noise': 'Gaussian Noise',
    'color_jitter': 'Color Jitter',
    'cropping': 'Cropping',
}

# Attack colors for consistent plotting
ATTACK_COLORS = {
    'none': '#2E86AB',
    'jpeg': '#A23B72',
    'blurring': '#F18F01',
    'noise': '#C73E1D',
    'color_jitter': '#6A994E',
    'cropping': '#BC4B51',
}

# Attack markers
ATTACK_MARKERS = {
    'none': 'o',
    'jpeg': 's',
    'blurring': '^',
    'noise': 'D',
    'color_jitter': 'v',
    'cropping': 'p',
}


def extract_attack_info(filename):
    """Extract attack type and parameter value from filename."""
    filename = filename.replace('.json', '')

    if filename == 'attack_none':
        return 'none', 0

    # JPEG: attack_jpeg_q90
    if 'jpeg_q' in filename:
        quality = int(filename.split('_q')[1])
        return 'jpeg', quality

    # Blurring: attack_blur_k19
    if 'blur_k' in filename:
        kernel = int(filename.split('_k')[1])
        return 'blurring', kernel

    # Noise: attack_noise_std0_1
    if 'noise_std' in filename:
        std_str = filename.split('_std')[1].replace('_', '.')
        std = float(std_str)
        return 'noise', std

    # Color jitter: attack_colorjitter_b2_5
    if 'colorjitter_b' in filename:
        brightness_str = filename.split('_b')[1].replace('_', '.')
        brightness = float(brightness_str)
        return 'color_jitter', brightness

    # Cropping: attack_crop_s50
    if 'crop_s' in filename:
        scale = int(filename.split('_s')[1])
        scale_fraction = scale / 100.0
        return 'cropping', scale_fraction

    return None, None


def load_attack_results(base_dir):
    """Load all attack results from the directory structure.

    Returns:
        dict: Organized as results[attack_type][resolution][algorithm] = [(param_value, tpr_at_fpr), ...]
    """
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Warning: Directory {base_dir} does not exist")
        return results

    # Traverse: {algorithm}/{delta}/{resolution}/attack_*.json
    # Or for pairwise: {algorithm}/{resolution}/attack_*.json
    for algo_dir in base_path.iterdir():
        if not algo_dir.is_dir():
            continue
        algorithm = algo_dir.name

        # Check if this directory contains resolution folders directly (e.g., pairwise)
        # or if it contains delta subdirectories (e.g., spectral/delta2.0/512)
        subdirs = [d for d in algo_dir.iterdir() if d.is_dir()]

        # Check if subdirs look like resolution folders (numeric names)
        if subdirs and all(d.name.replace('px', '').isdigit() for d in subdirs):
            # Direct structure: {algorithm}/{resolution}/attack_*.json
            for res_dir in subdirs:
                resolution = res_dir.name
                delta = 'baseline'  # No delta for pairwise

                # Load all attack JSON files
                for json_file in res_dir.glob('attack_*.json'):
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)

                        # Extract attack info
                        attack_type, param_value = extract_attack_info(json_file.name)
                        if attack_type is None:
                            print(f"Warning: Could not parse filename {json_file.name}")
                            continue

                        # Get TPR@FPR
                        tpr_at_fpr = data['summary'].get('tpr_at_fpr', None)
                        if tpr_at_fpr is None:
                            print(f"Warning: No tpr_at_fpr in {json_file}")
                            continue

                        # Create algorithm label
                        algo_label = algorithm

                        # Store result: organized by attack type, then resolution, then algorithm
                        results[attack_type][resolution][algo_label].append({
                            'param_value': param_value,
                            'tpr_at_fpr': tpr_at_fpr,
                            'file': str(json_file)
                        })

                    except Exception as e:
                        print(f"Error loading {json_file}: {e}")
        else:
            # Nested structure: {algorithm}/{delta}/{resolution}/attack_*.json
            for delta_dir in algo_dir.iterdir():
                if not delta_dir.is_dir():
                    continue
                delta = delta_dir.name

                for res_dir in delta_dir.iterdir():
                    if not res_dir.is_dir():
                        continue
                    resolution = res_dir.name

                    # Load all attack JSON files
                    for json_file in res_dir.glob('attack_*.json'):
                        try:
                            with open(json_file, 'r') as f:
                                data = json.load(f)

                            # Extract attack info
                            attack_type, param_value = extract_attack_info(json_file.name)
                            if attack_type is None:
                                print(f"Warning: Could not parse filename {json_file.name}")
                                continue

                            # Get TPR@FPR
                            tpr_at_fpr = data['summary'].get('tpr_at_fpr', None)
                            if tpr_at_fpr is None:
                                print(f"Warning: No tpr_at_fpr in {json_file}")
                                continue

                            # Create algorithm label with delta info
                            if delta == 'baseline' or delta == '0.0':
                                algo_label = f'{algorithm} (baseline)'
                            else:
                                algo_label = f'{algorithm} (δ={delta.replace("delta", "")})'

                            # Store result: organized by attack type, then resolution, then algorithm
                            results[attack_type][resolution][algo_label].append({
                                'param_value': param_value,
                                'tpr_at_fpr': tpr_at_fpr,
                                'file': str(json_file)
                            })

                        except Exception as e:
                            print(f"Error loading {json_file}: {e}")

    return results


def create_attack_plot(attack_type, resolution, resolution_data, output_dir, task_type):
    """Create a plot for a specific attack showing all algorithms.

    Args:
        attack_type: Type of attack (e.g., 'jpeg', 'blurring')
        resolution: Resolution (e.g., '256', '512')
        resolution_data: Dict mapping algorithm labels to list of data points
        output_dir: Output directory for plots
        task_type: 't2i' or 'c2i'
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Colors for different algorithms
    algo_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    color_idx = 0

    # Plot each algorithm
    for algo_label in sorted(resolution_data.keys()):
        data_points = resolution_data[algo_label]

        # Sort by parameter value
        data_points = sorted(data_points, key=lambda x: x['param_value'])

        param_values = [d['param_value'] for d in data_points]
        tpr_values = [d['tpr_at_fpr'] for d in data_points]

        # Plot line
        ax.plot(param_values, tpr_values,
                label=algo_label,
                marker='o',
                color=algo_colors[color_idx % 10],
                linewidth=2.5,
                markersize=8,
                alpha=0.8)

        color_idx += 1

    # Customize plot based on attack type
    display_name = ATTACK_DISPLAY_NAMES.get(attack_type, attack_type)

    if attack_type in ATTACK_PARAMS:
        param_info = ATTACK_PARAMS[attack_type]
        xlabel = param_info['param_name'].replace('_', ' ').title()
    else:
        xlabel = 'Attack Strength'

    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel('TPR @ FPR = 1%', fontsize=14, fontweight='bold')
    ax.set_title(f'{task_type.upper()} - {display_name} Attack Robustness ({resolution}px)',
                 fontsize=16, fontweight='bold', pad=20)

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='best', framealpha=0.9)

    # Set y-axis limits
    ax.set_ylim([0, 1.05])

    # Add horizontal line at y=0.5 for reference
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    plt.tight_layout()

    # Save plot
    output_filename = f'{task_type}_{attack_type}_{resolution}_robustness.png'
    output_path = output_dir / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved plot: {output_path}")


def main():
    # Setup directories
    base_dir = Path('/nfshomes/anirudhs/GraphWatermark')
    results_dir = base_dir / 'results'
    output_dir = base_dir / 'plots'
    output_dir.mkdir(exist_ok=True)

    # Process both T2I and C2I results
    for task_type in ['attacks_t2i', 'attacks_c2i']:
        task_results_dir = results_dir / task_type

        print(f"\n{'='*80}")
        print(f"Processing {task_type.upper()}")
        print(f"{'='*80}")

        # Load results: organized as results[attack_type][resolution][algorithm]
        results = load_attack_results(task_results_dir)

        if not results:
            print(f"No results found in {task_results_dir}")
            continue

        # Count configurations
        total_configs = sum(len(res_dict) for attack_data in results.values() for res_dict in attack_data.values())
        print(f"Found {len(results)} attack types, {total_configs} total configurations")

        # Create plots: one plot per attack type per resolution
        for attack_type in sorted(results.keys()):
            print(f"\n  Attack: {attack_type}")

            for resolution in sorted(results[attack_type].keys()):
                algo_data = results[attack_type][resolution]
                print(f"    Resolution {resolution}: {len(algo_data)} algorithms")

                # Create plot for this attack type and resolution
                create_attack_plot(
                    attack_type=attack_type,
                    resolution=resolution,
                    resolution_data=algo_data,
                    output_dir=output_dir,
                    task_type=task_type.replace('attacks_', '')
                )

    print(f"\n{'='*80}")
    print(f"All plots saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
