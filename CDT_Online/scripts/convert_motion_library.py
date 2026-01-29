#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_motion_library.py - Convert Motion Library to JSON for Pavlovia

This script converts the .npy motion library files to JSON format for use
in the online version of the CDT experiment on Pavlovia.

Features:
- Converts core_pool.npy to JSON
- Pre-selects best trajectories to reduce file size
- Validates trajectory quality
- Compresses output for faster loading

Author: CDT Online Conversion for Simon Knogler's PhD Project
"""

import numpy as np
import json
from pathlib import Path
import gzip
import shutil

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MOTION_LIB_DIR = PROJECT_ROOT / "Motion_Library"
OUTPUT_DIR = SCRIPT_DIR.parent / "resources"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Number of trajectories to include (balance between coverage and file size)
MAX_TRAJECTORIES = 800  # Reduced from full library for web performance

# Quality thresholds
MIN_MEAN_SPEED = 1.0
MAX_ZERO_MOVEMENT_RATIO = 0.3
MAX_JITTER_RATIO = 0.1
MAX_JERKINESS = 1.5

# =============================================================================
# Trajectory Quality Analysis
# =============================================================================

def analyze_trajectory_quality(trajectory):
    """Analyze trajectory quality metrics."""
    velocities = np.diff(trajectory, axis=0)
    if len(velocities) == 0:
        return {
            'mean_speed': 0,
            'std_speed': 0,
            'zero_movement_ratio': 1.0,
            'high_jitter_ratio': 0,
            'jerkiness': 0,
            'path_length': 0,
            'net_displacement': 0
        }
    
    speeds = np.linalg.norm(velocities, axis=1)
    mean_speed = np.mean(speeds)
    std_speed = np.std(speeds)
    
    # Quality metrics
    zero_movement_ratio = np.sum(speeds < 0.5) / len(speeds)
    high_jitter_ratio = np.sum(speeds > mean_speed + 3 * std_speed) / len(speeds) if std_speed > 0 else 0
    
    # Jerkiness (smoothness of direction changes)
    if len(velocities) > 1:
        unit_velocities = velocities / (speeds.reshape(-1, 1) + 1e-9)
        dot_products = np.sum(unit_velocities[:-1] * unit_velocities[1:], axis=1)
        angle_changes = np.arccos(np.clip(dot_products, -1, 1))
        jerkiness = np.std(angle_changes)
    else:
        jerkiness = 0
    
    # Path characteristics
    path_length = np.sum(speeds)
    net_displacement = np.linalg.norm(trajectory[-1] - trajectory[0]) if len(trajectory) > 1 else 0
    
    return {
        'mean_speed': float(mean_speed),
        'std_speed': float(std_speed),
        'zero_movement_ratio': float(zero_movement_ratio),
        'high_jitter_ratio': float(high_jitter_ratio),
        'jerkiness': float(jerkiness),
        'path_length': float(path_length),
        'net_displacement': float(net_displacement)
    }


def is_trajectory_valid(quality):
    """Check if trajectory meets quality thresholds."""
    if quality['mean_speed'] < MIN_MEAN_SPEED:
        return False, "mean_speed_too_low"
    if quality['zero_movement_ratio'] > MAX_ZERO_MOVEMENT_RATIO:
        return False, "too_much_zero_movement"
    if quality['high_jitter_ratio'] > MAX_JITTER_RATIO:
        return False, "too_much_jitter"
    if quality['jerkiness'] > MAX_JERKINESS:
        return False, "too_jerky"
    return True, "valid"


def score_trajectory(quality):
    """Score trajectory for ranking (higher is better)."""
    # Prefer trajectories with:
    # - Medium speed (not too slow, not too fast)
    # - Low zero movement
    # - Low jitter
    # - Smooth direction changes
    
    speed_score = 1.0 / (1.0 + abs(quality['mean_speed'] - 8.0))
    variability_score = 1.0 / (1.0 + abs(quality['std_speed'] - 3.0))
    smoothness_score = 1.0 / (1.0 + quality['jerkiness'])
    movement_score = 1.0 - quality['zero_movement_ratio']
    
    return speed_score * variability_score * smoothness_score * movement_score


# =============================================================================
# Conversion Functions
# =============================================================================

def convert_motion_library():
    """Convert motion library to JSON format with quality filtering."""
    
    print("=" * 70)
    print("MOTION LIBRARY CONVERSION FOR PAVLOVIA")
    print("=" * 70)
    print()
    
    # Load motion pool
    motion_pool_path = MOTION_LIB_DIR / "core_pool.npy"
    if not motion_pool_path.exists():
        raise FileNotFoundError(f"Motion pool not found: {motion_pool_path}")
    
    motion_pool = np.load(motion_pool_path)
    print(f"Loaded motion pool: {motion_pool.shape}")
    print(f"  - {motion_pool.shape[0]} trajectories")
    print(f"  - {motion_pool.shape[1]} frames each")
    print()
    
    # Analyze and score all trajectories
    print("Analyzing trajectory quality...")
    trajectory_data = []
    
    for i in range(len(motion_pool)):
        snippet = motion_pool[i]
        # Convert to cumulative positions for analysis
        trajectory = np.cumsum(snippet, axis=0)
        quality = analyze_trajectory_quality(trajectory)
        is_valid, reason = is_trajectory_valid(quality)
        score = score_trajectory(quality) if is_valid else 0
        
        trajectory_data.append({
            'index': i,
            'quality': quality,
            'is_valid': is_valid,
            'reason': reason,
            'score': score
        })
    
    # Filter valid trajectories
    valid_trajectories = [t for t in trajectory_data if t['is_valid']]
    invalid_count = len(trajectory_data) - len(valid_trajectories)
    
    print(f"Quality filtering results:")
    print(f"  - Valid: {len(valid_trajectories)}")
    print(f"  - Invalid: {invalid_count}")
    print()
    
    # Sort by score and select top trajectories
    valid_trajectories.sort(key=lambda x: x['score'], reverse=True)
    selected = valid_trajectories[:MAX_TRAJECTORIES]
    
    print(f"Selected top {len(selected)} trajectories for online version")
    print(f"  - Best score: {selected[0]['score']:.4f}")
    print(f"  - Worst selected score: {selected[-1]['score']:.4f}")
    print()
    
    # Extract selected trajectory data
    selected_indices = [t['index'] for t in selected]
    selected_motion_data = motion_pool[selected_indices].tolist()
    
    # Create output structure
    output = {
        'metadata': {
            'source': 'core_pool.npy',
            'total_original': len(motion_pool),
            'total_selected': len(selected),
            'frames_per_trajectory': motion_pool.shape[1],
            'selection_criteria': {
                'min_mean_speed': MIN_MEAN_SPEED,
                'max_zero_movement_ratio': MAX_ZERO_MOVEMENT_RATIO,
                'max_jitter_ratio': MAX_JITTER_RATIO,
                'max_jerkiness': MAX_JERKINESS
            }
        },
        'trajectories': selected_motion_data,
        'indices': selected_indices  # Original indices for reference
    }
    
    # Save as JSON
    json_path = OUTPUT_DIR / "motion_library.json"
    print(f"Saving to {json_path}...")
    
    with open(json_path, 'w') as f:
        json.dump(output, f)
    
    file_size_mb = json_path.stat().st_size / (1024 * 1024)
    print(f"  - File size: {file_size_mb:.2f} MB")
    
    # Also create compressed version
    gz_path = OUTPUT_DIR / "motion_library.json.gz"
    with open(json_path, 'rb') as f_in:
        with gzip.open(gz_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    gz_size_mb = gz_path.stat().st_size / (1024 * 1024)
    print(f"  - Compressed size: {gz_size_mb:.2f} MB")
    print()
    
    return output


def convert_supporting_files():
    """Convert other supporting files (already JSON, just copy)."""
    
    print("Copying supporting files...")
    
    # cluster_centroids.json
    centroids_src = MOTION_LIB_DIR / "cluster_centroids.json"
    centroids_dst = OUTPUT_DIR / "cluster_centroids.json"
    if centroids_src.exists():
        shutil.copy(centroids_src, centroids_dst)
        print(f"  - Copied: cluster_centroids.json")
    
    # scaler_params.json
    scaler_src = MOTION_LIB_DIR / "scaler_params.json"
    scaler_dst = OUTPUT_DIR / "scaler_params.json"
    if scaler_src.exists():
        shutil.copy(scaler_src, scaler_dst)
        print(f"  - Copied: scaler_params.json")
    
    print()


def create_precomputed_pairs():
    """Pre-compute trajectory pairs for faster online loading."""
    
    print("Creating pre-computed trajectory pairs...")
    
    # Load the converted motion library
    json_path = OUTPUT_DIR / "motion_library.json"
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    trajectories = data['trajectories']
    n_trajectories = len(trajectories)
    
    # Create pairs with similar characteristics
    # For simplicity, create random pairs ensuring no self-pairing
    np.random.seed(42)  # Reproducible
    
    n_pairs = 500  # Pre-compute 500 pairs
    pairs = []
    
    for _ in range(n_pairs):
        idx1 = np.random.randint(0, n_trajectories)
        idx2 = np.random.randint(0, n_trajectories)
        while idx2 == idx1:
            idx2 = np.random.randint(0, n_trajectories)
        pairs.append([int(idx1), int(idx2)])
    
    # Save pairs
    pairs_path = OUTPUT_DIR / "trajectory_pairs.json"
    with open(pairs_path, 'w') as f:
        json.dump({
            'n_pairs': n_pairs,
            'pairs': pairs
        }, f)
    
    print(f"  - Created {n_pairs} pre-computed pairs")
    print(f"  - Saved to: {pairs_path}")
    print()


# =============================================================================
# Main
# =============================================================================

def main():
    """Run full conversion process."""
    
    print()
    print("=" * 70)
    print("CDT MOTION LIBRARY CONVERSION FOR PAVLOVIA")
    print("=" * 70)
    print()
    print(f"Source: {MOTION_LIB_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print()
    
    # Convert motion library
    convert_motion_library()
    
    # Copy supporting files
    convert_supporting_files()
    
    # Create pre-computed pairs
    create_precomputed_pairs()
    
    print("=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)
    print()
    print("Output files:")
    for f in OUTPUT_DIR.iterdir():
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name}: {size_kb:.1f} KB")
    print()
    print("Next steps:")
    print("  1. Upload resources/ folder to Pavlovia")
    print("  2. Reference motion_library.json in PsychoPy Builder")
    print("  3. Use trajectory_pairs.json for pre-computed pairs")
    print()


if __name__ == "__main__":
    main()
