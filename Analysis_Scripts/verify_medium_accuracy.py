#!/usr/bin/env python3
"""
Verify Medium Difficulty Accuracy
==================================

This script checks if the calibrated 70% threshold actually produces
70% accuracy during test phase medium trials.

Usage:
    python verify_medium_accuracy.py [data_file.csv]
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def analyze_medium_accuracy(csv_file):
    """
    Analyze accuracy specifically for test_medium trials.
    """
    print("=" * 80)
    print("MEDIUM DIFFICULTY ACCURACY VERIFICATION")
    print("=" * 80)
    print(f"\nAnalyzing: {csv_file.name}")
    
    # Load data
    df = pd.read_csv(csv_file)
    
    # Filter to test phase
    test_df = df[df['phase'].fillna("").astype(str).str.contains('test', case=False)].copy()
    
    if len(test_df) == 0:
        print("[ERROR] No test phase data found!")
        return
    
    print(f"\n[OK] Found {len(test_df)} test phase trials")
    
    # Filter to medium difficulty trials only
    if 'difficulty_type' not in test_df.columns:
        print("[ERROR] No 'difficulty_type' column found!")
        return
    
    medium_df = test_df[test_df['difficulty_type'] == 'test_medium'].copy()
    
    if len(medium_df) == 0:
        print("[ERROR] No medium difficulty trials found!")
        return
    
    print(f"[OK] Found {len(medium_df)} medium difficulty trials")
    
    # Exclude timeout trials
    if 'resp_shape' in medium_df.columns:
        timeout_mask = medium_df['resp_shape'] == 'timeout'
        n_timeouts = timeout_mask.sum()
        medium_df = medium_df[~timeout_mask].copy()
        print(f"  - Excluded {n_timeouts} timeout trials")
        print(f"  - Analyzing {len(medium_df)} valid medium trials")
    
    if len(medium_df) == 0:
        print("[ERROR] No valid medium trials after excluding timeouts!")
        return
    
    # Calculate accuracy
    if 'accuracy' not in medium_df.columns:
        print("[ERROR] No 'accuracy' column found!")
        return
    
    overall_acc = medium_df['accuracy'].mean()
    overall_std = medium_df['accuracy'].std()
    overall_sem = overall_std / np.sqrt(len(medium_df))
    
    print("\n" + "=" * 80)
    print("RESULTS: MEDIUM DIFFICULTY ACCURACY")
    print("=" * 80)
    print(f"\n[RESULT] Overall Medium Accuracy: {overall_acc:.1%} +/- {overall_sem:.1%} (SEM)")
    print(f"   Target: 70%")
    print(f"   Difference: {(overall_acc - 0.70)*100:+.1f} percentage points")
    
    # Assessment
    if abs(overall_acc - 0.70) < 0.05:
        status = "[EXCELLENT] Within 5% of target"
    elif abs(overall_acc - 0.70) < 0.10:
        status = "[GOOD] Within 10% of target"
    elif abs(overall_acc - 0.70) < 0.15:
        status = "[ACCEPTABLE] Within 15% of target"
    else:
        status = "[POOR] More than 15% from target"
    
    print(f"\n{status}\n")
    
    # Break down by angle condition if available
    if 'angle_bias' in medium_df.columns:
        print("-" * 80)
        print("BREAKDOWN BY ANGLE CONDITION")
        print("-" * 80)
        
        for angle in sorted(medium_df['angle_bias'].unique()):
            angle_data = medium_df[medium_df['angle_bias'] == angle]
            angle_acc = angle_data['accuracy'].mean()
            angle_sem = angle_data['accuracy'].std() / np.sqrt(len(angle_data))
            
            print(f"\n{int(angle)}° Rotation:")
            print(f"  Accuracy: {angle_acc:.1%} ± {angle_sem:.1%} (n={len(angle_data)})")
            print(f"  Difference from 70%: {(angle_acc - 0.70)*100:+.1f} pp")
    
    # Break down by cue color if available
    if 'cue_color' in medium_df.columns:
        print("\n" + "-" * 80)
        print("BREAKDOWN BY CUE COLOR (This should be identical difficulty!)")
        print("-" * 80)
        
        for color in sorted(medium_df['cue_color'].unique()):
            color_data = medium_df[medium_df['cue_color'] == color]
            color_acc = color_data['accuracy'].mean()
            color_sem = color_data['accuracy'].std() / np.sqrt(len(color_data))
            
            print(f"\n{color.capitalize()} cue:")
            print(f"  Accuracy: {color_acc:.1%} ± {color_sem:.1%} (n={len(color_data)})")
            print(f"  Difference from 70%: {(color_acc - 0.70)*100:+.1f} pp")
        
        # Statistical test for difference between colors
        colors = sorted(medium_df['cue_color'].unique())
        if len(colors) == 2:
            from scipy import stats
            color1_data = medium_df[medium_df['cue_color'] == colors[0]]['accuracy']
            color2_data = medium_df[medium_df['cue_color'] == colors[1]]['accuracy']
            
            t_stat, p_val = stats.ttest_ind(color1_data, color2_data)
            
            print(f"\n[STATS] Statistical Test (t-test):")
            print(f"   {colors[0]} vs {colors[1]}: t={t_stat:.3f}, p={p_val:.4f}")
            
            if p_val < 0.05:
                print(f"   [WARNING] SIGNIFICANT DIFFERENCE (p < 0.05)")
                print(f"   This suggests the two colors may have different objective difficulty!")
            else:
                print(f"   [OK] No significant difference (p >= 0.05)")
                print(f"   The two colors have statistically equivalent difficulty.")
    
    # Check prop_used values to verify they're identical
    if 'prop_used' in medium_df.columns and 'cue_color' in medium_df.columns:
        print("\n" + "-" * 80)
        print("PROP_USED VERIFICATION (Should be identical for all medium trials)")
        print("-" * 80)
        
        for color in sorted(medium_df['cue_color'].unique()):
            color_data = medium_df[medium_df['cue_color'] == color]
            prop_mean = color_data['prop_used'].mean()
            prop_std = color_data['prop_used'].std()
            prop_min = color_data['prop_used'].min()
            prop_max = color_data['prop_used'].max()
            
            print(f"\n{color.capitalize()} cue:")
            print(f"  prop_used: {prop_mean:.4f} ± {prop_std:.4f}")
            print(f"  Range: [{prop_min:.4f}, {prop_max:.4f}]")
        
        # Check if all prop_used values are identical
        unique_props = medium_df['prop_used'].nunique()
        if unique_props == 1:
            print(f"\n[OK] All medium trials use IDENTICAL prop_used value")
        else:
            print(f"\n[WARNING] Found {unique_props} different prop_used values")
            print(f"   (Small variations may be due to rounding or adaptive adjustments)")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80 + "\n")

def main():
    if len(sys.argv) > 1:
        csv_file = Path(sys.argv[1])
    else:
        # Default to the subjects directory
        script_dir = Path(__file__).resolve().parent
        subjects_dir = script_dir.parent / "Main_Experiment" / "data" / "subjects"
        
        # Find CSV files
        csv_files = [f for f in subjects_dir.glob("*.csv") if "kinematics" not in f.name]
        
        if not csv_files:
            print("[ERROR] No data files found!")
            return
        
        if len(csv_files) == 1:
            csv_file = csv_files[0]
        else:
            print(f"Found {len(csv_files)} data files:")
            for i, f in enumerate(csv_files, 1):
                print(f"  {i}. {f.name}")
            
            choice = input("\nEnter file number to analyze (or 'all' for all files): ").strip()
            
            if choice.lower() == 'all':
                for csv_file in csv_files:
                    analyze_medium_accuracy(csv_file)
                    print("\n")
                return
            else:
                try:
                    idx = int(choice) - 1
                    csv_file = csv_files[idx]
                except (ValueError, IndexError):
                    print("[ERROR] Invalid choice!")
                    return
    
    if not csv_file.exists():
        print(f"[ERROR] File not found: {csv_file}")
        return
    
    analyze_medium_accuracy(csv_file)

if __name__ == "__main__":
    main()

