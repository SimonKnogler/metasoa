#!/usr/bin/env python3
"""
Analyze the relationship between trial difficulty, evidence accumulation, and response patterns.
This script examines whether easy trials (high prop) accumulate more evidence than difficult trials.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_evidence_difficulty_relationship():
    """
    Analyze how evidence accumulation relates to trial difficulty.
    Uses behavioral CSVs (not kinematics) from Main_Experiment/data/subjects.
    """
    print("EVIDENCE vs DIFFICULTY ANALYSIS")
    print("=" * 50)

    # Prefer subjects directory; fallback to data root if empty
    subjects_dir = Path("../Main_Experiment/data/subjects")
    data_dir = subjects_dir if subjects_dir.exists() else Path("../Main_Experiment/data")
    csv_files = [f for f in data_dir.glob("*.csv") if "kinematics" not in f.name]

    if not csv_files:
        print("No behavioral data CSV files found!")
        return

    for data_file in csv_files:
        print(f"\nAnalyzing: {data_file.name}")
        print("-" * 40)

        try:
            df = pd.read_csv(data_file)
            print(f"Total rows: {len(df)}")

            # Exclude timeout trials if present
            if 'resp_shape' in df.columns:
                df = df[df['resp_shape'] != 'timeout'].copy()

            # Map to expected column names if needed
            if 'stimulus_intensity_s' not in df.columns and 'prop_used' in df.columns:
                df['stimulus_intensity_s'] = df['prop_used']
            if 'response_correct' not in df.columns and 'accuracy' in df.columns:
                df['response_correct'] = df['accuracy']

            # Check required columns and evidence metrics
            required_cols = ['stimulus_intensity_s', 'response_correct']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Missing required columns: {missing_cols}")
                continue

            # Optional columns
            has_rt = 'rt_choice' in df.columns
            has_early = 'early_response' in df.columns
            has_evidence = all(c in df.columns for c in ['sum_evidence', 'mean_evidence', 'var_evidence'])
            if not has_evidence:
                print("Warning: Evidence metrics not found in behavioral CSV; only intensity-performance stats will be shown.")

            # Ensure numeric
            for c in ['stimulus_intensity_s', 'response_correct', 'rt_choice', 'early_response', 'sum_evidence', 'mean_evidence', 'var_evidence']:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')

            # Report stimulus range
            print(f"Stimulus intensity range: {df['stimulus_intensity_s'].min():.3f} – {df['stimulus_intensity_s'].max():.3f}")

            # Create difficulty bins
            df['difficulty_bin'] = pd.cut(df['stimulus_intensity_s'], bins=5,
                                          labels=['Very Easy', 'Easy', 'Medium', 'Hard', 'Very Hard'])

            # Group aggregations
            agg_spec = {
                'stimulus_intensity_s': ['mean', 'std'],
                'response_correct': ['count', 'mean'],
            }
            if has_rt:
                agg_spec['rt_choice'] = ['mean', 'std']
            if has_early:
                agg_spec['early_response'] = 'mean'
            if has_evidence:
                agg_spec['sum_evidence'] = ['mean', 'std']
                agg_spec['mean_evidence'] = ['mean', 'std']
                agg_spec['var_evidence'] = ['mean', 'std']

            difficulty_analysis = df.groupby('difficulty_bin').agg(agg_spec).round(3)
            print("\nPerformance by difficulty level:")
            print(difficulty_analysis)

            # Correlation analysis
            print("\nCorrelation analysis:")
            print(f"  Intensity vs Accuracy: {df['stimulus_intensity_s'].corr(df['response_correct']):.3f}")
            if has_rt:
                print(f"  Intensity vs RT: {df['stimulus_intensity_s'].corr(df['rt_choice']):.3f}")
            if has_early:
                print(f"  Intensity vs Early Response: {df['stimulus_intensity_s'].corr(df['early_response']):.3f}")
            if has_evidence:
                print(f"  Intensity vs Sum Evidence: {df['stimulus_intensity_s'].corr(df['sum_evidence']):.3f}")
                print(f"  Intensity vs Mean Evidence: {df['stimulus_intensity_s'].corr(df['mean_evidence']):.3f}")

            # Show sample rows
            sample_cols = ['stimulus_intensity_s', 'response_correct']
            if has_rt:
                sample_cols.append('rt_choice')
            if has_early:
                sample_cols.append('early_response')
            if has_evidence:
                sample_cols += ['sum_evidence', 'mean_evidence', 'var_evidence']
            print("\nSample trials:")
            print(df[sample_cols].head(10).to_string(index=False))

        except Exception as e:
            print(f"Error analyzing {data_file}: {e}")

def explain_evidence_difficulty_relationship():
    """
    Explain the theoretical relationship between evidence and difficulty.
    """
    print(f"\n{'='*60}")
    print("EVIDENCE vs DIFFICULTY THEORY")
    print(f"{'='*60}")
    
    print("""
THEORETICAL RELATIONSHIP:

1. EASY TRIALS (High prop, e.g., 0.8):
   - High control over target movement
   - Mouse movements align well with target direction
   - High cos_T (cosine similarity with target)
   - Low cos_D (cosine similarity with distractor)
   - Evidence = speed * (cos_T - cos_D) = HIGH POSITIVE
   - Result: High evidence accumulation, fast responses

2. DIFFICULT TRIALS (Low prop, e.g., 0.2):
   - Low control over target movement
   - Mouse movements don't align well with target
   - Low cos_T (cosine similarity with target)
   - High cos_D (cosine similarity with distractor)
   - Evidence = speed * (cos_T - cos_D) = LOW or NEGATIVE
   - Result: Low evidence accumulation, slow responses

3. EVIDENCE ACCUMULATION PATTERNS:
   - Easy trials: High positive evidence → Quick decision → Early response
   - Difficult trials: Low/negative evidence → Uncertain decision → Late response
   - BUT: Both run for full 5 seconds, so total evidence depends on:
     * How much evidence per frame (quality)
     * How many frames of movement (quantity)

4. KEY INSIGHT:
   - Easy trials: High evidence per frame × Fewer frames = High total evidence
   - Difficult trials: Low evidence per frame × More frames = Variable total evidence
   - The relationship depends on BOTH evidence quality AND movement duration

5. EXPECTED PATTERN:
   - Easy trials should have HIGHER total evidence (sum_evidence)
   - Difficult trials should have LOWER total evidence (sum_evidence)
   - This reflects the participant's confidence in their choice
""")

def main():
    """Main function."""
    explain_evidence_difficulty_relationship()
    analyze_evidence_difficulty_relationship()

if __name__ == "__main__":
    main()
