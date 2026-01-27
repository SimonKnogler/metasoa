#!/usr/bin/env python3
"""
Integrated CDT Analysis Script
==============================

This script combines:
1. Staircase convergence analysis
2. Group posterior analysis with plots and summary panels
3. Comprehensive behavioral data analysis

Usage:
    python integrated_cdt_analysis.py [data_directory]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
import json
from typing import Dict, List, Tuple

# Import the group analysis functions
from analyze_group_posteriors import (
    ReplayedQuestPlus, logit, inv_logit, clamp_prop,
    analyze_group, analyze_calibration_group, find_data_files, extract_mapping, extract_participant_id
)

def check_staircase_convergence(filename, last_n=20):
    """
    Analyze staircase convergence for CDT experiment data.
    """
    print(f"Analyzing: {filename}")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(filename)
    print(f"Total trials: {len(df)}")
    
    # Filter for practice trials
    practice_df = df[df['phase'].str.contains('practice', na=False)].copy()
    print(f"Practice trials: {len(practice_df)}")
    
    if len(practice_df) == 0:
        print("No practice trials found!")
        return
    
    # Group by staircase type
    for cond, subdf in practice_df.groupby("chosen_by"):
        print(f"\n{'='*40}")
        print(f"Staircase: {cond}")
        print(f"{'='*40}")
        
        # Analyze recent trials
        recent = subdf.tail(last_n)
        
        # Observed accuracy
        acc = recent["response_correct"].mean()
        print(f"Observed accuracy (last {len(recent)} trials): {acc:.3f}")
        
        # QUEST parameters
        if 'quest_alpha_mean' in recent.columns:
            mean_est = recent["quest_alpha_mean"].iloc[-1]
            sd_est = recent["quest_alpha_sd"].iloc[-1]
            print(f"QUEST alpha: {mean_est:.3f} ± {sd_est:.3f}")
        
        # Target accuracy
        if 'blue' in cond.lower():
            target = 0.90
            difficulty = "Easy (High Precision)"
        else:
            target = 0.65
            difficulty = "Hard (Low Precision)"
        
        print(f"Target: {target:.1f} ({difficulty})")
        print(f"Difference: {acc - target:+.3f}")
        
        # Convergence status
        if abs(acc - target) < 0.1:
            status = "✓ CONVERGED"
        elif abs(acc - target) < 0.15:
            status = "~ PARTIALLY CONVERGED"
        else:
            status = "✗ NOT CONVERGED"
        
        print(f"Status: {status}")
        
        # Stimulus intensity
        if 'stimulus_intensity_s' in recent.columns:
            intensities = recent['stimulus_intensity_s'].values
            print(f"Intensity range: {intensities.min():.3f} - {intensities.max():.3f}")

def analyze_test_performance(filename):
    """
    Quick analysis of test phase performance.
    """
    print(f"\n{'='*60}")
    print("TEST PHASE PERFORMANCE")
    print(f"{'='*60}")
    
    df = pd.read_csv(filename)
    test_df = df[df['phase'].str.contains('test', na=False)].copy()
    
    if len(test_df) == 0:
        print("No test trials found!")
        return
    
    print(f"Found {len(test_df)} test trials")
    
    # Overall performance
    overall_acc = test_df['response_correct'].mean()
    print(f"Overall test accuracy: {overall_acc:.3f}")
    
    # Performance by difficulty condition
    if 'difficulty_condition' in test_df.columns:
        print(f"\nPerformance by difficulty condition:")
        for condition, subdf in test_df.groupby('difficulty_condition'):
            acc = subdf['response_correct'].mean()
            print(f"  {condition}: {acc:.3f} ({len(subdf)} trials)")
    
    # Performance by difficulty type
    if 'difficulty_type' in test_df.columns:
        print(f"\nPerformance by difficulty type:")
        for dtype, subdf in test_df.groupby('difficulty_type'):
            acc = subdf['response_correct'].mean()
            print(f"  {dtype}: {acc:.3f} ({len(subdf)} trials)")

def run_individual_analysis(data_dir):
    """
    Run individual staircase convergence analysis on all data files.
    """
    print(f"\n{'='*80}")
    print("INDIVIDUAL STAIRCASE CONVERGENCE ANALYSIS")
    print(f"{'='*80}")
    
    # Find data files
    csv_files = [f for f in data_dir.glob("*.csv") if "kinematics" not in f.name]
    
    if not csv_files:
        print("No behavioral data CSV files found!")
        return
    
    print(f"Found {len(csv_files)} data files")
    
    # Analyze each file
    for data_file in csv_files:
        print(f"\n{'='*80}")
        print(f"FILE: {data_file.name}")
        print(f"{'='*80}")
        
        try:
            check_staircase_convergence(data_file)
            analyze_test_performance(data_file)
        except Exception as e:
            print(f"Error analyzing {data_file}: {e}")

def run_group_analysis(data_dir):
    """
    Run group posterior analysis with plots and summary panels.
    Automatically detects whether data is calibration or practice/learning phase.
    """
    print(f"\n{'='*80}")
    print("GROUP POSTERIOR ANALYSIS WITH PLOTS")
    print(f"{'='*80}")
    
    try:
        # Check if we have calibration or practice data
        csv_files = [f for f in data_dir.glob("*.csv") if "kinematics" not in f.name]
        has_calibration = False
        has_practice = False
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if 'phase' in df.columns:
                    phases = df['phase'].astype(str).unique()
                    if any('calibration' in p.lower() for p in phases):
                        has_calibration = True
                    if any('practice' in p.lower() or 'learning' in p.lower() for p in phases):
                        has_practice = True
            except Exception:
                continue
        
        if has_calibration:
            print("Detected calibration data - running calibration-specific analysis...")
            analyze_calibration_group(data_dir)
        
        if has_practice:
            print("Detected practice/learning data - running practice-specific analysis...")
            analyze_group(data_dir)
        
        if not has_calibration and not has_practice:
            print("No calibration or practice data found.")
            return
        
        print("Group analysis completed successfully!")
        print(f"Results saved to: {data_dir}/quest_group_analysis/")
        
        # List generated files
        output_dir = data_dir / "quest_group_analysis"
        if output_dir.exists():
            print(f"\nGenerated files:")
            for file in output_dir.glob("*"):
                print(f"  - {file.name}")
        
    except Exception as e:
        print(f"Error in group analysis: {e}")
        import traceback
        traceback.print_exc()

def run_evidence_accuracy_analysis(data_dir: Path, phase_filter: str = 'calibration') -> None:
    """
    Analyze the relationship between evidence strength and accuracy.
    
    Args:
        data_dir: Directory containing subject CSV files
        phase_filter: Which phase to analyze ('calibration', 'learning', 'test', or 'all')
    """
    print(f"\n{'='*80}")
    print(f"EVIDENCE-ACCURACY RELATIONSHIP ANALYSIS ({phase_filter.upper()} PHASE)")
    print(f"{'='*80}")
    
    csv_files = [f for f in data_dir.glob("*.csv") if "kinematics" not in f.name]
    if not csv_files:
        print("No behavioral data CSV files found!")
        return
    
    all_data = []
    
    for data_file in csv_files:
        try:
            df = pd.read_csv(data_file)
            
            # Filter by phase
            if 'phase' not in df.columns:
                continue
            
            if phase_filter != 'all':
                df = df[df['phase'].fillna("").astype(str).str.contains(phase_filter, case=False)].copy()
            
            if df.empty:
                continue
            
            # Check for required columns
            if 'mean_evidence' not in df.columns or 'accuracy' not in df.columns:
                print(f"Skipping {data_file.name}: missing evidence or accuracy columns")
                continue
            
            # Exclude timeout trials
            if 'resp_shape' in df.columns:
                df = df[df['resp_shape'] != 'timeout'].copy()
            
            # Add participant ID
            participant_id = extract_participant_id(df, default_id=data_file.stem)
            df['participant'] = participant_id
            
            all_data.append(df)
            
        except Exception as e:
            print(f"Error reading {data_file.name}: {e}")
    
    if not all_data:
        print("No valid data found for analysis!")
        return
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Ensure numeric columns
    combined_df['mean_evidence'] = pd.to_numeric(combined_df['mean_evidence'], errors='coerce')
    combined_df['sum_evidence'] = pd.to_numeric(combined_df.get('sum_evidence', np.nan), errors='coerce')
    combined_df['accuracy'] = pd.to_numeric(combined_df['accuracy'], errors='coerce')
    # New pre-RT evidence columns (optional)
    for col in [
        'mean_evidence_preRT','sum_evidence_preRT','var_evidence_preRT',
        'cum_evidence_preRT','max_cum_evidence_preRT','min_cum_evidence_preRT',
        'max_abs_cum_evidence_preRT','prop_positive_evidence_preRT'
    ]:
        if col in combined_df.columns:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
    
    # Prefer pre-RT evidence when available
    evidence_primary = 'mean_evidence_preRT' if 'mean_evidence_preRT' in combined_df.columns else 'mean_evidence'
    sum_evidence_primary = 'sum_evidence_preRT' if 'sum_evidence_preRT' in combined_df.columns else 'sum_evidence'

    # Remove NaN values
    cols_for_valid = [evidence_primary, sum_evidence_primary, 'accuracy', 'participant']
    cols_for_valid = [c for c in cols_for_valid if c in combined_df.columns]
    valid_data = combined_df[cols_for_valid].dropna(subset=[evidence_primary, 'accuracy'])
    
    if len(valid_data) < 10:
        print(f"Insufficient valid trials ({len(valid_data)}) for analysis!")
        return
    
    print(f"\nAnalyzing {len(valid_data)} trials from {valid_data['participant'].nunique()} participant(s)")
    
    # 1. Correlation analysis - Overall and by condition
    from scipy import stats
    corr_mean, p_mean = stats.pointbiserialr(valid_data['accuracy'], valid_data[evidence_primary])
    print(f"\n{'─'*60}")
    print("1. CORRELATION ANALYSIS")
    print(f"{'─'*60}")
    print(f"OVERALL:")
    print(f"  Mean Evidence vs Accuracy: r = {corr_mean:.3f}, p = {p_mean:.4f} {'***' if p_mean < 0.001 else '**' if p_mean < 0.01 else '*' if p_mean < 0.05 else 'ns'}")
    
    if sum_evidence_primary in valid_data.columns and valid_data[sum_evidence_primary].notna().any():
        corr_sum, p_sum = stats.pointbiserialr(valid_data['accuracy'], valid_data[sum_evidence_primary])
        print(f"  Sum Evidence vs Accuracy: r = {corr_sum:.3f}, p = {p_sum:.4f} {'***' if p_sum < 0.001 else '**' if p_sum < 0.01 else '*' if p_sum < 0.05 else 'ns'}")
    
    # By angle bias condition
    if 'angle_bias' in combined_df.columns:
        valid_data_with_angle = combined_df[[evidence_primary, 'accuracy', 'participant', 'angle_bias']].dropna(subset=[evidence_primary, 'accuracy', 'angle_bias'])
        if len(valid_data_with_angle) > 0:
            print(f"\nBY ANGLE CONDITION:")
            for angle in sorted(valid_data_with_angle['angle_bias'].unique()):
                angle_data = valid_data_with_angle[valid_data_with_angle['angle_bias'] == angle]
                if len(angle_data) >= 10:
                    try:
                        r_angle, p_angle = stats.pointbiserialr(angle_data['accuracy'], angle_data[evidence_primary])
                        print(f"  {int(angle)}° rotation (n={len(angle_data)}): r = {r_angle:.3f}, p = {p_angle:.4f} {'***' if p_angle < 0.001 else '**' if p_angle < 0.01 else '*' if p_angle < 0.05 else 'ns'}")
                    except:
                        print(f"  {int(angle)}° rotation (n={len(angle_data)}): insufficient variance")
    
    # By difficulty (stimulus intensity tertiles)
    if 'prop_used' in combined_df.columns:
        valid_data_with_prop = combined_df[[evidence_primary, 'accuracy', 'participant', 'prop_used']].dropna(subset=[evidence_primary, 'accuracy', 'prop_used'])
        if len(valid_data_with_prop) > 0:
            valid_data_with_prop['difficulty'] = pd.qcut(valid_data_with_prop['prop_used'], q=3, labels=['Hard', 'Medium', 'Easy'], duplicates='drop')
            print(f"\nBY DIFFICULTY (stimulus intensity):")
            for diff in ['Hard', 'Medium', 'Easy']:
                diff_data = valid_data_with_prop[valid_data_with_prop['difficulty'] == diff]
                if len(diff_data) >= 10:
                    try:
                        r_diff, p_diff = stats.pointbiserialr(diff_data['accuracy'], diff_data[evidence_primary])
                        mean_prop = diff_data['prop_used'].mean()
                        print(f"  {diff} (prop≈{mean_prop:.2f}, n={len(diff_data)}): r = {r_diff:.3f}, p = {p_diff:.4f} {'***' if p_diff < 0.001 else '**' if p_diff < 0.01 else '*' if p_diff < 0.05 else 'ns'}")
                    except:
                        print(f"  {diff} (n={len(diff_data)}): insufficient variance")
    
    # 2. Binned accuracy analysis
    print(f"\n{'─'*60}")
    print("2. BINNED ACCURACY BY EVIDENCE STRENGTH")
    print(f"{'─'*60}")
    
    # Create quintiles (5 bins) of evidence
    valid_data['evidence_bin'] = pd.qcut(valid_data[evidence_primary], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'], duplicates='drop')
    
    binned_stats = valid_data.groupby('evidence_bin').agg({
        'accuracy': ['mean', 'std', 'count'],
        evidence_primary: ['mean', 'min', 'max']
    }).round(3)
    
    print("\nEvidence Bin → Accuracy:")
    print(binned_stats)
    
    # 3. Logistic regression (optional - requires sklearn)
    print(f"\n{'─'*60}")
    print("3. LOGISTIC REGRESSION (Evidence → Accuracy)")
    print(f"{'─'*60}")
    
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report, roc_auc_score
        
        X = valid_data[[evidence_primary]].values
        y = valid_data['accuracy'].values
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, y)
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        
        print(f"Coefficient (β): {model.coef_[0][0]:.4f}")
        print(f"Intercept: {model.intercept_[0]:.4f}")
        print(f"AUC-ROC: {roc_auc_score(y, y_prob):.3f}")
        print(f"\nClassification accuracy: {(y_pred == y).mean():.3f}")
        
        has_sklearn = True
    except ImportError:
        print("⚠️  scikit-learn not installed - skipping logistic regression")
        print("   Install with: pip install scikit-learn")
        has_sklearn = False
        model = None
        y_prob = None
    
    # 4. Create visualization
    print(f"\n{'─'*60}")
    print("4. GENERATING VISUALIZATIONS")
    print(f"{'─'*60}")
    
    out_dir = data_dir.parent / "quest_group_analysis"
    out_dir.mkdir(exist_ok=True)
    
    # Check if we have angle_bias for condition-specific plots
    has_angle_data = 'angle_bias' in combined_df.columns and combined_df['angle_bias'].notna().sum() > 0
    
    if has_angle_data:
        # Create 1x2 panel: Evidence distributions for each angle condition
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        valid_data_with_angle = combined_df[[evidence_primary, 'accuracy', 'angle_bias']].dropna()
        
        for idx, angle in enumerate([0, 90]):
            ax = axes[idx]
            angle_data = valid_data_with_angle[valid_data_with_angle['angle_bias'] == angle]
            
            if len(angle_data) >= 10:
                correct_angle = angle_data[angle_data['accuracy'] == 1]
                incorrect_angle = angle_data[angle_data['accuracy'] == 0]
                
                # Histogram of evidence distributions
                ax.hist(correct_angle[evidence_primary], bins=15, alpha=0.6, color='green', 
                       label=f'Correct (n={len(correct_angle)})', density=True, edgecolor='darkgreen')
                ax.hist(incorrect_angle[evidence_primary], bins=15, alpha=0.6, color='red', 
                       label=f'Incorrect (n={len(incorrect_angle)})', density=True, edgecolor='darkred')
                
                # Add statistics
                try:
                    r_cond, p_cond = stats.pointbiserialr(angle_data['accuracy'], angle_data[evidence_primary])
                    acc = angle_data['accuracy'].mean()
                    ax.set_title(f'{int(angle)}° Rotation\nAccuracy={acc:.1%}, r={r_cond:.3f}, p={p_cond:.3f}', fontsize=12, fontweight='bold')
                except:
                    ax.set_title(f'{int(angle)}° Rotation (n={len(angle_data)})', fontsize=12, fontweight='bold')
                
                ax.set_xlabel('Pre-RT Evidence' if evidence_primary.endswith('preRT') else 'Mean Evidence', fontsize=11)
                ax.set_ylabel('Density', fontsize=11)
                ax.legend(loc='best', fontsize=10)
                ax.grid(alpha=0.3, axis='y')
                ax.axvline(0, color='black', linestyle='--', alpha=0.3, linewidth=1)
            else:
                ax.text(0.5, 0.5, f'{int(angle)}° Rotation\nInsufficient data', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_xlabel('Mean Evidence')
                ax.set_ylabel('Density')
        
        plt.tight_layout()
        out_path = out_dir / f'evidence_accuracy_analysis_{phase_filter}.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        print("⚠️  No angle_bias data found - skipping visualization")
        print("   (Visualization requires angle_bias column in data)")
    
    print(f"✓ Visualization saved: {out_path}")
    
    # Save summary statistics - overall and by condition
    summary_rows = []
    
    # Calculate overall correct/incorrect evidence means
    correct_overall = valid_data[valid_data['accuracy'] == 1]
    incorrect_overall = valid_data[valid_data['accuracy'] == 0]
    
    # Overall summary
    summary_dict = {
        'condition': 'Overall',
        'phase': phase_filter,
        'n_trials': len(valid_data),
        'n_participants': valid_data['participant'].nunique(),
        'correlation_mean_evidence': corr_mean,
        'p_value_mean_evidence': p_mean,
        'mean_evidence_correct': correct_overall[evidence_primary].mean(),
        'mean_evidence_incorrect': incorrect_overall[evidence_primary].mean(),
        'accuracy': valid_data['accuracy'].mean(),
    }
    
    if has_sklearn and model is not None:
        summary_dict.update({
            'logistic_coefficient': model.coef_[0][0],
            'logistic_intercept': model.intercept_[0],
            'auc_roc': roc_auc_score(y, y_prob),
        })
    
    summary_rows.append(summary_dict)
    
    # Add condition-specific summaries
    if 'angle_bias' in combined_df.columns:
        valid_data_with_angle = combined_df[[evidence_primary, 'accuracy', 'angle_bias']].dropna()
        for angle in sorted(valid_data_with_angle['angle_bias'].unique()):
            angle_data = valid_data_with_angle[valid_data_with_angle['angle_bias'] == angle]
            if len(angle_data) >= 10:
                try:
                    r_angle, p_angle = stats.pointbiserialr(angle_data['accuracy'], angle_data['mean_evidence'])
                    corr_data = angle_data[angle_data['accuracy'] == 1]
                    incorr_data = angle_data[angle_data['accuracy'] == 0]
                    
                    summary_rows.append({
                        'condition': f'{int(angle)}deg',
                        'phase': phase_filter,
                        'n_trials': len(angle_data),
                        'n_participants': valid_data['participant'].nunique(),
                        'correlation_mean_evidence': r_angle,
                        'p_value_mean_evidence': p_angle,
                        'mean_evidence_correct': corr_data[evidence_primary].mean() if len(corr_data) > 0 else np.nan,
                        'mean_evidence_incorrect': incorr_data[evidence_primary].mean() if len(incorr_data) > 0 else np.nan,
                        'accuracy': angle_data['accuracy'].mean(),
                    })
                except:
                    pass
    
    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / f'evidence_accuracy_summary_{phase_filter}.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Summary statistics saved: {summary_path}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


def run_calibration_only_analysis(data_dir: Path) -> None:
    """
    Calibration-only analysis:
    - Scans CSVs in data_dir (excluding kinematics)
    - Filters to calibration rows (phase contains 'calibration')
    - Summarizes per-staircase trial counts, alpha SDs, and accuracy
    - Writes a summary CSV to quest_group_analysis/calibration_only_summary.csv
    """
    print(f"\n{'='*80}")
    print("CALIBRATION-ONLY ANALYSIS")
    print(f"{'='*80}")
    
    # Create output directory
    out_dir = data_dir.parent / "quest_group_analysis"
    out_dir.mkdir(exist_ok=True)

    csv_files = [f for f in data_dir.glob("*.csv") if "kinematics" not in f.name]
    if not csv_files:
        print("No behavioral data CSV files found!")
        return

    summary_rows: List[Dict] = []

    for data_file in csv_files:
        try:
            df = pd.read_csv(data_file)
        except Exception as e:
            print(f"Skipping {data_file.name} (read error): {e}")
            continue

        # Filter to calibration phase
        if 'phase' not in df.columns:
            print(f"{data_file.name}: no 'phase' column found, skipping.")
            continue
        
        calib_mask = df['phase'].fillna("").astype(str).str.contains('calibration', case=False)
        calib_df = df[calib_mask].copy()

        if calib_df.empty:
            print(f"{data_file.name}: no calibration trials found.")
            continue

        # Filter to actual trial rows (exclude timeout trials for accuracy calc)
        trial_df = calib_df[calib_df['resp_shape'].notna()].copy()
        
        # Identify timeout trials
        timeout_mask = (trial_df['resp_shape'] == 'timeout') | (trial_df.get('is_timeout', False) == True)
        
        # Columns that may exist
        staircase_col = 'staircase_id' if 'staircase_id' in trial_df.columns else None
        alpha_sd_cols = [c for c in ['quest_alpha_sd', 'alpha_sd_current'] if c in trial_df.columns]

        print(f"\nFile: {data_file.name}")
        if not staircase_col:
            # Fallback: treat all calibration trials as a single staircase summary
            total_trials = len(trial_df)
            timeout_trials = timeout_mask.sum()
            non_timeout = trial_df[~timeout_mask]
            
            last_alpha_sd = None
            for c in alpha_sd_cols:
                sd_series = trial_df[c].dropna()
                if not sd_series.empty:
                    last_alpha_sd = float(sd_series.iloc[-1])
                    break
            
            accuracy = non_timeout['accuracy'].mean() if len(non_timeout) > 0 and 'accuracy' in non_timeout.columns else None
            
            print(f"  Staircases: not logged")
            print(f"    Total trials: {total_trials} (timeouts: {timeout_trials})")
            print(f"    Accuracy (excl. timeout): {accuracy:.3f}" if accuracy is not None else "    Accuracy: N/A")
            print(f"    Last alpha SD: {last_alpha_sd:.4f}" if last_alpha_sd is not None else "    Last alpha SD: N/A")
            
            summary_rows.append({
                'file': data_file.name,
                'staircase_id': 'unknown',
                'total_trials': total_trials,
                'timeout_trials': timeout_trials,
                'non_timeout_trials': total_trials - timeout_trials,
                'accuracy_excl_timeout': accuracy,
                'last_alpha_sd': last_alpha_sd
            })
            continue

        # Group by staircase
        for staircase_id, sub in trial_df.groupby(staircase_col):
            total_trials = len(sub)
            timeout_sub = timeout_mask[sub.index]
            timeout_trials = timeout_sub.sum()
            non_timeout = sub[~timeout_sub]
            
            last_alpha_sd = None
            for c in alpha_sd_cols:
                sd_series = sub[c].dropna()
                if not sd_series.empty:
                    last_alpha_sd = float(sd_series.iloc[-1])
                    break

            # Calculate accuracy excluding timeouts
            accuracy = non_timeout['accuracy'].mean() if len(non_timeout) > 0 and 'accuracy' in non_timeout.columns else None
            
            # Get trials_completed_this_staircase if present
            trials_completed = None
            if 'trials_completed_this_staircase' in sub.columns:
                try:
                    trials_completed = int(sub['trials_completed_this_staircase'].max())
                except Exception:
                    trials_completed = None

            print(f"  Staircase {staircase_id}:")
            print(f"    Total trials: {total_trials} (timeouts: {timeout_trials})")
            print(f"    Accuracy (excl. timeout): {accuracy:.3f}" if accuracy is not None else "    Accuracy: N/A")
            print(f"    Last alpha SD: {last_alpha_sd:.4f}" if last_alpha_sd is not None else "    Last alpha SD: N/A")
            print(f"    Trials completed: {trials_completed}" if trials_completed is not None else "")
            
            summary_rows.append({
                'file': data_file.name,
                'staircase_id': staircase_id,
                'total_trials': total_trials,
                'timeout_trials': timeout_trials,
                'non_timeout_trials': total_trials - timeout_trials,
                'accuracy_excl_timeout': accuracy,
                'last_alpha_sd': last_alpha_sd,
                'last_trials_completed': trials_completed
            })

    if summary_rows:
        out_df = pd.DataFrame(summary_rows)
        out_path = out_dir / 'calibration_only_summary.csv'
        out_df.to_csv(out_path, index=False)
        print(f"\n{'='*80}")
        print(f"Calibration-only summary written to: {out_path}")
        print(f"{'='*80}\n")
    else:
        print("No calibration data summarized.")

def create_comprehensive_summary(data_dir):
    """
    Create a comprehensive summary of all analyses.
    """
    print(f"\n{'='*80}")
    print("COMPREHENSIVE SUMMARY")
    print(f"{'='*80}")
    
    # Count data files
    csv_files = [f for f in data_dir.glob("*.csv") if "kinematics" not in f.name]
    print(f"Data files analyzed: {len(csv_files)}")
    
    # Check for group analysis results
    output_dir = data_dir / "quest_group_analysis"
    if output_dir.exists():
        result_files = list(output_dir.glob("*"))
        print(f"Group analysis results: {len(result_files)} files generated")
        
        # List key result files
        key_files = [
            "group_summary_panels_0deg.png",
            "group_summary_panels_90deg.png", 
            "group_thresholds_0deg.csv",
            "group_thresholds_90deg.csv"
        ]
        
        print(f"\nKey result files:")
        for key_file in key_files:
            file_path = output_dir / key_file
            if file_path.exists():
                print(f"  ✓ {key_file}")
            else:
                print(f"  ✗ {key_file} (not found)")
    
    print(f"\nAnalysis complete! Check the following directories for results:")
    print(f"  - Individual analysis: Console output above")
    print(f"  - Group analysis plots: {output_dir}")
    print(f"  - Summary panels: {output_dir}/group_summary_panels_*.png")

def main():
    """Main function to run integrated analysis."""
    print("INTEGRATED CDT ANALYSIS")
    print("=" * 50)
    
    parser = argparse.ArgumentParser(description="Integrated CDT Analysis")
    parser.add_argument("data_dir", nargs='?', default="../Main_Experiment/data/subjects", help="Path to directory with CSVs")
    parser.add_argument("--calibration-only", "-c", action="store_true", help="Analyze calibration trials only and exit")
    parser.add_argument("--evidence-accuracy", "-e", action="store_true", help="Run evidence-accuracy relationship analysis")
    parser.add_argument("--phase", default="calibration", choices=["calibration", "learning", "test", "all"], help="Phase to analyze for evidence-accuracy (default: calibration)")
    args = parser.parse_args()

    # Resolve data directory robustly relative to this script's location
    script_dir = Path(__file__).resolve().parent
    raw_path = Path(args.data_dir)
    data_dir_candidate = raw_path if raw_path.is_absolute() else (script_dir / raw_path)

    fallback_candidates = [
        data_dir_candidate,
        script_dir.parent / "Main_Experiment/data/subjects",
        script_dir.parent / "Main_Experiment/data",
        Path.cwd() / "Main_Experiment/data/subjects",
        Path.cwd() / "Main_Experiment/data",
    ]

    data_dir = None
    for cand in fallback_candidates:
        if cand.exists():
            data_dir = cand
            break

    if data_dir is None:
        print(f"Data directory not found. Tried:")
        for cand in fallback_candidates:
            print(f"  - {cand}")
        return

    print(f"Analyzing data in: {data_dir}")

    if args.calibration_only:
        run_calibration_only_analysis(data_dir)
        return
    
    if args.evidence_accuracy:
        run_evidence_accuracy_analysis(data_dir, phase_filter=args.phase)
        return

    # Run individual staircase convergence analysis
    run_individual_analysis(data_dir)
    
    # Run group posterior analysis with plots
    run_group_analysis(data_dir)
    
    # Run evidence-accuracy analysis for calibration phase
    print("\n" + "=" * 80)
    print("Running Evidence-Accuracy Analysis (Calibration Phase)...")
    print("=" * 80)
    try:
        run_evidence_accuracy_analysis(data_dir, phase_filter='calibration')
    except Exception as e:
        print(f"Error in evidence-accuracy analysis: {e}")
        import traceback
        traceback.print_exc()
    
    # Create comprehensive summary
    create_comprehensive_summary(data_dir)

if __name__ == "__main__":
    main()
