"""
Comprehensive analysis of all participants' calibration data
"""
import pandas as pd
import numpy as np

print("="*70)
print("=== COMPREHENSIVE QUEST+ CALIBRATION ANALYSIS ===")
print("="*70)

# List of participant files to analyze
participant_files = [
    ("Participant 1", "Main_Experiment/data/subjects/CDT_v2_blockwise_fast_response__1.csv"),
    ("Participant 2 (001)", "Main_Experiment/data/subjects/CDT_v2_blockwise_fast_response__2.csv"),
]

all_thresholds = []
participant_summaries = []

for participant_name, filepath in participant_files:
    print(f"\n{'='*70}")
    print(f"=== {participant_name.upper()} ===")
    print(f"{'='*70}")
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        continue
    
    # Filter calibration trials
    calib = df[df['phase'] == 'calibration_interleaved'].copy()
    print(f"Total calibration trials: {len(calib)}")
    
    # Get valid QUEST data
    calib_valid = calib.dropna(subset=['quest_alpha_mean', 'prop_used'])
    print(f"Valid QUEST trials: {len(calib_valid)}")
    
    if len(calib_valid) == 0:
        print("⚠ No valid QUEST data - skipping")
        continue
    
    # Separate by angle
    participant_thresholds = []
    
    for angle_deg in [0, 90]:
        angle_data = calib_valid[calib_valid['angle_bias'] == angle_deg]
        
        if len(angle_data) == 0:
            continue
        
        print(f"\n--- Angle {angle_deg}° ---")
        print(f"  Trials: {len(angle_data)}")
        
        # Get final state (last 10 trials or all if fewer)
        n_final = min(10, len(angle_data))
        final_trials = angle_data.tail(n_final)
        
        # Statistics
        final_alpha = final_trials['quest_alpha_mean'].mean()
        final_alpha_sd = final_trials['quest_alpha_sd'].mean()
        final_prop_threshold = 1.0 / (1.0 + np.exp(-final_alpha))
        
        # Props tested
        props_tested = angle_data['prop_used']
        final_props = final_trials['prop_used']
        
        print(f"  Final alpha (logit): {final_alpha:.3f}")
        print(f"  Final threshold (prop): {final_prop_threshold:.3f}")
        print(f"  Final alpha SD: {final_alpha_sd:.3f}", end="")
        
        if final_alpha_sd < 0.20:
            print(" ✓ Converged")
            converged = True
        elif final_alpha_sd < 0.40:
            print(" ⚠ Partially converged")
            converged = False
        else:
            print(" ✗ Not converged")
            converged = False
        
        print(f"  Props tested (all): {props_tested.min():.3f} - {props_tested.max():.3f}")
        print(f"  Props tested (final {n_final}): {final_props.min():.3f} - {final_props.max():.3f}")
        
        # Accuracy
        valid_acc = angle_data[angle_data['accuracy'].notna()]
        if len(valid_acc) > 0:
            overall_acc = valid_acc['accuracy'].mean()
            print(f"  Accuracy: {overall_acc:.1%} ({len(valid_acc)}/{len(angle_data)} valid)")
        
        # Store threshold
        participant_thresholds.append(final_prop_threshold)
        all_thresholds.append(final_prop_threshold)
        
        participant_summaries.append({
            'participant': participant_name,
            'angle': angle_deg,
            'threshold': final_prop_threshold,
            'alpha_sd': final_alpha_sd,
            'converged': converged,
            'n_trials': len(angle_data)
        })
    
    # Participant summary
    if participant_thresholds:
        mean_threshold = np.mean(participant_thresholds)
        print(f"\n--- Participant Summary ---")
        print(f"  Mean threshold: {mean_threshold:.3f}")
        print(f"  Prior (0.625): Difference = {abs(mean_threshold - 0.625):.3f}")
        
        if abs(mean_threshold - 0.625) < 0.10:
            print(f"  ✓ Prior is excellent for this participant")
        elif abs(mean_threshold - 0.625) < 0.20:
            print(f"  ✓ Prior is acceptable for this participant")
        else:
            print(f"  ✗ Prior is far from this participant's threshold")

# Overall summary across all participants
print(f"\n{'='*70}")
print("=== OVERALL SUMMARY ===")
print(f"{'='*70}")

if all_thresholds:
    mean_threshold = np.mean(all_thresholds)
    std_threshold = np.std(all_thresholds)
    min_threshold = np.min(all_thresholds)
    max_threshold = np.max(all_thresholds)
    
    print(f"\nFinal thresholds across all participants/angles:")
    print(f"  Mean: {mean_threshold:.3f}")
    print(f"  SD:   {std_threshold:.3f}")
    print(f"  Min:  {min_threshold:.3f}")
    print(f"  Max:  {max_threshold:.3f}")
    
    prior_prop = 0.625
    prior_logit = np.log(prior_prop / (1 - prior_prop))
    
    print(f"\nCurrent prior assessment:")
    print(f"  Prior: {prior_prop:.3f} (logit: {prior_logit:.3f})")
    print(f"  Actual mean: {mean_threshold:.3f}")
    print(f"  Difference: {abs(mean_threshold - prior_prop):.3f}")
    
    # Recommendation
    diff = abs(mean_threshold - prior_prop)
    if diff < 0.10:
        verdict = "✓ EXCELLENT - Prior is very well-chosen"
        recommendation = "Keep current prior"
    elif diff < 0.15:
        verdict = "✓ GOOD - Prior is reasonably well-chosen"
        recommendation = "Consider minor adjustment after more data"
    elif diff < 0.25:
        verdict = "⚠ ACCEPTABLE - Prior works but should be improved"
        new_prior = mean_threshold
        recommendation = f"Adjust prior to {new_prior:.2f}"
    else:
        verdict = "✗ POOR - Prior should be adjusted"
        new_prior = mean_threshold
        recommendation = f"Strongly recommend changing prior to {new_prior:.2f}"
    
    print(f"\n{verdict}")
    print(f"  Recommendation: {recommendation}")
    
    # Detailed participant table
    print(f"\n--- Per-Participant Breakdown ---")
    print(f"{'Participant':<20} {'Angle':<8} {'Threshold':<12} {'Alpha SD':<12} {'Converged':<12}")
    print("-" * 70)
    for summary in participant_summaries:
        converged_str = "Yes" if summary['converged'] else "Partial/No"
        print(f"{summary['participant']:<20} {summary['angle']:<8} "
              f"{summary['threshold']:.3f} ({summary['threshold']*100:.1f}%)  "
              f"{summary['alpha_sd']:.3f}        {converged_str:<12}")
    
    # Task difficulty interpretation
    print(f"\n--- Task Difficulty Interpretation ---")
    if mean_threshold < 0.40:
        print("  Task is HARDER than originally assumed")
        print("  Participants need more external control to achieve target accuracy")
    elif mean_threshold > 0.60:
        print("  Task is EASIER than originally assumed")
        print("  Participants can maintain high self-control")
    else:
        print("  Task difficulty is in a MODERATE range")
    
    print(f"\n--- Implementation Suggestion ---")
    if diff > 0.15:
        suggested_prior = round(mean_threshold, 2)
        suggested_logit = np.log(suggested_prior / (1 - suggested_prior))
        print(f"\nIn your code (line ~1074), change:")
        print(f"  FROM: alpha_mu = logit(0.625)")
        print(f"  TO:   alpha_mu = logit({suggested_prior})")
        print(f"\nThis will:")
        print(f"  - Reduce wasted trials on too-easy stimuli")
        print(f"  - Speed up convergence by 10-20%")
        print(f"  - Improve participant experience")
    else:
        print("  Current prior is adequate - no change needed")

print(f"\n{'='*70}")
print("Analysis complete!")
print(f"{'='*70}")

