"""
Detailed QUEST+ calibration analysis with convergence tracking
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load participant 001 data (the only one with substantial calibration data)
df = pd.read_csv("Main_Experiment/data/subjects/CDT_v2_blockwise_fast_response__2.csv")

# Filter calibration trials
calib = df[df['phase'] == 'calibration_interleaved'].copy()

print("="*70)
print("=== DETAILED QUEST+ CALIBRATION ANALYSIS ===")
print("="*70)
print(f"\nParticipant: {calib['participant'].iloc[0]}")
print(f"Total calibration trials: {len(calib)}")

# Separate by angle
calib_0 = calib[calib['angle_bias'] == 0].copy()
calib_90 = calib[calib['angle_bias'] == 90].copy()

print(f"  Angle 0°: {len(calib_0)} trials")
print(f"  Angle 90°: {len(calib_90)} trials")

# Analysis for each angle
for angle, angle_data in [('0°', calib_0), ('90°', calib_90)]:
    print(f"\n{'='*70}")
    print(f"=== ANGLE {angle} ===")
    print(f"{'='*70}")
    
    # Remove NaN values
    angle_data = angle_data.dropna(subset=['quest_alpha_mean', 'prop_used'])
    
    if len(angle_data) == 0:
        print("  No valid QUEST data for this angle")
        continue
    
    # Initial vs Final comparison
    initial_trials = angle_data.head(10)
    final_trials = angle_data.tail(10)
    
    # Initial state
    initial_alpha_mean = initial_trials['quest_alpha_mean'].mean()
    initial_alpha_sd = initial_trials['quest_alpha_sd'].mean()
    initial_prop_mean = initial_trials['prop_used'].mean()
    
    # Final state
    final_alpha_mean = final_trials['quest_alpha_mean'].mean()
    final_alpha_sd = final_trials['quest_alpha_sd'].mean()
    final_prop_mean = final_trials['prop_used'].mean()
    final_prop_threshold = 1.0 / (1.0 + np.exp(-final_alpha_mean))
    
    print(f"\n--- Initial State (first 10 trials) ---")
    print(f"  Alpha mean (logit): {initial_alpha_mean:.3f}")
    print(f"  Alpha SD (logit): {initial_alpha_sd:.3f}")
    print(f"  Props tested: {initial_trials['prop_used'].min():.3f} - {initial_trials['prop_used'].max():.3f}")
    print(f"  Mean prop: {initial_prop_mean:.3f}")
    
    print(f"\n--- Final State (last 10 trials) ---")
    print(f"  Alpha mean (logit): {final_alpha_mean:.3f}")
    print(f"  Alpha SD (logit): {final_alpha_sd:.3f}")
    print(f"  Threshold (prop space): {final_prop_threshold:.3f}")
    print(f"  Props tested: {final_trials['prop_used'].min():.3f} - {final_trials['prop_used'].max():.3f}")
    print(f"  Mean prop: {final_prop_mean:.3f}")
    
    # Convergence
    print(f"\n--- Convergence ---")
    print(f"  Change in alpha: {abs(final_alpha_mean - initial_alpha_mean):.3f} logit units")
    print(f"  Change in SD: {initial_alpha_sd - final_alpha_sd:.3f} (reduction)")
    print(f"  Final SD: {final_alpha_sd:.3f}", end="")
    if final_alpha_sd < 0.20:
        print(" ✓ Converged")
    elif final_alpha_sd < 0.40:
        print(" ⚠ Partially converged")
    else:
        print(" ✗ Not converged")
    
    # Accuracy over time
    valid_accuracy = angle_data[angle_data['accuracy'].notna()]
    if len(valid_accuracy) > 0:
        early_acc = valid_accuracy.head(10)['accuracy'].mean()
        late_acc = valid_accuracy.tail(10)['accuracy'].mean()
        overall_acc = valid_accuracy['accuracy'].mean()
        
        print(f"\n--- Performance ---")
        print(f"  Early accuracy (first 10): {early_acc:.1%}")
        print(f"  Late accuracy (last 10): {late_acc:.1%}")
        print(f"  Overall accuracy: {overall_acc:.1%}")
    
    # Stimulus range exploration
    prop_range = angle_data['prop_used']
    print(f"\n--- Stimulus Exploration ---")
    print(f"  Full range tested: {prop_range.min():.3f} - {prop_range.max():.3f}")
    print(f"  Range width: {prop_range.max() - prop_range.min():.3f}")
    print(f"  Mean prop tested: {prop_range.mean():.3f}")
    print(f"  SD of props: {prop_range.std():.3f}")

# Overall Prior Assessment
print(f"\n{'='*70}")
print("=== PRIOR ASSESSMENT ===")
print(f"{'='*70}")

# Get final thresholds
final_thresholds = []
for angle_data in [calib_0, calib_90]:
    angle_data = angle_data.dropna(subset=['quest_alpha_mean'])
    if len(angle_data) > 0:
        final_alpha = angle_data.tail(10)['quest_alpha_mean'].mean()
        final_prop = 1.0 / (1.0 + np.exp(-final_alpha))
        final_thresholds.append(final_prop)

if final_thresholds:
    mean_threshold = np.mean(final_thresholds)
    
    prior_prop = 0.625
    prior_logit = np.log(prior_prop / (1 - prior_prop))
    
    print(f"\nCurrent prior (neutral): {prior_prop:.3f} (logit: {prior_logit:.3f})")
    print(f"Actual thresholds: {[f'{t:.3f}' for t in final_thresholds]}")
    print(f"Mean actual threshold: {mean_threshold:.3f}")
    print(f"Difference from prior: {mean_threshold - prior_prop:.3f}")
    
    # Assessment
    diff = abs(mean_threshold - prior_prop)
    if diff < 0.10:
        verdict = "✓ EXCELLENT - Prior is very well-chosen"
    elif diff < 0.15:
        verdict = "✓ GOOD - Prior is reasonably well-chosen"
    elif diff < 0.25:
        verdict = "⚠ ACCEPTABLE - Prior works but could be improved"
    else:
        verdict = "✗ POOR - Consider adjusting prior"
    
    print(f"\n{verdict}")
    
    if diff > 0.15:
        print(f"  Suggested prior: {mean_threshold:.3f} (logit: {np.log(mean_threshold / (1 - mean_threshold)):.3f})")
    
    # Check if QUEST+ had to adapt significantly
    all_props = pd.concat([calib_0, calib_90])['prop_used'].dropna()
    if len(all_props) > 0:
        print(f"\nStimulus adaptation:")
        print(f"  Started testing around: {all_props.head(20).mean():.3f}")
        print(f"  Ended testing around: {all_props.tail(20).mean():.3f}")
        print(f"  Shift in focus: {abs(all_props.tail(20).mean() - all_props.head(20).mean()):.3f}")

print(f"\n{'='*70}")
print("=== KEY INSIGHTS ===")
print(f"{'='*70}")

print("""
1. QUEST+ BEHAVIOR:
   - The algorithm starts with the prior (0.625) and gradually adapts
   - It concentrates testing where the psychometric function is steepest
   - This is EFFICIENT - it doesn't waste trials on uninformative extremes

2. WHY FOCUS ON MID-RANGE:
   - At very low props (< 0.3): Performance near chance, uninformative
   - At very high props (> 0.7): Performance near ceiling, uninformative
   - Around 0.4-0.6: Maximum information gain about threshold

3. PRIOR SELECTION:
   - The 0.625 prior was chosen as a neutral starting point
   - Even if it's not perfect, QUEST+ adapts quickly (within 10-20 trials)
   - Prior uncertainty (SD = 1.0 logit) allows wide exploration if needed
""")

print(f"{'='*70}")

