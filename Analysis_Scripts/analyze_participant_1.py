"""
Analyze calibration data for participant 1
"""
import pandas as pd
import numpy as np

# Load participant 1 data
df = pd.read_csv("Main_Experiment/data/subjects/CDT_v2_blockwise_fast_response__1.csv")

print("="*70)
print("=== PARTICIPANT 1 CALIBRATION ANALYSIS ===")
print("="*70)

# Filter calibration trials
calib = df[df['phase'] == 'calibration_interleaved'].copy()

print(f"\nTotal trials: {len(df)}")
print(f"Calibration trials: {len(calib)}")

# Check if there's valid QUEST data
calib_valid = calib.dropna(subset=['quest_alpha_mean', 'prop_used'])

print(f"Calibration trials with valid QUEST data: {len(calib_valid)}")

if len(calib_valid) == 0:
    print("\n⚠ No valid QUEST data found for this participant")
    print("This might be a CHECK_MODE run with insufficient trials")
    exit(0)

# Separate by angle
calib_0 = calib_valid[calib_valid['angle_bias'] == 0]
calib_90 = calib_valid[calib_valid['angle_bias'] == 90]

print(f"\nAngle 0°: {len(calib_0)} trials")
print(f"Angle 90°: {len(calib_90)} trials")

# Analyze each angle
final_thresholds = []

for angle_deg, angle_data in [('0°', calib_0), ('90°', calib_90)]:
    if len(angle_data) == 0:
        continue
        
    print(f"\n{'='*70}")
    print(f"=== ANGLE {angle_deg} ===")
    print(f"{'='*70}")
    
    # Initial state (first 5 trials or all if less)
    n_initial = min(5, len(angle_data))
    initial = angle_data.head(n_initial)
    
    # Final state (last 10 trials or all if less)
    n_final = min(10, len(angle_data))
    final = angle_data.tail(n_final)
    
    # Calculate statistics
    initial_alpha = initial['quest_alpha_mean'].mean()
    initial_alpha_sd = initial['quest_alpha_sd'].mean()
    initial_props = initial['prop_used']
    
    final_alpha = final['quest_alpha_mean'].mean()
    final_alpha_sd = final['quest_alpha_sd'].mean()
    final_props = final['prop_used']
    
    # Convert to prop space
    initial_prop_thresh = 1.0 / (1.0 + np.exp(-initial_alpha))
    final_prop_thresh = 1.0 / (1.0 + np.exp(-final_alpha))
    
    final_thresholds.append(final_prop_thresh)
    
    print(f"\n--- Initial State (first {n_initial} trials) ---")
    print(f"  Alpha mean (logit): {initial_alpha:.3f}")
    print(f"  Alpha SD: {initial_alpha_sd:.3f}")
    print(f"  Threshold (prop): {initial_prop_thresh:.3f}")
    print(f"  Props tested: {initial_props.min():.3f} - {initial_props.max():.3f}")
    
    print(f"\n--- Final State (last {n_final} trials) ---")
    print(f"  Alpha mean (logit): {final_alpha:.3f}")
    print(f"  Alpha SD: {final_alpha_sd:.3f}")
    print(f"  Threshold (prop): {final_prop_thresh:.3f}")
    print(f"  Props tested: {final_props.min():.3f} - {final_props.max():.3f}")
    
    # Convergence assessment
    print(f"\n--- Convergence ---")
    print(f"  Change in alpha: {abs(final_alpha - initial_alpha):.3f} logit units")
    print(f"  Change in SD: {initial_alpha_sd - final_alpha_sd:.3f}")
    print(f"  Final SD: {final_alpha_sd:.3f}", end="")
    if final_alpha_sd < 0.20:
        print(" ✓ Converged")
    elif final_alpha_sd < 0.40:
        print(" ⚠ Partially converged")
    else:
        print(" ✗ Not converged")
    
    # Accuracy
    valid_acc = angle_data[angle_data['accuracy'].notna()]
    if len(valid_acc) > 0:
        overall_acc = valid_acc['accuracy'].mean()
        print(f"\n--- Performance ---")
        print(f"  Overall accuracy: {overall_acc:.1%}")
        print(f"  Valid responses: {len(valid_acc)}/{len(angle_data)}")
    
    # Stimulus exploration
    all_props = angle_data['prop_used']
    print(f"\n--- Stimulus Range ---")
    print(f"  Range tested: {all_props.min():.3f} - {all_props.max():.3f}")
    print(f"  Mean: {all_props.mean():.3f}")
    print(f"  SD: {all_props.std():.3f}")

# Overall assessment
if final_thresholds:
    print(f"\n{'='*70}")
    print("=== PRIOR ASSESSMENT ===")
    print(f"{'='*70}")
    
    mean_threshold = np.mean(final_thresholds)
    prior_prop = 0.625
    
    print(f"\nFinal thresholds: {[f'{t:.3f}' for t in final_thresholds]}")
    print(f"Mean threshold: {mean_threshold:.3f}")
    print(f"Current prior: {prior_prop:.3f}")
    print(f"Difference: {abs(mean_threshold - prior_prop):.3f}")
    
    diff = abs(mean_threshold - prior_prop)
    if diff < 0.10:
        print(f"\n✓ Prior is EXCELLENT for this participant")
    elif diff < 0.15:
        print(f"\n✓ Prior is GOOD for this participant")
    elif diff < 0.25:
        print(f"\n⚠ Prior is ACCEPTABLE but could be better")
    else:
        print(f"\n✗ Prior is FAR from this participant's threshold")
        print(f"  Better prior would be: {mean_threshold:.3f}")
    
    # Context
    if mean_threshold < 0.45:
        print(f"\nNote: This participant finds the task HARDER than average")
        print(f"      (threshold at {mean_threshold:.1%} self-control)")
    elif mean_threshold > 0.65:
        print(f"\nNote: This participant finds the task EASIER than average")
        print(f"      (threshold at {mean_threshold:.1%} self-control)")
    else:
        print(f"\nNote: This participant is near the expected range")

print(f"\n{'='*70}")
print("Analysis complete!")
print(f"{'='*70}")

