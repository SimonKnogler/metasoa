#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
control_detection_task_v2_blockwise_fast_response.py - Fast response version
Windows-compatible version with early response capability

"""

import os, sys, math, random, pathlib, datetime, atexit, hashlib, json, subprocess

# Check if we're running with the correct Python interpreter
def check_and_run_with_correct_python():
    # If psychopy import fails, try to find anaconda Python
    try:
        import numpy as np
        import pandas as pd
        from psychopy import visual, event, core, data, gui
        return False  # Continue with current interpreter
    except ImportError as e:
        print(f"Missing required packages: {e}")
        print("Trying to find Python with required packages...")
        
        # Try common Python paths on different systems
        python_paths = [
            "C:/Program Files/PsychoPy/python.exe",  # Standalone PsychoPy (most reliable)
            "C:/Users/knogl/Miniconda3/envs/psychopy_env/python.exe",  # Your PsychoPy conda environment
            "C:/Users/knogl/Miniconda3/python.exe",  # Your base Miniconda3
            "/opt/anaconda3/bin/python",  # macOS
            "/usr/bin/python3",  # Linux
        ]
        
        for path in python_paths:
            if os.path.exists(path):
                print(f"Found Python at: {path}")
                result = subprocess.run([path] + sys.argv, check=False)
                sys.exit(result.returncode)
        
        print("Error: Python with required packages not found. Please install psychopy, numpy, and pandas.")
        sys.exit(1)

# Check interpreter and switch if needed
if check_and_run_with_correct_python():
    sys.exit(0)

# Import statements (will work after interpreter check)
import numpy as np
import pandas as pd
from psychopy import visual, event, core, data, gui

# ───────────────────────────────────────────────────────
#  Global variable for kinematics data
# ───────────────────────────────────────────────────────
kinematics_data = []
kinematics_csv_path = ""

# ───────────────────────────────────────────────────────
#  Auto‐save on quit
# ───────────────────────────────────────────────────────
_saved = False
def _save():
    global _saved
    if not _saved:
        if 'thisExp' in globals() and thisExp is not None:
            thisExp.saveAsWideText(csv_path)
            print("Main data auto‐saved ➜", csv_path)
            if kinematics_data:
                kinematics_df = pd.DataFrame(kinematics_data)
                kinematics_df.to_csv(kinematics_csv_path, index=False)
                print("Kinematics data auto‐saved ➜", kinematics_csv_path)
        else:
            print("Experiment not initialized - no data to save")
        _saved = True
atexit.register(_save)

# ───────────────────────────────────────────────────────
#  Participant dialog
# ───────────────────────────────────────────────────────
expName = "ControlDetection_v2_blockwise_fast_response"
expInfo = {"participant": "", "session": "001", "simulate": False, "check_mode": True}
dlg = gui.DlgFromDict(expInfo, order=["participant", "session", "simulate", "check_mode"], title=expName)
if not dlg.OK:
    core.quit()
SIMULATE = bool(expInfo.pop("simulate"))
CHECK_MODE = bool(expInfo.pop("check_mode"))
if SIMULATE:
    expInfo["participant"] = "SIM"

# Feature flag: use QUEST+ training/test design
USE_QUEST_TRAINING = True
if CHECK_MODE:
    # Check mode: minimal trials to run through entire experiment quickly
    # Calibration: 6 trials per staircase (4 staircases = 24 total)
    CHECK_CALIBRATION_TRIALS = 6
    # Learning: 6 trials per cue (2 cues = 12 total per learning block)
    CHECK_LEARNING_TRIALS_PER_CUE = 6
    # Test: 6 medium + 3 learning-level per cue (18 total per test block)
    CHECK_TEST_MEDIUM_PER_CUE = 6
    CHECK_TEST_LEARNING_PER_CUE = 3
else:
    # Full experiment settings
    # Calibration: 50 trials per staircase (4 staircases = ~200 total)
    CHECK_CALIBRATION_TRIALS = 50
    # Learning: 30 trials per cue (2 cues = 60 total per learning block)
    CHECK_LEARNING_TRIALS_PER_CUE = 30
    # Test: 50 medium + 25 learning-level per cue (150 total per test block)
    CHECK_TEST_MEDIUM_PER_CUE = 50
    CHECK_TEST_LEARNING_PER_CUE = 25

# counter‐balance cue colours (will be set per block based on angle)
# Initialize with placeholder - will be updated in block loop
low_col, high_col = None, None

# Print check mode status
if CHECK_MODE:
    print("=" * 60)
    print("⚠️  CHECK MODE ENABLED - Running minimal trials")
    print(f"   Calibration: {CHECK_CALIBRATION_TRIALS} trials/staircase (24 total)")
    print(f"   Learning: {CHECK_LEARNING_TRIALS_PER_CUE} trials/cue (12 total per block)")
    print(f"   Test: {CHECK_TEST_MEDIUM_PER_CUE} medium + {CHECK_TEST_LEARNING_PER_CUE} learning/cue (18 total per block)")
    print(f"   Total experiment: ~78 trials")
    print("=" * 60)
else:
    print("Running FULL experiment mode")
    print(f"   Calibration: {CHECK_CALIBRATION_TRIALS} trials/staircase (~200 total)")
    print(f"   Learning: {CHECK_LEARNING_TRIALS_PER_CUE} trials/cue (60 total per block)")
    print(f"   Test: {CHECK_TEST_MEDIUM_PER_CUE} medium + {CHECK_TEST_LEARNING_PER_CUE} learning/cue (150 total per block)")
    print(f"   Total experiment: ~620 trials")

# Learning order will be set by counterbalancing system below
# (Placeholder - assigned after participant ID is processed)
learning_order = None

# Experiment structure:
# Block 1: Calibration (both 0° and 90° interleaved)
# Block 2: Learning (first angle from learning_order)
# Block 3: Test (same angle as Block 2)
# Block 4: Learning (second angle from learning_order) 
# Block 5: Test (same angle as Block 4)

# ───────────────────────────────────────────────────────
#  Load motion library with cluster information
# ───────────────────────────────────────────────────────
script_dir = pathlib.Path(__file__).parent
LIB_NAME = script_dir.parent / "Motion_Library" / "core_pool.npy"
FEATS_NAME = script_dir.parent / "Motion_Library" / "core_pool_feats.npy"
LABELS_NAME = script_dir.parent / "Motion_Library" / "core_pool_labels.npy"

motion_pool = np.load(LIB_NAME)
snippet_features = np.load(FEATS_NAME)
snippet_labels = np.load(LABELS_NAME)

SNIP_LEN = motion_pool.shape[1]
TOTAL_SNIPS = motion_pool.shape[0]
K_CLUST = 4

print(f"Loaded {TOTAL_SNIPS} snippets × {SNIP_LEN} frames from {LIB_NAME}")
print(f"Cluster distribution: {np.bincount(snippet_labels)}")

with open(script_dir.parent / "Motion_Library" / "scaler_params.json", "r") as f:
    scp = json.load(f)
scaler_mean = np.array(scp["mean"], dtype=np.float32)
scaler_std = np.array(scp["scale"], dtype=np.float32)

with open(script_dir.parent / "Motion_Library" / "cluster_centroids.json", "r") as f:
    CLUSTER_CENTROIDS = np.array(json.load(f), dtype=np.float32)

participant_clusters = None
seed = int(hashlib.sha256(expInfo["participant"].encode()).hexdigest(),16) & 0xFFFFFFFF
rng = np.random.default_rng(seed)

# ───────────────────────────────────────────────────────
#  Two color palette system - counterbalanced across blocks
# ───────────────────────────────────────────────────────
# Define two distinct color palettes for low/high cues across angle blocks
PALETTE_SET_1 = ("blue", "green")
PALETTE_SET_2 = ("red", "yellow")

# PROPER COUNTERBALANCING: Convert participant ID to index (0-7 for 8 conditions)
try:
    participant_num = int(expInfo["participant"])
except ValueError:
    participant_num = int(hashlib.sha256(expInfo["participant"].encode()).hexdigest(), 16) & 0xFFFF

cb_index = participant_num % 8

# Factor 1: Learning order (0° vs 90° first)
learning_order_first_angle = 0 if (cb_index & 1) == 0 else 90
learning_order = [learning_order_first_angle, 90 if learning_order_first_angle == 0 else 0]

# Factor 2: Which palette for first angle (blue/green vs red/yellow)
palette_first_is_blue_green = ((cb_index >> 1) & 1) == 0
if palette_first_is_blue_green:
    PALETTE_FOR_FIRST_ANGLE = PALETTE_SET_1
    PALETTE_FOR_SECOND_ANGLE = PALETTE_SET_2
else:
    PALETTE_FOR_FIRST_ANGLE = PALETTE_SET_2
    PALETTE_FOR_SECOND_ANGLE = PALETTE_SET_1

# Factor 3: Color-difficulty mapping within first palette
first_palette_flip = ((cb_index >> 2) & 1) == 1
if first_palette_flip:
    PALETTE_FOR_FIRST_ANGLE = (PALETTE_FOR_FIRST_ANGLE[1], PALETTE_FOR_FIRST_ANGLE[0])

# Second angle always flips to ensure variety
PALETTE_FOR_SECOND_ANGLE = (PALETTE_FOR_SECOND_ANGLE[1], PALETTE_FOR_SECOND_ANGLE[0])

# Log counterbalancing assignment
expInfo["learning_order"] = str(learning_order)
expInfo["counterbalance_index"] = cb_index
print("=" * 60)
print("COUNTERBALANCING ASSIGNMENT:")
print(f"  Participant: {expInfo['participant']} → CB Index: {cb_index}/8")
print(f"  Learning order: {learning_order}")
print(f"  First angle ({learning_order[0]}°): {PALETTE_FOR_FIRST_ANGLE} (low=hard, high=easy)")
print(f"  Second angle ({learning_order[1]}°): {PALETTE_FOR_SECOND_ANGLE} (low=hard, high=easy)")
print("=" * 60)

# ───────────────────────────────────────────────────────
#  Trajectory signature function (needed for universal selection)
# ───────────────────────────────────────────────────────

def get_trajectory_signature(trajectory):
    """Key movement characteristics for matching"""
    velocities = np.diff(trajectory, axis=0)
    if len(velocities) == 0:
        return {'mean_speed':0,'speed_variability':0,'path_length':0,'net_displacement':0,'speed_percentiles':np.array([0,0,0])}
    speeds = np.linalg.norm(velocities, axis=1)
    return {
        'mean_speed': np.mean(speeds),
        'speed_variability': np.std(speeds),
        'path_length': np.sum(speeds),
        'net_displacement': np.linalg.norm(trajectory[-1] - trajectory[0]),
        'speed_percentiles': np.percentile(speeds, [25, 50, 75])
    }

# ───────────────────────────────────────────────────────
#  Universal trajectory set (same for all participants)
# ───────────────────────────────────────────────────────

def select_universal_trajectory_set():
    """Pre-select trajectories for all participants with deterministic ranking.
    Produces two sets:
      - Primary: best 1,240 trajectories used for all standard trials
      - Overflow: next-best 40 trajectories used only if extra trials are needed
    """
    global valid_snippet_indices, universal_trajectory_set_primary, universal_trajectory_set_overflow, universal_trajectory_set

    total_valid = len(valid_snippet_indices)
    if total_valid < 1240:
        print(f"Warning: Only {total_valid} valid trajectories, fewer than 1,240 required for primary set")
        universal_trajectory_set_primary = valid_snippet_indices.copy()
        universal_trajectory_set_overflow = []
        universal_trajectory_set = universal_trajectory_set_primary.copy()
        return universal_trajectory_set.copy()

    # Use a fixed seed for deterministic selection across all participants (kept for clarity)
    selection_rng = np.random.default_rng(42)

    print("Selecting universal trajectory sets (Primary 1,240 + Overflow 40)...")

    # Score all valid trajectories by quality
    trajectory_scores = []
    for idx in valid_snippet_indices:
        trajectory = motion_pool[idx]
        traj_cumsum = np.cumsum(trajectory, axis=0)
        sig = get_trajectory_signature(traj_cumsum)

        # Quality score: prefer moderate speeds, good variability, reasonable length
        speed_score = 1.0 / (1.0 + abs(sig['mean_speed'] - 8.0))
        variability_score = 1.0 / (1.0 + abs(sig['speed_variability'] - 3.0))
        length_score = min(1.0, sig['path_length'] / 100.0)

        overall_score = speed_score * variability_score * length_score
        trajectory_scores.append((overall_score, idx))

    # Sort by quality score (best first) and take primary + overflow slices
    trajectory_scores.sort(reverse=True)
    primary_indices = [idx for score, idx in trajectory_scores[:1240]]
    overflow_indices = [idx for score, idx in trajectory_scores[1240:1280]] if total_valid >= 1280 else []

    universal_trajectory_set_primary = primary_indices
    universal_trajectory_set_overflow = overflow_indices
    universal_trajectory_set = primary_indices + overflow_indices

    print(f"Selected Primary: {len(primary_indices)} (best={trajectory_scores[0][0]:.3f})")
    if overflow_indices:
        print(f"Selected Overflow: {len(overflow_indices)} (worst_primary={trajectory_scores[1239][0]:.3f}, best_overflow={trajectory_scores[1240][0]:.3f} if available)")
    else:
        print("No overflow set available (valid < 1,280)")

    return universal_trajectory_set.copy()

# Will be initialized after preprocessing
universal_trajectory_set_primary = []
universal_trajectory_set_overflow = []
universal_trajectory_set = []
used_trajectory_indices = set()  # Track which ones from universal set have been used
trajectory_usage_stats = {"used_count": 0, "total_needed": 1600}  # For monitoring

# Global trial counter (persists across all phases for consistent trial numbering)
global_trial_counter = 0

# -------------------------------------------------------------------
#  Helpers for simulation mode
# -------------------------------------------------------------------
class SimulatedMouse:
    def __init__(self):
        self._pos = np.array([0.0, 0.0], dtype=float)
    def setPos(self, pos=(0, 0)):
        self._pos = np.array(pos, dtype=float)
    def getPos(self):
        self._pos += rng.normal(0, 3, 2)
        return self._pos.tolist()

def wait_keys(keys=None):
    if SIMULATE:
        if keys is None:
            core.wait(0.2)
            return ["space"]
        allowed = [k for k in keys if k != "escape"] or ["space"]
        return [rng.choice(allowed)]
    return event.waitKeys(keyList=keys)

def show_break_screen(trials_completed, total_trials_in_block, block_num):
    """Show a 30-second break screen with countdown timer."""
    
    # Create break message
    break_msg = visual.TextStim(
        win=win,
        text=f"""BREAK TIME
        
Please take a short break. You have completed {trials_completed} trials.
Progress: {trials_completed}/{total_trials_in_block} trials in Block {block_num}

Break time remaining: 30 seconds""",
        pos=(0, 50),
        color='white',
        height=30,
        wrapWidth=800
    )
    
    # Create countdown text
    countdown_text = visual.TextStim(
        win=win,
        text='30',
        pos=(0, -100),
        color='yellow',
        height=60
    )
    
    # 30-second countdown
    break_clock = core.Clock()
    while break_clock.getTime() < 30.0:
        remaining_time = 30 - int(break_clock.getTime())
        countdown_text.text = str(remaining_time)
        
        # Update break message with current countdown
        break_msg.text = f"""BREAK TIME
        
Please take a short break. You have completed {trials_completed} trials.
Progress: {trials_completed}/{total_trials_in_block} trials in Block {block_num}

Break time remaining: {remaining_time} seconds"""
        
        break_msg.draw()
        countdown_text.draw()
        win.flip()
        
        # Check for escape during break
        if not SIMULATE:
            keys = event.getKeys(['escape'])
            if keys and 'escape' in keys:
                _save()
                core.quit()
        
        core.wait(0.1)  # Small delay for smooth countdown
    
    # After countdown, show continue option
    final_msg = visual.TextStim(
        win=win,
        text=f"""BREAK COMPLETE
        
You have completed {trials_completed} trials.
Progress: {trials_completed}/{total_trials_in_block} trials in Block {block_num}

Press SPACE to continue""",
        pos=(0, 0),
        color='white',
        height=30,
        wrapWidth=800
    )
    
    final_msg.draw()
    win.flip()
    wait_keys(['space', 'escape'])

def show_phase_transition(completed_block, next_block_info, rest_duration=30):
    """
    Show a rest screen between phases with countdown timer.
    
    Args:
        completed_block: The block number just completed (1-5)
        next_block_info: Brief neutral description of what comes next
        rest_duration: Duration of mandatory rest in seconds (default 30)
    """
    # Create transition message
    transition_msg = visual.TextStim(
        win=win,
        text="",
        pos=(0, 100),
        color='white',
        height=30,
        wrapWidth=800,
        alignText='center'
    )
    
    # Create countdown text
    countdown_text = visual.TextStim(
        win=win,
        text='',
        pos=(0, -50),
        color='yellow',
        height=80
    )
    
    # Create "please wait" text
    wait_text = visual.TextStim(
        win=win,
        text='Please rest. The experiment will continue shortly.',
        pos=(0, -150),
        color='gray',
        height=20
    )
    
    # Countdown phase
    rest_clock = core.Clock()
    while rest_clock.getTime() < rest_duration:
        remaining_time = rest_duration - int(rest_clock.getTime())
        
        transition_msg.text = f"""Block {completed_block} Complete

{next_block_info}"""
        
        countdown_text.text = str(remaining_time)
        
        transition_msg.draw()
        countdown_text.draw()
        wait_text.draw()
        win.flip()
        
        # Check for escape during rest
        if not SIMULATE:
            keys = event.getKeys(['escape'])
            if keys and 'escape' in keys:
                _save()
                core.quit()
        
        core.wait(0.1)
    
    # After countdown, show ready message
    ready_msg = visual.TextStim(
        win=win,
        text=f"""Block {completed_block} Complete

{next_block_info}

Press SPACE when you are ready to continue.""",
        pos=(0, 0),
        color='white',
        height=30,
        wrapWidth=800,
        alignText='center'
    )
    
    ready_msg.draw()
    win.flip()
    wait_keys(['space', 'escape'])

# ───────────────────────────────────────────────────────
#  Trajectory Quality Control and Preprocessing
# ───────────────────────────────────────────────────────

def analyze_trajectory_quality(trajectory):
    velocities = np.diff(trajectory, axis=0)
    speeds = np.linalg.norm(velocities, axis=1)
    mean_speed = np.mean(speeds)
    std_speed = np.std(speeds)
    max_speed = np.max(speeds)
    min_speed = np.min(speeds)
    zero_movement_ratio = np.sum(speeds < 0.5) / len(speeds)
    high_jitter_ratio = np.sum(speeds > mean_speed + 3*std_speed) / len(speeds)
    if len(velocities) > 1:
        unit_velocities = velocities / (sps := (speeds.reshape(-1, 1) + 1e-9))
        angle_changes = np.arccos(np.clip(np.sum(unit_velocities[:-1] * unit_velocities[1:], axis=1), -1, 1))
        mean_angle_change = np.mean(angle_changes)
        jerkiness = np.std(angle_changes)
    else:
        mean_angle_change = 0
        jerkiness = 0
    return {
        'mean_speed': mean_speed,
        'std_speed': std_speed,
        'zero_movement_ratio': zero_movement_ratio,
        'high_jitter_ratio': high_jitter_ratio,
        'mean_angle_change': mean_angle_change,
        'jerkiness': jerkiness,
        'speed_range': max_speed - min_speed
    }

def is_trajectory_valid(trajectory, min_speed=1.0, max_zero_ratio=0.3, max_jitter_ratio=0.1, max_jerkiness=1.5):
    quality = analyze_trajectory_quality(trajectory)
    if quality['mean_speed'] < min_speed:
        return False, "mean_speed_too_low"
    if quality['zero_movement_ratio'] > max_zero_ratio:
        return False, "too_much_zero_movement"
    if quality['high_jitter_ratio'] > max_jitter_ratio:
        return False, "too_much_jitter"
    if quality['jerkiness'] > max_jerkiness:
        return False, "too_jerky"
    return True, "valid"

def normalize_trajectory(trajectory, target_speed_range=(3.0, 12.0), smooth_factor=0.45):
	if len(trajectory) < 2:
		return trajectory
	velocities = np.diff(trajectory, axis=0)
	speeds = np.linalg.norm(velocities, axis=1)
	current_mean_speed = np.mean(speeds)
	if current_mean_speed > 0:
		target_mean_speed = np.mean(target_speed_range)
		speed_scale = target_mean_speed / current_mean_speed
		velocities = velocities * speed_scale
	smoothed_velocities = velocities.copy()
	for i in range(1, len(velocities)):
		smoothed_velocities[i] = smooth_factor * smoothed_velocities[i-1] + (1 - smooth_factor) * velocities[i]
	normalized_trajectory = [trajectory[0]]
	for vel in smoothed_velocities:
		next_point = normalized_trajectory[-1] + vel
		normalized_trajectory.append(next_point)
	return np.array(normalized_trajectory)

def preprocess_motion_pool():
    """Preprocess motion pool to ensure quality and consistency"""
    global motion_pool, snippet_features, snippet_labels
    print("Preprocessing motion pool for quality control...")
    initial_count = len(motion_pool)
    processed_snippets = []
    processed_features = []
    processed_labels = []
    for i, snippet in enumerate(motion_pool):
        trajectory = np.cumsum(snippet, axis=0)
        is_valid, reason = is_trajectory_valid(trajectory)
        if is_valid:
            normalized_trajectory = normalize_trajectory(trajectory)
            velocities = np.diff(normalized_trajectory, axis=0)
            processed_snippets.append(velocities)
            processed_features.append(snippet_features[i])
            processed_labels.append(snippet_labels[i])
        else:
            print(f"Removed snippet {i}: {reason}")
    motion_pool = np.array(processed_snippets)
    snippet_features = np.array(processed_features)
    snippet_labels = np.array(processed_labels)
    global SNIP_LEN
    SNIP_LEN = motion_pool.shape[1] if len(motion_pool) > 0 else 0
    print(f"Motion pool preprocessed: kept {len(processed_snippets)}/{initial_count} snippets")
    return list(range(len(processed_snippets)))

valid_snippet_indices = preprocess_motion_pool()

# Now initialize the universal trajectory sets after preprocessing
universal_trajectory_set = select_universal_trajectory_set()
print(
    f"Universal trajectory sets initialized: Primary={len(universal_trajectory_set_primary)}; "
    f"Overflow={len(universal_trajectory_set_overflow)}; Total={len(universal_trajectory_set)}"
)

def find_matched_trajectory_pair():
    """Find two UNUSED trajectories with similar movement characteristics.
    Prefer primary set; only use overflow if necessary.
    """
    global universal_trajectory_set, universal_trajectory_set_primary, universal_trajectory_set_overflow
    global used_trajectory_indices, trajectory_usage_stats
    
    # Safety check: ensure universal sets are initialized
    if not universal_trajectory_set:
        universal_trajectory_set = valid_snippet_indices.copy()
        universal_trajectory_set_primary = universal_trajectory_set.copy()
        universal_trajectory_set_overflow = []
    
    # Available (unused) trajectories by priority
    available_primary = [idx for idx in universal_trajectory_set_primary if idx not in used_trajectory_indices]
    available_overflow = [idx for idx in universal_trajectory_set_overflow if idx not in used_trajectory_indices]
    available_indices = available_primary if len(available_primary) >= 2 else (available_primary + available_overflow)
    
    # Check if we have enough unused trajectories
    if len(available_indices) < 2:
        pass  # Low number of unused trajectories
        # Emergency fallback: use any available trajectories (even if used before)
        if len(available_indices) == 1:
            # Can't use same trajectory - need to find a different one
            target_idx = available_indices[0]
            # Find different trajectory from universal set (even if used before)
            different_options = [idx for idx in universal_trajectory_set if idx != target_idx]
            if different_options:
                distractor_idx = rng.choice(different_options)
                return target_idx, distractor_idx
            else:
                # Absolute emergency - use different random trajectory
                return None, None  # Will trigger fallback
        elif len(available_indices) == 0:
            print("No unused trajectories! Using random valid trajectories.")
            return None, None  # Will trigger fallback in calling function
    
    # Sample from unused trajectories for matching (prefer primary pool)
    sample_pool = available_primary if len(available_primary) >= 2 else available_indices
    sample_size = min(100, len(sample_pool))
    candidate_indices = rng.choice(sample_pool, size=sample_size, replace=False)
    
    # Get signatures for all candidates
    signatures = []
    for idx in candidate_indices:
        trajectory = motion_pool[idx]
        sig = get_trajectory_signature(np.cumsum(trajectory, axis=0))
        signatures.append((idx, sig))
    
    # Find the best matching pair among unused trajectories
    best_score = float('inf')
    best_pair = (None, None)
    
    for i in range(len(signatures)):
        for j in range(i + 1, len(signatures)):
            idx1, sig1 = signatures[i]
            idx2, sig2 = signatures[j]
            
            # Calculate similarity score (lower is better)
            speed_diff = abs(sig1['mean_speed'] - sig2['mean_speed'])
            var_diff = abs(sig1['speed_variability'] - sig2['speed_variability'])
            length_diff = abs(sig1['path_length'] - sig2['path_length']) / max(sig1['path_length'], sig2['path_length'])
            
            # Combined similarity score
            similarity_score = speed_diff + var_diff + length_diff * 10
            
            if similarity_score < best_score:
                best_score = similarity_score
                best_pair = (idx1, idx2)
    
    # Mark trajectories as used if we found a valid pair
    if best_pair[0] is not None and best_pair[1] is not None:
        used_trajectory_indices.add(best_pair[0])
        used_trajectory_indices.add(best_pair[1])
        trajectory_usage_stats["used_count"] += 2
        
        # Progress monitoring
        remaining = trajectory_usage_stats["total_needed"] - trajectory_usage_stats["used_count"]
        unused_count = len([idx for idx in universal_trajectory_set if idx not in used_trajectory_indices])
        
        if trajectory_usage_stats["used_count"] % 100 == 0:  # Print every 50 trials
            pass  # Trajectory usage tracked silently
    
    return best_pair

def apply_consistent_smoothing(trajectory1, trajectory2):
    """Apply consistent smoothing to both trajectories"""
    def smooth_trajectory(traj, window_size=3):
        if len(traj) < window_size:
            return traj
        smoothed = traj.copy()
        for i in range(len(traj)):
            start = max(0, i - window_size // 2)
            end = min(len(traj), i + window_size // 2 + 1)
            smoothed[i] = np.mean(traj[start:end], axis=0)
        return smoothed
    pos1 = np.cumsum(trajectory1, axis=0)
    pos2 = np.cumsum(trajectory2, axis=0)
    smooth_pos1 = smooth_trajectory(pos1)
    smooth_pos2 = smooth_trajectory(pos2)
    vel1 = np.diff(smooth_pos1, axis=0)
    vel2 = np.diff(smooth_pos2, axis=0)
    return vel1, vel2

def sample_from_all_trajectories(n_samples=20):
    if len(valid_snippet_indices) >= n_samples:
        return rng.choice(valid_snippet_indices, size=n_samples, replace=False)
    else:
        print(f"Warning: Only {len(valid_snippet_indices)} trajectories available")
        return valid_snippet_indices.copy()

# ───────────────────────────────────────────────────────
#  Constants and Parameters
# ───────────────────────────────────────────────────────
OFFSET_X = 300
LOWPASS = 0.5
START_BOOST_EASY = 0.05
START_BOOST_EASY_TRIALS = 6

# ───────────────────────────────────────────────────────
#  Paths & ExperimentHandler
# ───────────────────────────────────────────────────────
root = pathlib.Path(__file__).parent / "data"
subjects_dir = root / "subjects"
subjects_dir.mkdir(parents=True, exist_ok=True)

participant_id = expInfo['participant']
base_filename = f"CDT_v2_blockwise_fast_response_{participant_id}"
csv_path = subjects_dir / f"{base_filename}.csv"
kinematics_csv_path = subjects_dir / f"{base_filename}_kinematics.csv"

i = 1
while csv_path.exists():
    new_filename = f"CDT_v2_blockwise_fast_response_{participant_id}_{i}"
    csv_path = subjects_dir / f"{new_filename}.csv"
    kinematics_csv_path = subjects_dir / f"{new_filename}_kinematics.csv"
    i += 1

thisExp = data.ExperimentHandler(
    name=expName, extraInfo=expInfo,
    savePickle=False, saveWideText=False,
    dataFileName=str(root / base_filename)
)

# ───────────────────────────────────────────────────────
#  Window & stimuli
# ───────────────────────────────────────────────────────
win = visual.Window((1920,1080), fullscr=not SIMULATE, color=[0.5]*3, units="pix", allowGUI=True)
win.setMouseVisible(False)
square = visual.Rect(win, 40, 40, fillColor="black", lineColor="black")
dot = visual.Circle(win, 20, fillColor="black", lineColor="black")
fix = visual.TextStim(win, "+", color="white", height=60)
msg = visual.TextStim(win, "", color="white", height=26, wrapWidth=1000)
feedbackTxt = visual.TextStim(win, "", color="black", height=80)

confine = lambda p, l=250: p if (r := math.hypot(*p)) <= l else (p[0]*l/r, p[1]*l/r)
rotate = lambda vx, vy, a: (
    vx * math.cos(math.radians(a)) - vy * math.sin(math.radians(a)),
    vx * math.sin(math.radians(a)) + vy * math.cos(math.radians(a))
)

# ───────────────────────────────────────────────────────
#  Progress Bar Class
# ───────────────────────────────────────────────────────
class ProgressBar:
    """Displays experiment progress at top of screen."""
    
    def __init__(self, win, total_blocks=5):
        self.win = win
        self.total_blocks = total_blocks
        self.current_block = 1
        self.current_trial = 0
        self.total_trials_in_block = 0
        
        # Progress bar background (gray)
        self.bar_bg = visual.Rect(
            win, width=400, height=20,
            pos=(0, win.size[1]//2 - 40),
            fillColor='gray', lineColor='white'
        )
        
        # Progress bar fill (green)
        self.bar_fill = visual.Rect(
            win, width=0, height=18,
            pos=(0, win.size[1]//2 - 40),
            fillColor='green', lineColor=None
        )
        
        # Block text
        self.block_text = visual.TextStim(
            win, text="Block 1 of 5",
            pos=(0, win.size[1]//2 - 70),
            color='white', height=18
        )
        
    def set_block(self, block_num, total_trials=0):
        """Set current block and reset trial counter."""
        self.current_block = block_num
        self.current_trial = 0
        self.total_trials_in_block = total_trials
        self.block_text.text = f"Block {block_num} of {self.total_blocks}"
    
    def set_trial(self, trial_num):
        """Update current trial number."""
        self.current_trial = trial_num
    
    def draw(self):
        """Draw progress bar (visual only, no numbers)."""
        # Calculate fill width based on trial progress
        if self.total_trials_in_block > 0:
            progress = self.current_trial / self.total_trials_in_block
            fill_width = 398 * progress
            # Adjust position so bar fills from left to right
            self.bar_fill.width = fill_width
            self.bar_fill.pos = (-199 + fill_width/2, self.win.size[1]//2 - 40)
        else:
            self.bar_fill.width = 0
        
        self.bar_bg.draw()
        self.bar_fill.draw()
        self.block_text.draw()

# Create global progress bar instance
progress_bar = ProgressBar(win)

# Demo trial function removed as requested

# ───────────────────────────────────────────────────────
#  Basic condition labels
# ───────────────────────────────────────────────────────

EXPECT = ["low", "high"]

# Separation for learning levels in logit space (perceptually symmetric)
LEARNING_LOGIT_SEPARATION = 1.2

# Target performance level for defining medium difficulty during test phase
MEDIUM_TARGET_PCORR = 0.70

# Store per-angle learning levels to reuse in test phase
learning_levels_by_angle = {}

# ───────────────────────────────────────────────────────
#  Trial function with fast response capability
# ───────────────────────────────────────────────────────
def run_trial(
    trial_num, phase, angle_bias, expect_level, mode, catch_type="", target_shape=None, block_num=1,
    prop_override=None, cue_dur_range=None, motion_dur=None, response_window=None
):
    if catch_type == "full":
        prop = 1.0
    elif prop_override is not None:
        prop = float(np.clip(prop_override, 0.02, 0.90))
    elif mode == "true":
        # Legacy path removed; treat as medium if no override is provided
        prop = 0.40
    else:
        prop = 0.40

    # Use black for calibration phase, colored cues for fixed practice phase
    if phase == "calibration":
        cue = "black"
    else:
        cue = low_col if expect_level == "low" else high_col
    fix.color = cue; square.fillColor = square.lineColor = cue; dot.fillColor = dot.lineColor = cue

    fix.draw(); win.flip()
    if cue_dur_range is not None:
        core.wait(float(rng.uniform(cue_dur_range[0], cue_dur_range[1])))
    else:
        core.wait(1.0)

    left_shape = random.choice(['square', 'dot'])
    if left_shape == 'square':
        square.pos = (-OFFSET_X, 0); dot.pos = (OFFSET_X, 0)
    else:
        square.pos = (OFFSET_X, 0); dot.pos = (-OFFSET_X, 0)
    square.draw(); dot.draw(); win.flip()
    
    mouse = SimulatedMouse() if SIMULATE else event.Mouse(win=win, visible=False)
    mouse.setPos((0, 0))
    last = mouse.getPos()
    while True:
        square.draw(); dot.draw(); win.flip()
        x, y = mouse.getPos()
        if math.hypot(x - last[0], y - last[1]) > 0 or SIMULATE: break
        if not SIMULATE and event.getKeys(["escape"]): _save(); core.quit()

    target = target_shape if target_shape is not None else random.choice(["square", "dot"])
    target_snippet_idx, distractor_snippet_idx = find_matched_trajectory_pair()
    if target_snippet_idx is None or distractor_snippet_idx is None:
        pass  # Trajectory matching failed, using fallback
        # Fallback: try to get unused trajectories, preferring primary then overflow
        available_primary = [idx for idx in universal_trajectory_set_primary if idx not in used_trajectory_indices]
        available_overflow = [idx for idx in universal_trajectory_set_overflow if idx not in used_trajectory_indices]
        available_indices = available_primary if len(available_primary) >= 2 else (available_primary + available_overflow)
        if len(available_indices) >= 2:
            selected = rng.choice(available_indices, size=2, replace=False)
            target_snippet_idx, distractor_snippet_idx = selected[0], selected[1]
            # Mark as used
            used_trajectory_indices.add(target_snippet_idx)
            used_trajectory_indices.add(distractor_snippet_idx)
            trajectory_usage_stats["used_count"] += 2
        elif len(available_indices) == 1:
            # Can't use same trajectory for both - use available + one random different from universal set
            target_snippet_idx = available_indices[0]
            # Find a different trajectory from primary then overflow
            combined_sets = universal_trajectory_set_primary + universal_trajectory_set_overflow
            different_options = [idx for idx in combined_sets if idx != target_snippet_idx]
            if different_options:
                distractor_snippet_idx = rng.choice(different_options)
            else:
                # Absolute emergency: use different random trajectory
                distractor_snippet_idx = rng.choice([idx for idx in range(len(motion_pool)) if idx != target_snippet_idx])
            used_trajectory_indices.add(target_snippet_idx)
            used_trajectory_indices.add(distractor_snippet_idx)
            trajectory_usage_stats["used_count"] += 2
        else:
            pass  # No unused trajectories in universal set
            combined_sets = universal_trajectory_set_primary + universal_trajectory_set_overflow if (universal_trajectory_set_primary or universal_trajectory_set_overflow) else universal_trajectory_set
            if len(combined_sets) >= 2:
                selected = rng.choice(combined_sets, size=2, replace=False)
                target_snippet_idx, distractor_snippet_idx = selected[0], selected[1]
            else:
                pass  # Insufficient trajectories in universal set
                # Ensure different trajectories even in emergency
                available_range = list(range(len(motion_pool)))
                selected = rng.choice(available_range, size=2, replace=False)
                target_snippet_idx, distractor_snippet_idx = selected[0], selected[1]
    target_snippet = motion_pool[target_snippet_idx]
    distractor_snippet = motion_pool[distractor_snippet_idx]
    target_snippet, distractor_snippet = apply_consistent_smoothing(target_snippet, distractor_snippet)

    trial_kinematics = []
    clk = core.Clock(); frame = 0
    vt = vd = np.zeros(2, np.float32)
    mag_m_lp = 0.0
    prev_d = np.zeros(2, np.float32)
    
    # Variables for early response capability
    resp_shape = None
    rt_choice = np.nan
    early_response = False

    total_motion_duration = 5.0  # Fixed 5 seconds
    response_start_time = core.getTime()
    
    # Clear any existing events
    event.clearEvents(eventType='keyboard')
    
    # Determine applied rotation angle: randomize ±90° when angle_bias == 90
    applied_angle = angle_bias
    if angle_bias == 90:
        applied_angle = int(rng.choice([90, -90]))
    
    rt_frame = None
    while clk.getTime() < total_motion_duration and resp_shape is None:
        x, y = mouse.getPos()
        dx, dy = x - last[0], y - last[1]
        last = (x, y)
        dx, dy = rotate(dx, dy, applied_angle)
        target_ou_dx, target_ou_dy = target_snippet[frame % len(target_snippet)]
        distractor_ou_dx, distractor_ou_dy = distractor_snippet[frame % len(distractor_snippet)]
        frame += 1
        mag_m = math.hypot(dx, dy)
        MAX_SPEED = 20.0
        mag_m = min(mag_m, MAX_SPEED)
        if mag_m > 0:
            original_mag = math.hypot(dx, dy)
            if original_mag > MAX_SPEED:
                scale_factor = MAX_SPEED / original_mag
                dx = dx * scale_factor; dy = dy * scale_factor
        if frame == 1: mag_m_lp = mag_m
        else: mag_m_lp = 0.5 * mag_m_lp + 0.5 * mag_m
        mag_target = math.hypot(target_ou_dx, target_ou_dy)
        if mag_target > 0:
            dir_target_x, dir_target_y = target_ou_dx / mag_target, target_ou_dy / mag_target
        else:
            dir_target_x, dir_target_y = 0, 0
        mag_distractor = math.hypot(distractor_ou_dx, distractor_ou_dy)
        if mag_distractor > 0:
            dir_distractor_x, dir_distractor_y = distractor_ou_dx / mag_distractor, distractor_ou_dy / mag_distractor
        else:
            dir_distractor_x, dir_distractor_y = 0, 0
        target_ou_dx = dir_target_x * mag_m_lp; target_ou_dy = dir_target_y * mag_m_lp
        distractor_ou_dx = dir_distractor_x * mag_m_lp; distractor_ou_dy = dir_distractor_y * mag_m_lp
        tdx = prop * dx + (1 - prop) * target_ou_dx
        tdy = prop * dy + (1 - prop) * target_ou_dy
        mouse_speed = math.hypot(dx, dy); linear_bias = 0.0
        if mouse_speed > 0 and frame > 10 and len(trial_kinematics) >= 5:
            recent_positions = [(d['mouse_x'], d['mouse_y']) for d in trial_kinematics[-5:]] + [(x, y)]
            if len(recent_positions) >= 3:
                total_dist = sum(math.hypot(recent_positions[i+1][0] - recent_positions[i][0],
                                            recent_positions[i+1][1] - recent_positions[i][1])
                                 for i in range(len(recent_positions)-1))
                straight_dist = math.hypot(recent_positions[-1][0] - recent_positions[0][0],
                                           recent_positions[-1][1] - recent_positions[0][1])
                if total_dist > 0:
                    linearity = straight_dist / total_dist
                    if mouse_speed < 10.0 and linearity > 0.8:
                        linear_bias = min(0.3, linearity * 0.4)
        if linear_bias > 0:
            perp_dx, perp_dy = -dy, dx
            perp_mag = math.hypot(perp_dx, perp_dy)
            if perp_mag > 0:
                perp_dx /= perp_mag; perp_dy /= perp_mag
                cursor_mag = math.hypot(dx, dy)
                perp_dx *= cursor_mag; perp_dy *= cursor_mag
            else:
                perp_dx = perp_dy = 0
            ddx = (1 - linear_bias) * distractor_ou_dx + linear_bias * perp_dx
            ddy = (1 - linear_bias) * distractor_ou_dy + linear_bias * perp_dy
        else:
            ddx = distractor_ou_dx; ddy = distractor_ou_dy
        ddx_smooth, ddy_smooth = 0.4 * prev_d[0] + 0.6 * ddx, 0.4 * prev_d[1] + 0.6 * ddy
        prev_d = np.array([ddx_smooth, ddy_smooth])
        vt = LOWPASS * vt + (1 - LOWPASS) * np.array([tdx, tdy])
        vd = LOWPASS * vd + (1 - LOWPASS) * np.array([ddx_smooth, ddy_smooth])

        # --- EVIDENCE (momentary, displayed velocities) ---
        vm = np.array([dx, dy], dtype=float)
        mouse_speed = np.linalg.norm(vm) + 1e-9

        # use on-screen (mixed + low-pass) velocities for target and distractor
        vt_disp = np.array(vt, dtype=float)     # displayed target velocity this frame
        vd_disp = np.array(vd, dtype=float)     # displayed distractor velocity this frame

        ut = vt_disp / (np.linalg.norm(vt_disp) + 1e-9)
        ud = vd_disp / (np.linalg.norm(vd_disp) + 1e-9)

        cos_T = np.dot(vm, ut) / mouse_speed
        cos_D = np.dot(vm, ud) / mouse_speed

        evidence = (cos_T - cos_D) * (mouse_speed - 1e-9)  # identical scale as before

        if target == "square":
            square.pos = confine(tuple(square.pos + vt))
            dot.pos = confine(tuple(dot.pos + vd))
        else:
            dot.pos = confine(tuple(dot.pos + vt))
            square.pos = confine(tuple(square.pos + vd))

        trial_kinematics.append({
            'timestamp': clk.getTime(), 'frame': frame, 'mouse_x': x, 'mouse_y': y,
            'square_x': square.pos[0], 'square_y': square.pos[1], 'dot_x': dot.pos[0], 'dot_y': dot.pos[1],
            'evidence': evidence
        })
        
        # Check for early response during motion
        if not SIMULATE:
            keys = event.getKeys(['a', 's', 'escape'], timeStamped=True)
            if keys:
                key, key_time = keys[0]
                if key == "escape": 
                    _save(); core.quit()
                elif key == "a":
                    resp_shape = "square"
                    rt_choice = key_time - response_start_time
                    rt_frame = frame
                    early_response = True
                elif key == "s":
                    resp_shape = "dot"
                    rt_choice = key_time - response_start_time
                    rt_frame = frame
                    early_response = True
        else:
            # Simulation: random early response sometimes
            if frame > 60 and rng.random() < 0.1:  # 10% chance after 1 second
                resp_shape = rng.choice(["square", "dot"])
                rt_choice = clk.getTime()
                early_response = True
        
        square.draw(); dot.draw(); win.flip()
    
    # If no response during motion, mark timeout and skip remaining screens
    if resp_shape is None:
        msg.text = "Too slow!\n\nPlease respond faster next time."
        msg.draw(); win.flip(); core.wait(2.0)
        resp_shape = "timeout"
        correct = np.nan
        # Log minimal kinematics metadata for timeout trial
        for frame_data in trial_kinematics:
            frame_data.update({
                'participant': expInfo['participant'], 'session': expInfo['session'],
                'trial_num': trial_num, 'phase': phase, 'angle_bias': angle_bias,
                'applied_angle_bias': applied_angle,
                'expect_level': expect_level, 'prop_used': prop, 'confidence_rating': np.nan,
                'agency_rating': np.nan, 'block_num': block_num, 'early_response': False,
                'true_shape': target, 'resp_shape': resp_shape
            })
            kinematics_data.append(frame_data)
        
        # Aggregate evidence across trial (even for timeout)
        frame_evidence = [d['evidence'] for d in trial_kinematics]
        mean_evidence = np.mean(frame_evidence)
        sum_evidence = np.sum(frame_evidence)
        var_evidence = np.var(frame_evidence)
        
        return dict(
            target_snippet_id=target_snippet_idx, distractor_snippet_id=distractor_snippet_idx,
            catch_type=catch_type, phase=phase, block_num=block_num,
            angle_bias=angle_bias, applied_angle_bias=applied_angle, expect_level=expect_level, true_shape=target, resp_shape=resp_shape,
            confidence_rating=np.nan, accuracy=np.nan, rt_choice=np.nan,
            agency_rating=np.nan, prop_used=prop, early_response=False,
            mean_evidence=mean_evidence, sum_evidence=sum_evidence, var_evidence=var_evidence,
            # Pre-response evidence metrics not applicable for timeout
            rt_frame=np.nan, num_frames_preRT=np.nan,
            mean_evidence_preRT=np.nan, sum_evidence_preRT=np.nan, var_evidence_preRT=np.nan,
            cum_evidence_preRT=np.nan, max_cum_evidence_preRT=np.nan, min_cum_evidence_preRT=np.nan,
            max_abs_cum_evidence_preRT=np.nan, prop_positive_evidence_preRT=np.nan
        )

    correct = int(resp_shape == target)

    # Confidence rating (1-4 scale) - separate screen (skip for calibration phase)
    confidence_rating = np.nan
    if phase not in ["calibration", "practice"]:  # Only ask for confidence in test phase
        if SIMULATE:
            confidence_rating = float(rng.integers(1, 5))
        else:
            msg.text = "How confident are you in your choice?\n\n1 = Guessing\n2 = Slightly confident\n3 = Fairly confident\n4 = Certain"
            msg.draw(); win.flip()
            
            confidence_keys = ['1', '2', '3', '4']
            conf_key = wait_keys(confidence_keys + ["escape"])[0]
            if conf_key == "escape": 
                _save(); core.quit()
            else:
                confidence_rating = int(conf_key)
            core.wait(0.2)

    # Show feedback for calibration and learning trials (no feedback in test)
    if phase == "calibration" or phase == "practice":
        feedbackTxt.text = "Right" if correct else "Wrong"
        feedbackTxt.draw(); win.flip(); core.wait(0.8)
        win.flip(); core.wait(0.3)

    # Agency rating (only for test trials) - using memory preference [[memory:4144075]]
    agency_rating = np.nan
    if phase == "test":
        if SIMULATE:
            agency_rating = float(rng.integers(1, 8))
        else:
            # Clear keyboard buffer to prevent carryover from confidence rating
            event.clearEvents(eventType='keyboard')
            
            msg.text = "How much control did you feel over the shape's movement?\n\nUse the full range of the scale."
            scale_positions = [(-450, -100), (-300, -100), (-150, -100), (0, -100), (150, -100), (300, -100), (450, -100)]
            scale_labels = ["1\nNo control\n(moved on\nits own)","2","3","4\nUncertain","5","6","7\nComplete\ncontrol"]
            scale_stimuli = [visual.TextStim(win, text=label, pos=pos, height=18, color='white', alignText='center')
                             for pos, label in zip(scale_positions, scale_labels)]
            rating = None
            while rating is None:
                msg.draw()
                for stim in scale_stimuli: stim.draw()
                win.flip()
                keys = event.getKeys(['1','2','3','4','5','6','7','escape'])
                if keys:
                    if 'escape' in keys: _save(); core.quit()
                    else: rating = int(keys[0])
                core.wait(0.01)
            agency_rating = rating
            core.wait(0.2)

    for frame_data in trial_kinematics:
        frame_data.update({
            'participant': expInfo['participant'], 'session': expInfo['session'],
            'trial_num': trial_num, 'phase': phase, 'angle_bias': angle_bias,
            'applied_angle_bias': applied_angle,
            'expect_level': expect_level, 'prop_used': prop, 'confidence_rating': confidence_rating,
            'agency_rating': agency_rating, 'block_num': block_num, 'early_response': early_response,
            'true_shape': target, 'resp_shape': resp_shape
        })
        kinematics_data.append(frame_data)

    # Aggregate evidence across trial
    frame_evidence = [d['evidence'] for d in trial_kinematics]
    mean_evidence = np.mean(frame_evidence)
    sum_evidence = np.sum(frame_evidence)
    var_evidence = np.var(frame_evidence)

    # Pre-response evidence metrics (using all recorded frames which are pre-RT by construction)
    if early_response and len(trial_kinematics) > 0:
        pre_rt_evidence = frame_evidence
        cum = np.cumsum(pre_rt_evidence)
        mean_evidence_preRT = float(np.mean(pre_rt_evidence))
        sum_evidence_preRT = float(np.sum(pre_rt_evidence))
        var_evidence_preRT = float(np.var(pre_rt_evidence))
        max_cum_evidence_preRT = float(np.max(cum))
        min_cum_evidence_preRT = float(np.min(cum))
        max_abs_cum_evidence_preRT = float(np.max(np.abs(cum)))
        prop_positive_evidence_preRT = float(np.mean(np.array(pre_rt_evidence) > 0))
        rt_frame_out = int(trial_kinematics[-1]['frame']) if rt_frame is None else int(rt_frame)
        num_frames_preRT = int(len(pre_rt_evidence))
    else:
        mean_evidence_preRT = np.nan
        sum_evidence_preRT = np.nan
        var_evidence_preRT = np.nan
        max_cum_evidence_preRT = np.nan
        min_cum_evidence_preRT = np.nan
        max_abs_cum_evidence_preRT = np.nan
        prop_positive_evidence_preRT = np.nan
        rt_frame_out = np.nan
        num_frames_preRT = np.nan

    return dict(
        target_snippet_id=target_snippet_idx, distractor_snippet_id=distractor_snippet_idx,
        catch_type=catch_type, phase=phase, block_num=block_num,
        angle_bias=angle_bias, applied_angle_bias=applied_angle, expect_level=expect_level, true_shape=target, resp_shape=resp_shape,
        confidence_rating=confidence_rating, accuracy=correct, rt_choice=rt_choice, 
        agency_rating=agency_rating, prop_used=prop, early_response=early_response,
        mean_evidence=mean_evidence, sum_evidence=sum_evidence, var_evidence=var_evidence,
        rt_frame=rt_frame_out, num_frames_preRT=num_frames_preRT,
        mean_evidence_preRT=mean_evidence_preRT, sum_evidence_preRT=sum_evidence_preRT, var_evidence_preRT=var_evidence_preRT,
        cum_evidence_preRT=sum_evidence_preRT, max_cum_evidence_preRT=max_cum_evidence_preRT, min_cum_evidence_preRT=min_cum_evidence_preRT,
        max_abs_cum_evidence_preRT=max_abs_cum_evidence_preRT, prop_positive_evidence_preRT=prop_positive_evidence_preRT
    )

# ───────────────────────────────────────────────────────
#  QUEST+ style adaptive training (Green=0.90, Blue=0.65)
# ───────────────────────────────────────────────────────

def logit(x):
    x = float(np.clip(x, 1e-6, 1-1e-6))
    return float(np.log(x/(1-x)))

def inv_logit(z):
    return float(1.0/(1.0 + np.exp(-z)))

def clamp_prop(s):
    return float(np.clip(s, 0.02, 0.90))

class QuestPlusStaircase:
    def __init__(self, target_type):
        """
        QUEST+ implementation with proper entropy-based stimulus selection
        target_type: "high" for 80% target, "low" for 60% target, or "neutral" for calibration
        """
        # Grids as specified
        self.s_grid = np.linspace(logit(0.05), logit(0.90), 61)  # Stimulus grid (logit domain)
        self.alpha_grid = np.linspace(logit(0.05), logit(0.90), 61)  # Threshold grid
        self.beta_grid = np.geomspace(1.0, 12.0, 25)  # Slope grid  
        self.lambda_grid = np.array([0.00, 0.01, 0.02, 0.04, 0.06])  # Lapse grid
        self.gamma = 0.5  # 2AFC chance level
        
        self.target_type = target_type
        
        # Priors adjusted based on pilot data (shifted down by ~0.22 from original)
        if target_type == "high":
            alpha_mu = logit(0.48)  # "high target" prior mean (was 0.70)
        elif target_type == "low":
            alpha_mu = logit(0.33)  # "low target" prior mean (was 0.55)
        else:
            alpha_mu = logit(0.40)  # Neutral prior from pilot participants (was 0.625)
        alpha_sd = 1.0  # Prior SD in logits
        
        self.prior_alpha = np.exp(-0.5 * ((self.alpha_grid - alpha_mu) / alpha_sd)**2)
        self.prior_alpha /= self.prior_alpha.sum()
        
        # Beta prior: log-normal with mean 2.5, gsd 2.0
        beta_mean = 2.5
        beta_gsd = 2.0
        ln_beta_mean = np.log(beta_mean)
        ln_beta_sd = np.log(beta_gsd)
        self.prior_beta = np.exp(-0.5 * ((np.log(self.beta_grid) - ln_beta_mean) / ln_beta_sd)**2)
        self.prior_beta /= self.prior_beta.sum()

        # Lambda prior: uniform
        self.prior_lambda = np.ones_like(self.lambda_grid) / len(self.lambda_grid)
        
        # Initialize posteriors to priors
        self.post_alpha = self.prior_alpha.copy()
        self.post_beta = self.prior_beta.copy()
        self.post_lambda = self.prior_lambda.copy()

        # Trial history
        self.trial_count = 0
        self.responses = []  # List of (stimulus_logit, correct) pairs
        
    def psychometric(self, s_logit, alpha, beta, lapse):
        """Psychometric function: p(correct | s; α, β, λ) = γ + (1 - γ - λ) σ(β [s - α])"""
        sigmoid = 1.0 / (1.0 + np.exp(-beta * (s_logit - alpha)))
        return self.gamma + (1.0 - self.gamma - lapse) * sigmoid
    
    def compute_entropy(self, posterior):
        """Compute entropy of a probability distribution"""
        posterior = posterior + 1e-12  # Avoid log(0)
        return -np.sum(posterior * np.log(posterior))
    
    def select_stimulus_entropy_fast(self):
        """Fast approximation of entropy-based stimulus selection for speed"""
        # Use a smaller subset of stimuli for entropy calculation to speed up
        # This maintains the core QUEST+ principle while being much faster
        
        # Subsample stimulus grid for faster computation (every 3rd point)
        s_grid_subset = self.s_grid[::3]  # Reduces from 61 to ~20 points
        
        current_entropy = self.compute_entropy(self.post_alpha)
        best_stimulus = None
        max_info_gain = -np.inf
        
        for s_logit in s_grid_subset:
            # Fast posterior predictive probability using current means
            alpha_mean = np.sum(self.alpha_grid * self.post_alpha)
            beta_mean = np.sum(self.beta_grid * self.post_beta)
            lambda_mean = np.sum(self.lambda_grid * self.post_lambda)
            
            p_correct = self.psychometric(s_logit, alpha_mean, beta_mean, lambda_mean)
            p_incorrect = 1.0 - p_correct
            
            # Skip if probability is too extreme
            if p_correct < 1e-6 or p_incorrect < 1e-6:
                continue
            
            # Simplified entropy approximation using only alpha updates
            # (since alpha/threshold is what we care most about)
            post_alpha_correct = np.zeros_like(self.post_alpha)
            post_alpha_incorrect = np.zeros_like(self.post_alpha)
            
            for i, alpha in enumerate(self.alpha_grid):
                # Use mean beta and lambda for speed
                like_correct = self.psychometric(s_logit, alpha, beta_mean, lambda_mean)
                like_incorrect = 1.0 - like_correct
                
                post_alpha_correct[i] = self.post_alpha[i] * like_correct
                post_alpha_incorrect[i] = self.post_alpha[i] * like_incorrect
            
            # Normalize
            post_alpha_correct /= (post_alpha_correct.sum() + 1e-12)
            post_alpha_incorrect /= (post_alpha_incorrect.sum() + 1e-12)
            
            # Expected entropy focusing on alpha
            entropy_correct = self.compute_entropy(post_alpha_correct)
            entropy_incorrect = self.compute_entropy(post_alpha_incorrect)
            expected_entropy = p_correct * entropy_correct + p_incorrect * entropy_incorrect
            
            # Information gain
            info_gain = current_entropy - expected_entropy
            
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_stimulus = s_logit
        
        # Convert back to probability space and clamp
        if best_stimulus is None:
            best_stimulus = self.s_grid[len(self.s_grid)//2]  # Fallback to middle
            
        return clamp_prop(inv_logit(best_stimulus))
    
    def select_stimulus_entropy(self):
        """Select stimulus using fast entropy approximation"""
        # Use the fast version to avoid delays
        return self.select_stimulus_entropy_fast()
    
    def update(self, stimulus_prop, correct):
        """Update posterior after observing response"""
        s_logit = logit(clamp_prop(stimulus_prop))
        
        # Bayesian update using full joint posterior
        new_post = np.zeros((len(self.alpha_grid), len(self.beta_grid), len(self.lambda_grid)))
        
        for i, alpha in enumerate(self.alpha_grid):
            for j, beta in enumerate(self.beta_grid):
                for k, lapse in enumerate(self.lambda_grid):
                    prior_weight = self.post_alpha[i] * self.post_beta[j] * self.post_lambda[k]
                    likelihood = self.psychometric(s_logit, alpha, beta, lapse) if correct else (1.0 - self.psychometric(s_logit, alpha, beta, lapse))
                    new_post[i, j, k] = prior_weight * likelihood
        
        # Normalize
        new_post /= (new_post.sum() + 1e-12)
        
        # Marginalize to get individual posteriors
        self.post_alpha = new_post.sum(axis=(1, 2))
        self.post_beta = new_post.sum(axis=(0, 2))
        self.post_lambda = new_post.sum(axis=(0, 1))
        
        # Store trial data
        self.trial_count += 1
        self.responses.append((s_logit, correct))
    
    def get_threshold_sd(self):
        """Get standard deviation of alpha (threshold) posterior in logits"""
        alpha_mean = np.sum(self.alpha_grid * self.post_alpha)
        alpha_var = np.sum(self.post_alpha * (self.alpha_grid - alpha_mean)**2)
        return float(np.sqrt(alpha_var))
    
    def get_threshold_mean(self):
        """Get mean of alpha (threshold) posterior in logits"""
        return float(np.sum(self.alpha_grid * self.post_alpha))

    def posterior_summary(self):
        """Get summary statistics of posteriors"""
        alpha_mean = np.sum(self.alpha_grid * self.post_alpha)
        alpha_sd = np.sqrt(np.sum(self.post_alpha * (self.alpha_grid - alpha_mean)**2))
        
        beta_mean = np.sum(self.beta_grid * self.post_beta)
        beta_sd = np.sqrt(np.sum(self.post_beta * (self.beta_grid - beta_mean)**2))
        
        lambda_mean = np.sum(self.lambda_grid * self.post_lambda)
        lambda_sd = np.sqrt(np.sum(self.post_lambda * (self.lambda_grid - lambda_mean)**2))
        
        return {
            'alpha_mean': float(alpha_mean),
            'alpha_sd': float(alpha_sd),
            'beta_mean': float(beta_mean), 
            'beta_sd': float(beta_sd),
            'lambda_mean': float(lambda_mean),
            'lambda_sd': float(lambda_sd)
        }

    def threshold_for_target(self, p_target):
        """Compute threshold for target percentage correct using posterior predictive"""
        # Ensure feasibility: p_target <= 1 - lambda_hat
        lambda_hat = np.sum(self.lambda_grid * self.post_lambda)
        max_achievable = 1.0 - lambda_hat
        
        if p_target > max_achievable:
            p_target = min(0.85, max_achievable - 0.02)
        
        # Find stimulus that gives closest to target performance
        best_diff = float('inf')
        best_s = 0.5
        
        for s_logit in self.s_grid:
            # Posterior predictive probability
            p_pred = 0.0
            for i, alpha in enumerate(self.alpha_grid):
                for j, beta in enumerate(self.beta_grid):
                    for k, lapse in enumerate(self.lambda_grid):
                        weight = self.post_alpha[i] * self.post_beta[j] * self.post_lambda[k]
                        p_pred += weight * self.psychometric(s_logit, alpha, beta, lapse)
            
            diff = abs(p_pred - p_target)
            if diff < best_diff:
                best_diff = diff
                best_s = inv_logit(s_logit)
        
        return clamp_prop(best_s)

# Global 4-staircase system: one per angle-color combination
global_quest = None

def initialize_global_quest():
    """Initialize 2 staircases for angle conditions only (0° and 90°)"""
    global global_quest
    global_quest = {
        '0': QuestPlusStaircase("neutral"),
        '90': QuestPlusStaircase("neutral"),
    }
    # Initialize 2 QUEST+ staircases silently

def reset_quest_for_angle(angle_bias):
    """Reset QUEST algorithm for a specific angle condition (0° or 90°)"""
    global global_quest
    abs_angle = abs(angle_bias)  # Treat +90 and -90 as same
    
    print(f"Resetting QUEST algorithm for angle condition: {angle_bias}°")
    
    # Reset the QUEST staircase for this angle condition
    global_quest[f'{abs_angle}'] = QuestPlusStaircase("neutral")

def run_calibration_both_angles(max_trials_per_staircase=70, min_trials_per_staircase=40, sd_threshold=0.20):
    """
    Block 1: Calibration phase with both 0° and 90° angles randomly interleaved
    Estimates psychometric functions for all 4 staircases: 0°_low, 0°_high, 90°_low, 90°_high
    """
    global global_quest, global_trial_counter
    if global_quest is None:
        raise ValueError("Global QUEST not initialized. Call initialize_global_quest() first.")
    
    print("Starting Calibration Block - both angles interleaved")
    
    # Track trials per staircase (angles only: 0, 90)
    staircase_keys = ['0', '90']
    trials_per_staircase = {key: 0 for key in staircase_keys}

    trial_counter = 0
    
    # Continue until all staircases meet stopping criteria
    safety_cap_total = max_trials_per_staircase * len(staircase_keys)
    while trial_counter < safety_cap_total:  # Safety cap based on number of staircases
        # Check if all staircases are done
        all_converged = True
        for staircase_key in staircase_keys:
            q = global_quest[staircase_key]
            trials_done = trials_per_staircase[staircase_key]
            alpha_sd = q.get_threshold_sd()
            
            # Stopping criteria: reach minimum trials OR max trials reached
            # (End calibration deterministically after minimum trials per staircase)
            converged = (trials_done >= min_trials_per_staircase)
            max_reached = (trials_done >= max_trials_per_staircase)
            
            if not (converged or max_reached):
                all_converged = False
                break
        
        if all_converged:
            print(f"All staircases converged after {trial_counter} trials")
            break
            
        # Select which staircase to run next (prioritize unconverged staircases)
        available_staircases = []
        for staircase_key in staircase_keys:
            q = global_quest[staircase_key]
            trials_done = trials_per_staircase[staircase_key]
            alpha_sd = q.get_threshold_sd()
            
            converged = (trials_done >= min_trials_per_staircase)
            max_reached = (trials_done >= max_trials_per_staircase)
            
            if not (converged or max_reached):
                available_staircases.append(staircase_key)
        
        if not available_staircases:
            break
            
        # Select staircase (balance between both staircases)
        selected_staircase = min(available_staircases, key=lambda k: trials_per_staircase[k])
        
        # Parse staircase info
        angle_bias = int(selected_staircase)
        
        trial_counter += 1
        global_trial_counter += 1
        trials_per_staircase[selected_staircase] += 1
        
        q = global_quest[selected_staircase]
        
        # Use entropy-based stimulus selection (proper QUEST+)
        s_candidate = q.select_stimulus_entropy()
        
        # During calibration, cue is always black and expect level is not used
        expect_level = 'low'  # placeholder, not functionally used in calibration
        res = run_trial(
            trial_counter, "calibration", angle_bias=angle_bias, expect_level=expect_level, mode="calibration",
            prop_override=s_candidate, cue_dur_range=(0.5, 0.8), motion_dur=5.0
        )

        # Only update QUEST for valid responses (exclude timeouts)
        if res.get('resp_shape') != 'timeout':
            correct = int(res.get('accuracy', 0))
            q.update(s_candidate, correct)

        # Log trial data (optimized for speed during calibration)
        # Only compute expensive posterior summary every 10 trials for speed
        if trial_counter % 10 == 0 or trial_counter < 10:
            summ = q.posterior_summary()
            quest_alpha_sd = summ['alpha_sd']
        else:
            quest_alpha_sd = q.get_threshold_sd()  # Faster than full summary
            summ = None
        
        thisExp.addData('trial_num', global_trial_counter)
        thisExp.addData('participant', expInfo['participant'])
        thisExp.addData('session', expInfo['session'])
        thisExp.addData('phase', 'calibration_interleaved')
        thisExp.addData('cue_color', 'black')  # Always black during calibration
        # During calibration there is no color; omit target_difficulty or set to 'unknown'
        thisExp.addData('target_difficulty', 'unknown')
        thisExp.addData('trial_index_within_staircase', trials_per_staircase[selected_staircase])
        used_prop = res.get('prop_used', s_candidate)
        thisExp.addData('prop_used', used_prop)
        thisExp.addData('stimulus_logit', logit(used_prop))
        thisExp.addData('staircase_id', selected_staircase)
        thisExp.addData('accuracy', res.get('accuracy', 0))
        thisExp.addData('is_timeout', res.get('resp_shape') == 'timeout')
        thisExp.addData('rt_choice', res.get('rt_choice', np.nan))
        thisExp.addData('early_response', res.get('early_response', False))
        thisExp.addData('true_shape', res.get('true_shape', ''))
        thisExp.addData('resp_shape', res.get('resp_shape', ''))
        
        # QUEST+ logging (only when summary computed)
        if summ is not None:
            thisExp.addData('quest_alpha_mean', summ['alpha_mean'])
            thisExp.addData('quest_alpha_sd', summ['alpha_sd'])
            thisExp.addData('quest_beta_mean', summ['beta_mean'])
            thisExp.addData('quest_beta_sd', summ['beta_sd'])
            thisExp.addData('quest_lambda_mean', summ['lambda_mean'])
            thisExp.addData('quest_lambda_sd', summ['lambda_sd'])
        else:
            thisExp.addData('quest_alpha_mean', np.nan)
            thisExp.addData('quest_alpha_sd', quest_alpha_sd)
            thisExp.addData('quest_beta_mean', np.nan)
            thisExp.addData('quest_beta_sd', np.nan)
            thisExp.addData('quest_lambda_mean', np.nan)
            thisExp.addData('quest_lambda_sd', np.nan)
        
        thisExp.addData('angle_bias', angle_bias)
        thisExp.addData('applied_angle_bias', res.get('applied_angle_bias', angle_bias))
        thisExp.addData('trials_completed_this_staircase', trials_per_staircase[selected_staircase])
        thisExp.addData('alpha_sd_current', quest_alpha_sd)
        # Evidence metrics
        thisExp.addData('mean_evidence', res.get('mean_evidence', np.nan))
        thisExp.addData('sum_evidence', res.get('sum_evidence', np.nan))
        thisExp.addData('var_evidence', res.get('var_evidence', np.nan))
        # Pre-RT evidence metrics
        thisExp.addData('rt_frame', res.get('rt_frame', np.nan))
        thisExp.addData('num_frames_preRT', res.get('num_frames_preRT', np.nan))
        thisExp.addData('mean_evidence_preRT', res.get('mean_evidence_preRT', np.nan))
        thisExp.addData('sum_evidence_preRT', res.get('sum_evidence_preRT', np.nan))
        thisExp.addData('var_evidence_preRT', res.get('var_evidence_preRT', np.nan))
        thisExp.addData('cum_evidence_preRT', res.get('cum_evidence_preRT', np.nan))
        thisExp.addData('max_cum_evidence_preRT', res.get('max_cum_evidence_preRT', np.nan))
        thisExp.addData('min_cum_evidence_preRT', res.get('min_cum_evidence_preRT', np.nan))
        thisExp.addData('max_abs_cum_evidence_preRT', res.get('max_abs_cum_evidence_preRT', np.nan))
        thisExp.addData('prop_positive_evidence_preRT', res.get('prop_positive_evidence_preRT', np.nan))
        thisExp.nextEntry()

        # Show break screen every 100 trials, but not at or after planned total
        planned_total = min_trials_per_staircase * len(staircase_keys)
        if (trial_counter % 100 == 0) and (trial_counter < planned_total):
            show_break_screen(trial_counter, planned_total, "Calibration")

def run_learning_phase_for_angle(angle_bias, learning_trials_per_cue=30):
    """
    Learning phase: Fixed practice with colored cues for one specific angle
    Uses thresholds estimated from calibration phase
    """
    global global_quest, global_trial_counter
    
    # Get the thresholds from calibration for this angle
    quest_angle = global_quest[f'{angle_bias}']
    
    # Calculate individualized thresholds (60% and 80% targets)
    s_hat_low = quest_angle.threshold_for_target(0.60)    # 60% target
    s_hat_high = quest_angle.threshold_for_target(0.80)  # 80% target

    # Symmetric, participant-specific separation in logit space
    D = LEARNING_LOGIT_SEPARATION  # desired half-gap in logit units for strong distinctness
    z_low = logit(s_hat_low); z_high = logit(s_hat_high)
    z_mid = 0.5 * (z_low + z_high)
    z_hard = z_mid - D
    z_easy = z_mid + D
    s_hard_learn = clamp_prop(inv_logit(z_hard))
    s_easy_learn = clamp_prop(inv_logit(z_easy))

    # Persist for test phase reuse
    learning_levels_by_angle[angle_bias] = (s_hard_learn, s_easy_learn)
    
    print(f"Learning phase for {angle_bias}°")
    print(f"  Calibration: Hard={s_hat_low:.3f}, Easy={s_hat_high:.3f}")
    print(f"  Learning (logit±{D:.1f}): Hard={s_hard_learn:.3f}, Easy={s_easy_learn:.3f}")
    
    # Create learning trials using individualized thresholds
    learning_trials = []
    for _ in range(learning_trials_per_cue):
        learning_trials.append((low_col, 'learning_fixed', s_hard_learn))   # Hard level (more distinct)
        learning_trials.append((high_col, 'learning_fixed', s_easy_learn))  # Easy level (more distinct)
    
    # Randomly shuffle learning trials
    rng.shuffle(learning_trials)
    
    trial_counter = 0
    for idx, (cue_color, difficulty_type, prop_value) in enumerate(learning_trials, 1):
        trial_counter += 1
        global_trial_counter += 1
        expect_level = 'low' if cue_color == low_col else 'high'
        
        res = run_trial(
            idx, "practice", angle_bias=angle_bias, expect_level=expect_level, mode=difficulty_type,
            prop_override=prop_value, cue_dur_range=(0.5, 0.8), motion_dur=5.0
        )
        used_prop = res.get('prop_used', prop_value)
        
        # Count trials of each type up to this point for indexing
        trials_so_far = learning_trials[:idx]
        trial_index_within_type = sum(1 for t in trials_so_far if t[0] == cue_color and t[1] == difficulty_type)
        
        # Log trial data
        thisExp.addData('trial_num', global_trial_counter)
        thisExp.addData('participant', expInfo['participant'])
        thisExp.addData('session', expInfo['session'])
        thisExp.addData('phase', f'learning_{angle_bias}')
        thisExp.addData('cue_color', cue_color)  # Colored cues in learning phase
        thisExp.addData('target_difficulty', 'high' if cue_color == high_col else 'low')
        thisExp.addData('difficulty_type', difficulty_type)
        thisExp.addData('trial_index_within_type', trial_index_within_type)
        thisExp.addData('prop_used', used_prop)
        thisExp.addData('stimulus_logit', logit(used_prop))
        thisExp.addData('threshold_source', 'individualized_calibration')
        thisExp.addData('accuracy', res.get('accuracy', 0))
        thisExp.addData('is_timeout', res.get('resp_shape') == 'timeout')
        thisExp.addData('rt_choice', res.get('rt_choice', np.nan))
        thisExp.addData('early_response', res.get('early_response', False))
        thisExp.addData('true_shape', res.get('true_shape', ''))
        thisExp.addData('resp_shape', res.get('resp_shape', ''))
        thisExp.addData('s_threshold_60pct', s_hat_low)
        thisExp.addData('s_threshold_80pct', s_hat_high)
        thisExp.addData('angle_bias', angle_bias)
        thisExp.addData('applied_angle_bias', res.get('applied_angle_bias', angle_bias))
        # Evidence metrics
        thisExp.addData('mean_evidence', res.get('mean_evidence', np.nan))
        thisExp.addData('sum_evidence', res.get('sum_evidence', np.nan))
        thisExp.addData('var_evidence', res.get('var_evidence', np.nan))
        # Pre-RT evidence metrics
        thisExp.addData('rt_frame', res.get('rt_frame', np.nan))
        thisExp.addData('num_frames_preRT', res.get('num_frames_preRT', np.nan))
        thisExp.addData('mean_evidence_preRT', res.get('mean_evidence_preRT', np.nan))
        thisExp.addData('sum_evidence_preRT', res.get('sum_evidence_preRT', np.nan))
        thisExp.addData('var_evidence_preRT', res.get('var_evidence_preRT', np.nan))
        thisExp.addData('cum_evidence_preRT', res.get('cum_evidence_preRT', np.nan))
        thisExp.addData('max_cum_evidence_preRT', res.get('max_cum_evidence_preRT', np.nan))
        thisExp.addData('min_cum_evidence_preRT', res.get('min_cum_evidence_preRT', np.nan))
        thisExp.addData('max_abs_cum_evidence_preRT', res.get('max_abs_cum_evidence_preRT', np.nan))
        thisExp.addData('prop_positive_evidence_preRT', res.get('prop_positive_evidence_preRT', np.nan))
        thisExp.nextEntry()

        # Show break screen every 100 trials
        if trial_counter % 100 == 0:
            show_break_screen(trial_counter, len(learning_trials), f"Learning {angle_bias}°")

def run_test_phase_for_angle(angle_bias, medium_trials_per_cue=50, learning_test_trials_per_cue=25):
    """
    Test phase: 150 trials total with mixed difficulty levels
    - 100 medium-level trials (50 per cue color) - tests generalization
    - 50 learning-level trials (25 per difficulty) - maintains learned associations
    """
    global global_quest, global_trial_counter
    
    # Get the thresholds from calibration for this angle
    quest_angle = global_quest[f'{angle_bias}']
    
    # Calculate individualized thresholds
    s_hat_low = quest_angle.threshold_for_target(0.60)   # Hard level (60% target)
    s_hat_high = quest_angle.threshold_for_target(0.80)  # Easy level (80% target)

    # Get learning levels from learning phase if available; otherwise construct them now
    if angle_bias in learning_levels_by_angle:
        s_hard_test, s_easy_test = learning_levels_by_angle[angle_bias]
    else:
        z_mid = 0.5 * (logit(s_hat_low) + logit(s_hat_high))
        D = LEARNING_LOGIT_SEPARATION
        s_hard_test = clamp_prop(inv_logit(z_mid - D))
        s_easy_test = clamp_prop(inv_logit(z_mid + D))
        learning_levels_by_angle[angle_bias] = (s_hard_test, s_easy_test)

    # Define medium at calibrated 70% performance level (participant-specific)
    s_medium = clamp_prop(quest_angle.threshold_for_target(MEDIUM_TARGET_PCORR))
    # Guard: ensure medium lies between hard/easy; otherwise fallback to perceptual midpoint
    if not (min(s_hard_test, s_easy_test) <= s_medium <= max(s_hard_test, s_easy_test)):
        z_mid_learn = 0.5 * (logit(s_hard_test) + logit(s_easy_test))
        s_medium = clamp_prop(inv_logit(z_mid_learn))
    
    print(f"Test phase for {angle_bias}° - Medium={s_medium:.3f} (70%); Learning-level: Hard={s_hard_test:.3f}, Easy={s_easy_test:.3f}")
    
    # Create test trials - 150 total
    test_trials = []
    
    # 100 medium-level trials (50 per cue color)
    # These test generalization to new difficulty level
    for _ in range(medium_trials_per_cue):
        test_trials.append((low_col, 'test_medium', s_medium))   # Medium with low-difficulty cue
        test_trials.append((high_col, 'test_medium', s_medium))  # Medium with high-difficulty cue
    
    # 50 learning-level trials (25 per difficulty)
    # These maintain learned associations and prevent rapid medium-level learning
    for _ in range(learning_test_trials_per_cue):
        test_trials.append((low_col, 'test_learning_hard', s_hard_test))   # Hard level (more distinct)
        test_trials.append((high_col, 'test_learning_easy', s_easy_test))  # Easy level (more distinct)
    
    # Randomly shuffle all test trials
    rng.shuffle(test_trials)
    
    print(f"Test trials created: {len(test_trials)} total (100 medium + 50 learning-level)")
    
    trial_counter = 0
    for idx, (cue_color, difficulty_type, prop_value) in enumerate(test_trials, 1):
        trial_counter += 1
        global_trial_counter += 1
        expect_level = 'low' if cue_color == low_col else 'high'
        
        res = run_trial(
            idx, "test", angle_bias=angle_bias, expect_level=expect_level, mode=difficulty_type,
            prop_override=prop_value, cue_dur_range=(0.5, 0.8), motion_dur=5.0
        )
        used_prop = res.get('prop_used', prop_value)
        
        # Count trials of each type up to this point for indexing
        trials_so_far = test_trials[:idx]
        trial_index_within_type = sum(1 for t in trials_so_far if t[0] == cue_color and t[1] == difficulty_type)
        
        # Determine the actual difficulty level for logging
        if difficulty_type == 'test_medium':
            actual_difficulty = 'medium'
            target_percentage = '70pct'
        elif difficulty_type == 'test_learning_hard':
            actual_difficulty = 'hard'
            target_percentage = '60pct'
        elif difficulty_type == 'test_learning_easy':
            actual_difficulty = 'easy'
            target_percentage = '80pct'
        else:
            actual_difficulty = 'unknown'
            target_percentage = 'unknown'
        
        # Log trial data
        thisExp.addData('trial_num', global_trial_counter)
        thisExp.addData('participant', expInfo['participant'])
        thisExp.addData('session', expInfo['session'])
        thisExp.addData('phase', f'test_{angle_bias}')
        thisExp.addData('cue_color', cue_color)
        thisExp.addData('cue_difficulty_prediction', 'high' if cue_color == high_col else 'low')  # What cue predicts
        thisExp.addData('actual_difficulty_level', actual_difficulty)  # Actual difficulty presented
        thisExp.addData('target_percentage', target_percentage)  # Target performance level
        thisExp.addData('difficulty_type', difficulty_type)
        thisExp.addData('trial_index_within_type', trial_index_within_type)
        thisExp.addData('prop_used', used_prop)
        thisExp.addData('stimulus_logit', logit(used_prop))
        thisExp.addData('threshold_source', 'individualized_calibration')
        thisExp.addData('accuracy', res.get('accuracy', 0))
        thisExp.addData('is_timeout', res.get('resp_shape') == 'timeout')
        thisExp.addData('rt_choice', res.get('rt_choice', np.nan))
        thisExp.addData('confidence_rating', res.get('confidence_rating', np.nan))  # Now included in test
        thisExp.addData('agency_rating', res.get('agency_rating', np.nan))  # Now included in test
        thisExp.addData('early_response', res.get('early_response', False))
        thisExp.addData('true_shape', res.get('true_shape', ''))
        thisExp.addData('resp_shape', res.get('resp_shape', ''))
        thisExp.addData('s_threshold_60pct', s_hat_low)   # Hard threshold
        thisExp.addData('s_threshold_80pct', s_hat_high)  # Easy threshold
        thisExp.addData('s_threshold_70pct', s_medium)    # Medium threshold (70%)
        thisExp.addData('angle_bias', angle_bias)
        thisExp.addData('applied_angle_bias', res.get('applied_angle_bias', angle_bias))
        # Evidence metrics
        thisExp.addData('mean_evidence', res.get('mean_evidence', np.nan))
        thisExp.addData('sum_evidence', res.get('sum_evidence', np.nan))
        thisExp.addData('var_evidence', res.get('var_evidence', np.nan))
        # Pre-RT evidence metrics
        thisExp.addData('rt_frame', res.get('rt_frame', np.nan))
        thisExp.addData('num_frames_preRT', res.get('num_frames_preRT', np.nan))
        thisExp.addData('mean_evidence_preRT', res.get('mean_evidence_preRT', np.nan))
        thisExp.addData('sum_evidence_preRT', res.get('sum_evidence_preRT', np.nan))
        thisExp.addData('var_evidence_preRT', res.get('var_evidence_preRT', np.nan))
        thisExp.addData('cum_evidence_preRT', res.get('cum_evidence_preRT', np.nan))
        thisExp.addData('max_cum_evidence_preRT', res.get('max_cum_evidence_preRT', np.nan))
        thisExp.addData('min_cum_evidence_preRT', res.get('min_cum_evidence_preRT', np.nan))
        thisExp.addData('max_abs_cum_evidence_preRT', res.get('max_abs_cum_evidence_preRT', np.nan))
        thisExp.addData('prop_positive_evidence_preRT', res.get('prop_positive_evidence_preRT', np.nan))
        thisExp.nextEntry()
        
        # Show break screen every 100 trials
        if trial_counter % 100 == 0:
            show_break_screen(trial_counter, len(test_trials), f"Test {angle_bias}°")

# ───────────────────────────────────────────────────────
#  Initial Instructions
# ───────────────────────────────────────────────────────

def show_initial_instructions():
    instructions = [
        # Page 1: Welcome and overview
        """Welcome to the Control Detection Study.

In this experiment, you will see two shapes moving on screen.
Your task is to identify which shape you are controlling with your mouse.

The experiment consists of 5 blocks with short breaks in between.
Total duration: approximately 60 minutes.

Press SPACE to continue...""",

        # Page 2: Response instructions
        """Response Keys:

Press A if you think you controlled the SQUARE
Press S if you think you controlled the CIRCLE

Please respond as quickly and accurately as possible.
If you are unsure, make your best guess.

You have up to 5 seconds to respond on each trial.

Press SPACE to begin..."""
    ]
    
    for instruction in instructions:
        msg.text = instruction
        msg.draw()
        win.flip()
        keys = wait_keys(['space', 'escape'])
        if 'escape' in keys:
            _save()
            core.quit()

# ───────────────────────────────────────────────────────
#  Main Experiment - Block-wise Design
# ───────────────────────────────────────────────────────

# Show initial instructions
show_initial_instructions()

# Demo trial removed as requested

if USE_QUEST_TRAINING:
    # Initialize global 4-staircase system
    initialize_global_quest()
    
    # EXPERIMENT STRUCTURE (internal names - not shown to participants):
    # Block 1: Calibration (both 0° and 90° interleaved)
    # Block 2: Learning (first angle from learning_order)
    # Block 3: Test (same angle as Block 2)
    # Block 4: Learning (second angle from learning_order)
    # Block 5: Test (same angle as Block 4)
    
    # === BLOCK 1: CALIBRATION PHASE ===
    # Set placeholder colors for calibration (black cues used anyway)
    low_col, high_col = "black", "black"
    
    # Update progress bar for Block 1
    progress_bar.set_block(1, CHECK_CALIBRATION_TRIALS * 4)  # 4 staircases
    
    msg.text = """Block 1 of 5

In this block, you will practice the task.
You will receive feedback after each response.

Press SPACE to start..."""
    msg.draw(); win.flip(); wait_keys()
    
    run_calibration_both_angles(
        max_trials_per_staircase=CHECK_CALIBRATION_TRIALS + 20 if not CHECK_MODE else CHECK_CALIBRATION_TRIALS,
        min_trials_per_staircase=CHECK_CALIBRATION_TRIALS,
        sd_threshold=0.20
    )
    
    # 30-second transition: Block 1 → Block 2
    show_phase_transition(
        completed_block=1,
        next_block_info="In the next blocks, the shapes will appear in different colors.\nContinue responding the same way as before.",
        rest_duration=30
    )
    
    # === BLOCKS 2-5: LEARNING AND TEST PHASES ===
    first_angle, second_angle = learning_order[0], learning_order[1]
    
    for phase_num, current_angle in enumerate([first_angle, first_angle, second_angle, second_angle], 2):
        # Activate palette for this angle block
        low_col, high_col = (PALETTE_FOR_FIRST_ANGLE if current_angle == first_angle else PALETTE_FOR_SECOND_ANGLE)
        print(f"Block {phase_num} ({current_angle}°): Using colors {low_col} (low) and {high_col} (high)")
        
        # Determine phase type (internal - not revealed to participant)
        is_learning_phase = (phase_num % 2 == 0)  # Even numbers (2,4) are learning, odd (3,5) are test
        
        # Update progress bar
        if is_learning_phase:
            progress_bar.set_block(phase_num, CHECK_LEARNING_TRIALS_PER_CUE * 2)
        else:
            progress_bar.set_block(phase_num, (CHECK_TEST_MEDIUM_PER_CUE + CHECK_TEST_LEARNING_PER_CUE) * 2)
        
        if is_learning_phase:
            # Learning phase - neutral messaging (no mention of color-difficulty association)
            if phase_num == 2:
                # First learning block
                msg.text = """Block 2 of 5

Continue identifying which shape you control.
You will receive feedback after each response.

Press SPACE to start..."""
            else:
                # Second learning block (Block 4)
                msg.text = """Block 4 of 5

The shapes will now appear in new colors.
Continue responding as before.
You will receive feedback after each response.

Press SPACE to start..."""
            
            msg.draw(); win.flip(); wait_keys()
            run_learning_phase_for_angle(current_angle, learning_trials_per_cue=CHECK_LEARNING_TRIALS_PER_CUE)
            
            # 30-second transition after learning phase
            if phase_num == 2:
                show_phase_transition(
                    completed_block=2,
                    next_block_info="In the next block, you will also rate your confidence\nand sense of control after each trial.",
                    rest_duration=30
                )
            else:
                show_phase_transition(
                    completed_block=4,
                    next_block_info="One more block to go!",
                    rest_duration=30
                )
            
        else:
            # Test phase - neutral messaging
            if phase_num == 3:
                msg.text = """Block 3 of 5

Continue responding as before.

After each trial, you will:
1. Rate how confident you are (1 = Guessing to 4 = Certain)
2. Rate how much control you felt (1 = No control to 7 = Complete control)

Please use the FULL RANGE of each scale.
No feedback will be shown in this block.

Press SPACE to start..."""
            else:
                # Final block (Block 5)
                msg.text = """Final Block (5 of 5)

Continue responding and giving ratings as before.
Remember to use the full range of the scales.
No feedback will be shown.

Press SPACE to start..."""
            
            msg.draw(); win.flip(); wait_keys()
            run_test_phase_for_angle(current_angle, medium_trials_per_cue=CHECK_TEST_MEDIUM_PER_CUE, learning_test_trials_per_cue=CHECK_TEST_LEARNING_PER_CUE)
            
            # 60-second transition after Block 3 (halfway point)
            if phase_num == 3:
                show_phase_transition(
                    completed_block=3,
                    next_block_info="Halfway Complete!\n\nYou have finished 3 of 5 blocks.\nTake a longer rest before continuing.",
                    rest_duration=60
                )

    # Final trajectory usage report (internal logging only)
    final_used = len(used_trajectory_indices)
    primary_total = len(universal_trajectory_set_primary)
    overflow_total = len(universal_trajectory_set_overflow)
    used_primary = len([i for i in used_trajectory_indices if i in set(universal_trajectory_set_primary)])
    used_overflow = len([i for i in used_trajectory_indices if i in set(universal_trajectory_set_overflow)])
    total_available = primary_total + overflow_total if (primary_total or overflow_total) else len(universal_trajectory_set)
    usage_percentage = (final_used / max(1, total_available)) * 100
    
    # Final QUEST convergence summary (logged to data file only, not shown to participant)

    # All reports logged to data file only - no output shown to participant
    
    # End screen
    msg.text = """Experiment Complete

Thank you for participating!
Your responses have been saved.

Press SPACE to exit."""
    msg.draw(); win.flip(); wait_keys()
    win.close(); core.quit()
