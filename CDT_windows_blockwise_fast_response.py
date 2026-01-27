#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-timescale inference variant of the control detection task.

Calibration:
    - 3-down-1-up staircase on two-shape trials estimates the self-motion proportion T.

Main experiment:
    - Easy trials contain 2 shapes (1 target, 1 distractor) -> 2AFC using keys A/S.
    - Complex trials contain 4 shapes (1 target, 3 distractors) -> 4AFC using keys 1-4.

All trials reuse the motion library and trajectory pairing logic from the
fast-response CDT, but remove the multi-angle QUEST+/cue-learning system.
"""

import os
import sys
import math
import random
import pathlib
import atexit
import hashlib
import json
import subprocess


def check_and_run_with_correct_python():
    """Ensure required PsychoPy dependencies are available; relaunch if needed."""
    try:
        import numpy as np  # noqa: F401
        import pandas as pd  # noqa: F401
        from psychopy import visual, event, core, data, gui  # noqa: F401
        return False
    except ImportError as e:
        print(f"Missing required packages: {e}")
        print("Trying to find Python with required packages...")

        python_paths = [
            "C:/Program Files/PsychoPy/python.exe",
            "C:/Users/knogl/Miniconda3/envs/psychopy_env/python.exe",
            "C:/Users/knogl/Miniconda3/python.exe",
            "/opt/anaconda3/bin/python",
            "/usr/bin/python3",
        ]

        for path in python_paths:
            if os.path.exists(path):
                print(f"Found Python at: {path}")
                result = subprocess.run([path] + sys.argv, check=False)
                sys.exit(result.returncode)

        print("Error: Python with required packages not found. "
              "Please install psychopy, numpy, and pandas.")
        sys.exit(1)


if check_and_run_with_correct_python():
    sys.exit(0)

import numpy as np
import pandas as pd
from psychopy import visual, event, core, data, gui

# ───────────────────────────────────────────────────────
#  Global experiment state
# ───────────────────────────────────────────────────────
kinematics_data = []
kinematics_csv_path = ""

_saved = False


def _save():
    global _saved
    if not _saved:
        if 'thisExp' in globals() and thisExp is not None:
            thisExp.saveAsWideText(csv_path)
            print("Main data auto-saved ->", csv_path)
            if kinematics_data:
                kinematics_df = pd.DataFrame(kinematics_data)
                kinematics_df.to_csv(kinematics_csv_path, index=False)
                print("Kinematics data auto-saved ->", kinematics_csv_path)
        else:
            print("Experiment not initialized - no data to save")
        _saved = True


atexit.register(_save)

# ───────────────────────────────────────────────────────
#  Participant dialog
# ───────────────────────────────────────────────────────
expName = "MultiTimescaleInference"
expInfo = {"participant": "", "session": "001", "simulate": False, "check_mode": False}
dlg = gui.DlgFromDict(expInfo, order=["participant", "session", "simulate", "check_mode"], title=expName)
if not dlg.OK:
    core.quit()

SIMULATE = bool(expInfo.pop("simulate"))
CHECK_MODE = bool(expInfo.pop("check_mode"))
if SIMULATE:
    expInfo["participant"] = "SIM"

if CHECK_MODE:
    print("=" * 60)
    print("CHECK MODE ENABLED - running shortened calibration and experiment.")
    print("=" * 60)
else:
    print("Running full experiment mode.")

# ───────────────────────────────────────────────────────
#  Motion library
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
print(f"Loaded {TOTAL_SNIPS} snippets x {SNIP_LEN} frames from {LIB_NAME}")

with open(script_dir.parent / "Motion_Library" / "scaler_params.json", "r") as f:
    scp = json.load(f)
scaler_mean = np.array(scp["mean"], dtype=np.float32)
scaler_std = np.array(scp["scale"], dtype=np.float32)

with open(script_dir.parent / "Motion_Library" / "cluster_centroids.json", "r") as f:
    CLUSTER_CENTROIDS = np.array(json.load(f), dtype=np.float32)

seed = int(hashlib.sha256(expInfo["participant"].encode()).hexdigest(), 16) & 0xFFFFFFFF
rng = np.random.default_rng(seed)

universal_trajectory_set_primary = []
universal_trajectory_set_overflow = []
universal_trajectory_set = []
used_trajectory_indices = set()
trajectory_usage_stats = {"used_count": 0, "total_needed": 2000}


def get_trajectory_signature(trajectory):
    """Key movement characteristics for matching."""
    velocities = np.diff(trajectory, axis=0)
    if len(velocities) == 0:
        return {
            'mean_speed': 0,
            'speed_variability': 0,
            'path_length': 0,
            'net_displacement': 0,
            'speed_percentiles': np.array([0, 0, 0])
        }
    speeds = np.linalg.norm(velocities, axis=1)
    return {
        'mean_speed': np.mean(speeds),
        'speed_variability': np.std(speeds),
        'path_length': np.sum(speeds),
        'net_displacement': np.linalg.norm(trajectory[-1] - trajectory[0]),
        'speed_percentiles': np.percentile(speeds, [25, 50, 75])
    }


def select_universal_trajectory_set():
    """Pre-select trajectories for all participants with deterministic ranking."""
    global valid_snippet_indices, universal_trajectory_set_primary, universal_trajectory_set_overflow, universal_trajectory_set

    total_valid = len(valid_snippet_indices)
    if total_valid < 1240:
        print(f"Warning: Only {total_valid} valid trajectories, fewer than 1,240 required for primary set")
        universal_trajectory_set_primary = valid_snippet_indices.copy()
        universal_trajectory_set_overflow = []
        universal_trajectory_set = universal_trajectory_set_primary.copy()
        return universal_trajectory_set.copy()

    selection_rng = np.random.default_rng(42)

    print("Selecting universal trajectory sets (Primary 1,240 + Overflow 40)...")

    trajectory_scores = []
    for idx in valid_snippet_indices:
        trajectory = motion_pool[idx]
        traj_cumsum = np.cumsum(trajectory, axis=0)
        sig = get_trajectory_signature(traj_cumsum)

        speed_score = 1.0 / (1.0 + abs(sig['mean_speed'] - 8.0))
        variability_score = 1.0 / (1.0 + abs(sig['speed_variability'] - 3.0))
        length_score = min(1.0, sig['path_length'] / 100.0)

        overall_score = speed_score * variability_score * length_score
        trajectory_scores.append((overall_score, idx))

    trajectory_scores.sort(reverse=True)
    primary_indices = [idx for score, idx in trajectory_scores[:1240]]
    overflow_indices = [idx for score, idx in trajectory_scores[1240:1280]] if total_valid >= 1280 else []

    universal_trajectory_set_primary = primary_indices
    universal_trajectory_set_overflow = overflow_indices
    universal_trajectory_set = primary_indices + overflow_indices

    print(f"Selected Primary: {len(primary_indices)} (best={trajectory_scores[0][0]:.3f})")
    if overflow_indices:
        print(f"Selected Overflow: {len(overflow_indices)} "
              f"(worst_primary={trajectory_scores[1239][0]:.3f}, best_overflow={trajectory_scores[1240][0]:.3f} if available)")
    else:
        print("No overflow set available (valid < 1,280)")

    return universal_trajectory_set.copy()


def analyze_trajectory_quality(trajectory):
    velocities = np.diff(trajectory, axis=0)
    speeds = np.linalg.norm(velocities, axis=1)
    mean_speed = np.mean(speeds)
    std_speed = np.std(speeds)
    max_speed = np.max(speeds) if len(speeds) else 0
    min_speed = np.min(speeds) if len(speeds) else 0
    zero_movement_ratio = np.sum(speeds < 0.5) / len(speeds) if len(speeds) else 1.0
    high_jitter_ratio = np.sum(speeds > mean_speed + 3 * std_speed) / len(speeds) if len(speeds) else 0.0
    if len(velocities) > 1:
        unit_velocities = velocities / (speeds.reshape(-1, 1) + 1e-9)
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
        smoothed_velocities[i] = smooth_factor * smoothed_velocities[i - 1] + (1 - smooth_factor) * velocities[i]
    normalized_trajectory = [trajectory[0]]
    for vel in smoothed_velocities:
        next_point = normalized_trajectory[-1] + vel
        normalized_trajectory.append(next_point)
    return np.array(normalized_trajectory)


def preprocess_motion_pool():
    """Preprocess motion pool to ensure quality and consistency."""
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


def find_matched_trajectory_pair():
    """Find two unused trajectories with similar movement characteristics."""
    global universal_trajectory_set, universal_trajectory_set_primary, universal_trajectory_set_overflow
    global used_trajectory_indices, trajectory_usage_stats

    if not universal_trajectory_set:
        universal_trajectory_set = valid_snippet_indices.copy()
        universal_trajectory_set_primary = universal_trajectory_set.copy()
        universal_trajectory_set_overflow = []

    available_primary = [idx for idx in universal_trajectory_set_primary if idx not in used_trajectory_indices]
    available_overflow = [idx for idx in universal_trajectory_set_overflow if idx not in used_trajectory_indices]
    available_indices = available_primary if len(available_primary) >= 2 else (available_primary + available_overflow)

    if len(available_indices) < 2:
        if len(available_indices) == 1:
            target_idx = available_indices[0]
            different_options = [idx for idx in universal_trajectory_set if idx != target_idx]
            if different_options:
                distractor_idx = rng.choice(different_options)
                return target_idx, distractor_idx
            return None, None
        elif len(available_indices) == 0:
            print("No unused trajectories! Using random valid trajectories.")
            return None, None

    sample_pool = available_primary if len(available_primary) >= 2 else available_indices
    sample_size = min(100, len(sample_pool))
    candidate_indices = rng.choice(sample_pool, size=sample_size, replace=False)

    signatures = []
    for idx in candidate_indices:
        trajectory = motion_pool[idx]
        sig = get_trajectory_signature(np.cumsum(trajectory, axis=0))
        signatures.append((idx, sig))

    best_score = float('inf')
    best_pair = (None, None)

    for i in range(len(signatures)):
        for j in range(i + 1, len(signatures)):
            idx1, sig1 = signatures[i]
            idx2, sig2 = signatures[j]

            speed_diff = abs(sig1['mean_speed'] - sig2['mean_speed'])
            var_diff = abs(sig1['speed_variability'] - sig2['speed_variability'])
            length_diff = abs(sig1['path_length'] - sig2['path_length']) / max(sig1['path_length'], sig2['path_length'])

            similarity_score = speed_diff + var_diff + length_diff * 10

            if similarity_score < best_score:
                best_score = similarity_score
                best_pair = (idx1, idx2)

    if best_pair[0] is not None and best_pair[1] is not None:
        used_trajectory_indices.add(best_pair[0])
        used_trajectory_indices.add(best_pair[1])
        trajectory_usage_stats["used_count"] += 2

    return best_pair


def apply_consistent_smoothing(trajectory1, trajectory2):
    """Apply consistent smoothing to both trajectories."""
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


valid_snippet_indices = preprocess_motion_pool()
universal_trajectory_set = select_universal_trajectory_set()
print(
    f"Universal trajectory sets initialized: Primary={len(universal_trajectory_set_primary)}; "
    f"Overflow={len(universal_trajectory_set_overflow)}; Total={len(universal_trajectory_set)}"
)

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


# ───────────────────────────────────────────────────────
#  Constants and Parameters
# ───────────────────────────────────────────────────────
OFFSET_X = 200  # Reduced from 300 so shapes start within confine limit (250)
LOWPASS = 0.5

# ───────────────────────────────────────────────────────
#  Paths & ExperimentHandler
# ───────────────────────────────────────────────────────
root = script_dir / "data"
subjects_dir = root / "subjects"
subjects_dir.mkdir(parents=True, exist_ok=True)

participant_id = expInfo['participant']
base_filename = f"MTI_{participant_id}"
csv_path = subjects_dir / f"{base_filename}.csv"
kinematics_csv_path = subjects_dir / f"{base_filename}_kinematics.csv"

i = 1
while csv_path.exists():
    new_filename = f"{base_filename}_{i}"
    csv_path = subjects_dir / f"{new_filename}.csv"
    kinematics_csv_path = subjects_dir / f"{new_filename}_kinematics.csv"
    i += 1

data_file_stem = csv_path.stem
thisExp = data.ExperimentHandler(
    name=expName, extraInfo=expInfo,
    savePickle=False, saveWideText=False,
    dataFileName=str(root / data_file_stem)
)

# ───────────────────────────────────────────────────────
#  Window & stimuli
# ───────────────────────────────────────────────────────
win = visual.Window((1920, 1080), fullscr=not SIMULATE, color=[0.5] * 3, units="pix", allowGUI=True)
win.setMouseVisible(False)

fix = visual.TextStim(win, "+", color="white", height=60)
msg = visual.TextStim(win, "", color="white", height=26, wrapWidth=1000)
feedbackTxt = visual.TextStim(win, "", color="black", height=80)

def create_shape(index):
    """Create a new shape instance based on index.
    
    Uses same sizes as original CDT: 40x40 rect, radius=20 circle.
    Colors are black on gray background to match original.
    """
    if index % 2 == 0:
        return visual.Rect(win, width=40, height=40, fillColor="black", lineColor="black")
    else:
        return visual.Circle(win, radius=20, fillColor="black", lineColor="black")


def confine(point, limit=250):
    """Confine point to within limit pixels of center (matching original)."""
    r = math.hypot(*point)
    if r <= limit:
        return point
    return (point[0] * limit / r, point[1] * limit / r)


def rotate(vx, vy, angle_deg):
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    return (
        vx * cos_a - vy * sin_a,
        vx * sin_a + vy * cos_a
    )


def record_shape_positions(shapes):
    data = {}
    for idx in range(4):
        if idx < len(shapes):
            pos = shapes[idx].pos
            data[f'shape{idx}_x'] = float(pos[0])
            data[f'shape{idx}_y'] = float(pos[1])
        else:
            data[f'shape{idx}_x'] = np.nan
            data[f'shape{idx}_y'] = np.nan
    return data


global_trial_counter = 0


def run_trial(
    trial_num,
    phase,
    complexity,
    prop_self,
    block_num=1,
):
    prop = float(np.clip(prop_self, 0.02, 0.90))
    n_shapes = 2 if complexity == "easy" else 4

    # Create shapes first so we can set colors consistently
    shapes = [create_shape(i) for i in range(n_shapes)]
    
    # Set fixation and shape colors to black (matching original calibration style)
    fix.color = "black"
    for shape in shapes:
        shape.fillColor = "black"
        shape.lineColor = "black"
    
    fix.draw()
    win.flip()
    # Variable cue duration like original (0.5-0.8s)
    core.wait(float(rng.uniform(0.5, 0.8)))
    
    # Randomize positions like original (which shape goes where)
    if complexity == "easy":
        start_positions = [(-OFFSET_X, 0), (OFFSET_X, 0)]
        # Randomize which shape is left vs right
        position_order = list(range(n_shapes))
        random.shuffle(position_order)
        for i, pos in enumerate(start_positions):
            shapes[position_order[i]].pos = pos
    else:
        start_positions = [(-OFFSET_X, 150), (OFFSET_X, 150), (-OFFSET_X, -150), (OFFSET_X, -150)]
        # Randomize positions for 4 shapes
        position_order = list(range(n_shapes))
        random.shuffle(position_order)
        for i, pos in enumerate(start_positions):
            shapes[position_order[i]].pos = pos

    target_index = int(rng.integers(0, n_shapes))

    # Show shapes and wait for mouse movement to start (like original)
    for shape in shapes:
        shape.draw()
    win.flip()
    
    mouse = SimulatedMouse() if SIMULATE else event.Mouse(win=win, visible=False)
    mouse.setPos((0, 0))
    last = mouse.getPos()

    while True:
        for shape in shapes:
            shape.draw()
        win.flip()
        x, y = mouse.getPos()
        if math.hypot(x - last[0], y - last[1]) > 0 or SIMULATE:
            break
        if not SIMULATE and event.getKeys(["escape"]):
            _save()
            core.quit()

    target_snippet_idx, distractor_snippet_idx = find_matched_trajectory_pair()
    if target_snippet_idx is None or distractor_snippet_idx is None:
        available_primary = [idx for idx in universal_trajectory_set_primary if idx not in used_trajectory_indices]
        available_overflow = [idx for idx in universal_trajectory_set_overflow if idx not in used_trajectory_indices]
        available_indices = available_primary if len(available_primary) >= 2 else (available_primary + available_overflow)
        if len(available_indices) >= 2:
            selected = rng.choice(available_indices, size=2, replace=False)
            target_snippet_idx, distractor_snippet_idx = selected[0], selected[1]
            used_trajectory_indices.add(target_snippet_idx)
            used_trajectory_indices.add(distractor_snippet_idx)
            trajectory_usage_stats["used_count"] += 2
        elif len(available_indices) == 1:
            target_snippet_idx = available_indices[0]
            combined_sets = universal_trajectory_set_primary + universal_trajectory_set_overflow
            different_options = [idx for idx in combined_sets if idx != target_snippet_idx]
            if different_options:
                distractor_snippet_idx = rng.choice(different_options)
            else:
                distractor_snippet_idx = rng.choice([idx for idx in range(len(motion_pool)) if idx != target_snippet_idx])
            used_trajectory_indices.add(target_snippet_idx)
            used_trajectory_indices.add(distractor_snippet_idx)
            trajectory_usage_stats["used_count"] += 2
        else:
            combined_sets = universal_trajectory_set_primary + universal_trajectory_set_overflow if (universal_trajectory_set_primary or universal_trajectory_set_overflow) else universal_trajectory_set
            if len(combined_sets) >= 2:
                selected = rng.choice(combined_sets, size=2, replace=False)
                target_snippet_idx, distractor_snippet_idx = selected[0], selected[1]
            else:
                available_range = list(range(len(motion_pool)))
                selected = rng.choice(available_range, size=2, replace=False)
                target_snippet_idx, distractor_snippet_idx = selected[0], selected[1]

    target_snippet = motion_pool[target_snippet_idx]
    distractor_snippet = motion_pool[distractor_snippet_idx]
    target_snippet, distractor_snippet = apply_consistent_smoothing(target_snippet, distractor_snippet)

    trial_kinematics = []
    clk = core.Clock()
    frame = 0
    vt = np.zeros(2, np.float32)
    vd = np.zeros(2, np.float32)
    mag_m_lp = 0.0
    prev_d = np.zeros(2, np.float32)

    resp_index = None
    rt_choice = np.nan
    early_response = False
    total_motion_duration = 5.0
    response_start_time = core.getTime()
    event.clearEvents(eventType='keyboard')
    rt_frame = None

    if complexity == "easy":
        response_keys = ['a', 's']
        key_to_index = {'a': 0, 's': 1}
    else:
        response_keys = ['1', '2', '3', '4']
        key_to_index = {'1': 0, '2': 1, '3': 2, '4': 3}

    while clk.getTime() < total_motion_duration and resp_index is None:
        x, y = mouse.getPos()
        dx = x - last[0]
        dy = y - last[1]
        last = (x, y)
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
                dx = dx * scale_factor
                dy = dy * scale_factor
        if frame == 1:
            mag_m_lp = mag_m
        else:
            mag_m_lp = 0.5 * mag_m_lp + 0.5 * mag_m
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
        target_ou_dx = dir_target_x * mag_m_lp
        target_ou_dy = dir_target_y * mag_m_lp
        distractor_ou_dx = dir_distractor_x * mag_m_lp
        distractor_ou_dy = dir_distractor_y * mag_m_lp
        tdx = prop * dx + (1 - prop) * target_ou_dx
        tdy = prop * dy + (1 - prop) * target_ou_dy
        mouse_speed = math.hypot(dx, dy)
        linear_bias = 0.0
        if mouse_speed > 0 and frame > 10 and len(trial_kinematics) >= 5:
            recent_positions = [(d['mouse_x'], d['mouse_y']) for d in trial_kinematics[-5:]] + [(x, y)]
            if len(recent_positions) >= 3:
                total_dist = sum(math.hypot(recent_positions[i + 1][0] - recent_positions[i][0],
                                            recent_positions[i + 1][1] - recent_positions[i][1])
                                 for i in range(len(recent_positions) - 1))
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
                perp_dx /= perp_mag
                perp_dy /= perp_mag
                cursor_mag = math.hypot(dx, dy)
                perp_dx *= cursor_mag
                perp_dy *= cursor_mag
            else:
                perp_dx = perp_dy = 0
            ddx = (1 - linear_bias) * distractor_ou_dx + linear_bias * perp_dx
            ddy = (1 - linear_bias) * distractor_ou_dy + linear_bias * perp_dy
        else:
            ddx = distractor_ou_dx
            ddy = distractor_ou_dy
        ddx_smooth = 0.4 * prev_d[0] + 0.6 * ddx
        ddy_smooth = 0.4 * prev_d[1] + 0.6 * ddy
        prev_d = np.array([ddx_smooth, ddy_smooth])
        vt = LOWPASS * vt + (1 - LOWPASS) * np.array([tdx, tdy])
        vd = LOWPASS * vd + (1 - LOWPASS) * np.array([ddx_smooth, ddy_smooth])

        vm = np.array([dx, dy], dtype=float)
        mouse_speed = np.linalg.norm(vm) + 1e-9

        vt_disp = np.array(vt, dtype=float)
        vd_disp = np.array(vd, dtype=float)

        ut = vt_disp / (np.linalg.norm(vt_disp) + 1e-9)
        ud = vd_disp / (np.linalg.norm(vd_disp) + 1e-9)

        cos_T = np.dot(vm, ut) / mouse_speed
        cos_D = np.dot(vm, ud) / mouse_speed

        evidence = (cos_T - cos_D) * (mouse_speed - 1e-9)

        # Update positions like original: shape.pos + velocity, then confine
        if complexity == "easy":
            # Easy: one target, one distractor - exactly like original
            if target_index == 0:
                shapes[0].pos = confine(tuple(shapes[0].pos + vt))
                shapes[1].pos = confine(tuple(shapes[1].pos + vd))
            else:
                shapes[0].pos = confine(tuple(shapes[0].pos + vd))
                shapes[1].pos = confine(tuple(shapes[1].pos + vt))
        else:
            # Complex: one target, three distractors with rotated velocities
            rotation_angles = [0, 20, -20]
            d_idx = 0
            for i, shape in enumerate(shapes):
                if i == target_index:
                    shape.pos = confine(tuple(shape.pos + vt))
                else:
                    ang = rotation_angles[d_idx % len(rotation_angles)]
                    rx, ry = rotate(vd[0], vd[1], ang)
                    shape.pos = confine(tuple(shape.pos + np.array([rx, ry])))
                    d_idx += 1

        frame_data = {
            'timestamp': clk.getTime(),
            'frame': frame,
            'mouse_x': x,
            'mouse_y': y,
            'evidence': evidence
        }
        frame_data.update(record_shape_positions(shapes))
        trial_kinematics.append(frame_data)

        for shape in shapes:
            shape.draw()
        win.flip()

        if not SIMULATE:
            keys = event.getKeys(response_keys + ['escape'], timeStamped=True)
            if keys:
                key, key_time = keys[0]
                if key == "escape":
                    _save()
                    core.quit()
                elif key in key_to_index:
                    resp_index = key_to_index[key]
                    rt_choice = key_time - response_start_time
                    rt_frame = frame
                    early_response = True
        else:
            if frame > 60 and rng.random() < 0.1:
                resp_index = int(rng.integers(0, n_shapes))
                rt_choice = clk.getTime()
                rt_frame = frame
                early_response = True

    if resp_index is None:
        msg.text = "Too slow!\n\nPlease respond faster next time."
        msg.draw()
        win.flip()
        core.wait(2.0)
        resp_value = "timeout"
        correct = np.nan
    else:
        resp_value = resp_index
        correct = int(resp_index == target_index)

    agency_rating = np.nan
    if phase == "test":
        if SIMULATE:
            agency_rating = float(rng.integers(1, 8))
        else:
            event.clearEvents(eventType='keyboard')
            msg.text = "How much control did you feel over the shape's movement?"
            scale_positions = [(-450, -100), (-300, -100), (-150, -100), (0, -100), (150, -100), (300, -100), (450, -100)]
            scale_labels = [
                "1\nVery weak", "2\nWeak", "3\nSomewhat weak",
                "4\nModerate", "5\nSomewhat strong", "6\nStrong", "7\nVery strong"
            ]
            scale_stimuli = [
                visual.TextStim(win, text=label, pos=pos, height=18, color='white', alignText='center')
                for pos, label in zip(scale_positions, scale_labels)
            ]
            rating = None
            while rating is None:
                msg.draw()
                for stim in scale_stimuli:
                    stim.draw()
                win.flip()
                keys = event.getKeys(['1', '2', '3', '4', '5', '6', '7', 'escape'])
                if keys:
                    if 'escape' in keys:
                        _save()
                        core.quit()
                    else:
                        rating = int(keys[0])
                core.wait(0.01)
            agency_rating = rating
            core.wait(0.2)

    if phase == "calibration" and resp_value != "timeout":
        feedbackTxt.text = "Right" if correct else "Wrong"
        feedbackTxt.draw()
        win.flip()
        core.wait(0.8)
        win.flip()
        core.wait(0.2)

    confidence_rating = np.nan

    for frame_data in trial_kinematics:
        frame_data.update({
            'participant': expInfo['participant'],
            'session': expInfo['session'],
            'trial_num': trial_num,
            'phase': phase,
            'complexity': complexity,
            'prop_used': prop,
            'confidence_rating': confidence_rating,
            'agency_rating': agency_rating,
            'block_num': block_num,
            'early_response': early_response,
            'true_shape': target_index,
            'resp_shape': resp_value,
            'num_shapes': n_shapes,
            'accuracy': correct
        })
        kinematics_data.append(frame_data)

    frame_evidence = [d['evidence'] for d in trial_kinematics]
    if frame_evidence:
        mean_evidence = float(np.mean(frame_evidence))
        sum_evidence = float(np.sum(frame_evidence))
        var_evidence = float(np.var(frame_evidence))
    else:
        mean_evidence = sum_evidence = var_evidence = np.nan

    if early_response and frame_evidence:
        pre_rt_evidence = frame_evidence
        cum = np.cumsum(pre_rt_evidence)
        mean_evidence_preRT = float(np.mean(pre_rt_evidence))
        sum_evidence_preRT = float(np.sum(pre_rt_evidence))
        var_evidence_preRT = float(np.var(pre_rt_evidence))
        max_cum_evidence_preRT = float(np.max(cum))
        min_cum_evidence_preRT = float(np.min(cum))
        max_abs_cum_evidence_preRT = float(np.max(np.abs(cum)))
        prop_positive_evidence_preRT = float(np.mean(np.array(pre_rt_evidence) > 0))
        rt_frame_out = int(rt_frame) if rt_frame is not None else int(trial_kinematics[-1]['frame'])
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
        trial_num=trial_num,
        target_snippet_id=target_snippet_idx,
        distractor_snippet_id=distractor_snippet_idx,
        phase=phase,
        complexity=complexity,
        block_num=block_num,
        prop_used=prop,
        true_shape=target_index,
        resp_shape=resp_value,
        confidence_rating=confidence_rating,
        accuracy=correct,
        rt_choice=rt_choice,
        agency_rating=agency_rating,
        early_response=early_response,
        mean_evidence=mean_evidence,
        sum_evidence=sum_evidence,
        var_evidence=var_evidence,
        rt_frame=rt_frame_out,
        num_frames_preRT=num_frames_preRT,
        mean_evidence_preRT=mean_evidence_preRT,
        sum_evidence_preRT=sum_evidence_preRT,
        var_evidence_preRT=var_evidence_preRT,
        cum_evidence_preRT=sum_evidence_preRT,
        max_cum_evidence_preRT=max_cum_evidence_preRT,
        min_cum_evidence_preRT=min_cum_evidence_preRT,
        max_abs_cum_evidence_preRT=max_abs_cum_evidence_preRT,
        prop_positive_evidence_preRT=prop_positive_evidence_preRT,
    )


def log_trial_result(res, phase_label, complexity_label, prop_level, extra_fields=None):
    thisExp.addData('trial_num', res.get('trial_num', np.nan))
    thisExp.addData('participant', expInfo['participant'])
    thisExp.addData('session', expInfo['session'])
    thisExp.addData('phase', phase_label)
    thisExp.addData('complexity', complexity_label)
    thisExp.addData('prop_used', res.get('prop_used', prop_level))
    thisExp.addData('control_level', prop_level)
    thisExp.addData('accuracy', res.get('accuracy', np.nan))
    thisExp.addData('rt_choice', res.get('rt_choice', np.nan))
    thisExp.addData('agency_rating', res.get('agency_rating', np.nan))
    thisExp.addData('early_response', res.get('early_response', False))
    thisExp.addData('true_shape', res.get('true_shape', np.nan))
    thisExp.addData('resp_shape', res.get('resp_shape', np.nan))
    thisExp.addData('target_snippet_id', res.get('target_snippet_id', np.nan))
    thisExp.addData('distractor_snippet_id', res.get('distractor_snippet_id', np.nan))
    thisExp.addData('mean_evidence', res.get('mean_evidence', np.nan))
    thisExp.addData('sum_evidence', res.get('sum_evidence', np.nan))
    thisExp.addData('var_evidence', res.get('var_evidence', np.nan))
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
    if extra_fields:
        for key, value in extra_fields.items():
            thisExp.addData(key, value)
    thisExp.nextEntry()


def run_calibration_staircase(
    max_trials=80,
    min_reversals=12,
    start_prop=0.8,
    step_large=0.10,
    step_small=0.05
):
    """3-down-1-up staircase on self-motion proportion."""
    global global_trial_counter
    print("Starting calibration staircase...")
    prop_self = float(np.clip(start_prop, 0.05, 0.95))
    step = step_large
    correct_counter = 0
    reversals = []
    last_direction = None
    trial_number_local = 0

    while trial_number_local < max_trials and len(reversals) < min_reversals:
        trial_number_local += 1
        global_trial_counter += 1
        current_prop = prop_self
        res = run_trial(
            trial_num=global_trial_counter,
            phase="calibration",
            complexity="easy",
            prop_self=current_prop,
            block_num=1,
        )

        extra_fields = {"staircase_prop": current_prop}
        if res.get('resp_shape') == "timeout":
            extra_fields["staircase_direction"] = "timeout"
            log_trial_result(res, "calibration", "easy", current_prop, extra_fields)
            continue

        is_correct = int(res.get('accuracy', 0))
        direction = None
        new_prop = current_prop

        if is_correct:
            correct_counter += 1
            if correct_counter == 3:
                direction = "down"
                new_prop = current_prop - step
                correct_counter = 0
        else:
            direction = "up"
            new_prop = current_prop + step
            correct_counter = 0

        if direction is not None:
            new_prop = float(np.clip(new_prop, 0.05, 0.95))
            if last_direction is not None and direction != last_direction:
                reversals.append(current_prop)
                if len(reversals) == 4:
                    step = step_small
            last_direction = direction
            prop_self = new_prop

        extra_fields["staircase_direction"] = direction or "hold"
        log_trial_result(res, "calibration", "easy", current_prop, extra_fields)

    if reversals:
        tail = reversals[-6:] if len(reversals) >= 6 else reversals
        threshold = float(np.mean(tail))
    else:
        threshold = prop_self

    print(f"Calibration staircase finished with {len(reversals)} reversals; T = {threshold:.3f}")
    return threshold


def make_control_levels(threshold):
    """Generate four control levels around the estimated threshold."""
    levels = [
        max(0.05, threshold - 0.10),
        max(0.05, threshold - 0.05),
        min(0.95, threshold),
        min(0.95, threshold + 0.05),
    ]
    return sorted(levels)


def run_main_experiment(control_levels, trials_per_condition=100):
    """Main SoA task mixing easy (2AFC) and complex (4AFC) trials."""
    global global_trial_counter
    conditions = []
    for complexity in ["easy", "complex"]:
        for level in control_levels:
            for _ in range(trials_per_condition):
                conditions.append((complexity, level))
    if not conditions:
        print("No conditions scheduled for main experiment.")
        return

    order = rng.permutation(len(conditions))
    total_trials = len(order)
    print(f"Starting main experiment with {total_trials} trials.")

    for order_idx, cond_idx in enumerate(order, 1):
        complexity, prop_self = conditions[cond_idx]
        global_trial_counter += 1
        res = run_trial(
            trial_num=global_trial_counter,
            phase="test",
            complexity=complexity,
            prop_self=prop_self,
            block_num=1,
        )
        log_trial_result(
            res,
            phase_label="test",
            complexity_label=complexity,
            prop_level=prop_self,
            extra_fields={"trial_index_within_block": order_idx}
        )
        if order_idx % 100 == 0 or order_idx == total_trials:
            print(f"... completed {order_idx}/{total_trials} main trials")


def show_initial_instructions():
    instructions = [
        """Welcome to the study.

On each trial you will see moving shapes. Move your mouse as usual and decide which shape you had more control over.

Press SPACE to continue...""",
        """When there are TWO shapes, they will appear to the left and right of center.

Press A for the LEFT shape and S for the RIGHT shape.

Respond quickly but accurately. Press SPACE to continue...""",
        """When there are FOUR shapes, they form a square:

1 = top-left, 2 = top-right, 3 = bottom-left, 4 = bottom-right.

Pick the number that matches the shape you controlled. Press SPACE to continue...""",
        """After the calibration, the main task will ask for an agency rating (1-7) after each trial.
Try to use the full scale.

Press SPACE to continue..."""
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
#  Main Experiment Flow
# ───────────────────────────────────────────────────────
show_initial_instructions()

msg.text = "First, we will run a short calibration to find a challenging control level.\n\nPress SPACE to start."
msg.draw()
win.flip()
wait_keys(['space', 'escape'])

calibration_kwargs = dict(
    max_trials=40 if CHECK_MODE else 80,
    min_reversals=6 if CHECK_MODE else 12,
    start_prop=0.8,
    step_large=0.10,
    step_small=0.05,
)
threshold_T = run_calibration_staircase(**calibration_kwargs)
control_levels = make_control_levels(threshold_T)
print("Derived control levels:", [f"{level:.3f}" for level in control_levels])

msg.text = (
    "Now the main task will begin.\n\n"
    "Sometimes you will see two shapes (A/S response), sometimes four shapes (1-4 response).\n"
    "Always pick the shape you felt you controlled the most.\n\n"
    "Press SPACE to continue."
)
msg.draw()
win.flip()
wait_keys(['space', 'escape'])

main_trials_per_condition = 10 if CHECK_MODE else 100
run_main_experiment(control_levels, trials_per_condition=main_trials_per_condition)

msg.text = "Thank you for participating.\n\nPress SPACE to exit."
msg.draw()
win.flip()
wait_keys(['space', 'escape'])
win.close()
core.quit()

