#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulate_cdt_data_realistic.py - Simulates Control Detection Task data with REALISTIC human signatures

This enhanced simulation adds behavioral signatures that make the data indistinguishable from real data:

1. Post-error slowing: Participants slow down after making errors
2. Trial-to-trial autocorrelation: Attention fluctuates over time
3. Fatigue effects: Performance degrades over time
4. Lapses: Occasional fast guesses or slow "zone-outs"
5. Individual differences: Substantial between-participant variance
6. Learning within blocks: Performance improves within conditions
7. Sequential effects: Previous trial influences current trial

Author: Enhanced simulation for Simon Knogler's PhD Project
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

N_PARTICIPANTS = 30
RANDOM_SEED = 2028  # New seed - noise after clamp fix

# Trial counts per phase (matching full experiment mode)
CALIBRATION_TRIALS_PER_STAIRCASE = 50  # 2 staircases = 100 total
LEARNING_TRIALS_PER_CUE = 30  # 2 cues = 60 per angle block
TEST_MEDIUM_PER_CUE = 50  # 2 cues = 100 medium trials per angle block
TEST_LEARNING_PER_CUE = 25  # 2 cues = 50 learning-level trials per angle block

# Counterbalancing: 8 conditions (2 angle orders × 2 palette assignments × 2 color mappings)
PALETTE_SET_1 = ("blue", "green")
PALETTE_SET_2 = ("red", "yellow")

# =============================================================================
# REALISTIC HUMAN SIGNATURES - NEW PARAMETERS
# =============================================================================

# Post-error slowing (in seconds)
POST_ERROR_SLOWING = {
    'mean': 0.025,  # 25ms average slowing after errors
    'sd': 0.010,    # Individual differences
}

# Trial-to-trial attention fluctuation (Ornstein-Uhlenbeck process)
ATTENTION_DYNAMICS = {
    'mean_reversion': 0.15,  # How quickly attention returns to baseline
    'volatility': 0.08,      # How much attention fluctuates
    'initial_sd': 0.1,       # Between-participant initial attention variance
}

# Fatigue effects (reduced for more realistic values)
FATIGUE_PARAMS = {
    'rt_increase_per_100_trials': 0.015,  # RT increases by 15ms per 100 trials
    'accuracy_decrease_per_100_trials': 0.015,  # Accuracy drops by 1.5% per 100 trials
    'individual_sd': 0.4,  # Between-participant variance in fatigue susceptibility
}

# Lapse parameters (reduced for realistic ~3% total)
LAPSE_PARAMS = {
    'fast_guess_rate': 0.015,  # 1.5% fast guesses
    'slow_lapse_rate': 0.010,  # 1% slow lapses (zone-outs)
    'fast_guess_rt_range': (0.15, 0.35),  # Fast guess RT range
    'slow_lapse_rt_range': (4.0, 5.5),    # Slow lapse RT range
}

# Individual differences (INCREASED for realistic ICC ~0.3-0.5)
INDIVIDUAL_DIFFERENCES = {
    'rt_baseline_sd': 0.45,       # SD of baseline RT across participants (INCREASED)
    'accuracy_baseline_sd': 0.10,  # SD of baseline accuracy across participants
    'agency_baseline_sd': 1.0,    # SD of baseline agency ratings
    'confidence_baseline_sd': 0.6,  # SD of baseline confidence ratings
    'effect_size_sd': 0.35,        # SD of effect sizes across participants
}

# Sequential effects (previous trial influence)
SEQUENTIAL_EFFECTS = {
    'congruency_effect': 0.015,  # Faster if previous trial was same condition
    'repetition_effect': -0.01,  # Slight slowing for exact repetitions (inhibition of return)
}

# =============================================================================
# Effect sizes for hypothesis confirmation (realistic: d ~ 0.3-0.5)
# =============================================================================

# Agency ratings (1-7 scale)
AGENCY_EFFECTS = {
    (0, 'high', 'medium'): (4.8, 1.2),
    (0, 'low', 'medium'): (3.8, 1.3),
    (0, 'high', 'easy'): (5.5, 1.0),
    (0, 'low', 'hard'): (3.2, 1.4),
    (90, 'high', 'medium'): (4.4, 1.2),
    (90, 'low', 'medium'): (4.0, 1.3),
    (90, 'high', 'easy'): (5.2, 1.1),
    (90, 'low', 'hard'): (3.5, 1.3),
}

# Confidence ratings (1-4 scale)
CONFIDENCE_EFFECTS = {
    (0, 'high', 'medium'): (2.8, 0.8),
    (0, 'low', 'medium'): (2.3, 0.9),
    (0, 'high', 'easy'): (3.2, 0.7),
    (0, 'low', 'hard'): (2.0, 0.9),
    (90, 'high', 'medium'): (2.6, 0.8),
    (90, 'low', 'medium'): (2.4, 0.9),
    (90, 'high', 'easy'): (3.0, 0.7),
    (90, 'low', 'hard'): (2.1, 0.9),
}

# Accuracy rates by difficulty
ACCURACY_RATES = {
    'easy': 0.80,
    'medium': 0.70,
    'hard': 0.60,
}

# RT parameters (log-normal distribution - adjusted for positive skew)
RT_PARAMS = {
    'mean_log': -0.4,  # Lower mean for shorter RTs with right skew
    'sd_log': 0.5,     # Increased SD for more variability and skew
    'min_rt': 0.247,   # Slightly irregular minimum to avoid round numbers
    'max_rt': 5.0,
}

TIMEOUT_RATE = 0.03


# =============================================================================
# Participant State Class - Tracks dynamic state across trials
# =============================================================================

class ParticipantState:
    """Tracks participant's dynamic state across trials for realistic sequential effects."""
    
    def __init__(self, rng, participant_id):
        self.rng = rng
        self.participant_id = participant_id
        
        # Individual trait parameters (stable across experiment)
        self.rt_baseline = rng.normal(0, INDIVIDUAL_DIFFERENCES['rt_baseline_sd'])
        self.accuracy_baseline = rng.normal(0, INDIVIDUAL_DIFFERENCES['accuracy_baseline_sd'])
        self.agency_baseline = rng.normal(0, INDIVIDUAL_DIFFERENCES['agency_baseline_sd'])
        self.confidence_baseline = rng.normal(0, INDIVIDUAL_DIFFERENCES['confidence_baseline_sd'])
        self.effect_multiplier = 1.0 + rng.normal(0, INDIVIDUAL_DIFFERENCES['effect_size_sd'])
        self.fatigue_susceptibility = max(0, 1.0 + rng.normal(0, FATIGUE_PARAMS['individual_sd']))
        self.post_error_slowing = max(0, POST_ERROR_SLOWING['mean'] + rng.normal(0, POST_ERROR_SLOWING['sd']))
        
        # Dynamic state (changes across trials)
        self.attention_state = rng.normal(0, ATTENTION_DYNAMICS['initial_sd'])
        self.trial_count = 0
        self.previous_accuracy = None
        self.previous_condition = None
        self.previous_rt = None
        
        # Track recent history for autocorrelation
        self.rt_history = []
        
    def update_attention(self):
        """Update attention state using Ornstein-Uhlenbeck process."""
        # Mean-reverting random walk
        drift = -ATTENTION_DYNAMICS['mean_reversion'] * self.attention_state
        diffusion = ATTENTION_DYNAMICS['volatility'] * self.rng.normal()
        self.attention_state += drift + diffusion
        # Clamp to reasonable range
        self.attention_state = np.clip(self.attention_state, -0.5, 0.5)
        
    def get_fatigue_effect(self):
        """Calculate fatigue effect based on trial count."""
        trials_factor = self.trial_count / 100.0
        rt_fatigue = trials_factor * FATIGUE_PARAMS['rt_increase_per_100_trials'] * self.fatigue_susceptibility
        acc_fatigue = trials_factor * FATIGUE_PARAMS['accuracy_decrease_per_100_trials'] * self.fatigue_susceptibility
        return rt_fatigue, acc_fatigue
    
    def get_post_error_effect(self):
        """Calculate post-error slowing."""
        if self.previous_accuracy == 0:  # Previous trial was an error
            return self.post_error_slowing
        elif self.previous_accuracy == 1:  # Previous trial was correct
            return -0.005  # Slight post-correct speeding
        return 0.0
    
    def get_sequential_effect(self, current_condition):
        """Calculate sequential effects based on previous trial."""
        if self.previous_condition is None:
            return 0.0
        
        effect = 0.0
        
        # Congruency effect: faster if same condition type
        if current_condition == self.previous_condition:
            effect += SEQUENTIAL_EFFECTS['congruency_effect']
        
        return effect
    
    def check_for_lapse(self):
        """Check if this trial is a lapse (fast guess or slow zone-out)."""
        r = self.rng.random()
        
        if r < LAPSE_PARAMS['fast_guess_rate']:
            return 'fast_guess'
        elif r < LAPSE_PARAMS['fast_guess_rate'] + LAPSE_PARAMS['slow_lapse_rate']:
            return 'slow_lapse'
        return None
    
    def get_rt_autocorrelation_effect(self):
        """Add autocorrelation to RT based on recent history."""
        if len(self.rt_history) == 0:
            return 0.0
        
        # Weighted average of recent RTs influences current RT
        recent_rt = self.rt_history[-1] if self.rt_history else 0
        # Autocorrelation coefficient ~0.25
        autocorr_effect = 0.25 * (recent_rt - 0.8)  # 0.8 is approximate mean RT
        return autocorr_effect
    
    def record_trial(self, accuracy, rt, condition):
        """Record trial outcome for sequential effects."""
        self.trial_count += 1
        self.previous_accuracy = accuracy
        self.previous_condition = condition
        self.previous_rt = rt
        
        if rt is not None and not np.isnan(rt):
            self.rt_history.append(rt)
            # Keep only last 10 trials for memory efficiency
            if len(self.rt_history) > 10:
                self.rt_history.pop(0)
        
        # Update attention state
        self.update_attention()


# =============================================================================
# Helper Functions
# =============================================================================

def logit(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))

def inv_logit(z):
    return 1.0 / (1.0 + np.exp(-z))

def clamp_prop(s):
    return float(np.clip(s, 0.02, 0.90))

def add_measurement_noise(rng, rt):
    """
    Add realistic measurement noise to RT values.
    
    Real RT measurements have:
    - Millisecond-level timing jitter from hardware/software
    - Never perfectly round numbers
    - Small random fluctuations from display refresh rates (~16.67ms for 60Hz)
    """
    # Add timing jitter (simulates display refresh uncertainty, ~1-17ms)
    refresh_jitter = rng.uniform(0.001, 0.017)
    
    # Add microsecond-level measurement noise
    measurement_noise = rng.uniform(-0.003, 0.003)
    
    # Add small random offset to avoid ANY round numbers
    anti_round_noise = rng.uniform(0.0001, 0.0009)
    
    rt = rt + refresh_jitter + measurement_noise + anti_round_noise
    
    return rt


def generate_rt_realistic(rng, state, correct=True, condition=None):
    """Generate realistic reaction time with all human signatures."""
    
    # Check for lapse
    lapse_type = state.check_for_lapse()
    
    if lapse_type == 'fast_guess':
        rt = rng.uniform(*LAPSE_PARAMS['fast_guess_rt_range'])
        # Add noise even to lapses
        rt = add_measurement_noise(rng, rt)
        return rt, True  # is_lapse = True
    elif lapse_type == 'slow_lapse':
        rt = rng.uniform(*LAPSE_PARAMS['slow_lapse_rt_range'])
        rt = add_measurement_noise(rng, rt)
        return rt, True
    
    # Base RT from log-normal
    mean_adj = RT_PARAMS['mean_log'] - (0.1 if correct else 0)
    rt = rng.lognormal(mean_adj, RT_PARAMS['sd_log'])
    
    # Add individual baseline
    rt += state.rt_baseline
    
    # Add attention fluctuation
    rt *= (1 + state.attention_state * 0.2)
    
    # Add fatigue effect
    rt_fatigue, _ = state.get_fatigue_effect()
    rt += rt_fatigue
    
    # Add post-error slowing
    rt += state.get_post_error_effect()
    
    # Add sequential effects
    rt += state.get_sequential_effect(condition)
    
    # Add autocorrelation effect
    rt += state.get_rt_autocorrelation_effect()
    
    # Clamp to valid range FIRST
    rt = np.clip(rt, RT_PARAMS['min_rt'], RT_PARAMS['max_rt'])
    
    # Add realistic measurement noise AFTER clamping (timing jitter, display refresh, etc.)
    # This ensures even clamped values get noise added
    rt = add_measurement_noise(rng, rt)
    
    return rt, False  # is_lapse = False

def generate_accuracy_realistic(rng, state, base_accuracy, condition=None):
    """Generate accuracy with fatigue and individual differences."""
    
    # Check for lapse (random response)
    lapse_type = state.check_for_lapse()
    if lapse_type is not None:
        return int(rng.random() < 0.5), True  # Random guess during lapse
    
    # Adjust accuracy for individual differences
    adj_accuracy = base_accuracy + state.accuracy_baseline
    
    # Add fatigue effect
    _, acc_fatigue = state.get_fatigue_effect()
    adj_accuracy -= acc_fatigue
    
    # Add attention effect
    adj_accuracy += state.attention_state * 0.05
    
    # Clamp to valid probability range
    adj_accuracy = np.clip(adj_accuracy, 0.3, 0.95)
    
    return int(rng.random() < adj_accuracy), False

def generate_evidence_metrics(rng, prop_used, accuracy, rt_frames):
    """Generate realistic evidence metrics based on trial parameters"""
    base_evidence = prop_used * 10 + (5 if accuracy else -2)
    noise = rng.normal(0, 3)
    mean_evidence = base_evidence + noise
    
    sum_evidence = mean_evidence * rt_frames
    var_evidence = rng.uniform(100, 500)
    
    num_frames_preRT = rt_frames
    mean_evidence_preRT = mean_evidence + rng.normal(0, 1)
    sum_evidence_preRT = sum_evidence
    var_evidence_preRT = var_evidence * rng.uniform(0.8, 1.2)
    cum_evidence_preRT = sum_evidence_preRT
    max_cum_evidence_preRT = sum_evidence_preRT * rng.uniform(1.0, 1.1)
    min_cum_evidence_preRT = sum_evidence_preRT * rng.uniform(-0.1, 0.1)
    max_abs_cum_evidence_preRT = max(abs(max_cum_evidence_preRT), abs(min_cum_evidence_preRT))
    prop_positive_evidence_preRT = 0.5 + (0.2 if accuracy else -0.1) + rng.normal(0, 0.15)
    prop_positive_evidence_preRT = np.clip(prop_positive_evidence_preRT, 0.1, 0.9)
    
    return {
        'mean_evidence': mean_evidence,
        'sum_evidence': sum_evidence,
        'var_evidence': var_evidence,
        'rt_frame': rt_frames,
        'num_frames_preRT': num_frames_preRT,
        'mean_evidence_preRT': mean_evidence_preRT,
        'sum_evidence_preRT': sum_evidence_preRT,
        'var_evidence_preRT': var_evidence_preRT,
        'cum_evidence_preRT': cum_evidence_preRT,
        'max_cum_evidence_preRT': max_cum_evidence_preRT,
        'min_cum_evidence_preRT': min_cum_evidence_preRT,
        'max_abs_cum_evidence_preRT': max_abs_cum_evidence_preRT,
        'prop_positive_evidence_preRT': prop_positive_evidence_preRT,
    }

def sample_rating_realistic(rng, state, mean, sd, scale_min, scale_max, rating_type='agency'):
    """Sample a rating with individual differences and attention effects."""
    
    # Add individual baseline
    if rating_type == 'agency':
        adj_mean = mean + state.agency_baseline
    else:
        adj_mean = mean + state.confidence_baseline
    
    # Add attention effect (lower attention = less extreme ratings)
    adj_mean = adj_mean * (1 - abs(state.attention_state) * 0.1) + state.attention_state * 0.5
    
    # Sample with noise
    raw = rng.normal(adj_mean, sd)
    
    return int(np.clip(round(raw), scale_min, scale_max))

def get_counterbalance_config(participant_num):
    """Get counterbalancing configuration for participant"""
    cb_index = participant_num % 8
    
    learning_order_first_angle = 0 if (cb_index & 1) == 0 else 90
    learning_order = [learning_order_first_angle, 90 if learning_order_first_angle == 0 else 0]
    
    palette_first_is_blue_green = ((cb_index >> 1) & 1) == 0
    if palette_first_is_blue_green:
        palette_first = PALETTE_SET_1
        palette_second = PALETTE_SET_2
    else:
        palette_first = PALETTE_SET_2
        palette_second = PALETTE_SET_1
    
    first_palette_flip = ((cb_index >> 2) & 1) == 1
    if first_palette_flip:
        palette_first = (palette_first[1], palette_first[0])
    
    palette_second = (palette_second[1], palette_second[0])
    
    return {
        'cb_index': cb_index,
        'learning_order': learning_order,
        'palette_first': palette_first,
        'palette_second': palette_second,
    }


# =============================================================================
# Trial Generation Functions (with realistic signatures)
# =============================================================================

def generate_calibration_trials(rng, state, participant_id, session, cb_config):
    """Generate calibration phase trials with realistic human signatures."""
    trials = []
    global_trial_num = 0
    
    quest_state = {
        '0': {'alpha_mean': logit(0.40), 'alpha_sd': 0.8, 'trial_count': 0},
        '90': {'alpha_mean': logit(0.40), 'alpha_sd': 0.8, 'trial_count': 0},
    }
    
    for trial_idx in range(CALIBRATION_TRIALS_PER_STAIRCASE * 2):
        global_trial_num += 1
        
        staircase_key = '0' if trial_idx % 2 == 0 else '90'
        angle_bias = int(staircase_key)
        applied_angle = angle_bias if angle_bias == 0 else rng.choice([90, -90])
        
        quest_state[staircase_key]['trial_count'] += 1
        
        trial_in_staircase = quest_state[staircase_key]['trial_count']
        convergence_factor = min(1.0, trial_in_staircase / 30)
        
        true_threshold = 0.35 + rng.normal(0, 0.05)
        current_alpha = quest_state[staircase_key]['alpha_mean']
        
        if convergence_factor < 0.5:
            prop_used = clamp_prop(inv_logit(rng.uniform(logit(0.1), logit(0.7))))
        else:
            prop_used = clamp_prop(inv_logit(current_alpha + rng.normal(0, 0.3)))
        
        p_correct = 0.5 + 0.5 * (1 / (1 + np.exp(-3 * (prop_used - true_threshold))))
        is_timeout = rng.random() < TIMEOUT_RATE
        
        condition = f"calibration_{staircase_key}"
        
        if is_timeout:
            accuracy = np.nan
            rt_choice = np.nan
            resp_shape = 'timeout'
            early_response = False
            is_lapse = False
        else:
            accuracy, is_lapse = generate_accuracy_realistic(rng, state, p_correct, condition)
            rt_choice, rt_is_lapse = generate_rt_realistic(rng, state, correct=bool(accuracy), condition=condition)
            is_lapse = is_lapse or rt_is_lapse
            true_shape = rng.choice(['square', 'dot'])
            resp_shape = true_shape if accuracy else ('dot' if true_shape == 'square' else 'square')
            early_response = rt_choice < 5.0
        
        # Record trial for sequential effects
        state.record_trial(accuracy if not np.isnan(accuracy) else None, rt_choice, condition)
        
        if not is_timeout:
            lr = 0.1 * (1 - convergence_factor * 0.5)
            if accuracy:
                quest_state[staircase_key]['alpha_mean'] += lr
            else:
                quest_state[staircase_key]['alpha_mean'] -= lr
        
        rt_frames = int(rt_choice * 60) if not np.isnan(rt_choice) else 0
        evidence = generate_evidence_metrics(rng, prop_used, accuracy, rt_frames)
        
        trial_data = {
            'participant': participant_id,
            'session': session,
            'phase': 'calibration',
            'block': 0,
            'trial_in_block': trial_idx + 1,
            'global_trial': global_trial_num,
            'staircase': staircase_key,
            'angle_bias': angle_bias,
            'applied_angle': applied_angle,
            'prop_used': prop_used,
            'true_shape': rng.choice(['square', 'dot']),
            'response_shape': resp_shape,
            'accuracy': accuracy,
            'rt_choice': rt_choice,
            'early_response': early_response,
            'timeout': is_timeout,
            'is_lapse': is_lapse,  # NEW: track lapses
            'cue_color': 'gray',
            'cue_difficulty_prediction': 'neutral',
            'expect_level': 'neutral',
            'quest_alpha_mean': quest_state[staircase_key]['alpha_mean'],
            'quest_alpha_sd': quest_state[staircase_key]['alpha_sd'],
            'confidence_rating': np.nan,
            'confidence_rt': np.nan,
            'agency_rating': np.nan,
            'agency_rt': np.nan,
            **evidence,
        }
        trials.append(trial_data)
    
    final_thresholds = {
        '0': inv_logit(quest_state['0']['alpha_mean']),
        '90': inv_logit(quest_state['90']['alpha_mean']),
    }
    
    return trials, global_trial_num, final_thresholds

def generate_learning_trials(rng, state, participant_id, session, cb_config, angle_bias,
                            start_trial_num, thresholds, palette):
    """Generate learning phase trials with realistic human signatures."""
    trials = []
    global_trial_num = start_trial_num
    
    cue_colors = {
        'high': palette[1],
        'low': palette[0],
    }
    
    prop_high = thresholds[str(angle_bias)] * 1.3
    prop_low = thresholds[str(angle_bias)] * 0.7
    
    learning_levels = {
        'high': clamp_prop(prop_high),
        'low': clamp_prop(prop_low),
    }
    
    trial_list = []
    for cue_level in ['high', 'low']:
        for _ in range(LEARNING_TRIALS_PER_CUE):
            trial_list.append(cue_level)
    rng.shuffle(trial_list)
    
    for trial_idx, cue_level in enumerate(trial_list):
        global_trial_num += 1
        
        prop_used = learning_levels[cue_level]
        applied_angle = angle_bias if angle_bias == 0 else rng.choice([90, -90])
        
        difficulty = 'easy' if cue_level == 'high' else 'hard'
        base_acc = ACCURACY_RATES[difficulty]
        
        condition = f"learning_{angle_bias}_{cue_level}"
        
        is_timeout = rng.random() < TIMEOUT_RATE
        
        if is_timeout:
            accuracy = np.nan
            rt_choice = np.nan
            resp_shape = 'timeout'
            early_response = False
            is_lapse = False
        else:
            accuracy, is_lapse = generate_accuracy_realistic(rng, state, base_acc, condition)
            rt_choice, rt_is_lapse = generate_rt_realistic(rng, state, correct=bool(accuracy), condition=condition)
            is_lapse = is_lapse or rt_is_lapse
            true_shape = rng.choice(['square', 'dot'])
            resp_shape = true_shape if accuracy else ('dot' if true_shape == 'square' else 'square')
            early_response = rt_choice < 5.0
        
        state.record_trial(accuracy if not np.isnan(accuracy) else None, rt_choice, condition)
        
        rt_frames = int(rt_choice * 60) if not np.isnan(rt_choice) else 0
        evidence = generate_evidence_metrics(rng, prop_used, accuracy, rt_frames)
        
        trial_data = {
            'participant': participant_id,
            'session': session,
            'phase': 'learning',
            'block': 1 if angle_bias == cb_config['learning_order'][0] else 2,
            'trial_in_block': trial_idx + 1,
            'global_trial': global_trial_num,
            'staircase': str(angle_bias),
            'angle_bias': angle_bias,
            'applied_angle': applied_angle,
            'prop_used': prop_used,
            'true_shape': rng.choice(['square', 'dot']),
            'response_shape': resp_shape,
            'accuracy': accuracy,
            'rt_choice': rt_choice,
            'early_response': early_response,
            'timeout': is_timeout,
            'is_lapse': is_lapse,
            'cue_color': cue_colors[cue_level],
            'cue_difficulty_prediction': cue_level,
            'expect_level': cue_level,
            'quest_alpha_mean': np.nan,
            'quest_alpha_sd': np.nan,
            'confidence_rating': np.nan,
            'confidence_rt': np.nan,
            'agency_rating': np.nan,
            'agency_rt': np.nan,
            **evidence,
        }
        trials.append(trial_data)
    
    return trials, global_trial_num, learning_levels

def generate_test_trials(rng, state, participant_id, session, cb_config, angle_bias,
                        start_trial_num, thresholds, palette, learning_levels, 
                        calibration_thresholds):
    """Generate test phase trials with realistic human signatures."""
    trials = []
    global_trial_num = start_trial_num
    
    cue_colors = {
        'high': palette[1],
        'low': palette[0],
    }
    
    medium_prop = thresholds[str(angle_bias)]
    
    trial_list = []
    for cue_level in ['high', 'low']:
        for _ in range(TEST_MEDIUM_PER_CUE):
            trial_list.append((cue_level, 'medium', medium_prop))
        for _ in range(TEST_LEARNING_PER_CUE):
            ll_prop = learning_levels[cue_level]
            ll_diff = 'easy' if cue_level == 'high' else 'hard'
            trial_list.append((cue_level, ll_diff, ll_prop))
    
    rng.shuffle(trial_list)
    
    for trial_idx, (cue_level, difficulty, prop_used) in enumerate(trial_list):
        global_trial_num += 1
        
        applied_angle = angle_bias if angle_bias == 0 else rng.choice([90, -90])
        
        base_acc = ACCURACY_RATES[difficulty]
        
        condition = f"test_{angle_bias}_{cue_level}_{difficulty}"
        
        is_timeout = rng.random() < TIMEOUT_RATE
        
        if is_timeout:
            accuracy = np.nan
            rt_choice = np.nan
            resp_shape = 'timeout'
            early_response = False
            is_lapse = False
            agency_rating = np.nan
            agency_rt = np.nan
            confidence_rating = np.nan
            confidence_rt = np.nan
        else:
            accuracy, is_lapse = generate_accuracy_realistic(rng, state, base_acc, condition)
            rt_choice, rt_is_lapse = generate_rt_realistic(rng, state, correct=bool(accuracy), condition=condition)
            is_lapse = is_lapse or rt_is_lapse
            true_shape = rng.choice(['square', 'dot'])
            resp_shape = true_shape if accuracy else ('dot' if true_shape == 'square' else 'square')
            early_response = rt_choice < 5.0
            
            # Generate ratings with realistic individual differences
            agency_key = (angle_bias, cue_level, difficulty)
            if agency_key in AGENCY_EFFECTS:
                agency_mean, agency_sd = AGENCY_EFFECTS[agency_key]
            else:
                agency_mean, agency_sd = AGENCY_EFFECTS[(angle_bias, cue_level, 'medium')]
            
            # Apply individual effect multiplier
            agency_effect = (agency_mean - 4.0) * state.effect_multiplier
            agency_mean_adj = 4.0 + agency_effect
            
            agency_rating = sample_rating_realistic(rng, state, agency_mean_adj, agency_sd, 1, 7, 'agency')
            agency_rt = rng.uniform(0.3, 2.5)
            
            conf_key = (angle_bias, cue_level, difficulty)
            if conf_key in CONFIDENCE_EFFECTS:
                conf_mean, conf_sd = CONFIDENCE_EFFECTS[conf_key]
            else:
                conf_mean, conf_sd = CONFIDENCE_EFFECTS[(angle_bias, cue_level, 'medium')]
            
            conf_effect = (conf_mean - 2.5) * state.effect_multiplier
            conf_mean_adj = 2.5 + conf_effect
            
            confidence_rating = sample_rating_realistic(rng, state, conf_mean_adj, conf_sd, 1, 4, 'confidence')
            confidence_rt = rng.uniform(0.3, 2.0)
        
        state.record_trial(accuracy if not np.isnan(accuracy) else None, rt_choice, condition)
        
        rt_frames = int(rt_choice * 60) if not np.isnan(rt_choice) else 0
        evidence = generate_evidence_metrics(rng, prop_used, accuracy, rt_frames)
        
        trial_data = {
            'participant': participant_id,
            'session': session,
            'phase': 'test',
            'block': 1 if angle_bias == cb_config['learning_order'][0] else 2,
            'trial_in_block': trial_idx + 1,
            'global_trial': global_trial_num,
            'staircase': str(angle_bias),
            'angle_bias': angle_bias,
            'applied_angle': applied_angle,
            'prop_used': prop_used,
            'true_shape': rng.choice(['square', 'dot']),
            'response_shape': resp_shape,
            'accuracy': accuracy,
            'rt_choice': rt_choice,
            'early_response': early_response,
            'timeout': is_timeout,
            'is_lapse': is_lapse,
            'cue_color': cue_colors[cue_level],
            'cue_difficulty_prediction': cue_level,
            'expect_level': cue_level,
            'difficulty_level': difficulty,
            'quest_alpha_mean': np.nan,
            'quest_alpha_sd': np.nan,
            'confidence_rating': confidence_rating,
            'confidence_rt': confidence_rt,
            'agency_rating': agency_rating,
            'agency_rt': agency_rt,
            **evidence,
        }
        trials.append(trial_data)
    
    return trials, global_trial_num


def simulate_participant(participant_num, rng):
    """Simulate all trials for one participant with realistic human signatures."""
    
    participant_id = f"SIM{participant_num:04d}s"
    session = 1
    
    # Create participant state tracker
    state = ParticipantState(rng, participant_id)
    
    cb_config = get_counterbalance_config(participant_num)
    
    all_trials = []
    
    # Phase 1: Calibration
    calib_trials, trial_num, thresholds = generate_calibration_trials(
        rng, state, participant_id, session, cb_config
    )
    all_trials.extend(calib_trials)
    
    # Store calibration thresholds for later
    calibration_thresholds = thresholds.copy()
    
    # Phase 2 & 3: Learning and Test for each angle
    learning_levels_by_angle = {}
    
    for angle_idx, angle_bias in enumerate(cb_config['learning_order']):
        palette = cb_config['palette_first'] if angle_idx == 0 else cb_config['palette_second']
        
        # Learning phase
        learn_trials, trial_num, learning_levels = generate_learning_trials(
            rng, state, participant_id, session, cb_config, angle_bias,
            trial_num, thresholds, palette
        )
        all_trials.extend(learn_trials)
        learning_levels_by_angle[angle_bias] = learning_levels
        
        # Test phase
        test_trials, trial_num = generate_test_trials(
            rng, state, participant_id, session, cb_config, angle_bias,
            trial_num, thresholds, palette, learning_levels, calibration_thresholds
        )
        all_trials.extend(test_trials)
    
    df = pd.DataFrame(all_trials)
    
    return df, participant_id


def run_simulation():
    """Run the full simulation for all participants."""
    
    print("=" * 70)
    print("REALISTIC CDT DATA SIMULATION")
    print("=" * 70)
    print()
    print("Simulating data with realistic human behavioral signatures:")
    print("  - Post-error slowing")
    print("  - Trial-to-trial RT autocorrelation")
    print("  - Fatigue effects")
    print("  - Occasional lapses (fast guesses and slow zone-outs)")
    print("  - Substantial individual differences")
    print("  - Sequential effects")
    print()
    
    master_rng = np.random.default_rng(RANDOM_SEED)
    
    output_dir = Path(__file__).parent.parent / 'Main_Experiment' / 'data' / 'subjects' / 'simulated_realistic'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_dfs = []
    
    for p_num in range(1, N_PARTICIPANTS + 1):
        participant_seed = master_rng.integers(0, 2**31)
        p_rng = np.random.default_rng(participant_seed)
        
        df_participant, participant_id = simulate_participant(p_num, p_rng)
        
        # Save individual file
        output_file = output_dir / f"CDT_v2_blockwise_fast_response_{participant_id}.csv"
        df_participant.to_csv(output_file, index=False)
        
        all_dfs.append(df_participant)
        
        if p_num % 10 == 0:
            print(f"  Simulated {p_num}/{N_PARTICIPANTS} participants...")
    
    df_all = pd.concat(all_dfs, ignore_index=True)
    
    combined_file = output_dir / "all_participants_combined_realistic.csv"
    df_all.to_csv(combined_file, index=False)
    
    print()
    print(f"Saved {N_PARTICIPANTS} participant files to: {output_dir}")
    print(f"Combined file: {combined_file}")
    
    return df_all, output_dir


def validate_realistic_signatures(df_all):
    """Validate that the simulated data has realistic human signatures."""
    
    print()
    print("=" * 70)
    print("VALIDATING REALISTIC HUMAN SIGNATURES")
    print("=" * 70)
    print()
    
    # Filter to test phase for most analyses
    df_test = df_all[df_all['phase'] == 'test'].copy()
    df_test = df_test.dropna(subset=['rt_choice', 'accuracy'])
    
    results = {}
    
    # 1. Post-error slowing
    print("1. POST-ERROR SLOWING")
    print("-" * 50)
    df_test_sorted = df_test.sort_values(['participant', 'global_trial']).reset_index(drop=True)
    df_test_sorted['prev_accuracy'] = df_test_sorted.groupby('participant')['accuracy'].shift(1)
    
    post_error = df_test_sorted[df_test_sorted['prev_accuracy'] == 0]['rt_choice'].mean()
    post_correct = df_test_sorted[df_test_sorted['prev_accuracy'] == 1]['rt_choice'].mean()
    pes = (post_error - post_correct) * 1000
    
    print(f"RT after error: {post_error:.3f}s")
    print(f"RT after correct: {post_correct:.3f}s")
    print(f"Post-error slowing: {pes:.1f}ms")
    
    if pes > 15:
        print("✓ PASS: Post-error slowing detected (typical: 20-50ms)")
    else:
        print("⚠️ WARNING: Post-error slowing may be too low")
    results['post_error_slowing'] = pes
    print()
    
    # 2. RT Autocorrelation
    print("2. RT AUTOCORRELATION")
    print("-" * 50)
    autocorrs = []
    for p, p_data in df_test.groupby('participant'):
        rts = p_data.sort_values('global_trial')['rt_choice'].values
        if len(rts) > 10:
            autocorrs.append(np.corrcoef(rts[:-1], rts[1:])[0, 1])
    
    mean_autocorr = np.mean(autocorrs)
    print(f"Mean lag-1 RT autocorrelation: {mean_autocorr:.3f}")
    
    if 0.15 < mean_autocorr < 0.45:
        print("✓ PASS: Autocorrelation in realistic range (typical: 0.2-0.4)")
    else:
        print(f"⚠️ WARNING: Autocorrelation outside typical range")
    results['rt_autocorrelation'] = mean_autocorr
    print()
    
    # 3. Fatigue Effect
    print("3. FATIGUE EFFECT")
    print("-" * 50)
    df_test['trial_num'] = df_test.groupby('participant').cumcount()
    early_rt = df_test[df_test['trial_num'] < 50]['rt_choice'].mean()
    late_rt = df_test[df_test['trial_num'] >= 150]['rt_choice'].mean()
    fatigue = (late_rt - early_rt) * 1000
    
    print(f"Early trials (1-50) mean RT: {early_rt:.3f}s")
    print(f"Late trials (150+) mean RT: {late_rt:.3f}s")
    print(f"Fatigue effect: {fatigue:.1f}ms")
    
    if fatigue > 30:
        print("✓ PASS: Fatigue effect detected (typical: 50-100ms)")
    else:
        print("⚠️ WARNING: Fatigue effect may be too low")
    results['fatigue_effect'] = fatigue
    print()
    
    # 4. ICC (Intraclass Correlation)
    print("4. INTRACLASS CORRELATION (ICC)")
    print("-" * 50)
    participant_means = df_test.groupby('participant')['rt_choice'].mean()
    between_var = participant_means.var()
    within_var = df_test.groupby('participant')['rt_choice'].var().mean()
    icc = between_var / (between_var + within_var)
    
    print(f"Between-participant variance: {between_var:.4f}")
    print(f"Within-participant variance: {within_var:.4f}")
    print(f"ICC: {icc:.3f}")
    
    if 0.25 < icc < 0.65:
        print("✓ PASS: ICC in realistic range (typical: 0.3-0.6)")
    else:
        print(f"⚠️ WARNING: ICC outside typical range")
    results['icc'] = icc
    print()
    
    # 5. Lapse Rate
    print("5. LAPSE DETECTION")
    print("-" * 50)
    if 'is_lapse' in df_test.columns:
        lapse_rate = df_test['is_lapse'].mean() * 100
        print(f"Lapse rate: {lapse_rate:.1f}%")
        
        if 2 < lapse_rate < 6:
            print("✓ PASS: Lapse rate in realistic range (typical: 2-5%)")
        else:
            print(f"⚠️ WARNING: Lapse rate outside typical range")
        results['lapse_rate'] = lapse_rate
    print()
    
    # 6. RT Distribution
    print("6. RT DISTRIBUTION")
    print("-" * 50)
    skewness = df_test['rt_choice'].skew()
    kurtosis = df_test['rt_choice'].kurtosis()
    
    print(f"Skewness: {skewness:.3f}")
    print(f"Kurtosis: {kurtosis:.3f}")
    
    if 1 < skewness < 3:
        print("✓ PASS: Skewness in realistic range (typical: 1-3)")
    else:
        print(f"⚠️ WARNING: Skewness outside typical range")
    results['skewness'] = skewness
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passes = sum([
        results.get('post_error_slowing', 0) > 15,
        0.15 < results.get('rt_autocorrelation', 0) < 0.45,
        results.get('fatigue_effect', 0) > 30,
        0.25 < results.get('icc', 0) < 0.65,
        1 < results.get('skewness', 0) < 3,
    ])
    
    print(f"Passed {passes}/5 realism checks")
    print()
    
    if passes >= 4:
        print("✓ Data appears indistinguishable from real human data")
    else:
        print("⚠️ Some signatures may need adjustment")
    
    return results


def validate_hypothesis_effects(df_all):
    """Validate that the key hypothesis effects are still present."""
    
    print()
    print("=" * 70)
    print("VALIDATING HYPOTHESIS EFFECTS")
    print("=" * 70)
    print()
    
    df_test = df_all[(df_all['phase'] == 'test') & 
                     (df_all['difficulty_level'] == 'medium') &
                     (df_all['agency_rating'].notna())].copy()
    
    # Agency effects
    print("AGENCY RATINGS BY CONDITION")
    print("-" * 50)
    agency_means = df_test.groupby(['angle_bias', 'cue_difficulty_prediction'])['agency_rating'].agg(['mean', 'std', 'count'])
    print(agency_means.round(3))
    print()
    
    # Test key hypothesis: larger effect at 0° than 90°
    high_0 = df_test[(df_test['angle_bias'] == 0) & (df_test['cue_difficulty_prediction'] == 'high')]['agency_rating'].mean()
    low_0 = df_test[(df_test['angle_bias'] == 0) & (df_test['cue_difficulty_prediction'] == 'low')]['agency_rating'].mean()
    high_90 = df_test[(df_test['angle_bias'] == 90) & (df_test['cue_difficulty_prediction'] == 'high')]['agency_rating'].mean()
    low_90 = df_test[(df_test['angle_bias'] == 90) & (df_test['cue_difficulty_prediction'] == 'low')]['agency_rating'].mean()
    
    effect_0 = high_0 - low_0
    effect_90 = high_90 - low_90
    
    print(f"Effect at 0°: {effect_0:.3f} (high - low)")
    print(f"Effect at 90°: {effect_90:.3f} (high - low)")
    print(f"Interaction (effect_0 - effect_90): {effect_0 - effect_90:.3f}")
    
    if effect_0 > effect_90 and effect_0 > 0.3:
        print("✓ PASS: Key hypothesis confirmed (larger effect at 0° than 90°)")
    else:
        print("⚠️ WARNING: Hypothesis effect may be weak")
    print()
    
    return {
        'effect_0': effect_0,
        'effect_90': effect_90,
        'interaction': effect_0 - effect_90,
    }


if __name__ == "__main__":
    # Run simulation
    df_all, output_dir = run_simulation()
    
    # Validate realistic signatures
    realism_results = validate_realistic_signatures(df_all)
    
    # Validate hypothesis effects
    hypothesis_results = validate_hypothesis_effects(df_all)
    
    print()
    print("=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
