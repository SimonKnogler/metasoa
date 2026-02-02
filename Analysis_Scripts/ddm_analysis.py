#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ddm_analysis.py - Drift Diffusion Model Analysis for CDT Data

Tests the mechanistic dissociation hypothesis:
- At 0° (prediction-based): Expectations modulate DRIFT RATE (evidence quality)
- At 90° (regularity-based): Expectations modulate BOUNDARY SEPARATION (decision policy)

This script:
1. Simulates DDM data with hypothesis-confirming parameters
2. Fits DDM to the simulated behavioral data
3. Tests whether drift rate and boundary parameters differ by condition

Author: Analysis script for Simon Knogler's PhD Project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

COLORS = {
    'high_exp': '#2E86AB',
    'low_exp': '#E94F37',
    'deg_0': '#1B4965',
    'deg_90': '#5FA8D3',
    'drift': '#2D6A4F',
    'boundary': '#9B2226',
}

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR.parent / "Main_Experiment" / "data" / "analysis_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DDM Simulation Functions
# =============================================================================

def simulate_ddm_trial(v, a, t0, z=0.5, dt=0.001, max_time=10.0, s=1.0):
    """
    Simulate a single DDM trial.
    
    Parameters:
    -----------
    v : float
        Drift rate (evidence accumulation rate)
    a : float
        Boundary separation (threshold)
    t0 : float
        Non-decision time
    z : float
        Starting point (proportion of boundary, 0.5 = unbiased)
    dt : float
        Time step
    max_time : float
        Maximum decision time
    s : float
        Diffusion coefficient (noise)
    
    Returns:
    --------
    rt : float
        Reaction time (including non-decision time)
    response : int
        1 = upper boundary (correct), 0 = lower boundary (error)
    """
    # Starting point in absolute units
    x = z * a
    
    # Simulate accumulation
    time = 0
    while time < max_time:
        # Wiener process: dx = v*dt + s*sqrt(dt)*N(0,1)
        noise = np.random.normal(0, s * np.sqrt(dt))
        x += v * dt + noise
        time += dt
        
        # Check boundaries
        if x >= a:
            return t0 + time, 1  # Upper boundary (correct)
        elif x <= 0:
            return t0 + time, 0  # Lower boundary (error)
    
    # Timeout
    return max_time + t0, np.random.choice([0, 1])


def simulate_ddm_condition(v, a, t0, z=0.5, n_trials=100):
    """Simulate multiple DDM trials for a condition."""
    results = []
    for _ in range(n_trials):
        rt, response = simulate_ddm_trial(v, a, t0, z)
        results.append({'rt': rt, 'response': response, 'accuracy': response})
    return pd.DataFrame(results)


def simulate_ddm_experiment(n_participants=30, n_trials_per_condition=50):
    """
    Simulate full DDM experiment with hypothesis-confirming parameters.
    
    Hypotheses:
    - At 0°: High expectation → higher drift rate (better evidence)
    - At 90°: High expectation → lower boundary (more liberal)
    
    Parameters are set to create dissociable effects.
    """
    print("=" * 70)
    print("DDM SIMULATION")
    print("=" * 70)
    print()
    
    # Base parameters
    base_params = {
        'v': 0.8,      # Base drift rate
        'a': 1.5,      # Base boundary separation
        't0': 0.3,     # Non-decision time
        'z': 0.5,      # Unbiased starting point
    }
    
    # Effect sizes (hypothesis-confirming)
    # At 0°: Expectation affects DRIFT RATE
    drift_effect_0deg = 0.4  # High exp increases drift by this amount
    
    # At 90°: Expectation affects BOUNDARY
    boundary_effect_90deg = -0.3  # High exp decreases boundary by this amount
    
    # Small cross-effects (for realism)
    drift_effect_90deg = 0.1  # Small drift effect at 90°
    boundary_effect_0deg = -0.05  # Small boundary effect at 0°
    
    # Condition parameters
    conditions = {
        'high_0': {
            'v': base_params['v'] + drift_effect_0deg,
            'a': base_params['a'] + boundary_effect_0deg,
            't0': base_params['t0'],
            'expectation': 'high',
            'angle': 0
        },
        'low_0': {
            'v': base_params['v'],
            'a': base_params['a'],
            't0': base_params['t0'],
            'expectation': 'low',
            'angle': 0
        },
        'high_90': {
            'v': base_params['v'] + drift_effect_90deg,
            'a': base_params['a'] + boundary_effect_90deg,
            't0': base_params['t0'],
            'expectation': 'high',
            'angle': 90
        },
        'low_90': {
            'v': base_params['v'],
            'a': base_params['a'],
            't0': base_params['t0'],
            'expectation': 'low',
            'angle': 90
        },
    }
    
    print("Simulation Parameters (Ground Truth):")
    print("-" * 50)
    print(f"{'Condition':<12} {'Drift (v)':<12} {'Boundary (a)':<12}")
    print("-" * 50)
    for cond, params in conditions.items():
        print(f"{cond:<12} {params['v']:<12.2f} {params['a']:<12.2f}")
    print()
    
    print("Expected Effects:")
    print(f"  Drift rate effect at 0°:  {drift_effect_0deg:.2f} (HIGH - should be significant)")
    print(f"  Drift rate effect at 90°: {drift_effect_90deg:.2f} (LOW - small/non-significant)")
    print(f"  Boundary effect at 0°:    {boundary_effect_0deg:.2f} (LOW - small/non-significant)")
    print(f"  Boundary effect at 90°:   {boundary_effect_90deg:.2f} (HIGH - should be significant)")
    print()
    
    # Simulate data
    all_data = []
    
    for p in range(n_participants):
        participant_id = f"SIM{p+1:03d}s"
        
        # Add participant-level variability
        p_drift_offset = np.random.normal(0, 0.15)
        p_boundary_offset = np.random.normal(0, 0.1)
        p_t0_offset = np.random.normal(0, 0.05)
        
        for cond_name, cond_params in conditions.items():
            # Participant-specific parameters
            v_p = cond_params['v'] + p_drift_offset
            a_p = max(0.5, cond_params['a'] + p_boundary_offset)  # Ensure positive
            t0_p = max(0.1, cond_params['t0'] + p_t0_offset)
            
            # Simulate trials
            for trial in range(n_trials_per_condition):
                rt, response = simulate_ddm_trial(v_p, a_p, t0_p)
                
                all_data.append({
                    'participant': participant_id,
                    'condition': cond_name,
                    'expectation': cond_params['expectation'],
                    'angle': cond_params['angle'],
                    'trial': trial,
                    'rt': rt,
                    'response': response,
                    'accuracy': response,
                    # Store true parameters for validation
                    'true_v': v_p,
                    'true_a': a_p,
                    'true_t0': t0_p,
                })
    
    df = pd.DataFrame(all_data)
    
    print(f"Simulated {len(df)} trials for {n_participants} participants")
    print(f"Trials per condition per participant: {n_trials_per_condition}")
    print()
    
    return df, conditions


# =============================================================================
# DDM Fitting Functions (Simplified EZ-Diffusion)
# =============================================================================

def ez_diffusion(pc, vrt, mrt):
    """
    EZ-diffusion model parameter estimation.
    
    A simplified method to estimate DDM parameters from summary statistics.
    
    Parameters:
    -----------
    pc : float
        Proportion correct (0.5 to 1.0)
    vrt : float
        Variance of RT for correct responses
    mrt : float
        Mean RT for correct responses
    
    Returns:
    --------
    v : float
        Drift rate estimate
    a : float
        Boundary separation estimate
    t0 : float
        Non-decision time estimate
    """
    s = 0.1  # Scaling parameter (fixed)
    s2 = s ** 2
    
    # Edge correction for extreme accuracy
    pc = np.clip(pc, 0.501, 0.999)
    
    # Logit transform of accuracy
    L = np.log(pc / (1 - pc))
    
    # Estimate drift rate
    x = L * (L * pc**2 - L * pc + pc - 0.5) / vrt
    v = np.sign(pc - 0.5) * s * x ** 0.25
    
    # Estimate boundary
    a = s2 * L / v
    
    # Estimate non-decision time
    y = -v * a / s2
    mdt = (a / (2 * v)) * (1 - np.exp(y)) / (1 + np.exp(y))
    t0 = mrt - mdt
    
    return v, a, t0


def fit_ez_diffusion_by_condition(df):
    """
    Fit EZ-diffusion model to each participant × condition.
    
    Returns DataFrame with estimated parameters.
    """
    results = []
    
    for participant in df['participant'].unique():
        for condition in df['condition'].unique():
            subset = df[(df['participant'] == participant) & 
                       (df['condition'] == condition)]
            
            if len(subset) < 10:
                continue
            
            # Compute summary statistics
            correct_trials = subset[subset['accuracy'] == 1]
            
            if len(correct_trials) < 5:
                continue
            
            pc = subset['accuracy'].mean()
            mrt = correct_trials['rt'].mean()
            vrt = correct_trials['rt'].var()
            
            if vrt <= 0 or pc <= 0.5:
                continue
            
            try:
                v, a, t0 = ez_diffusion(pc, vrt, mrt)
                
                # Get condition info
                cond_info = subset.iloc[0]
                
                results.append({
                    'participant': participant,
                    'condition': condition,
                    'expectation': cond_info['expectation'],
                    'angle': cond_info['angle'],
                    'v': v,
                    'a': a,
                    't0': t0,
                    'pc': pc,
                    'mrt': mrt,
                    'vrt': vrt,
                    'n_trials': len(subset),
                    # True parameters for validation
                    'true_v': subset['true_v'].iloc[0],
                    'true_a': subset['true_a'].iloc[0],
                })
            except:
                continue
    
    return pd.DataFrame(results)


# =============================================================================
# Statistical Analysis
# =============================================================================

def analyze_ddm_parameters(params_df):
    """
    Analyze DDM parameters to test mechanistic dissociation hypothesis.
    """
    print("=" * 70)
    print("DDM PARAMETER ANALYSIS")
    print("=" * 70)
    print()
    
    # =========================================================================
    # Descriptive Statistics
    # =========================================================================
    print("-" * 70)
    print("Descriptive Statistics: Estimated Parameters by Condition")
    print("-" * 70)
    print()
    
    desc_stats = params_df.groupby(['angle', 'expectation']).agg({
        'v': ['mean', 'std', 'sem'],
        'a': ['mean', 'std', 'sem'],
        't0': ['mean', 'std', 'sem'],
        'pc': ['mean'],
        'mrt': ['mean'],
    }).round(3)
    
    print(desc_stats)
    print()
    
    # =========================================================================
    # Test 1: Drift Rate Analysis
    # =========================================================================
    print("=" * 70)
    print("TEST 1: DRIFT RATE (v) - Evidence Accumulation Quality")
    print("=" * 70)
    print()
    print("Hypothesis: At 0°, high expectation increases drift rate")
    print("            At 90°, expectation has minimal effect on drift rate")
    print()
    
    # Simple effect at 0°
    v_high_0 = params_df[(params_df['angle'] == 0) & (params_df['expectation'] == 'high')]['v']
    v_low_0 = params_df[(params_df['angle'] == 0) & (params_df['expectation'] == 'low')]['v']
    t_v_0, p_v_0 = stats.ttest_ind(v_high_0, v_low_0)
    d_v_0 = (v_high_0.mean() - v_low_0.mean()) / np.sqrt((v_high_0.var() + v_low_0.var()) / 2)
    
    print(f"Simple effect of Expectation on Drift Rate at 0°:")
    print(f"  High: M = {v_high_0.mean():.3f}, SD = {v_high_0.std():.3f}")
    print(f"  Low:  M = {v_low_0.mean():.3f}, SD = {v_low_0.std():.3f}")
    print(f"  Difference: {v_high_0.mean() - v_low_0.mean():.3f}")
    print(f"  t = {t_v_0:.2f}, p = {p_v_0:.4f}, d = {d_v_0:.2f}")
    print()
    
    # Simple effect at 90°
    v_high_90 = params_df[(params_df['angle'] == 90) & (params_df['expectation'] == 'high')]['v']
    v_low_90 = params_df[(params_df['angle'] == 90) & (params_df['expectation'] == 'low')]['v']
    t_v_90, p_v_90 = stats.ttest_ind(v_high_90, v_low_90)
    d_v_90 = (v_high_90.mean() - v_low_90.mean()) / np.sqrt((v_high_90.var() + v_low_90.var()) / 2)
    
    print(f"Simple effect of Expectation on Drift Rate at 90°:")
    print(f"  High: M = {v_high_90.mean():.3f}, SD = {v_high_90.std():.3f}")
    print(f"  Low:  M = {v_low_90.mean():.3f}, SD = {v_low_90.std():.3f}")
    print(f"  Difference: {v_high_90.mean() - v_low_90.mean():.3f}")
    print(f"  t = {t_v_90:.2f}, p = {p_v_90:.4f}, d = {d_v_90:.2f}")
    print()
    
    # Interaction test for drift rate
    print("Interaction: Is the drift rate effect larger at 0° than 90°?")
    effect_v_0 = v_high_0.mean() - v_low_0.mean()
    effect_v_90 = v_high_90.mean() - v_low_90.mean()
    
    # Compute per-participant effects and test
    v_effects = []
    for p in params_df['participant'].unique():
        p_data = params_df[params_df['participant'] == p]
        v_h0 = p_data[(p_data['angle'] == 0) & (p_data['expectation'] == 'high')]['v'].values
        v_l0 = p_data[(p_data['angle'] == 0) & (p_data['expectation'] == 'low')]['v'].values
        v_h90 = p_data[(p_data['angle'] == 90) & (p_data['expectation'] == 'high')]['v'].values
        v_l90 = p_data[(p_data['angle'] == 90) & (p_data['expectation'] == 'low')]['v'].values
        
        if len(v_h0) > 0 and len(v_l0) > 0 and len(v_h90) > 0 and len(v_l90) > 0:
            eff_0 = v_h0[0] - v_l0[0]
            eff_90 = v_h90[0] - v_l90[0]
            v_effects.append({'effect_0': eff_0, 'effect_90': eff_90, 'diff': eff_0 - eff_90})
    
    v_effects_df = pd.DataFrame(v_effects)
    t_v_int, p_v_int = stats.ttest_1samp(v_effects_df['diff'], 0)
    
    print(f"  Effect at 0°:  {effect_v_0:.3f}")
    print(f"  Effect at 90°: {effect_v_90:.3f}")
    print(f"  Interaction (0° - 90°): {v_effects_df['diff'].mean():.3f}")
    print(f"  t({len(v_effects_df)-1}) = {t_v_int:.2f}, p = {p_v_int:.4f}")
    print()
    
    # =========================================================================
    # Test 2: Boundary Separation Analysis
    # =========================================================================
    print("=" * 70)
    print("TEST 2: BOUNDARY SEPARATION (a) - Decision Policy")
    print("=" * 70)
    print()
    print("Hypothesis: At 90°, high expectation lowers boundary (more liberal)")
    print("            At 0°, expectation has minimal effect on boundary")
    print()
    
    # Simple effect at 0°
    a_high_0 = params_df[(params_df['angle'] == 0) & (params_df['expectation'] == 'high')]['a']
    a_low_0 = params_df[(params_df['angle'] == 0) & (params_df['expectation'] == 'low')]['a']
    t_a_0, p_a_0 = stats.ttest_ind(a_high_0, a_low_0)
    d_a_0 = (a_high_0.mean() - a_low_0.mean()) / np.sqrt((a_high_0.var() + a_low_0.var()) / 2)
    
    print(f"Simple effect of Expectation on Boundary at 0°:")
    print(f"  High: M = {a_high_0.mean():.3f}, SD = {a_high_0.std():.3f}")
    print(f"  Low:  M = {a_low_0.mean():.3f}, SD = {a_low_0.std():.3f}")
    print(f"  Difference: {a_high_0.mean() - a_low_0.mean():.3f}")
    print(f"  t = {t_a_0:.2f}, p = {p_a_0:.4f}, d = {d_a_0:.2f}")
    print()
    
    # Simple effect at 90°
    a_high_90 = params_df[(params_df['angle'] == 90) & (params_df['expectation'] == 'high')]['a']
    a_low_90 = params_df[(params_df['angle'] == 90) & (params_df['expectation'] == 'low')]['a']
    t_a_90, p_a_90 = stats.ttest_ind(a_high_90, a_low_90)
    d_a_90 = (a_high_90.mean() - a_low_90.mean()) / np.sqrt((a_high_90.var() + a_low_90.var()) / 2)
    
    print(f"Simple effect of Expectation on Boundary at 90°:")
    print(f"  High: M = {a_high_90.mean():.3f}, SD = {a_high_90.std():.3f}")
    print(f"  Low:  M = {a_low_90.mean():.3f}, SD = {a_low_90.std():.3f}")
    print(f"  Difference: {a_high_90.mean() - a_low_90.mean():.3f}")
    print(f"  t = {t_a_90:.2f}, p = {p_a_90:.4f}, d = {d_a_90:.2f}")
    print()
    
    # Interaction test for boundary
    print("Interaction: Is the boundary effect larger at 90° than 0°?")
    effect_a_0 = a_high_0.mean() - a_low_0.mean()
    effect_a_90 = a_high_90.mean() - a_low_90.mean()
    
    # Compute per-participant effects
    a_effects = []
    for p in params_df['participant'].unique():
        p_data = params_df[params_df['participant'] == p]
        a_h0 = p_data[(p_data['angle'] == 0) & (p_data['expectation'] == 'high')]['a'].values
        a_l0 = p_data[(p_data['angle'] == 0) & (p_data['expectation'] == 'low')]['a'].values
        a_h90 = p_data[(p_data['angle'] == 90) & (p_data['expectation'] == 'high')]['a'].values
        a_l90 = p_data[(p_data['angle'] == 90) & (p_data['expectation'] == 'low')]['a'].values
        
        if len(a_h0) > 0 and len(a_l0) > 0 and len(a_h90) > 0 and len(a_l90) > 0:
            eff_0 = a_h0[0] - a_l0[0]
            eff_90 = a_h90[0] - a_l90[0]
            # Note: We expect effect at 90° to be MORE NEGATIVE than at 0°
            # So we test if (effect_90 - effect_0) < 0
            a_effects.append({'effect_0': eff_0, 'effect_90': eff_90, 'diff': eff_90 - eff_0})
    
    a_effects_df = pd.DataFrame(a_effects)
    t_a_int, p_a_int = stats.ttest_1samp(a_effects_df['diff'], 0)
    
    print(f"  Effect at 0°:  {effect_a_0:.3f}")
    print(f"  Effect at 90°: {effect_a_90:.3f}")
    print(f"  Interaction (90° - 0°): {a_effects_df['diff'].mean():.3f}")
    print(f"  t({len(a_effects_df)-1}) = {t_a_int:.2f}, p = {p_a_int:.4f}")
    print()
    
    # =========================================================================
    # Summary: Mechanistic Dissociation
    # =========================================================================
    print("=" * 70)
    print("SUMMARY: MECHANISTIC DISSOCIATION")
    print("=" * 70)
    print()
    print("The key question: Do expectations influence agency via different")
    print("mechanisms depending on processing mode?")
    print()
    print("Evidence for DRIFT RATE modulation at 0° (prediction-based):")
    print(f"  - Expectation effect on drift: Δv = {effect_v_0:.3f}, p = {p_v_0:.4f}")
    print(f"  - Expectation effect on boundary: Δa = {effect_a_0:.3f}, p = {p_a_0:.4f}")
    if p_v_0 < 0.05 and p_a_0 > 0.05:
        print("  → CONFIRMED: Drift rate affected, boundary unaffected")
    print()
    
    print("Evidence for BOUNDARY modulation at 90° (regularity-based):")
    print(f"  - Expectation effect on drift: Δv = {effect_v_90:.3f}, p = {p_v_90:.4f}")
    print(f"  - Expectation effect on boundary: Δa = {effect_a_90:.3f}, p = {p_a_90:.4f}")
    if p_a_90 < 0.05 and p_v_90 > 0.05:
        print("  → CONFIRMED: Boundary affected, drift rate unaffected")
    print()
    
    print("Cross-over interaction:")
    print(f"  - Drift rate effect larger at 0° than 90°: p = {p_v_int:.4f}")
    print(f"  - Boundary effect larger at 90° than 0°: p = {p_a_int:.4f}")
    print()
    
    return {
        'drift_effects': v_effects_df,
        'boundary_effects': a_effects_df,
        'stats': {
            'v_effect_0': {'diff': effect_v_0, 't': t_v_0, 'p': p_v_0, 'd': d_v_0},
            'v_effect_90': {'diff': effect_v_90, 't': t_v_90, 'p': p_v_90, 'd': d_v_90},
            'a_effect_0': {'diff': effect_a_0, 't': t_a_0, 'p': p_a_0, 'd': d_a_0},
            'a_effect_90': {'diff': effect_a_90, 't': t_a_90, 'p': p_a_90, 'd': d_a_90},
            'v_interaction': {'t': t_v_int, 'p': p_v_int},
            'a_interaction': {'t': t_a_int, 'p': p_a_int},
        }
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_ddm_results(params_df, analysis_results, output_path):
    """
    Create publication-quality figure showing DDM parameter results.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # =========================================================================
    # Row 1: Drift Rate
    # =========================================================================
    
    # Panel A: Drift rate by condition
    ax1 = axes[0, 0]
    
    conditions = ['high_0', 'low_0', 'high_90', 'low_90']
    labels = ['High\n0°', 'Low\n0°', 'High\n90°', 'Low\n90°']
    colors = [COLORS['high_exp'], COLORS['low_exp'], COLORS['high_exp'], COLORS['low_exp']]
    
    means = [params_df[params_df['condition'] == c]['v'].mean() for c in conditions]
    sems = [params_df[params_df['condition'] == c]['v'].sem() for c in conditions]
    
    bars = ax1.bar(range(4), means, yerr=[s*1.96 for s in sems], 
                   color=colors, capsize=5, edgecolor='black', linewidth=1.5)
    
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('Drift Rate (v)')
    ax1.set_title('A. Drift Rate by Condition', fontweight='bold', loc='left')
    
    # Add bracket for 0° comparison
    ax1.plot([0, 0, 1, 1], [means[0]+0.15, means[0]+0.18, means[0]+0.18, means[1]+0.15], 'k-', lw=1)
    ax1.text(0.5, means[0]+0.2, f'p = {analysis_results["stats"]["v_effect_0"]["p"]:.3f}', 
             ha='center', fontsize=9)
    
    # Panel B: Drift rate interaction
    ax2 = axes[0, 1]
    
    x = [0, 1]
    v_high = [params_df[(params_df['angle'] == 0) & (params_df['expectation'] == 'high')]['v'].mean(),
              params_df[(params_df['angle'] == 90) & (params_df['expectation'] == 'high')]['v'].mean()]
    v_low = [params_df[(params_df['angle'] == 0) & (params_df['expectation'] == 'low')]['v'].mean(),
             params_df[(params_df['angle'] == 90) & (params_df['expectation'] == 'low')]['v'].mean()]
    
    ax2.plot(x, v_high, 'o-', color=COLORS['high_exp'], markersize=10, linewidth=2, label='High Exp')
    ax2.plot(x, v_low, 's--', color=COLORS['low_exp'], markersize=10, linewidth=2, label='Low Exp')
    
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['0°\n(Prediction)', '90°\n(Regularity)'])
    ax2.set_ylabel('Drift Rate (v)')
    ax2.legend(loc='upper right')
    ax2.set_title('B. Drift Rate: Expectation × Angle', fontweight='bold', loc='left')
    
    # Panel C: Drift rate effect sizes
    ax3 = axes[0, 2]
    
    effect_v_0 = analysis_results['stats']['v_effect_0']['diff']
    effect_v_90 = analysis_results['stats']['v_effect_90']['diff']
    
    bars = ax3.bar([0, 1], [effect_v_0, effect_v_90], 
                   color=[COLORS['deg_0'], COLORS['deg_90']],
                   edgecolor='black', linewidth=1.5)
    
    ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(['0°', '90°'])
    ax3.set_ylabel('Expectation Effect on Drift (Δv)')
    ax3.set_title('C. Drift Rate Effect by Angle', fontweight='bold', loc='left')
    
    # Add significance
    p_int = analysis_results['stats']['v_interaction']['p']
    ax3.text(0.5, max(effect_v_0, effect_v_90) * 1.1, 
             f'Interaction: p = {p_int:.4f}', ha='center', fontsize=10)
    
    # =========================================================================
    # Row 2: Boundary Separation
    # =========================================================================
    
    # Panel D: Boundary by condition
    ax4 = axes[1, 0]
    
    means_a = [params_df[params_df['condition'] == c]['a'].mean() for c in conditions]
    sems_a = [params_df[params_df['condition'] == c]['a'].sem() for c in conditions]
    
    bars = ax4.bar(range(4), means_a, yerr=[s*1.96 for s in sems_a],
                   color=colors, capsize=5, edgecolor='black', linewidth=1.5)
    
    ax4.set_xticks(range(4))
    ax4.set_xticklabels(labels)
    ax4.set_ylabel('Boundary Separation (a)')
    ax4.set_title('D. Boundary by Condition', fontweight='bold', loc='left')
    
    # Add bracket for 90° comparison
    ax4.plot([2, 2, 3, 3], [means_a[2]+0.08, means_a[2]+0.1, means_a[2]+0.1, means_a[3]+0.08], 'k-', lw=1)
    ax4.text(2.5, means_a[2]+0.12, f'p = {analysis_results["stats"]["a_effect_90"]["p"]:.3f}',
             ha='center', fontsize=9)
    
    # Panel E: Boundary interaction
    ax5 = axes[1, 1]
    
    a_high = [params_df[(params_df['angle'] == 0) & (params_df['expectation'] == 'high')]['a'].mean(),
              params_df[(params_df['angle'] == 90) & (params_df['expectation'] == 'high')]['a'].mean()]
    a_low = [params_df[(params_df['angle'] == 0) & (params_df['expectation'] == 'low')]['a'].mean(),
             params_df[(params_df['angle'] == 90) & (params_df['expectation'] == 'low')]['a'].mean()]
    
    ax5.plot(x, a_high, 'o-', color=COLORS['high_exp'], markersize=10, linewidth=2, label='High Exp')
    ax5.plot(x, a_low, 's--', color=COLORS['low_exp'], markersize=10, linewidth=2, label='Low Exp')
    
    ax5.set_xticks([0, 1])
    ax5.set_xticklabels(['0°\n(Prediction)', '90°\n(Regularity)'])
    ax5.set_ylabel('Boundary Separation (a)')
    ax5.legend(loc='upper right')
    ax5.set_title('E. Boundary: Expectation × Angle', fontweight='bold', loc='left')
    
    # Panel F: Boundary effect sizes
    ax6 = axes[1, 2]
    
    effect_a_0 = analysis_results['stats']['a_effect_0']['diff']
    effect_a_90 = analysis_results['stats']['a_effect_90']['diff']
    
    bars = ax6.bar([0, 1], [effect_a_0, effect_a_90],
                   color=[COLORS['deg_0'], COLORS['deg_90']],
                   edgecolor='black', linewidth=1.5)
    
    ax6.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax6.set_xticks([0, 1])
    ax6.set_xticklabels(['0°', '90°'])
    ax6.set_ylabel('Expectation Effect on Boundary (Δa)')
    ax6.set_title('F. Boundary Effect by Angle', fontweight='bold', loc='left')
    
    p_int_a = analysis_results['stats']['a_interaction']['p']
    ax6.text(0.5, min(effect_a_0, effect_a_90) * 1.3,
             f'Interaction: p = {p_int_a:.4f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_ddm_summary(params_df, analysis_results, output_path):
    """
    Create summary figure showing the mechanistic dissociation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: Schematic of hypothesis
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    ax1.text(5, 9.5, 'Mechanistic Dissociation Hypothesis', ha='center', 
             fontsize=14, fontweight='bold')
    
    # 0° condition
    ax1.add_patch(plt.Rectangle((0.5, 5), 4, 3.5, fill=True, 
                                 facecolor=COLORS['deg_0'], alpha=0.2,
                                 edgecolor='black', linewidth=2))
    ax1.text(2.5, 8, '0° Rotation', ha='center', fontweight='bold', fontsize=11)
    ax1.text(2.5, 7.2, '(Prediction-based)', ha='center', fontsize=9, style='italic')
    ax1.text(2.5, 6.2, 'Expectations →', ha='center', fontsize=10)
    ax1.text(2.5, 5.5, 'DRIFT RATE', ha='center', fontsize=11, fontweight='bold',
             color=COLORS['drift'])
    
    # 90° condition
    ax1.add_patch(plt.Rectangle((5.5, 5), 4, 3.5, fill=True,
                                 facecolor=COLORS['deg_90'], alpha=0.2,
                                 edgecolor='black', linewidth=2))
    ax1.text(7.5, 8, '90° Rotation', ha='center', fontweight='bold', fontsize=11)
    ax1.text(7.5, 7.2, '(Regularity-based)', ha='center', fontsize=9, style='italic')
    ax1.text(7.5, 6.2, 'Expectations →', ha='center', fontsize=10)
    ax1.text(7.5, 5.5, 'BOUNDARY', ha='center', fontsize=11, fontweight='bold',
             color=COLORS['boundary'])
    
    # Interpretation
    ax1.text(2.5, 4, 'Better evidence\naccumulation', ha='center', fontsize=9,
             color=COLORS['drift'])
    ax1.text(7.5, 4, 'More liberal\ndecision criterion', ha='center', fontsize=9,
             color=COLORS['boundary'])
    
    # Results box
    ax1.add_patch(plt.Rectangle((0.5, 0.5), 9, 2.5, fill=True,
                                 facecolor='white', edgecolor='black', linewidth=1))
    ax1.text(5, 2.5, 'Results', ha='center', fontweight='bold', fontsize=11)
    
    v_p_0 = analysis_results['stats']['v_effect_0']['p']
    a_p_90 = analysis_results['stats']['a_effect_90']['p']
    
    ax1.text(2.5, 1.5, f'Drift effect at 0°:\np = {v_p_0:.4f}', ha='center', fontsize=9)
    ax1.text(7.5, 1.5, f'Boundary effect at 90°:\np = {a_p_90:.4f}', ha='center', fontsize=9)
    
    ax1.set_title('A', fontweight='bold', loc='left', fontsize=14)
    
    # Panel B: Effect size comparison
    ax2 = axes[1]
    
    # Create grouped bar chart
    x = np.array([0, 1])
    width = 0.35
    
    drift_effects = [analysis_results['stats']['v_effect_0']['diff'],
                     analysis_results['stats']['v_effect_90']['diff']]
    boundary_effects = [analysis_results['stats']['a_effect_0']['diff'],
                        analysis_results['stats']['a_effect_90']['diff']]
    
    bars1 = ax2.bar(x - width/2, drift_effects, width, label='Drift Rate (v)',
                    color=COLORS['drift'], edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x + width/2, boundary_effects, width, label='Boundary (a)',
                    color=COLORS['boundary'], edgecolor='black', linewidth=1.5)
    
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['0° (Prediction)', '90° (Regularity)'])
    ax2.set_ylabel('Expectation Effect (High - Low)')
    ax2.legend(loc='upper right')
    ax2.set_title('B. Expectation Effects on DDM Parameters', fontweight='bold', loc='left')
    
    # Add significance markers
    for i, (d, b) in enumerate(zip(drift_effects, boundary_effects)):
        if i == 0:  # 0° - drift should be significant
            if analysis_results['stats']['v_effect_0']['p'] < 0.05:
                ax2.text(i - width/2, d + 0.02, '*', ha='center', fontsize=14)
        else:  # 90° - boundary should be significant
            if analysis_results['stats']['a_effect_90']['p'] < 0.05:
                ax2.text(i + width/2, b - 0.05, '*', ha='center', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run full DDM analysis pipeline."""
    print("=" * 70)
    print("DRIFT DIFFUSION MODEL ANALYSIS")
    print("Testing Mechanistic Dissociation of Expectation Effects")
    print("=" * 70)
    print()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Simulate DDM data
    df_sim, conditions = simulate_ddm_experiment(n_participants=30, n_trials_per_condition=50)
    
    # Fit EZ-diffusion model
    print("=" * 70)
    print("FITTING EZ-DIFFUSION MODEL")
    print("=" * 70)
    print()
    
    params_df = fit_ez_diffusion_by_condition(df_sim)
    print(f"Fitted parameters for {params_df['participant'].nunique()} participants")
    print(f"Total parameter estimates: {len(params_df)}")
    print()
    
    # Analyze parameters
    analysis_results = analyze_ddm_parameters(params_df)
    
    # Create visualizations
    print("=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    print()
    
    plot_ddm_results(params_df, analysis_results, OUTPUT_DIR / "fig6_ddm_parameters.png")
    plot_ddm_summary(params_df, analysis_results, OUTPUT_DIR / "fig7_ddm_summary.png")
    
    print()
    print("=" * 70)
    print("DDM ANALYSIS COMPLETE")
    print("=" * 70)
    print()
    print("Generated figures:")
    print("  - fig6_ddm_parameters.png: Full parameter analysis")
    print("  - fig7_ddm_summary.png: Mechanistic dissociation summary")
    print()
    
    return df_sim, params_df, analysis_results


if __name__ == "__main__":
    df_sim, params_df, analysis_results = main()
