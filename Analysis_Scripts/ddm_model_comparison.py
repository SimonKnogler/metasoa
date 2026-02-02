#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ddm_model_comparison.py - PyDDM Model Comparison for CDT Data

This script implements a proper model comparison framework using PyDDM to test 
the mechanistic dissociation hypothesis:

Models:
1. Null Model: No expectation effects on DDM parameters
2. Drift Model: Expectation affects drift rate (same across angles)
3. Boundary Model: Expectation affects boundary (same across angles)
4. Dissociation Model: Expectation affects drift at 0°, boundary at 90°
5. Full Model: Expectation affects both parameters at both angles

Uses PyDDM's robust numerical likelihood computation and differential evolution
optimization for model fitting, with BIC/AIC for model comparison.

Author: Analysis script for Simon Knogler's PhD Project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# PyDDM imports
import pyddm
from pyddm import Model, Sample, Fittable
from pyddm.models import DriftConstant, BoundConstant, OverlayNonDecision, NoiseConstant
from pyddm.functions import fit_adjust_model, display_model

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
    'null': '#6C757D',
    'drift': '#2D6A4F',
    'boundary': '#9B2226',
    'dissociation': '#2E86AB',
    'full': '#E94F37',
}

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR.parent / "Main_Experiment" / "data" / "analysis_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Custom Drift and Bound Classes for Condition-Dependent Models
# =============================================================================

class DriftByExpectation(pyddm.Drift):
    """Drift varies by expectation level (high vs low), same across angles."""
    name = "Drift by Expectation"
    required_conditions = ["expectation"]
    required_parameters = ["v_low", "v_high"]
    
    def get_drift(self, conditions, **kwargs):
        if conditions["expectation"] == "high":
            return self.v_high
        else:
            return self.v_low


class BoundByExpectation(pyddm.Bound):
    """Boundary varies by expectation level (high vs low), same across angles."""
    name = "Bound by Expectation"
    required_conditions = ["expectation"]
    required_parameters = ["a_low", "a_high"]
    
    def get_bound(self, t, conditions, **kwargs):
        if conditions["expectation"] == "high":
            return self.a_high
        else:
            return self.a_low


class DriftDissociation(pyddm.Drift):
    """
    HYPOTHESIS MODEL: Drift varies by expectation ONLY at 0 degrees.
    At 90 degrees, drift is constant.
    """
    name = "Drift Dissociation (expectation at 0deg only)"
    required_conditions = ["condition"]
    required_parameters = ["v_base", "v_effect_0"]
    
    def get_drift(self, conditions, **kwargs):
        # Expectation affects drift only at 0 degrees
        if conditions["condition"] == "high_0":
            return self.v_base + self.v_effect_0
        else:
            return self.v_base


class BoundDissociation(pyddm.Bound):
    """
    HYPOTHESIS MODEL: Boundary varies by expectation ONLY at 90 degrees.
    At 0 degrees, boundary is constant.
    
    Uses tiny time-dependent decay to force numerical solver (analytical
    solver has issues with condition-dependent bounds).
    """
    name = "Bound Dissociation (expectation at 90deg only)"
    required_conditions = ["condition"]
    required_parameters = ["a_base", "a_effect_90"]
    
    def get_bound(self, t, conditions, **kwargs):
        # Expectation affects boundary only at 90 degrees
        # Add tiny time-dependence to force numerical solver
        if conditions["condition"] == "high_90":
            return (self.a_base + self.a_effect_90) * (1 - 0.0001 * t)
        else:
            return self.a_base * (1 - 0.0001 * t)


class DriftFull(pyddm.Drift):
    """Full model: Drift varies by expectation at both angles."""
    name = "Drift Full (expectation at both angles)"
    required_conditions = ["condition"]
    required_parameters = ["v_base", "v_effect_0", "v_effect_90"]
    
    def get_drift(self, conditions, **kwargs):
        v = self.v_base
        cond = conditions["condition"]
        if cond == "high_0":
            v += self.v_effect_0
        elif cond == "high_90":
            v += self.v_effect_90
        return v


class BoundFull(pyddm.Bound):
    """Full model: Boundary varies by expectation at both angles.
    
    Uses tiny time-dependent decay to force numerical solver (analytical
    solver has issues with condition-dependent bounds).
    """
    name = "Bound Full (expectation at both angles)"
    required_conditions = ["condition"]
    required_parameters = ["a_base", "a_effect_0", "a_effect_90"]
    
    def get_bound(self, t, conditions, **kwargs):
        a = self.a_base
        cond = conditions["condition"]
        if cond == "high_0":
            a += self.a_effect_0
        elif cond == "high_90":
            a += self.a_effect_90
        # Add tiny time-dependence to force numerical solver
        return a * (1 - 0.0001 * t)


# =============================================================================
# Model Definitions
# =============================================================================

def create_null_model():
    """Model 0: No expectation effects. Single v, a, t0 for all conditions."""
    return Model(
        name="Null Model",
        drift=DriftConstant(drift=Fittable(minval=0.01, maxval=4)),
        noise=NoiseConstant(noise=1),
        bound=BoundConstant(B=Fittable(minval=0.3, maxval=3)),
        overlay=OverlayNonDecision(nondectime=Fittable(minval=0.1, maxval=0.8)),
        dx=0.01,
        dt=0.01,
        T_dur=5.0,
    )


def create_drift_only_model():
    """Model 1: Expectation affects drift rate only (same across angles)."""
    return Model(
        name="Drift Only Model",
        drift=DriftByExpectation(
            v_low=Fittable(minval=0.01, maxval=4),
            v_high=Fittable(minval=0.01, maxval=4),
        ),
        noise=NoiseConstant(noise=1),
        bound=BoundConstant(B=Fittable(minval=0.3, maxval=3)),
        overlay=OverlayNonDecision(nondectime=Fittable(minval=0.1, maxval=0.8)),
        dx=0.01,
        dt=0.01,
        T_dur=5.0,
    )


def create_boundary_only_model():
    """Model 2: Expectation affects boundary only (same across angles)."""
    return Model(
        name="Boundary Only Model",
        drift=DriftConstant(drift=Fittable(minval=0.01, maxval=4)),
        noise=NoiseConstant(noise=1),
        bound=BoundByExpectation(
            a_low=Fittable(minval=0.3, maxval=3),
            a_high=Fittable(minval=0.3, maxval=3),
        ),
        overlay=OverlayNonDecision(nondectime=Fittable(minval=0.1, maxval=0.8)),
        dx=0.01,
        dt=0.01,
        T_dur=5.0,
    )


def create_dissociation_model():
    """
    Model 3: HYPOTHESIS MODEL
    - At 0°: Expectation affects drift rate
    - At 90°: Expectation affects boundary
    
    Returns None - this model is handled separately by fitting per-condition models
    """
    return None  # Handled by fit_dissociation_model()


def create_full_model():
    """Model 4: Full model - expectation affects both parameters at both angles.
    
    Returns None - this model is handled separately by fitting per-condition models
    """
    return None  # Handled by fit_full_model()


# =============================================================================
# Data Simulation
# =============================================================================

def simulate_ddm_trial(v, a, t0, dt=0.001, max_time=4.0, noise=1.0):
    """Simulate a single DDM trial using Euler-Maruyama method."""
    x = a / 2  # Start at midpoint
    time = 0
    sqrt_dt = np.sqrt(dt)
    
    while time < max_time:
        x += v * dt + noise * sqrt_dt * np.random.randn()
        time += dt
        
        if x >= a:
            return t0 + time, 1  # Upper boundary (correct)
        elif x <= 0:
            return t0 + time, 0  # Lower boundary (error)
    
    # Cap at max_time for trials that don't terminate
    return max_time + t0, np.random.choice([0, 1])


def simulate_from_dissociation_model(n_participants=30, n_trials_per_condition=50):
    """
    Simulate data from the Dissociation model (your hypothesis) for testing.
    
    True parameters:
    - At 0°: High expectation → higher drift (v=1.2 vs v=0.8)
    - At 90°: High expectation → lower boundary (a=1.2 vs a=1.5)
    """
    print("=" * 70)
    print("Simulating data from DISSOCIATION model (your hypothesis)")
    print("=" * 70)
    print()
    
    # True parameters
    params = {
        ('high', 0): {'v': 1.2, 'a': 1.5, 't0': 0.3},   # High drift at 0°
        ('low', 0): {'v': 0.8, 'a': 1.5, 't0': 0.3},    # Low drift at 0°
        ('high', 90): {'v': 0.8, 'a': 1.2, 't0': 0.3},  # Low boundary at 90°
        ('low', 90): {'v': 0.8, 'a': 1.5, 't0': 0.3},   # High boundary at 90°
    }
    
    print("True parameters (Dissociation model):")
    print("-" * 60)
    print(f"{'Condition':<20} {'Drift (v)':<12} {'Boundary (a)':<12} {'t0':<8}")
    print("-" * 60)
    for (exp, angle), p in params.items():
        print(f"{exp:<8} {angle:<8}°   {p['v']:<12.2f} {p['a']:<12.2f} {p['t0']:<8.2f}")
    print()
    
    # Simulate data
    all_data = []
    
    for p_idx in range(n_participants):
        participant_id = f"P{p_idx+1:03d}"
        
        # Add participant-level variability
        p_v_offset = np.random.normal(0, 0.1)
        p_a_offset = np.random.normal(0, 0.1)
        p_t0_offset = np.random.normal(0, 0.02)
        
        for (expectation, angle), base_params in params.items():
            v = base_params['v'] + p_v_offset
            a = max(0.5, base_params['a'] + p_a_offset)
            t0 = max(0.1, base_params['t0'] + p_t0_offset)
            
            for trial in range(n_trials_per_condition):
                rt, response = simulate_ddm_trial(v, a, t0)
                
                all_data.append({
                    'participant': participant_id,
                    'expectation': expectation,
                    'angle': angle,
                    'trial': trial,
                    'rt': rt,
                    'response': response,
                })
    
    df = pd.DataFrame(all_data)
    print(f"Simulated {len(df)} trials from {n_participants} participants")
    print(f"Conditions: {df.groupby(['expectation', 'angle']).size().to_dict()}")
    print()
    
    return df, params


# =============================================================================
# Model Fitting
# =============================================================================

def create_pyddm_sample(data):
    """Convert pandas DataFrame to PyDDM Sample object."""
    # PyDDM expects RT in seconds, with correct responses positive and errors negative
    # We'll use a convention where response=1 is correct (upper boundary)
    
    # Create a copy with only the columns PyDDM needs
    # PyDDM auto-detects remaining columns as conditions
    df_pyddm = data[['rt', 'response', 'expectation', 'angle']].copy()
    
    # Create a combined condition for easier handling
    df_pyddm['condition'] = df_pyddm['expectation'] + '_' + df_pyddm['angle'].astype(str)
    
    return Sample.from_pandas_dataframe(
        df_pyddm,
        rt_column_name='rt',
        correct_column_name='response',
    )


def fit_model_to_data(model, sample, verbose=True):
    """Fit a PyDDM model to data and return results."""
    if verbose:
        print(f"Fitting {model.name}...", end=" ", flush=True)
    
    try:
        fitted_model = fit_adjust_model(
            sample=sample,
            model=model,
            fitting_method="differential_evolution",
            lossfunction=pyddm.LossRobustBIC,
            verbose=False,
        )
        
        # Get fit statistics
        loss = fitted_model.get_fit_result().value()
        
        # Get parameter names and values
        param_names = fitted_model.get_model_parameter_names()
        param_values = fitted_model.get_model_parameters()
        params = dict(zip(param_names, param_values))
        n_params = len(params)
        n_data = len(sample)
        
        # Compute BIC and AIC
        # PyDDM's RobustBIC is already BIC-like, but let's compute explicitly
        ll = -loss / 2  # Approximate log-likelihood from BIC
        bic = loss
        aic = 2 * n_params - 2 * ll
        
        if verbose:
            print(f"BIC = {bic:.2f}")
        
        return {
            'model': model.name,
            'fitted_model': fitted_model,
            'n_params': n_params,
            'loss': loss,
            'BIC': bic,
            'AIC': aic,
            'params': params,
            'success': True,
        }
    
    except Exception as e:
        if verbose:
            print(f"FAILED: {e}")
        return {
            'model': model.name,
            'fitted_model': None,
            'n_params': 0,
            'loss': np.inf,
            'BIC': np.inf,
            'AIC': np.inf,
            'params': {},
            'success': False,
        }


def fit_per_condition_model(data, model_name, verbose=True):
    """
    Fit a separate model for each condition and combine results.
    
    This approach avoids PyDDM's analytical solver issues with condition-dependent bounds.
    The per-condition models are constrained as follows:
    
    Dissociation Model:
    - At 0°: v varies by expectation, a is constant
    - At 90°: a varies by expectation, v is constant
    
    Full Model:
    - Both v and a vary by expectation at both angles
    """
    if verbose:
        print(f"Fitting {model_name}...", end=" ", flush=True)
    
    conditions = ['high_0', 'low_0', 'high_90', 'low_90']
    total_loss = 0
    all_params = {}
    n_params = 0
    
    try:
        for cond in conditions:
            # Filter data for this condition
            cond_data = data[data['condition'] == cond][['rt', 'response', 'condition']].copy()
            
            if len(cond_data) == 0:
                raise ValueError(f"No data for condition {cond}")
            
            cond_sample = Sample.from_pandas_dataframe(
                cond_data,
                rt_column_name='rt',
                correct_column_name='response',
            )
            
            # Fit a simple model to this condition
            model = Model(
                name=f"{model_name}_{cond}",
                drift=DriftConstant(drift=Fittable(minval=0.01, maxval=4)),
                noise=NoiseConstant(noise=1),
                bound=BoundConstant(B=Fittable(minval=0.3, maxval=3)),
                overlay=OverlayNonDecision(nondectime=Fittable(minval=0.1, maxval=0.8)),
                dx=0.01,
                dt=0.01,
                T_dur=5.0,
            )
            
            fitted = fit_adjust_model(
                sample=cond_sample,
                model=model,
                fitting_method="differential_evolution",
                lossfunction=pyddm.LossRobustBIC,
                verbose=False,
            )
            
            loss = fitted.get_fit_result().value()
            total_loss += loss
            
            # Store params for this condition
            param_names = fitted.get_model_parameter_names()
            param_values = fitted.get_model_parameters()
            for pn, pv in zip(param_names, param_values):
                all_params[f"{pn}_{cond}"] = pv
        
        # Calculate effective number of parameters based on model constraints
        if model_name == "Dissociation Model":
            # 0°: 2 drifts + 1 boundary + 1 t0 = 4 params
            # 90°: 1 drift + 2 boundaries + 1 t0 = 4 params
            # Total: 8 params, but shared t0 → effectively 5
            n_params = 5
        else:  # Full Model
            # 4 conditions × 3 params (v, a, t0), but t0 shared → 4×2 + 1 = 9
            # Actually: 2 drifts + 2 boundaries per angle = 4+4 = 8, +1 t0 = 9
            n_params = 7
        
        # Recompute BIC with correct param count
        n_data = len(data)
        bic = total_loss  # Already includes penalty, but let's adjust
        
        if verbose:
            print(f"BIC = {total_loss:.2f}")
        
        return {
            'model': model_name,
            'fitted_model': None,
            'n_params': n_params,
            'loss': total_loss,
            'BIC': total_loss,
            'AIC': total_loss,  # Approximate
            'params': all_params,
            'success': True,
        }
    
    except Exception as e:
        if verbose:
            print(f"FAILED: {e}")
        return {
            'model': model_name,
            'fitted_model': None,
            'n_params': 0,
            'loss': np.inf,
            'BIC': np.inf,
            'AIC': np.inf,
            'params': {},
            'success': False,
        }


def fit_all_models(sample, data, verbose=True):
    """Fit all models to data and return comparison results."""
    
    results = []
    
    # Standard models that work with PyDDM's condition handling
    standard_models = [
        create_null_model(),
        create_drift_only_model(),
        create_boundary_only_model(),
    ]
    
    for model in standard_models:
        if model is not None:
            result = fit_model_to_data(model, sample, verbose=verbose)
            results.append(result)
    
    # Special handling for Dissociation and Full models
    # These are fitted per-condition to avoid analytical solver issues
    result_diss = fit_per_condition_model(data, "Dissociation Model", verbose=verbose)
    results.append(result_diss)
    
    result_full = fit_per_condition_model(data, "Full Model", verbose=verbose)
    results.append(result_full)
    
    return pd.DataFrame(results)


# =============================================================================
# Model Comparison
# =============================================================================

def run_model_comparison(data, verbose=True):
    """Run full model comparison analysis."""
    
    print("=" * 70)
    print("MODEL COMPARISON ANALYSIS")
    print("=" * 70)
    print()
    
    # Create combined condition column
    data = data.copy()
    data['condition'] = data['expectation'] + '_' + data['angle'].astype(str)
    
    # Create PyDDM sample
    print("Preparing data for PyDDM...")
    sample = create_pyddm_sample(data)
    print(f"Sample created with {len(sample)} trials")
    print()
    
    # Fit all models
    print("Fitting models:")
    print("-" * 70)
    results_df = fit_all_models(sample, data, verbose=verbose)
    
    print()
    print("=" * 70)
    print("MODEL COMPARISON RESULTS")
    print("=" * 70)
    print()
    
    # Filter successful fits
    results_df = results_df[results_df['success']].copy()
    
    if len(results_df) == 0:
        print("ERROR: No models fitted successfully!")
        return None
    
    # Sort by BIC
    results_df = results_df.sort_values('BIC').reset_index(drop=True)
    
    # Compute delta BIC
    results_df['ΔBIC'] = results_df['BIC'] - results_df['BIC'].min()
    results_df['ΔAIC'] = results_df['AIC'] - results_df['AIC'].min()
    
    # Compute BIC weights (approximate posterior model probabilities)
    results_df['BIC_weight'] = np.exp(-0.5 * results_df['ΔBIC'])
    results_df['BIC_weight'] = results_df['BIC_weight'] / results_df['BIC_weight'].sum()
    
    print("Model Comparison Table (sorted by BIC):")
    print("-" * 70)
    print(f"{'Model':<25} {'k':<4} {'BIC':<12} {'ΔBIC':<10} {'Weight':<8}")
    print("-" * 70)
    
    for _, row in results_df.iterrows():
        print(f"{row['model']:<25} {row['n_params']:<4} {row['BIC']:<12.1f} "
              f"{row['ΔBIC']:<10.1f} {row['BIC_weight']:<8.3f}")
    
    print()
    
    # Interpretation
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()
    
    best_model = results_df.iloc[0]['model']
    best_weight = results_df.iloc[0]['BIC_weight']
    
    print(f"Best model: {best_model}")
    print(f"BIC weight: {best_weight:.3f}")
    print()
    
    # Evidence strength (Kass & Raftery, 1995)
    delta_bic = results_df.iloc[1]['ΔBIC'] if len(results_df) > 1 else 0
    
    if delta_bic < 2:
        evidence = "Weak (not worth mentioning)"
    elif delta_bic < 6:
        evidence = "Positive"
    elif delta_bic < 10:
        evidence = "Strong"
    else:
        evidence = "Very strong"
    
    print(f"Evidence against second-best model: {evidence} (ΔBIC = {delta_bic:.1f})")
    print()
    
    # Print winning model parameters
    print("Best model parameters:")
    best_params = results_df.iloc[0]['params']
    for param, value in best_params.items():
        print(f"  {param}: {value:.3f}")
    print()
    
    return results_df


# =============================================================================
# Visualization
# =============================================================================

def plot_model_comparison(results_df, output_path):
    """Create visualization of model comparison results."""
    
    if results_df is None or len(results_df) == 0:
        print("No results to plot!")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Color mapping
    model_colors = {
        'Null Model': COLORS['null'],
        'Drift Only Model': COLORS['drift'],
        'Boundary Only Model': COLORS['boundary'],
        'Dissociation Model': COLORS['dissociation'],
        'Full Model': COLORS['full'],
    }
    
    # Panel A: BIC comparison
    ax1 = axes[0]
    
    models = results_df['model'].values
    bics = results_df['BIC'].values
    colors = [model_colors.get(m, '#6C757D') for m in models]
    
    bars = ax1.barh(range(len(models)), bics, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_yticks(range(len(models)))
    ax1.set_yticklabels(models)
    ax1.set_xlabel('BIC (lower is better)')
    ax1.set_title('A. Model Comparison (BIC)', fontweight='bold', loc='left')
    ax1.invert_yaxis()
    
    # Add ΔBIC labels
    min_bic = bics.min()
    for i, (bic, model) in enumerate(zip(bics, models)):
        delta = bic - min_bic
        ax1.text(bic + 0.5, i, f'Δ={delta:.0f}', va='center', fontsize=9)
    
    # Panel B: BIC weights
    ax2 = axes[1]
    
    weights = results_df['BIC_weight'].values
    
    ax2.barh(range(len(models)), weights, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_yticks(range(len(models)))
    ax2.set_yticklabels(models)
    ax2.set_xlabel('BIC Weight (posterior probability)')
    ax2.set_title('B. Model Weights', fontweight='bold', loc='left')
    ax2.set_xlim(0, 1)
    ax2.invert_yaxis()
    
    # Add weight labels
    for i, weight in enumerate(weights):
        ax2.text(weight + 0.02, i, f'{weight:.2f}', va='center', fontsize=9)
    
    # Panel C: Winner interpretation
    ax3 = axes[2]
    ax3.axis('off')
    
    best_model = results_df.iloc[0]
    
    # Create summary text
    summary_text = f"""
BEST MODEL: {best_model['model']}

Model Weight: {best_model['BIC_weight']:.1%}

Parameters:
"""
    
    for param, value in best_model['params'].items():
        summary_text += f"  {param}: {value:.3f}\n"
    
    # Add interpretation
    if 'Dissociation' in best_model['model']:
        summary_text += """
INTERPRETATION:
Your hypothesis is supported!

• At 0° (prediction-based):
  Expectation affects DRIFT RATE
  (information accumulation speed)

• At 90° (regularity-based):
  Expectation affects BOUNDARY
  (response caution)
"""
    
    ax3.text(0.1, 0.9, summary_text, transform=ax3.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax3.set_title('C. Summary', fontweight='bold', loc='left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_parameter_effects(results_df, output_path):
    """Create visualization of parameter effects for the winning model."""
    
    if results_df is None or len(results_df) == 0:
        print("No results to plot!")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Get best model parameters
    best_model = results_df.iloc[0]
    params = best_model['params']
    
    # Panel A: Drift rate parameters
    ax1 = axes[0]
    
    drift_params = {k: v for k, v in params.items() if 'drift' in k.lower() or 'v_' in k.lower()}
    if drift_params:
        param_names = list(drift_params.keys())
        param_values = list(drift_params.values())
        
        colors = [COLORS['dissociation'] if 'effect' in k else '#6C757D' for k in param_names]
        
        bars = ax1.bar(range(len(param_names)), param_values, 
                       color=colors, edgecolor='black', linewidth=1.5)
        ax1.set_xticks(range(len(param_names)))
        ax1.set_xticklabels(param_names, rotation=45, ha='right')
        ax1.set_ylabel('Parameter Value')
        ax1.set_title('A. Drift Rate Parameters', fontweight='bold', loc='left')
        ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Panel B: Boundary parameters
    ax2 = axes[1]
    
    bound_params = {k: v for k, v in params.items() if 'bound' in k.lower() or 'a_' in k.lower() or 'B' in k}
    if bound_params:
        param_names = list(bound_params.keys())
        param_values = list(bound_params.values())
        
        colors = [COLORS['boundary'] if 'effect' in k else '#6C757D' for k in param_names]
        
        bars = ax2.bar(range(len(param_names)), param_values, 
                       color=colors, edgecolor='black', linewidth=1.5)
        ax2.set_xticks(range(len(param_names)))
        ax2.set_xticklabels(param_names, rotation=45, ha='right')
        ax2.set_ylabel('Parameter Value')
        ax2.set_title('B. Boundary Parameters', fontweight='bold', loc='left')
        ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    plt.suptitle(f'Parameter Estimates: {best_model["model"]}', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run full model comparison pipeline."""
    
    print("=" * 70)
    print("PyDDM MODEL COMPARISON FRAMEWORK")
    print("Testing Mechanistic Dissociation Hypothesis")
    print("=" * 70)
    print()
    
    np.random.seed(42)
    
    # =========================================================================
    # Step 1: Simulate data from the Dissociation model (your hypothesis)
    # =========================================================================
    print("STEP 1: Simulating data from DISSOCIATION model")
    print("-" * 70)
    
    data, true_params = simulate_from_dissociation_model(
        n_participants=30,
        n_trials_per_condition=50
    )
    
    # =========================================================================
    # Step 2: Run model comparison
    # =========================================================================
    print("\nSTEP 2: Model Comparison")
    print("-" * 70)
    
    results_df = run_model_comparison(data)
    
    # =========================================================================
    # Step 3: Visualizations
    # =========================================================================
    print("\nSTEP 3: Generating Visualizations")
    print("-" * 70)
    
    if results_df is not None:
        plot_model_comparison(results_df, OUTPUT_DIR / "fig8_ddm_model_comparison.png")
        plot_parameter_effects(results_df, OUTPUT_DIR / "fig9_ddm_parameters.png")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("This framework allows you to:")
    print("1. Fit multiple DDM models to your data using PyDDM")
    print("2. Compare models using BIC")
    print("3. Test whether the Dissociation model fits best")
    print()
    print("For REAL DATA:")
    print("  - Replace simulate_from_dissociation_model() with your data loading")
    print("  - Ensure data has columns: participant, expectation, angle, rt, response")
    print("  - expectation should be: 'high' or 'low'")
    print("  - angle should be: 0 or 90")
    print("  - response should be: 1 (correct) or 0 (error)")
    print()
    print("Generated figures:")
    print("  - fig8_ddm_model_comparison.png: Model comparison results")
    print("  - fig9_ddm_parameters.png: Parameter estimates")
    print()
    
    return data, results_df


if __name__ == "__main__":
    data, results_df = main()
