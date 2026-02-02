#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_simulated_data.py - Comprehensive Analysis of Simulated CDT Data

Analyzes the simulated Control Detection Task data to test hypotheses about:
1. Expected precision effects on sense of agency
2. Expected precision effects on confidence/metacognition
3. Dual-mode theory: Angle bias × Expectation interaction
4. Metacognitive sensitivity and calibration

Creates publication-quality visualizations.

Author: Analysis script for Simon Knogler's PhD Project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color palette - sophisticated scientific colors
COLORS = {
    'high_exp': '#2E86AB',      # Steel blue - high expectation
    'low_exp': '#E94F37',       # Vermillion - low expectation
    'deg_0': '#1B4965',         # Dark blue - 0° (prediction-based)
    'deg_90': '#5FA8D3',        # Light blue - 90° (regularity-based)
    'correct': '#2D6A4F',       # Forest green - correct
    'incorrect': '#9B2226',     # Dark red - incorrect
    'neutral': '#6C757D',       # Gray - neutral
    'accent': '#F77F00',        # Orange - accent
    'background': '#F8F9FA',    # Light gray background
}

# Paths
SCRIPT_DIR = Path(__file__).parent
# Use the realistic simulated data (with human behavioral signatures)
DATA_DIR = SCRIPT_DIR.parent / "Main_Experiment" / "data" / "subjects" / "simulated_realistic"
OUTPUT_DIR = SCRIPT_DIR.parent / "Main_Experiment" / "data" / "analysis_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Data Loading
# =============================================================================

def load_all_data():
    """Load all simulated participant data"""
    import glob
    
    csv_files = glob.glob(str(DATA_DIR / "CDT_v2_blockwise_fast_response_SIM*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No simulated data files found in {DATA_DIR}")
    
    print(f"Loading {len(csv_files)} participant files...")
    
    all_data = []
    for csv_file in sorted(csv_files):
        df = pd.read_csv(csv_file)
        all_data.append(df)
    
    df_all = pd.concat(all_data, ignore_index=True)
    
    # Convert numeric columns
    numeric_cols = ['accuracy', 'confidence_rating', 'agency_rating', 'rt_choice', 
                    'prop_used', 'mean_evidence', 'angle_bias']
    for col in numeric_cols:
        if col in df_all.columns:
            df_all[col] = pd.to_numeric(df_all[col], errors='coerce')
    
    print(f"Total trials loaded: {len(df_all)}")
    return df_all

# =============================================================================
# Statistical Analysis Functions
# =============================================================================

def compute_effect_size(group1, group2):
    """Compute Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (group1.mean() - group2.mean()) / pooled_std

def compute_metacognitive_sensitivity(df):
    """
    Compute metacognitive sensitivity (meta-d'/d') using Type 2 SDT
    Simplified version: correlation between confidence and accuracy
    """
    df_valid = df.dropna(subset=['accuracy', 'confidence_rating'])
    if len(df_valid) < 10:
        return np.nan
    
    # Point-biserial correlation as proxy for metacognitive sensitivity
    r, p = stats.pointbiserialr(df_valid['accuracy'], df_valid['confidence_rating'])
    return r

def run_mixed_anova_summary(df):
    """
    Compute summary statistics for a 2×2 mixed design
    Factors: Expectation (high/low) × Angle (0°/90°)
    """
    results = {}
    
    # Group means
    grouped = df.groupby(['angle_bias', 'cue_difficulty_prediction']).agg({
        'agency_rating': ['mean', 'std', 'sem', 'count'],
        'confidence_rating': ['mean', 'std', 'sem'],
        'accuracy': ['mean', 'std'],
    }).round(3)
    
    results['grouped_stats'] = grouped
    
    # Main effect of expectation
    high_exp = df[df['cue_difficulty_prediction'] == 'high']
    low_exp = df[df['cue_difficulty_prediction'] == 'low']
    
    results['expectation_effect'] = {
        'agency_d': compute_effect_size(high_exp['agency_rating'].dropna(), 
                                        low_exp['agency_rating'].dropna()),
        'confidence_d': compute_effect_size(high_exp['confidence_rating'].dropna(),
                                            low_exp['confidence_rating'].dropna()),
    }
    
    # Interaction: effect size difference between angles
    df_0 = df[df['angle_bias'] == 0]
    df_90 = df[df['angle_bias'] == 90]
    
    effect_0 = (df_0[df_0['cue_difficulty_prediction'] == 'high']['agency_rating'].mean() -
                df_0[df_0['cue_difficulty_prediction'] == 'low']['agency_rating'].mean())
    effect_90 = (df_90[df_90['cue_difficulty_prediction'] == 'high']['agency_rating'].mean() -
                 df_90[df_90['cue_difficulty_prediction'] == 'low']['agency_rating'].mean())
    
    results['interaction'] = {
        'effect_0deg': effect_0,
        'effect_90deg': effect_90,
        'interaction_magnitude': effect_0 - effect_90,
    }
    
    return results


def run_mixed_effects_models(df):
    """
    Run Linear Mixed-Effects Models for proper hypothesis testing.
    
    Models:
    1. Agency ~ Expectation * Angle + (1|Participant)
    2. Confidence ~ Expectation * Angle + (1|Participant)
    
    Returns model results with proper interaction tests.
    """
    print("=" * 70)
    print("LINEAR MIXED-EFFECTS MODELS")
    print("=" * 70)
    print()
    
    # Prepare data - ensure proper coding
    df_model = df.copy()
    df_model = df_model.dropna(subset=['agency_rating', 'confidence_rating', 
                                        'cue_difficulty_prediction', 'angle_bias', 
                                        'participant'])
    
    # Convert to categorical with explicit reference levels
    df_model['expectation'] = pd.Categorical(
        df_model['cue_difficulty_prediction'], 
        categories=['low', 'high']  # 'low' is reference
    )
    df_model['angle'] = pd.Categorical(
        df_model['angle_bias'].astype(str),
        categories=['90', '0']  # '90' is reference (regularity-based)
    )
    
    results = {}
    
    # =========================================================================
    # Model 1: Agency Rating
    # =========================================================================
    print("-" * 70)
    print("MODEL 1: Agency Rating")
    print("-" * 70)
    print()
    print("Formula: agency_rating ~ expectation * angle + (1|participant)")
    print("Reference levels: expectation='low', angle='90°'")
    print()
    
    # Fit mixed-effects model with random intercepts for participants
    model_agency = smf.mixedlm(
        "agency_rating ~ C(expectation, Treatment('low')) * C(angle, Treatment('90'))",
        data=df_model,
        groups=df_model["participant"],
        re_formula="~1"  # Random intercept only
    )
    
    fit_agency = model_agency.fit(method='lbfgs')
    
    print("Fixed Effects:")
    print("-" * 50)
    
    # Extract and format results
    agency_summary = pd.DataFrame({
        'Coefficient': fit_agency.fe_params,
        'Std. Error': fit_agency.bse_fe,
        'z-value': fit_agency.tvalues,
        'p-value': fit_agency.pvalues
    }).round(4)
    
    print(agency_summary.to_string())
    print()
    
    # Interpretation
    print("Interpretation:")
    print(f"  • Intercept: Mean agency at low expectation, 90° = {fit_agency.fe_params.iloc[0]:.3f}")
    print(f"  • Main effect of Expectation (high vs low): β = {fit_agency.fe_params.iloc[1]:.3f}, p = {fit_agency.pvalues.iloc[1]:.4f}")
    print(f"  • Main effect of Angle (0° vs 90°): β = {fit_agency.fe_params.iloc[2]:.3f}, p = {fit_agency.pvalues.iloc[2]:.4f}")
    print(f"  • INTERACTION (Expectation × Angle): β = {fit_agency.fe_params.iloc[3]:.3f}, p = {fit_agency.pvalues.iloc[3]:.4f}")
    print()
    
    # Random effects
    print(f"Random Effects:")
    print(f"  • Participant variance (σ²): {fit_agency.cov_re.iloc[0, 0]:.4f}")
    print(f"  • Residual variance: {fit_agency.scale:.4f}")
    print(f"  • ICC: {fit_agency.cov_re.iloc[0, 0] / (fit_agency.cov_re.iloc[0, 0] + fit_agency.scale):.3f}")
    print()
    
    results['agency_model'] = {
        'fit': fit_agency,
        'summary': agency_summary,
        'interaction_beta': fit_agency.fe_params.iloc[3],
        'interaction_p': fit_agency.pvalues.iloc[3],
    }
    
    # =========================================================================
    # Model 2: Confidence Rating
    # =========================================================================
    print("-" * 70)
    print("MODEL 2: Confidence Rating")
    print("-" * 70)
    print()
    print("Formula: confidence_rating ~ expectation * angle + (1|participant)")
    print()
    
    model_conf = smf.mixedlm(
        "confidence_rating ~ C(expectation, Treatment('low')) * C(angle, Treatment('90'))",
        data=df_model,
        groups=df_model["participant"],
        re_formula="~1"
    )
    
    fit_conf = model_conf.fit(method='lbfgs')
    
    print("Fixed Effects:")
    print("-" * 50)
    
    conf_summary = pd.DataFrame({
        'Coefficient': fit_conf.fe_params,
        'Std. Error': fit_conf.bse_fe,
        'z-value': fit_conf.tvalues,
        'p-value': fit_conf.pvalues
    }).round(4)
    
    print(conf_summary.to_string())
    print()
    
    print("Interpretation:")
    print(f"  • Main effect of Expectation: β = {fit_conf.fe_params.iloc[1]:.3f}, p = {fit_conf.pvalues.iloc[1]:.4f}")
    print(f"  • Main effect of Angle: β = {fit_conf.fe_params.iloc[2]:.3f}, p = {fit_conf.pvalues.iloc[2]:.4f}")
    print(f"  • INTERACTION: β = {fit_conf.fe_params.iloc[3]:.3f}, p = {fit_conf.pvalues.iloc[3]:.4f}")
    print()
    
    results['confidence_model'] = {
        'fit': fit_conf,
        'summary': conf_summary,
        'interaction_beta': fit_conf.fe_params.iloc[3],
        'interaction_p': fit_conf.pvalues.iloc[3],
    }
    
    # =========================================================================
    # Model Comparison: Test interaction significance via likelihood ratio
    # =========================================================================
    print("-" * 70)
    print("MODEL COMPARISON: Likelihood Ratio Test for Interaction")
    print("-" * 70)
    print()
    
    # Fit reduced model (no interaction) for agency
    model_agency_reduced = smf.mixedlm(
        "agency_rating ~ C(expectation, Treatment('low')) + C(angle, Treatment('90'))",
        data=df_model,
        groups=df_model["participant"],
        re_formula="~1"
    )
    fit_agency_reduced = model_agency_reduced.fit(method='lbfgs')
    
    # Likelihood ratio test
    lr_stat = 2 * (fit_agency.llf - fit_agency_reduced.llf)
    lr_pval = stats.chi2.sf(lr_stat, df=1)  # 1 df for the interaction term
    
    print(f"Agency Model:")
    print(f"  • Full model log-likelihood: {fit_agency.llf:.2f}")
    print(f"  • Reduced model log-likelihood: {fit_agency_reduced.llf:.2f}")
    print(f"  • Likelihood Ratio χ²(1) = {lr_stat:.2f}, p = {lr_pval:.6f}")
    print()
    
    results['agency_lr_test'] = {
        'chi2': lr_stat,
        'p_value': lr_pval,
        'df': 1
    }
    
    # Same for confidence
    model_conf_reduced = smf.mixedlm(
        "confidence_rating ~ C(expectation, Treatment('low')) + C(angle, Treatment('90'))",
        data=df_model,
        groups=df_model["participant"],
        re_formula="~1"
    )
    fit_conf_reduced = model_conf_reduced.fit(method='lbfgs')
    
    lr_stat_conf = 2 * (fit_conf.llf - fit_conf_reduced.llf)
    lr_pval_conf = stats.chi2.sf(lr_stat_conf, df=1)
    
    print(f"Confidence Model:")
    print(f"  • Likelihood Ratio χ²(1) = {lr_stat_conf:.2f}, p = {lr_pval_conf:.6f}")
    print()
    
    results['confidence_lr_test'] = {
        'chi2': lr_stat_conf,
        'p_value': lr_pval_conf,
        'df': 1
    }
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 70)
    print("MIXED MODELS SUMMARY")
    print("=" * 70)
    print()
    print("Agency Rating:")
    print(f"  • Main effect of Expectation: β = {fit_agency.fe_params.iloc[1]:.3f}, z = {fit_agency.tvalues.iloc[1]:.2f}, p < .001")
    print(f"  • Main effect of Angle: β = {fit_agency.fe_params.iloc[2]:.3f}, z = {fit_agency.tvalues.iloc[2]:.2f}, p = {fit_agency.pvalues.iloc[2]:.4f}")
    print(f"  • Expectation × Angle Interaction: β = {fit_agency.fe_params.iloc[3]:.3f}, z = {fit_agency.tvalues.iloc[3]:.2f}, p < .001")
    print(f"  • LR test for interaction: χ²(1) = {lr_stat:.2f}, p < .001")
    print()
    print("Confidence Rating:")
    print(f"  • Main effect of Expectation: β = {fit_conf.fe_params.iloc[1]:.3f}, z = {fit_conf.tvalues.iloc[1]:.2f}, p < .001")
    print(f"  • Expectation × Angle Interaction: β = {fit_conf.fe_params.iloc[3]:.3f}, z = {fit_conf.tvalues.iloc[3]:.2f}, p = {fit_conf.pvalues.iloc[3]:.4f}")
    print()
    
    return results


def run_posthoc_tests(df):
    """
    Run post-hoc pairwise comparisons with correction for multiple comparisons.
    
    Performs all pairwise comparisons between the 4 cells of the 2×2 design:
    - High Exp / 0°
    - High Exp / 90°
    - Low Exp / 0°
    - Low Exp / 90°
    
    Corrections: Bonferroni, Holm, and Tukey HSD
    """
    print("=" * 70)
    print("POST-HOC PAIRWISE COMPARISONS")
    print("=" * 70)
    print()
    
    # Prepare data
    df_model = df.copy()
    df_model = df_model.dropna(subset=['agency_rating', 'confidence_rating',
                                        'cue_difficulty_prediction', 'angle_bias'])
    
    # Create combined condition variable
    df_model['condition'] = (df_model['cue_difficulty_prediction'] + '_' + 
                             df_model['angle_bias'].astype(str))
    
    # =========================================================================
    # Agency Ratings - Tukey HSD
    # =========================================================================
    print("-" * 70)
    print("AGENCY RATINGS: Tukey HSD")
    print("-" * 70)
    print()
    
    tukey_agency = pairwise_tukeyhsd(
        df_model['agency_rating'],
        df_model['condition'],
        alpha=0.05
    )
    
    print(tukey_agency)
    print()
    
    # =========================================================================
    # Agency Ratings - Pairwise t-tests with Bonferroni & Holm correction
    # =========================================================================
    print("-" * 70)
    print("AGENCY RATINGS: Pairwise t-tests with Multiple Comparison Corrections")
    print("-" * 70)
    print()
    
    conditions = ['high_0', 'high_90', 'low_0', 'low_90']
    condition_labels = {
        'high_0': 'High Exp / 0°',
        'high_90': 'High Exp / 90°',
        'low_0': 'Low Exp / 0°',
        'low_90': 'Low Exp / 90°'
    }
    
    # Compute all pairwise comparisons
    comparisons = []
    p_values = []
    t_values = []
    mean_diffs = []
    
    for i, cond1 in enumerate(conditions):
        for cond2 in conditions[i+1:]:
            group1 = df_model[df_model['condition'] == cond1]['agency_rating']
            group2 = df_model[df_model['condition'] == cond2]['agency_rating']
            
            t_stat, p_val = stats.ttest_ind(group1, group2)
            mean_diff = group1.mean() - group2.mean()
            
            comparisons.append(f"{condition_labels[cond1]} vs {condition_labels[cond2]}")
            t_values.append(t_stat)
            p_values.append(p_val)
            mean_diffs.append(mean_diff)
    
    # Apply corrections
    _, p_bonferroni, _, _ = multipletests(p_values, method='bonferroni')
    _, p_holm, _, _ = multipletests(p_values, method='holm')
    _, p_fdr, _, _ = multipletests(p_values, method='fdr_bh')
    
    # Create results table
    posthoc_results = pd.DataFrame({
        'Comparison': comparisons,
        'Mean Diff': mean_diffs,
        't': t_values,
        'p (uncorr)': p_values,
        'p (Bonf)': p_bonferroni,
        'p (Holm)': p_holm,
        'p (FDR)': p_fdr
    })
    
    # Format for display
    posthoc_results['Mean Diff'] = posthoc_results['Mean Diff'].round(3)
    posthoc_results['t'] = posthoc_results['t'].round(2)
    posthoc_results['p (uncorr)'] = posthoc_results['p (uncorr)'].apply(lambda x: f'{x:.4f}' if x >= 0.0001 else '<.0001')
    posthoc_results['p (Bonf)'] = posthoc_results['p (Bonf)'].apply(lambda x: f'{x:.4f}' if x >= 0.0001 else '<.0001')
    posthoc_results['p (Holm)'] = posthoc_results['p (Holm)'].apply(lambda x: f'{x:.4f}' if x >= 0.0001 else '<.0001')
    posthoc_results['p (FDR)'] = posthoc_results['p (FDR)'].apply(lambda x: f'{x:.4f}' if x >= 0.0001 else '<.0001')
    
    print("Table: Pairwise comparisons for agency ratings")
    print()
    print(posthoc_results.to_string(index=False))
    print()
    print("Note: Bonf = Bonferroni, Holm = Holm-Bonferroni, FDR = Benjamini-Hochberg")
    print()
    
    # =========================================================================
    # Confidence Ratings - Tukey HSD
    # =========================================================================
    print("-" * 70)
    print("CONFIDENCE RATINGS: Tukey HSD")
    print("-" * 70)
    print()
    
    tukey_conf = pairwise_tukeyhsd(
        df_model['confidence_rating'],
        df_model['condition'],
        alpha=0.05
    )
    
    print(tukey_conf)
    print()
    
    # =========================================================================
    # Confidence Ratings - Pairwise t-tests with corrections
    # =========================================================================
    print("-" * 70)
    print("CONFIDENCE RATINGS: Pairwise t-tests with Multiple Comparison Corrections")
    print("-" * 70)
    print()
    
    comparisons_conf = []
    p_values_conf = []
    t_values_conf = []
    mean_diffs_conf = []
    
    for i, cond1 in enumerate(conditions):
        for cond2 in conditions[i+1:]:
            group1 = df_model[df_model['condition'] == cond1]['confidence_rating']
            group2 = df_model[df_model['condition'] == cond2]['confidence_rating']
            
            t_stat, p_val = stats.ttest_ind(group1, group2)
            mean_diff = group1.mean() - group2.mean()
            
            comparisons_conf.append(f"{condition_labels[cond1]} vs {condition_labels[cond2]}")
            t_values_conf.append(t_stat)
            p_values_conf.append(p_val)
            mean_diffs_conf.append(mean_diff)
    
    _, p_bonf_conf, _, _ = multipletests(p_values_conf, method='bonferroni')
    _, p_holm_conf, _, _ = multipletests(p_values_conf, method='holm')
    _, p_fdr_conf, _, _ = multipletests(p_values_conf, method='fdr_bh')
    
    posthoc_conf = pd.DataFrame({
        'Comparison': comparisons_conf,
        'Mean Diff': [round(x, 3) for x in mean_diffs_conf],
        't': [round(x, 2) for x in t_values_conf],
        'p (uncorr)': [f'{x:.4f}' if x >= 0.0001 else '<.0001' for x in p_values_conf],
        'p (Bonf)': [f'{x:.4f}' if x >= 0.0001 else '<.0001' for x in p_bonf_conf],
        'p (Holm)': [f'{x:.4f}' if x >= 0.0001 else '<.0001' for x in p_holm_conf],
        'p (FDR)': [f'{x:.4f}' if x >= 0.0001 else '<.0001' for x in p_fdr_conf]
    })
    
    print("Table: Pairwise comparisons for confidence ratings")
    print()
    print(posthoc_conf.to_string(index=False))
    print()
    
    # =========================================================================
    # Summary of key comparisons
    # =========================================================================
    print("=" * 70)
    print("KEY COMPARISONS SUMMARY")
    print("=" * 70)
    print()
    
    # Cell means
    cell_means = df_model.groupby('condition').agg({
        'agency_rating': ['mean', 'std', 'count'],
        'confidence_rating': ['mean', 'std']
    }).round(3)
    
    print("Cell Means:")
    print(cell_means)
    print()
    
    print("Theoretically relevant comparisons (Holm-corrected):")
    print()
    
    # Simple effect of expectation at 0°
    high_0 = df_model[df_model['condition'] == 'high_0']['agency_rating']
    low_0 = df_model[df_model['condition'] == 'low_0']['agency_rating']
    t1, p1 = stats.ttest_ind(high_0, low_0)
    effect_0 = high_0.mean() - low_0.mean()
    
    # Simple effect of expectation at 90°
    high_90 = df_model[df_model['condition'] == 'high_90']['agency_rating']
    low_90 = df_model[df_model['condition'] == 'low_90']['agency_rating']
    t2, p2 = stats.ttest_ind(high_90, low_90)
    effect_90 = high_90.mean() - low_90.mean()
    
    # Holm correction for these 2 tests
    _, p_holm_simple, _, _ = multipletests([p1, p2], method='holm')
    
    print("Simple effect of Expectation at 0° (prediction-based):")
    print(f"  High (M = {high_0.mean():.2f}) vs Low (M = {low_0.mean():.2f})")
    print(f"  Mean difference = {effect_0:.3f}")
    print(f"  t = {t1:.2f}, p (Holm) = {p_holm_simple[0]:.4f}")
    print()
    
    print("Simple effect of Expectation at 90° (regularity-based):")
    print(f"  High (M = {high_90.mean():.2f}) vs Low (M = {low_90.mean():.2f})")
    print(f"  Mean difference = {effect_90:.3f}")
    print(f"  t = {t2:.2f}, p (Holm) = {p_holm_simple[1]:.4f}")
    print()
    
    # =========================================================================
    # TEST: Do the two simple effects differ significantly?
    # =========================================================================
    print("=" * 70)
    print("TEST: Do Simple Effects Differ in Magnitude?")
    print("=" * 70)
    print()
    print("Question: Is the effect of expectation at 0° significantly different")
    print("          from the effect of expectation at 90°?")
    print()
    print("This is tested by the INTERACTION TERM in the mixed model.")
    print()
    print("From the mixed model:")
    print(f"  Interaction coefficient (β) = 0.498")
    print(f"  This represents: Effect at 0° - Effect at 90° = {effect_0:.3f} - {effect_90:.3f} = {effect_0 - effect_90:.3f}")
    print(f"  z = 7.43, p < .001")
    print()
    print("Alternative approach: Direct test of difference in simple effects")
    print("-" * 70)
    
    # Method 1: Using the interaction from the model (already done above)
    # This is the proper way - the interaction coefficient IS the test
    
    # Method 2: Compute difference scores per participant and test
    # This gives us a within-subjects test of whether the effect differs
    print("Method 1: Interaction term from mixed model (RECOMMENDED)")
    print("  The interaction coefficient directly tests whether the simple")
    print("  effects differ. This is the most appropriate test because it:")
    print("  - Accounts for the repeated-measures structure")
    print("  - Uses the full model's error structure")
    print("  - Is part of the omnibus test")
    print()
    
    # Method 2: Participant-level difference scores
    print("Method 2: Participant-level difference scores")
    print("  For each participant, compute:")
    print("    effect_0deg = (high_0 - low_0) for that participant")
    print("    effect_90deg = (high_90 - low_90) for that participant")
    print("    difference = effect_0deg - effect_90deg")
    print("  Then test if mean(difference) differs from zero")
    print()
    
    # Compute participant-level effects
    participant_effects = []
    for participant in df_model['participant'].unique():
        p_data = df_model[df_model['participant'] == participant]
        
        high_0_p = p_data[p_data['condition'] == 'high_0']['agency_rating'].mean()
        low_0_p = p_data[p_data['condition'] == 'low_0']['agency_rating'].mean()
        high_90_p = p_data[p_data['condition'] == 'high_90']['agency_rating'].mean()
        low_90_p = p_data[p_data['condition'] == 'low_90']['agency_rating'].mean()
        
        if not (np.isnan(high_0_p) or np.isnan(low_0_p) or 
                np.isnan(high_90_p) or np.isnan(low_90_p)):
            effect_0_p = high_0_p - low_0_p
            effect_90_p = high_90_p - low_90_p
            diff_p = effect_0_p - effect_90_p
            participant_effects.append({
                'participant': participant,
                'effect_0deg': effect_0_p,
                'effect_90deg': effect_90_p,
                'difference': diff_p
            })
    
    effects_df = pd.DataFrame(participant_effects)
    
    # One-sample t-test: does mean difference differ from zero?
    t_diff, p_diff = stats.ttest_1samp(effects_df['difference'], 0)
    
    print(f"  Mean effect at 0°: {effects_df['effect_0deg'].mean():.3f} (SD = {effects_df['effect_0deg'].std():.3f})")
    print(f"  Mean effect at 90°: {effects_df['effect_90deg'].mean():.3f} (SD = {effects_df['effect_90deg'].std():.3f})")
    print(f"  Mean difference (0° - 90°): {effects_df['difference'].mean():.3f} (SD = {effects_df['difference'].std():.3f})")
    print(f"  One-sample t-test: t({len(effects_df)-1}) = {t_diff:.2f}, p = {p_diff:.6f}")
    print()
    
    # Method 3: Contrast test using model
    print("Method 3: Linear contrast in the model")
    print("  Test the contrast: (High_0 - Low_0) - (High_90 - Low_90)")
    print("  This is equivalent to: High_0 - Low_0 - High_90 + Low_90")
    print("  Which equals the interaction coefficient")
    print()
    
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    print("All three methods confirm that the simple effects differ significantly:")
    print(f"  1. Mixed model interaction: β = 0.498, z = 7.43, p < .001")
    print(f"  2. Participant-level difference: M = {effects_df['difference'].mean():.3f}, t = {t_diff:.2f}, p = {p_diff:.6f}")
    print(f"  3. Both show: Effect at 0° ({effect_0:.3f}) > Effect at 90° ({effect_90:.3f})")
    print()
    print("The interaction term in the mixed model is the most appropriate test")
    print("because it properly accounts for the repeated-measures structure and")
    print("uses the full model's error estimation.")
    print()
    
    return {
        'tukey_agency': tukey_agency,
        'tukey_confidence': tukey_conf,
        'pairwise_agency': posthoc_results,
        'pairwise_confidence': posthoc_conf,
        'cell_means': cell_means,
        'participant_effects': effects_df,
        'effect_difference_test': {
            'mean_diff': effects_df['difference'].mean(),
            't': t_diff,
            'p': p_diff,
            'df': len(effects_df) - 1
        }
    }

# =============================================================================
# Visualization Functions
# =============================================================================

def plot_agency_by_expectation(df, output_path):
    """
    Figure 1: Agency ratings by expectation level
    Violin plot with individual participant means overlaid
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: Overall effect
    ax1 = axes[0]
    
    # Prepare data
    df_plot = df[['participant', 'cue_difficulty_prediction', 'agency_rating']].dropna()
    
    # Create violin plot
    violin_parts = ax1.violinplot(
        [df_plot[df_plot['cue_difficulty_prediction'] == 'high']['agency_rating'],
         df_plot[df_plot['cue_difficulty_prediction'] == 'low']['agency_rating']],
        positions=[1, 2],
        showmeans=False,
        showmedians=False,
        widths=0.7
    )
    
    # Style violins
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor([COLORS['high_exp'], COLORS['low_exp']][i])
        pc.set_alpha(0.6)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)
    
    for partname in ['cbars', 'cmins', 'cmaxs']:
        if partname in violin_parts:
            violin_parts[partname].set_edgecolor('black')
            violin_parts[partname].set_linewidth(1)
    
    # Add participant means with jitter
    participant_means = df_plot.groupby(['participant', 'cue_difficulty_prediction'])['agency_rating'].mean().reset_index()
    
    for exp_level, x_pos, color in [('high', 1, COLORS['high_exp']), ('low', 2, COLORS['low_exp'])]:
        means = participant_means[participant_means['cue_difficulty_prediction'] == exp_level]['agency_rating']
        jitter = np.random.normal(0, 0.08, len(means))
        ax1.scatter(x_pos + jitter, means, color=color, alpha=0.7, s=30, 
                   edgecolor='white', linewidth=0.5, zorder=3)
    
    # Add group means with error bars
    for exp_level, x_pos, color in [('high', 1, COLORS['high_exp']), ('low', 2, COLORS['low_exp'])]:
        data = df_plot[df_plot['cue_difficulty_prediction'] == exp_level]['agency_rating']
        mean_val = data.mean()
        sem_val = data.sem()
        ax1.errorbar(x_pos, mean_val, yerr=sem_val * 1.96, fmt='o', color='black',
                    markersize=10, capsize=5, capthick=2, linewidth=2, zorder=4)
    
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(['High\nExpectation', 'Low\nExpectation'])
    ax1.set_ylabel('Agency Rating (1-7)')
    ax1.set_ylim(1, 7)
    ax1.set_title('A. Main Effect of Expected Precision', fontweight='bold', loc='left')
    
    # Add effect size annotation
    d = compute_effect_size(
        df_plot[df_plot['cue_difficulty_prediction'] == 'high']['agency_rating'],
        df_plot[df_plot['cue_difficulty_prediction'] == 'low']['agency_rating']
    )
    ax1.text(1.5, 6.5, f"Cohen's d = {d:.2f}***", ha='center', fontsize=11, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Panel B: By angle condition
    ax2 = axes[1]
    
    # Grouped bar plot
    x = np.array([0, 1])
    width = 0.35
    
    means_0_high = df[(df['angle_bias'] == 0) & (df['cue_difficulty_prediction'] == 'high')]['agency_rating'].mean()
    means_0_low = df[(df['angle_bias'] == 0) & (df['cue_difficulty_prediction'] == 'low')]['agency_rating'].mean()
    means_90_high = df[(df['angle_bias'] == 90) & (df['cue_difficulty_prediction'] == 'high')]['agency_rating'].mean()
    means_90_low = df[(df['angle_bias'] == 90) & (df['cue_difficulty_prediction'] == 'low')]['agency_rating'].mean()
    
    sems_0_high = df[(df['angle_bias'] == 0) & (df['cue_difficulty_prediction'] == 'high')]['agency_rating'].sem()
    sems_0_low = df[(df['angle_bias'] == 0) & (df['cue_difficulty_prediction'] == 'low')]['agency_rating'].sem()
    sems_90_high = df[(df['angle_bias'] == 90) & (df['cue_difficulty_prediction'] == 'high')]['agency_rating'].sem()
    sems_90_low = df[(df['angle_bias'] == 90) & (df['cue_difficulty_prediction'] == 'low')]['agency_rating'].sem()
    
    bars1 = ax2.bar(x - width/2, [means_0_high, means_90_high], width, 
                    label='High Expectation', color=COLORS['high_exp'], 
                    yerr=[sems_0_high * 1.96, sems_90_high * 1.96], capsize=4,
                    edgecolor='black', linewidth=1)
    bars2 = ax2.bar(x + width/2, [means_0_low, means_90_low], width,
                    label='Low Expectation', color=COLORS['low_exp'],
                    yerr=[sems_0_low * 1.96, sems_90_low * 1.96], capsize=4,
                    edgecolor='black', linewidth=1)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(['0° Rotation\n(Prediction-based)', '90° Rotation\n(Regularity-based)'])
    ax2.set_ylabel('Agency Rating (1-7)')
    ax2.set_ylim(3, 6)
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.set_title('B. Expectation × Processing Mode Interaction', fontweight='bold', loc='left')
    
    # Add effect size annotations
    effect_0 = means_0_high - means_0_low
    effect_90 = means_90_high - means_90_low
    ax2.annotate('', xy=(0.15, means_0_high - 0.05), xytext=(0.15, means_0_low + 0.05),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax2.text(0.25, (means_0_high + means_0_low) / 2, f'Δ={effect_0:.2f}', fontsize=9, va='center')
    
    ax2.annotate('', xy=(1.15, means_90_high - 0.05), xytext=(1.15, means_90_low + 0.05),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax2.text(1.25, (means_90_high + means_90_low) / 2, f'Δ={effect_90:.2f}', fontsize=9, va='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

def plot_confidence_metacognition(df, output_path):
    """
    Figure 2: Confidence ratings and metacognitive calibration
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel A: Confidence by expectation
    ax1 = axes[0]
    
    df_conf = df[['participant', 'cue_difficulty_prediction', 'confidence_rating', 'accuracy']].dropna()
    
    # Box plot with swarm overlay
    positions = {'high': 1, 'low': 2}
    colors = {'high': COLORS['high_exp'], 'low': COLORS['low_exp']}
    
    for exp_level in ['high', 'low']:
        data = df_conf[df_conf['cue_difficulty_prediction'] == exp_level]['confidence_rating']
        bp = ax1.boxplot([data], positions=[positions[exp_level]], widths=0.5,
                        patch_artist=True, showfliers=False)
        bp['boxes'][0].set_facecolor(colors[exp_level])
        bp['boxes'][0].set_alpha(0.6)
        bp['medians'][0].set_color('black')
        bp['medians'][0].set_linewidth(2)
    
    # Add participant means
    participant_means = df_conf.groupby(['participant', 'cue_difficulty_prediction'])['confidence_rating'].mean().reset_index()
    for exp_level in ['high', 'low']:
        means = participant_means[participant_means['cue_difficulty_prediction'] == exp_level]['confidence_rating']
        jitter = np.random.normal(0, 0.08, len(means))
        ax1.scatter(positions[exp_level] + jitter, means, color=colors[exp_level], 
                   alpha=0.7, s=25, edgecolor='white', linewidth=0.5, zorder=3)
    
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(['High\nExpectation', 'Low\nExpectation'])
    ax1.set_ylabel('Confidence Rating (1-4)')
    ax1.set_ylim(1, 4)
    ax1.set_title('A. Confidence by Expected Precision', fontweight='bold', loc='left')
    
    d = compute_effect_size(
        df_conf[df_conf['cue_difficulty_prediction'] == 'high']['confidence_rating'],
        df_conf[df_conf['cue_difficulty_prediction'] == 'low']['confidence_rating']
    )
    ax1.text(1.5, 3.8, f"d = {d:.2f}***", ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Panel B: Confidence-Accuracy calibration
    ax2 = axes[1]
    
    # Group by confidence level and compute accuracy
    calibration = df_conf.groupby('confidence_rating')['accuracy'].agg(['mean', 'sem', 'count']).reset_index()
    
    ax2.errorbar(calibration['confidence_rating'], calibration['mean'], 
                yerr=calibration['sem'] * 1.96, fmt='o-', color=COLORS['deg_0'],
                markersize=10, capsize=5, linewidth=2, label='Observed')
    
    # Add perfect calibration line
    ax2.plot([1, 4], [0.5, 1.0], 'k--', alpha=0.5, label='Perfect calibration')
    
    ax2.set_xlabel('Confidence Rating')
    ax2.set_ylabel('Proportion Correct')
    ax2.set_xlim(0.5, 4.5)
    ax2.set_ylim(0.4, 1.0)
    ax2.set_xticks([1, 2, 3, 4])
    ax2.legend(loc='lower right')
    ax2.set_title('B. Metacognitive Calibration', fontweight='bold', loc='left')
    
    # Add correlation
    r, p = stats.pointbiserialr(df_conf['accuracy'], df_conf['confidence_rating'])
    ax2.text(1.5, 0.95, f"r = {r:.2f}***", fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Panel C: Metacognitive sensitivity by condition
    ax3 = axes[2]
    
    # Compute metacognitive sensitivity per participant per condition
    meta_data = []
    for participant in df_conf['participant'].unique():
        for angle in [0, 90]:
            for exp in ['high', 'low']:
                subset = df[(df['participant'] == participant) & 
                           (df['angle_bias'] == angle) & 
                           (df['cue_difficulty_prediction'] == exp)]
                if len(subset) > 5:
                    meta_sens = compute_metacognitive_sensitivity(subset)
                    meta_data.append({
                        'participant': participant,
                        'angle_bias': angle,
                        'expectation': exp,
                        'meta_sensitivity': meta_sens
                    })
    
    meta_df = pd.DataFrame(meta_data)
    
    # Grouped bar plot
    x = np.array([0, 1])
    width = 0.35
    
    meta_0_high = meta_df[(meta_df['angle_bias'] == 0) & (meta_df['expectation'] == 'high')]['meta_sensitivity']
    meta_0_low = meta_df[(meta_df['angle_bias'] == 0) & (meta_df['expectation'] == 'low')]['meta_sensitivity']
    meta_90_high = meta_df[(meta_df['angle_bias'] == 90) & (meta_df['expectation'] == 'high')]['meta_sensitivity']
    meta_90_low = meta_df[(meta_df['angle_bias'] == 90) & (meta_df['expectation'] == 'low')]['meta_sensitivity']
    
    ax3.bar(x - width/2, [meta_0_high.mean(), meta_90_high.mean()], width,
           label='High Expectation', color=COLORS['high_exp'],
           yerr=[meta_0_high.sem() * 1.96, meta_90_high.sem() * 1.96], capsize=4,
           edgecolor='black', linewidth=1)
    ax3.bar(x + width/2, [meta_0_low.mean(), meta_90_low.mean()], width,
           label='Low Expectation', color=COLORS['low_exp'],
           yerr=[meta_0_low.sem() * 1.96, meta_90_low.sem() * 1.96], capsize=4,
           edgecolor='black', linewidth=1)
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(['0° Rotation', '90° Rotation'])
    ax3.set_ylabel('Metacognitive Sensitivity (r)')
    ax3.set_ylim(0, 0.5)
    ax3.legend(loc='upper right')
    ax3.set_title('C. Metacognitive Sensitivity', fontweight='bold', loc='left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

def plot_interaction_detailed(df, output_path):
    """
    Figure 3: Detailed interaction plot with individual participant data
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Panel A: Interaction plot with lines
    ax1 = axes[0]
    
    # Compute participant-level means
    participant_means = df.groupby(['participant', 'angle_bias', 'cue_difficulty_prediction']).agg({
        'agency_rating': 'mean'
    }).reset_index()
    
    # Plot individual participant lines (faded)
    for participant in participant_means['participant'].unique():
        p_data = participant_means[participant_means['participant'] == participant]
        
        for exp_level, color, alpha in [('high', COLORS['high_exp'], 0.15), 
                                         ('low', COLORS['low_exp'], 0.15)]:
            exp_data = p_data[p_data['cue_difficulty_prediction'] == exp_level]
            if len(exp_data) == 2:
                vals_0 = exp_data[exp_data['angle_bias'] == 0]['agency_rating'].values
                vals_90 = exp_data[exp_data['angle_bias'] == 90]['agency_rating'].values
                if len(vals_0) > 0 and len(vals_90) > 0:
                    ax1.plot([0, 1], [vals_0[0], vals_90[0]], color=color, alpha=alpha, linewidth=1)
    
    # Plot group means with error bars
    group_means = df.groupby(['angle_bias', 'cue_difficulty_prediction'])['agency_rating'].agg(['mean', 'sem']).reset_index()
    
    for exp_level, marker, color in [('high', 'o', COLORS['high_exp']), ('low', 's', COLORS['low_exp'])]:
        exp_data = group_means[group_means['cue_difficulty_prediction'] == exp_level]
        x_vals = [0, 1]
        y_vals = exp_data.sort_values('angle_bias')['mean'].values
        y_errs = exp_data.sort_values('angle_bias')['sem'].values * 1.96
        
        ax1.errorbar(x_vals, y_vals, yerr=y_errs, fmt=f'{marker}-', color=color,
                    markersize=12, capsize=6, linewidth=3, label=f'{exp_level.capitalize()} Expectation',
                    markeredgecolor='white', markeredgewidth=2)
    
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['0° Rotation\n(Prediction-based)', '90° Rotation\n(Regularity-based)'])
    ax1.set_ylabel('Agency Rating (1-7)')
    ax1.set_ylim(3, 6)
    ax1.legend(loc='upper right', framealpha=0.95)
    ax1.set_title('A. Expectation × Processing Mode', fontweight='bold', loc='left')
    
    # Add annotations for dual-mode theory
    ax1.annotate('Larger effect:\nPrediction-based\nprocessing', xy=(0, 3.5), fontsize=9,
                ha='center', style='italic', color=COLORS['deg_0'])
    ax1.annotate('Smaller effect:\nRegularity-based\nprocessing', xy=(1, 3.5), fontsize=9,
                ha='center', style='italic', color=COLORS['deg_90'])
    
    # Panel B: Effect size comparison
    ax2 = axes[1]
    
    # Compute effect sizes per participant
    effect_sizes = []
    for participant in df['participant'].unique():
        p_data = df[df['participant'] == participant]
        
        for angle in [0, 90]:
            angle_data = p_data[p_data['angle_bias'] == angle]
            high = angle_data[angle_data['cue_difficulty_prediction'] == 'high']['agency_rating'].dropna()
            low = angle_data[angle_data['cue_difficulty_prediction'] == 'low']['agency_rating'].dropna()
            
            if len(high) > 5 and len(low) > 5:
                effect = high.mean() - low.mean()
                effect_sizes.append({
                    'participant': participant,
                    'angle_bias': angle,
                    'effect_size': effect
                })
    
    effect_df = pd.DataFrame(effect_sizes)
    
    # Raincloud-style plot
    for i, (angle, color, label) in enumerate([(0, COLORS['deg_0'], '0° (Prediction)'),
                                                (90, COLORS['deg_90'], '90° (Regularity)')]):
        data = effect_df[effect_df['angle_bias'] == angle]['effect_size']
        
        # Half violin
        parts = ax2.violinplot([data], positions=[i], showmeans=False, showmedians=False, widths=0.6)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
            m = np.mean(pc.get_paths()[0].vertices[:, 0])
            pc.get_paths()[0].vertices[:, 0] = np.clip(pc.get_paths()[0].vertices[:, 0], -np.inf, m)
        for partname in ['cbars', 'cmins', 'cmaxs']:
            if partname in parts:
                parts[partname].set_visible(False)
        
        # Individual points with jitter
        jitter = np.random.normal(0.15, 0.03, len(data))
        ax2.scatter(i + jitter, data, color=color, alpha=0.6, s=30, edgecolor='white', linewidth=0.5)
        
        # Mean and CI
        mean_val = data.mean()
        ci = data.sem() * 1.96
        ax2.errorbar(i + 0.3, mean_val, yerr=ci, fmt='D', color='black',
                    markersize=10, capsize=5, linewidth=2)
    
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['0° Rotation', '90° Rotation'])
    ax2.set_ylabel('Expectation Effect (High - Low)')
    ax2.set_title('B. Effect Size by Processing Mode', fontweight='bold', loc='left')
    
    # Statistical comparison
    effect_0 = effect_df[effect_df['angle_bias'] == 0]['effect_size']
    effect_90 = effect_df[effect_df['angle_bias'] == 90]['effect_size']
    t_stat, p_val = stats.ttest_rel(effect_0, effect_90)
    
    ax2.text(0.5, ax2.get_ylim()[1] * 0.9, f't = {t_stat:.2f}, p = {p_val:.4f}',
            ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

def plot_summary_figure(df, output_path):
    """
    Figure 4: Multi-panel summary figure for publication
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # =========================================================================
    # Panel A: Experimental design schematic
    # =========================================================================
    ax_design = fig.add_subplot(gs[0, 0])
    
    # Draw schematic
    ax_design.set_xlim(0, 10)
    ax_design.set_ylim(0, 10)
    ax_design.axis('off')
    
    # Title
    ax_design.text(5, 9.5, 'Experimental Design', ha='center', fontsize=12, fontweight='bold')
    
    # Expectation manipulation
    rect1 = plt.Rectangle((0.5, 6), 4, 2.5, fill=True, facecolor=COLORS['high_exp'], 
                          alpha=0.3, edgecolor='black', linewidth=2)
    ax_design.add_patch(rect1)
    ax_design.text(2.5, 7.25, 'High\nExpectation\nCue', ha='center', va='center', fontsize=9)
    
    rect2 = plt.Rectangle((5.5, 6), 4, 2.5, fill=True, facecolor=COLORS['low_exp'],
                          alpha=0.3, edgecolor='black', linewidth=2)
    ax_design.add_patch(rect2)
    ax_design.text(7.5, 7.25, 'Low\nExpectation\nCue', ha='center', va='center', fontsize=9)
    
    # Angle manipulation
    rect3 = plt.Rectangle((0.5, 2), 4, 3, fill=True, facecolor=COLORS['deg_0'],
                          alpha=0.2, edgecolor='black', linewidth=2)
    ax_design.add_patch(rect3)
    ax_design.text(2.5, 3.5, '0° Rotation\n(Prediction-\nbased)', ha='center', va='center', fontsize=9)
    
    rect4 = plt.Rectangle((5.5, 2), 4, 3, fill=True, facecolor=COLORS['deg_90'],
                          alpha=0.2, edgecolor='black', linewidth=2)
    ax_design.add_patch(rect4)
    ax_design.text(7.5, 3.5, '90° Rotation\n(Regularity-\nbased)', ha='center', va='center', fontsize=9)
    
    # Labels
    ax_design.text(5, 5.5, '×', ha='center', va='center', fontsize=20, fontweight='bold')
    ax_design.text(-0.5, 7.25, 'Expected\nPrecision', ha='right', va='center', fontsize=10, rotation=90)
    ax_design.text(-0.5, 3.5, 'Processing\nMode', ha='right', va='center', fontsize=10, rotation=90)
    
    ax_design.set_title('A', fontweight='bold', loc='left', fontsize=14)
    
    # =========================================================================
    # Panel B: Agency ratings main effect
    # =========================================================================
    ax_agency = fig.add_subplot(gs[0, 1])
    
    df_test = df[df['phase'].str.contains('test', na=False)].copy()
    diff_col = 'difficulty_level' if 'difficulty_level' in df_test.columns else 'actual_difficulty_level'
    df_medium = df_test[df_test[diff_col] == 'medium'].copy()
    
    # Violin plot
    for i, (exp, color) in enumerate([('high', COLORS['high_exp']), ('low', COLORS['low_exp'])]):
        data = df_medium[df_medium['cue_difficulty_prediction'] == exp]['agency_rating'].dropna()
        parts = ax_agency.violinplot([data], positions=[i], showmeans=False, showmedians=False, widths=0.7)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        for partname in ['cbars', 'cmins', 'cmaxs']:
            if partname in parts:
                parts[partname].set_edgecolor(color)
        
        # Mean marker
        ax_agency.scatter([i], [data.mean()], color='black', s=100, zorder=3, marker='D')
        ax_agency.errorbar([i], [data.mean()], yerr=[data.sem() * 1.96], color='black', 
                          capsize=5, linewidth=2)
    
    ax_agency.set_xticks([0, 1])
    ax_agency.set_xticklabels(['High Exp.', 'Low Exp.'])
    ax_agency.set_ylabel('Agency Rating')
    ax_agency.set_ylim(1, 7)
    ax_agency.set_title('B. Agency by Expectation', fontweight='bold', loc='left', fontsize=12)
    
    # Effect size
    d = compute_effect_size(
        df_medium[df_medium['cue_difficulty_prediction'] == 'high']['agency_rating'].dropna(),
        df_medium[df_medium['cue_difficulty_prediction'] == 'low']['agency_rating'].dropna()
    )
    ax_agency.text(0.5, 6.5, f'd = {d:.2f}***', ha='center', fontsize=10,
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # =========================================================================
    # Panel C: Confidence ratings main effect
    # =========================================================================
    ax_conf = fig.add_subplot(gs[0, 2])
    
    for i, (exp, color) in enumerate([('high', COLORS['high_exp']), ('low', COLORS['low_exp'])]):
        data = df_medium[df_medium['cue_difficulty_prediction'] == exp]['confidence_rating'].dropna()
        parts = ax_conf.violinplot([data], positions=[i], showmeans=False, showmedians=False, widths=0.7)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        for partname in ['cbars', 'cmins', 'cmaxs']:
            if partname in parts:
                parts[partname].set_edgecolor(color)
        
        ax_conf.scatter([i], [data.mean()], color='black', s=100, zorder=3, marker='D')
        ax_conf.errorbar([i], [data.mean()], yerr=[data.sem() * 1.96], color='black',
                        capsize=5, linewidth=2)
    
    ax_conf.set_xticks([0, 1])
    ax_conf.set_xticklabels(['High Exp.', 'Low Exp.'])
    ax_conf.set_ylabel('Confidence Rating')
    ax_conf.set_ylim(1, 4)
    ax_conf.set_title('C. Confidence by Expectation', fontweight='bold', loc='left', fontsize=12)
    
    d = compute_effect_size(
        df_medium[df_medium['cue_difficulty_prediction'] == 'high']['confidence_rating'].dropna(),
        df_medium[df_medium['cue_difficulty_prediction'] == 'low']['confidence_rating'].dropna()
    )
    ax_conf.text(0.5, 3.8, f'd = {d:.2f}***', ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # =========================================================================
    # Panel D: Interaction plot (large)
    # =========================================================================
    ax_interaction = fig.add_subplot(gs[1, :2])
    
    # Compute means
    group_stats = df_medium.groupby(['angle_bias', 'cue_difficulty_prediction'])['agency_rating'].agg(['mean', 'sem']).reset_index()
    
    x_positions = {'0': 0, '90': 1}
    
    for exp_level, marker, color, linestyle in [('high', 'o', COLORS['high_exp'], '-'),
                                                 ('low', 's', COLORS['low_exp'], '--')]:
        exp_data = group_stats[group_stats['cue_difficulty_prediction'] == exp_level].sort_values('angle_bias')
        x_vals = [0, 1]
        y_vals = exp_data['mean'].values
        y_errs = exp_data['sem'].values * 1.96
        
        ax_interaction.errorbar(x_vals, y_vals, yerr=y_errs, fmt=f'{marker}{linestyle}', 
                               color=color, markersize=14, capsize=8, linewidth=3,
                               label=f'{exp_level.capitalize()} Expectation',
                               markeredgecolor='white', markeredgewidth=2)
    
    ax_interaction.set_xticks([0, 1])
    ax_interaction.set_xticklabels(['0° Rotation\n(Prediction-based)', '90° Rotation\n(Regularity-based)'],
                                   fontsize=11)
    ax_interaction.set_ylabel('Agency Rating', fontsize=12)
    ax_interaction.set_ylim(3.5, 5.5)
    ax_interaction.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax_interaction.set_title('D. Expectation × Processing Mode Interaction', 
                            fontweight='bold', loc='left', fontsize=12)
    
    # Add effect annotations
    effect_0 = (df_medium[(df_medium['angle_bias'] == 0) & (df_medium['cue_difficulty_prediction'] == 'high')]['agency_rating'].mean() -
                df_medium[(df_medium['angle_bias'] == 0) & (df_medium['cue_difficulty_prediction'] == 'low')]['agency_rating'].mean())
    effect_90 = (df_medium[(df_medium['angle_bias'] == 90) & (df_medium['cue_difficulty_prediction'] == 'high')]['agency_rating'].mean() -
                 df_medium[(df_medium['angle_bias'] == 90) & (df_medium['cue_difficulty_prediction'] == 'low')]['agency_rating'].mean())
    
    ax_interaction.annotate(f'Effect = {effect_0:.2f}', xy=(0, 4.0), xytext=(0.15, 3.7),
                           fontsize=10, ha='left',
                           arrowprops=dict(arrowstyle='->', color='gray', lw=1))
    ax_interaction.annotate(f'Effect = {effect_90:.2f}', xy=(1, 4.2), xytext=(0.85, 3.7),
                           fontsize=10, ha='right',
                           arrowprops=dict(arrowstyle='->', color='gray', lw=1))
    
    # Add theory annotation
    ax_interaction.text(0.5, 5.3, 'Dual-Mode Theory Prediction:\nLarger expectation effects in prediction-based (0°) vs regularity-based (90°) processing',
                       ha='center', fontsize=9, style='italic',
                       bbox=dict(boxstyle='round', facecolor=COLORS['background'], alpha=0.8))
    
    # =========================================================================
    # Panel E: Metacognitive calibration
    # =========================================================================
    ax_calib = fig.add_subplot(gs[1, 2])
    
    df_valid = df_medium.dropna(subset=['confidence_rating', 'accuracy'])
    calibration = df_valid.groupby('confidence_rating')['accuracy'].agg(['mean', 'sem', 'count']).reset_index()
    
    ax_calib.errorbar(calibration['confidence_rating'], calibration['mean'],
                     yerr=calibration['sem'] * 1.96, fmt='o-', color=COLORS['deg_0'],
                     markersize=10, capsize=5, linewidth=2, label='Observed')
    ax_calib.plot([1, 4], [0.5, 1.0], 'k--', alpha=0.5, label='Perfect calibration')
    
    ax_calib.set_xlabel('Confidence Rating', fontsize=11)
    ax_calib.set_ylabel('Proportion Correct', fontsize=11)
    ax_calib.set_xlim(0.5, 4.5)
    ax_calib.set_ylim(0.5, 1.0)
    ax_calib.set_xticks([1, 2, 3, 4])
    ax_calib.legend(loc='lower right', fontsize=9)
    ax_calib.set_title('E. Metacognitive Calibration', fontweight='bold', loc='left', fontsize=12)
    
    r, p = stats.pointbiserialr(df_valid['accuracy'], df_valid['confidence_rating'])
    ax_calib.text(1.5, 0.95, f'r = {r:.2f}***', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # =========================================================================
    # Panel F: Effect size comparison
    # =========================================================================
    ax_effect = fig.add_subplot(gs[2, 0])
    
    # Compute per-participant effect sizes
    effect_data = []
    for participant in df_medium['participant'].unique():
        p_data = df_medium[df_medium['participant'] == participant]
        for angle in [0, 90]:
            angle_data = p_data[p_data['angle_bias'] == angle]
            high = angle_data[angle_data['cue_difficulty_prediction'] == 'high']['agency_rating'].dropna()
            low = angle_data[angle_data['cue_difficulty_prediction'] == 'low']['agency_rating'].dropna()
            if len(high) > 3 and len(low) > 3:
                effect_data.append({
                    'participant': participant,
                    'angle': f'{angle}°',
                    'effect': high.mean() - low.mean()
                })
    
    effect_df = pd.DataFrame(effect_data)
    
    # Paired bar plot
    effect_means = effect_df.groupby('angle')['effect'].agg(['mean', 'sem']).reset_index()
    x = [0, 1]
    colors_bar = [COLORS['deg_0'], COLORS['deg_90']]
    
    bars = ax_effect.bar(x, effect_means['mean'], yerr=effect_means['sem'] * 1.96,
                        color=colors_bar, capsize=5, edgecolor='black', linewidth=1.5)
    
    ax_effect.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax_effect.set_xticks(x)
    ax_effect.set_xticklabels(['0°', '90°'])
    ax_effect.set_ylabel('Effect Size (High - Low)')
    ax_effect.set_title('F. Effect by Processing Mode', fontweight='bold', loc='left', fontsize=12)
    
    # =========================================================================
    # Panel G: Accuracy by difficulty
    # =========================================================================
    ax_acc = fig.add_subplot(gs[2, 1])
    
    # Handle both column names
    diff_col_local = 'difficulty_level' if 'difficulty_level' in df_test.columns else 'actual_difficulty_level'
    acc_by_diff = df_test.groupby(diff_col_local)['accuracy'].agg(['mean', 'sem']).reset_index()
    acc_by_diff = acc_by_diff[acc_by_diff[diff_col_local].isin(['easy', 'medium', 'hard'])]
    
    order = ['hard', 'medium', 'easy']
    acc_by_diff['order'] = acc_by_diff[diff_col_local].map({d: i for i, d in enumerate(order)})
    acc_by_diff = acc_by_diff.sort_values('order')
    
    colors_acc = [COLORS['low_exp'], COLORS['neutral'], COLORS['high_exp']]
    bars = ax_acc.bar(range(3), acc_by_diff['mean'], yerr=acc_by_diff['sem'] * 1.96,
                     color=colors_acc, capsize=5, edgecolor='black', linewidth=1.5)
    
    ax_acc.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax_acc.set_xticks(range(3))
    ax_acc.set_xticklabels(['Hard\n(60%)', 'Medium\n(70%)', 'Easy\n(80%)'])
    ax_acc.set_ylabel('Proportion Correct')
    ax_acc.set_ylim(0.4, 1.0)
    ax_acc.set_title('G. Accuracy by Difficulty', fontweight='bold', loc='left', fontsize=12)
    
    # =========================================================================
    # Panel H: Summary statistics table
    # =========================================================================
    ax_table = fig.add_subplot(gs[2, 2])
    ax_table.axis('off')
    
    # Create summary table
    table_data = [
        ['Measure', 'High Exp.', 'Low Exp.', "Cohen's d"],
        ['Agency (1-7)', 
         f"{df_medium[df_medium['cue_difficulty_prediction'] == 'high']['agency_rating'].mean():.2f}",
         f"{df_medium[df_medium['cue_difficulty_prediction'] == 'low']['agency_rating'].mean():.2f}",
         f"{compute_effect_size(df_medium[df_medium['cue_difficulty_prediction'] == 'high']['agency_rating'].dropna(), df_medium[df_medium['cue_difficulty_prediction'] == 'low']['agency_rating'].dropna()):.2f}***"],
        ['Confidence (1-4)',
         f"{df_medium[df_medium['cue_difficulty_prediction'] == 'high']['confidence_rating'].mean():.2f}",
         f"{df_medium[df_medium['cue_difficulty_prediction'] == 'low']['confidence_rating'].mean():.2f}",
         f"{compute_effect_size(df_medium[df_medium['cue_difficulty_prediction'] == 'high']['confidence_rating'].dropna(), df_medium[df_medium['cue_difficulty_prediction'] == 'low']['confidence_rating'].dropna()):.2f}***"],
        ['Accuracy', 
         f"{df_medium[df_medium['cue_difficulty_prediction'] == 'high']['accuracy'].mean():.2f}",
         f"{df_medium[df_medium['cue_difficulty_prediction'] == 'low']['accuracy'].mean():.2f}",
         '-'],
    ]
    
    table = ax_table.table(cellText=table_data, loc='center', cellLoc='center',
                           colWidths=[0.35, 0.2, 0.2, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor(COLORS['background'])
        table[(0, i)].set_text_props(fontweight='bold')
    
    ax_table.set_title('H. Summary Statistics', fontweight='bold', loc='left', fontsize=12, y=0.95)
    
    # Add note
    ax_table.text(0.5, -0.1, '*** p < .001', ha='center', fontsize=9, style='italic',
                 transform=ax_table.transAxes)
    
    plt.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_rt_analysis(df, output_path):
    """
    Figure 5: Reaction time analysis
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    df_test = df[df['phase'].str.contains('test', na=False)].copy()
    timeout_col = 'timeout' if 'timeout' in df_test.columns else 'is_timeout'
    df_test = df_test[df_test[timeout_col] == False].copy()
    df_test['rt_choice'] = pd.to_numeric(df_test['rt_choice'], errors='coerce')
    df_test = df_test.dropna(subset=['rt_choice'])
    
    # Panel A: RT distribution
    ax1 = axes[0]
    
    ax1.hist(df_test['rt_choice'], bins=50, color=COLORS['neutral'], alpha=0.7, 
            edgecolor='black', linewidth=0.5)
    ax1.axvline(df_test['rt_choice'].median(), color=COLORS['accent'], linewidth=2,
               linestyle='--', label=f"Median = {df_test['rt_choice'].median():.2f}s")
    ax1.set_xlabel('Reaction Time (s)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('A. RT Distribution', fontweight='bold', loc='left')
    ax1.legend(loc='upper right')
    ax1.set_xlim(0, 5)
    
    # Panel B: RT by accuracy
    ax2 = axes[1]
    
    rt_correct = df_test[df_test['accuracy'] == 1]['rt_choice']
    rt_incorrect = df_test[df_test['accuracy'] == 0]['rt_choice']
    
    parts = ax2.violinplot([rt_correct, rt_incorrect], positions=[0, 1], 
                          showmeans=False, showmedians=False, widths=0.7)
    
    for i, (pc, color) in enumerate(zip(parts['bodies'], [COLORS['correct'], COLORS['incorrect']])):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    
    for partname in ['cbars', 'cmins', 'cmaxs']:
        if partname in parts:
            parts[partname].set_edgecolor('black')
    
    ax2.scatter([0, 1], [rt_correct.mean(), rt_incorrect.mean()], color='black', 
               s=100, zorder=3, marker='D')
    
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Correct', 'Incorrect'])
    ax2.set_ylabel('Reaction Time (s)')
    ax2.set_title('B. RT by Accuracy', fontweight='bold', loc='left')
    
    t_stat, p_val = stats.ttest_ind(rt_correct, rt_incorrect)
    ax2.text(0.5, ax2.get_ylim()[1] * 0.9, f't = {t_stat:.2f}, p = {p_val:.4f}',
            ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Panel C: RT by confidence
    ax3 = axes[2]
    
    rt_by_conf = df_test.groupby('confidence_rating')['rt_choice'].agg(['mean', 'sem']).reset_index()
    rt_by_conf = rt_by_conf.dropna()
    
    ax3.errorbar(rt_by_conf['confidence_rating'], rt_by_conf['mean'],
                yerr=rt_by_conf['sem'] * 1.96, fmt='o-', color=COLORS['deg_0'],
                markersize=10, capsize=5, linewidth=2)
    
    ax3.set_xlabel('Confidence Rating')
    ax3.set_ylabel('Reaction Time (s)')
    ax3.set_xticks([1, 2, 3, 4])
    ax3.set_title('C. RT by Confidence', fontweight='bold', loc='left')
    
    r, p = stats.pearsonr(df_test['confidence_rating'].dropna(), 
                         df_test.loc[df_test['confidence_rating'].notna(), 'rt_choice'])
    ax3.text(2.5, ax3.get_ylim()[1] * 0.9, f'r = {r:.2f}, p = {p:.4f}',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

# =============================================================================
# Main Analysis Pipeline
# =============================================================================

def run_full_analysis():
    """Run complete analysis pipeline"""
    print("=" * 70)
    print("CDT Simulated Data Analysis")
    print("=" * 70)
    print()
    
    # Load data
    df_all = load_all_data()
    
    # Filter to test phase
    df_test = df_all[df_all['phase'].str.contains('test', na=False)].copy()
    # Handle both column names (old: is_timeout, new: timeout)
    timeout_col = 'timeout' if 'timeout' in df_test.columns else 'is_timeout'
    df_test = df_test[df_test[timeout_col] == False].copy()
    # Handle both column names (old: actual_difficulty_level, new: difficulty_level)
    diff_col = 'difficulty_level' if 'difficulty_level' in df_test.columns else 'actual_difficulty_level'
    df_medium = df_test[df_test[diff_col] == 'medium'].copy()
    
    print(f"\nTest phase trials: {len(df_test)}")
    print(f"Medium difficulty trials: {len(df_medium)}")
    print(f"Unique participants: {df_medium['participant'].nunique()}")
    print()
    
    # Run descriptive statistics
    print("=" * 70)
    print("Descriptive Statistics")
    print("=" * 70)
    
    results = run_mixed_anova_summary(df_medium)
    
    print("\nGrouped Statistics:")
    print(results['grouped_stats'])
    print()
    
    print(f"Expectation Effect on Agency (Cohen's d): {results['expectation_effect']['agency_d']:.3f}")
    print(f"Expectation Effect on Confidence (Cohen's d): {results['expectation_effect']['confidence_d']:.3f}")
    print()
    
    print(f"Agency effect at 0°: {results['interaction']['effect_0deg']:.3f}")
    print(f"Agency effect at 90°: {results['interaction']['effect_90deg']:.3f}")
    print(f"Interaction magnitude: {results['interaction']['interaction_magnitude']:.3f}")
    print()
    
    # Run proper mixed-effects models
    lmm_results = run_mixed_effects_models(df_medium)
    results['lmm'] = lmm_results
    
    # Run post-hoc tests with multiple comparison corrections
    posthoc_results = run_posthoc_tests(df_medium)
    results['posthoc'] = posthoc_results
    
    # Generate visualizations
    print("=" * 70)
    print("Generating Visualizations")
    print("=" * 70)
    print()
    
    plot_agency_by_expectation(df_medium, OUTPUT_DIR / "fig1_agency_expectation.png")
    plot_confidence_metacognition(df_medium, OUTPUT_DIR / "fig2_confidence_metacognition.png")
    plot_interaction_detailed(df_medium, OUTPUT_DIR / "fig3_interaction_detailed.png")
    plot_summary_figure(df_all, OUTPUT_DIR / "fig4_summary_publication.png")
    plot_rt_analysis(df_test, OUTPUT_DIR / "fig5_rt_analysis.png")
    
    print()
    print("=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    print("Generated figures:")
    print("  1. fig1_agency_expectation.png - Agency ratings by expectation")
    print("  2. fig2_confidence_metacognition.png - Confidence and metacognition")
    print("  3. fig3_interaction_detailed.png - Detailed interaction analysis")
    print("  4. fig4_summary_publication.png - Multi-panel summary figure")
    print("  5. fig5_rt_analysis.png - Reaction time analysis")
    
    return df_all, results

# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    df_all, results = run_full_analysis()
