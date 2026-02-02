#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explain_model_1.py - Explanation of Model 1 (Random Intercept Only)

This script explains what Model 1 means in the context of mixed models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# What is Model 1?
# =============================================================================

print("=" * 80)
print("WHAT IS MODEL 1?")
print("=" * 80)
print()
print("Model 1 is the BASELINE mixed model with:")
print("  • Fixed effects: expectation * angle (the main effects and interaction)")
print("  • Random effects: (1|sub) - ONLY a random intercept per participant")
print()
print("Formula: agency_rating ~ expectation * angle + (1|sub)")
print()
print("The '(1|sub)' part means:")
print("  - Each participant (sub) gets their own intercept")
print("  - But all participants share the SAME slopes for expectation and angle")
print()

# =============================================================================
# Visual Explanation
# =============================================================================

print("=" * 80)
print("VISUAL EXPLANATION")
print("=" * 80)
print()
print("Imagine plotting agency ratings for each participant:")
print()
print("Model 1 (Random Intercept Only):")
print("  ┌─────────────────────────────────────────┐")
print("  │ Participant 1:  y = 4.0 + β₁*exp + ... │")
print("  │ Participant 2:  y = 3.5 + β₁*exp + ... │  ← Different intercepts")
print("  │ Participant 3:  y = 4.2 + β₁*exp + ... │  ← (random intercept)")
print("  │ Participant 4:  y = 3.8 + β₁*exp + ... │")
print("  │ ...                                      │")
print("  │                                          │")
print("  │ BUT: All participants have the SAME β₁  │  ← Same slopes")
print("  │      (same effect of expectation)        │")
print("  └─────────────────────────────────────────┘")
print()
print("Model 2 (Random Intercept + Random Slope for Expectation):")
print("  ┌─────────────────────────────────────────┐")
print("  │ Participant 1:  y = 4.0 + 0.3*exp + ...│")
print("  │ Participant 2:  y = 3.5 + 0.5*exp + ...│  ← Different intercepts")
print("  │ Participant 3:  y = 4.2 + 0.2*exp + ... │  ← AND different slopes")
print("  │ Participant 4:  y = 3.8 + 0.4*exp + ... │  ← (random slope)")
print("  │ ...                                      │")
print("  │                                          │")
print("  │ Each participant has their OWN β₁       │  ← Individual differences")
print("  │ (different effect of expectation)       │")
print("  └─────────────────────────────────────────┘")
print()

# =============================================================================
# Mathematical Explanation
# =============================================================================

print("=" * 80)
print("MATHEMATICAL EXPLANATION")
print("=" * 80)
print()
print("Model 1 (Random Intercept Only):")
print()
print("  agency_ij = β₀ + u₀ᵢ + β₁*expectation_ij + β₂*angle_ij + β₃*interaction_ij + ε_ij")
print()
print("  Where:")
print("    • agency_ij = agency rating for participant i, trial j")
print("    • β₀ = fixed intercept (average across all participants)")
print("    • u₀ᵢ = random intercept for participant i (deviation from β₀)")
print("           ~ N(0, σ²ᵤ)  ← This varies across participants")
print("    • β₁, β₂, β₃ = fixed slopes (SAME for all participants)")
print("    • ε_ij = residual error ~ N(0, σ²)")
print()
print("Key point: The slopes (β₁, β₂, β₃) are FIXED - they don't vary")
print("           across participants. Only the intercept varies.")
print()
print("Model 2 (Random Intercept + Random Slope):")
print()
print("  agency_ij = β₀ + u₀ᵢ + (β₁ + u₁ᵢ)*expectation_ij + β₂*angle_ij + ... + ε_ij")
print()
print("  Where:")
print("    • u₁ᵢ = random slope for participant i (deviation from β₁)")
print("           ~ N(0, σ²ᵤ₁)  ← This also varies across participants")
print()
print("Key point: Now BOTH the intercept AND the expectation slope vary")
print("           across participants.")
print()

# =============================================================================
# Why Start with Model 1?
# =============================================================================

print("=" * 80)
print("WHY START WITH MODEL 1?")
print("=" * 80)
print()
print("Following the DesenderLab guide, we use a 'model building' approach:")
print()
print("1. Start simple (Model 1): Random intercept only")
print("   → This is the most parsimonious model")
print("   → Assumes all participants respond similarly to the experimental")
print("     manipulations (same slopes)")
print()
print("2. Test if adding random slopes improves fit:")
print("   → Model 2: Add random slope for expectation")
print("   → Model 3: Add random slope for angle")
print()
print("3. Compare models using:")
print("   → BIC (Bayesian Information Criterion) - lower is better")
print("   → Likelihood Ratio Test - tests if improvement is significant")
print()
print("4. Choose the best model:")
print("   → In our case, Model 2 had the lowest BIC (19,058 vs 19,065)")
print("   → The LR test showed Model 2 fits significantly better")
print("   → This means: participants DO differ in how expectation affects agency")
print()

# =============================================================================
# What Does This Mean for Your Data?
# =============================================================================

print("=" * 80)
print("WHAT DOES THIS MEAN FOR YOUR DATA?")
print("=" * 80)
print()
print("Model 1 (Random Intercept Only):")
print("  • Assumes: All 30 participants have the SAME effect of expectation")
print("  • Example: If expectation increases agency by 0.5 points on average,")
print("            ALL participants show this same 0.5-point increase")
print("  • Individual differences: Only in baseline agency (some participants")
print("            rate agency higher overall, but the effect is the same)")
print()
print("Model 2 (Random Slope for Expectation) - BEST MODEL:")
print("  • Assumes: Participants DIFFER in how expectation affects agency")
print("  • Example: Some participants might show a 0.8-point increase,")
print("            others might show only a 0.2-point increase")
print("  • Individual differences: Both in baseline AND in the effect size")
print()
print("Why Model 2 is better:")
print("  • BIC = 19,058 (vs 19,065 for Model 1) - lower is better")
print("  • LR test: χ²(2) = 24.22, p < .001 - significant improvement")
print("  • This means: There IS meaningful individual variation in how")
print("    participants respond to the expectation manipulation")
print()
print("Interpretation:")
print("  • The fixed effect (β = 0.388) is the AVERAGE effect across participants")
print("  • The random slope variance tells us how much participants vary")
print("  • Some participants are more sensitive to expectation cues than others")
print()

# =============================================================================
# Create a Visual Diagram
# =============================================================================

print("=" * 80)
print("CREATING VISUAL DIAGRAM")
print("=" * 80)
print()

# Simulate some example data to illustrate
np.random.seed(42)

# Model 1: Same slope, different intercepts
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left panel: Model 1 (Random Intercept Only)
ax1 = axes[0]

# Simulate 5 participants with different intercepts but same slope
intercepts = [3.5, 4.0, 3.8, 4.2, 3.6]
slope = 0.5  # Same for all

x = np.array([0, 1])  # Low expectation = 0, High = 1

for i, intercept in enumerate(intercepts):
    y = intercept + slope * x
    ax1.plot(x, y, 'o-', alpha=0.6, linewidth=2, label=f'Participant {i+1}' if i < 3 else '')

ax1.set_xlabel('Expectation (0=Low, 1=High)', fontsize=11)
ax1.set_ylabel('Agency Rating', fontsize=11)
ax1.set_title('Model 1: Random Intercept Only\n(Same slope, different intercepts)', 
              fontweight='bold', fontsize=12)
ax1.set_xticks([0, 1])
ax1.set_xticklabels(['Low', 'High'])
ax1.set_ylim(3.0, 5.5)
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)

# Add annotation
ax1.text(0.5, 3.2, 'All participants have\nthe SAME slope (0.5)', 
         ha='center', fontsize=10, 
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

# Right panel: Model 2 (Random Intercept + Random Slope)
ax2 = axes[1]

# Simulate 5 participants with different intercepts AND different slopes
intercepts_2 = [3.5, 4.0, 3.8, 4.2, 3.6]
slopes = [0.3, 0.7, 0.5, 0.6, 0.4]  # Different for each participant

for i, (intercept, slope) in enumerate(zip(intercepts_2, slopes)):
    y = intercept + slope * x
    ax2.plot(x, y, 'o-', alpha=0.6, linewidth=2, label=f'Participant {i+1}' if i < 3 else '')

ax2.set_xlabel('Expectation (0=Low, 1=High)', fontsize=11)
ax2.set_ylabel('Agency Rating', fontsize=11)
ax2.set_title('Model 2: Random Intercept + Random Slope\n(Different slopes for each participant)', 
              fontweight='bold', fontsize=12)
ax2.set_xticks([0, 1])
ax2.set_xticklabels(['Low', 'High'])
ax2.set_ylim(3.0, 5.5)
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)

# Add annotation
ax2.text(0.5, 3.2, 'Each participant has\na DIFFERENT slope', 
         ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.tight_layout()

output_path = Path(__file__).parent.parent / "Main_Experiment" / "data" / "analysis_output" / "fig_model_1_explanation.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, facecolor='white')
plt.close()

print(f"Visual diagram saved to: {output_path}")
print()

# =============================================================================
# Summary
# =============================================================================

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()
print("Model 1 = Random Intercept Only")
print("  • Formula: agency ~ expectation * angle + (1|sub)")
print("  • Meaning: Each participant has their own baseline (intercept),")
print("            but all participants respond the same way to the")
print("            experimental manipulations (same slopes)")
print()
print("Model 2 = Random Intercept + Random Slope for Expectation")
print("  • Formula: agency ~ expectation * angle + (1 + expectation|sub)")
print("  • Meaning: Participants differ in BOTH baseline AND in how")
print("            expectation affects their agency ratings")
print()
print("Your Results:")
print("  • Model 2 fits better (BIC = 19,058 vs 19,065)")
print("  • This means: There is meaningful individual variation in")
print("    how participants respond to expectation cues")
print("  • The fixed effect (β = 0.388) is the AVERAGE effect")
print("  • Individual participants vary around this average")
print()
