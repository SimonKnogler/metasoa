"""
Evidence-Accuracy Correlation Analysis
========================================

Visualizes the relationship between mean pre-response evidence 
and trial accuracy across all experimental phases.

Uses mean_evidence_preRT (not cumulative) because participants who respond
quickly have fewer frames, so cumulative evidence would be artificially low
even if they had strong evidence per frame.

Author: CDT Analysis Pipeline
Date: 2024
"""

# Check for required packages and provide helpful error message
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import pointbiserialr
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    import statsmodels.api as sm
    from pathlib import Path
    import glob
    import warnings
    warnings.filterwarnings('ignore')
except ImportError as e:
    print("\n" + "="*60)
    print("ERROR: Missing required Python packages")
    print("="*60)
    print(f"\nMissing module: {e.name}")
    print("\nTo install all required packages, run:")
    print("  pip install pandas numpy matplotlib scipy scikit-learn statsmodels")
    print("\nOr install from requirements.txt:")
    print("  pip install -r requirements.txt")
    print("="*60 + "\n")
    raise

# ========================================
# Configuration
# ========================================
DATA_DIR = Path(__file__).parent.parent / "Main_Experiment" / "data" / "subjects"
OUTPUT_DIR = Path(__file__).parent.parent / "Main_Experiment" / "data" / "quest_group_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ========================================
# Load and prepare data
# ========================================
print("Loading data files...")
all_files = sorted(glob.glob(str(DATA_DIR / "CDT_v2_blockwise_fast_response_*.csv")))
# Exclude kinematics files
data_files = [f for f in all_files if 'kinematics' not in f.lower()]
print(f"Found {len(data_files)} data files (excluding kinematics)")

if len(data_files) == 0:
    print("ERROR: No data files found!")
    exit(1)

# Load and combine all data
df_list = []
for file in data_files:
    try:
        df_temp = pd.read_csv(file)
        df_list.append(df_temp)
        print(f"  Loaded: {Path(file).name} ({len(df_temp)} trials)")
    except Exception as e:
        print(f"  Warning: Could not load {file}: {e}")

df = pd.concat(df_list, ignore_index=True)
print(f"\nTotal trials loaded: {len(df)}")

# ========================================
# Filter data
# ========================================
print("\nFiltering data...")
print(f"  Total trials: {len(df)}")

# Exclude timeout trials
df_no_timeout = df[df['is_timeout'] == False].copy()
print(f"  After removing timeouts: {len(df_no_timeout)}")

# Keep only trials with valid mean evidence
df_clean = df_no_timeout[df_no_timeout['mean_evidence_preRT'].notna()].copy()
print(f"  After removing NaN mean evidence: {len(df_clean)}")

# Keep only trials with valid accuracy (not NaN)
df_clean = df_clean[df_clean['accuracy'].notna()].copy()
print(f"  After removing NaN accuracy: {len(df_clean)}")

print(f"\nPhase distribution:")
for phase in df_clean['phase'].unique():
    count = len(df_clean[df_clean['phase'] == phase])
    print(f"  {phase}: {count} trials")

# ========================================
# Calculate statistics
# ========================================
print("\n" + "="*60)
print("STATISTICAL ANALYSIS")
print("="*60)

X = df_clean['mean_evidence_preRT'].values.reshape(-1, 1)
y = df_clean['accuracy'].values

# Point-biserial correlation
corr, pval = pointbiserialr(df_clean['mean_evidence_preRT'], df_clean['accuracy'])
print(f"\nPoint-Biserial Correlation:")
print(f"  r = {corr:.4f}")
print(f"  p-value = {pval:.4e}")

# Logistic regression with sklearn (for prediction)
print(f"\nLogistic Regression (sklearn):")
model = LogisticRegression(max_iter=1000)
model.fit(X, y)
print(f"  Intercept: {model.intercept_[0]:.4f}")
print(f"  Coefficient (evidence): {model.coef_[0][0]:.6f}")

# AUC
y_pred_proba = model.predict_proba(X)[:, 1]
auc = roc_auc_score(y, y_pred_proba)
print(f"  AUC: {auc:.4f}")

# Logistic regression with statsmodels (for p-value)
print(f"\nLogistic Regression (statsmodels - with p-value):")
X_with_const = sm.add_constant(X)
logit_model = sm.Logit(y, X_with_const)
logit_result = logit_model.fit(disp=0)  # disp=0 suppresses convergence messages
print(f"  Intercept: {logit_result.params[0]:.4f} (p = {logit_result.pvalues[0]:.4e})")
print(f"  Coefficient (evidence): {logit_result.params[1]:.6f} (p = {logit_result.pvalues[1]:.4e})")
print(f"  95% CI for coefficient: [{logit_result.conf_int()[1][0]:.6f}, {logit_result.conf_int()[1][1]:.6f}]")

# Store p-value for later use
logit_pvalue = logit_result.pvalues[1]
logit_coef = logit_result.params[1]

# Additional descriptive stats
print(f"\nDescriptive Statistics:")
print(f"  Mean evidence per frame (correct): {df_clean[df_clean['accuracy']==1]['mean_evidence_preRT'].mean():.2f}")
print(f"  Mean evidence per frame (incorrect): {df_clean[df_clean['accuracy']==0]['mean_evidence_preRT'].mean():.2f}")
print(f"  Overall accuracy: {df_clean['accuracy'].mean():.2%}")

# ========================================
# Create visualization
# ========================================
print("\nCreating visualization...")

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Logistic regression curve with binned data
print("  Creating logistic regression plot...")

# Create smooth prediction curve
evidence_range = np.linspace(df_clean['mean_evidence_preRT'].min(), 
                             df_clean['mean_evidence_preRT'].max(), 
                             300).reshape(-1, 1)
prob_predictions = model.predict_proba(evidence_range)[:, 1]

ax.plot(evidence_range, prob_predictions, 'b-', linewidth=3, 
         label='Logistic Regression', zorder=3)

# Add binned accuracy
n_bins = 10
df_clean['evidence_bin'] = pd.qcut(df_clean['mean_evidence_preRT'], 
                                     q=n_bins, duplicates='drop')
binned_stats = df_clean.groupby('evidence_bin', observed=True).agg({
    'mean_evidence_preRT': 'mean',
    'accuracy': ['mean', 'sem', 'count']
}).reset_index()

binned_stats.columns = ['bin', 'evidence_mean', 'accuracy_mean', 'accuracy_sem', 'n_trials']

# Plot binned data with error bars
ax.errorbar(
    binned_stats['evidence_mean'],
    binned_stats['accuracy_mean'],
    yerr=binned_stats['accuracy_sem'] * 1.96,  # 95% CI
    fmt='o',
    markersize=10,
    capsize=6,
    capthick=2,
    color='darkred',
    alpha=0.8,
    label='Binned Accuracy (±95% CI)',
    zorder=4
)

# Add reference lines
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='Chance level')
ax.axvline(0, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)

ax.set_xlabel('Mean Evidence per Frame (pre-response)', fontsize=14, fontweight='bold')
ax.set_ylabel('Probability of Correct Response', fontsize=14, fontweight='bold')
ax.set_title('Evidence-Accuracy Relationship: Logistic Regression', fontsize=16, fontweight='bold')
ax.set_ylim(-0.05, 1.05)
ax.grid(alpha=0.2)
ax.legend(loc='best', framealpha=0.9, fontsize=11)

# Add statistics annotation with p-value
stats_text = (f'Coefficient: {logit_coef:.4f}\n'
              f'p-value: {logit_pvalue:.4e}\n'
              f'AUC: {auc:.3f}\n'
              f'N = {len(df_clean)}')
ax.text(0.02, 0.98, stats_text,
         transform=ax.transAxes, fontsize=12,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()

# Save figure
output_fig = OUTPUT_DIR / "evidence_accuracy_correlation.png"
plt.savefig(output_fig, dpi=300, bbox_inches='tight')
print(f"\nFigure saved: {output_fig}")

# ========================================
# Save statistics to file
# ========================================
stats_file = OUTPUT_DIR / "evidence_accuracy_correlation_stats.txt"
with open(stats_file, 'w') as f:
    f.write("="*60 + "\n")
    f.write("Evidence-Accuracy Correlation Analysis\n")
    f.write("="*60 + "\n\n")
    
    f.write(f"Total trials analyzed: {len(df_clean)}\n\n")
    
    f.write("Phase Distribution:\n")
    for phase in sorted(df_clean['phase'].unique()):
        count = len(df_clean[df_clean['phase'] == phase])
        f.write(f"  {phase}: {count} trials\n")
    f.write("\n")
    
    f.write("Point-Biserial Correlation:\n")
    f.write(f"  r = {corr:.4f}\n")
    f.write(f"  p-value = {pval:.4e}\n\n")
    
    f.write("Logistic Regression (statsmodels):\n")
    f.write(f"  Intercept: {logit_result.params[0]:.4f} (p = {logit_result.pvalues[0]:.4e})\n")
    f.write(f"  Coefficient (evidence): {logit_result.params[1]:.6f} (p = {logit_result.pvalues[1]:.4e})\n")
    f.write(f"  95% CI for coefficient: [{logit_result.conf_int()[1][0]:.6f}, {logit_result.conf_int()[1][1]:.6f}]\n")
    f.write(f"  AUC: {auc:.4f}\n\n")
    
    f.write("Descriptive Statistics:\n")
    f.write(f"  Mean evidence per frame (correct trials): {df_clean[df_clean['accuracy']==1]['mean_evidence_preRT'].mean():.2f}\n")
    f.write(f"  Mean evidence per frame (incorrect trials): {df_clean[df_clean['accuracy']==0]['mean_evidence_preRT'].mean():.2f}\n")
    f.write(f"  SD evidence (correct trials): {df_clean[df_clean['accuracy']==1]['mean_evidence_preRT'].std():.2f}\n")
    f.write(f"  SD evidence (incorrect trials): {df_clean[df_clean['accuracy']==0]['mean_evidence_preRT'].std():.2f}\n")
    f.write(f"  Overall accuracy: {df_clean['accuracy'].mean():.2%}\n\n")
    
    f.write("Evidence Range (mean per frame):\n")
    f.write(f"  Minimum: {df_clean['mean_evidence_preRT'].min():.2f}\n")
    f.write(f"  Maximum: {df_clean['mean_evidence_preRT'].max():.2f}\n")
    f.write(f"  Mean: {df_clean['mean_evidence_preRT'].mean():.2f}\n")
    f.write(f"  Median: {df_clean['mean_evidence_preRT'].median():.2f}\n")

print(f"Statistics saved: {stats_file}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)

plt.show()

