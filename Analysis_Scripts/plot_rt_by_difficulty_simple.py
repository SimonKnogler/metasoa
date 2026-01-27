"""
Plot reaction times for easy vs hard trials in the test phase
"""
print("Script starting...")

import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
print("Matplotlib backend set to Agg")

import pandas as pd
print("Pandas imported")

import matplotlib.pyplot as plt
print("Pyplot imported")

import numpy as np
from pathlib import Path
from scipy import stats
print("All imports complete")

# Load data
data_file = Path(__file__).parent.parent / "Main_Experiment" / "data" / "subjects" / "CDT_v2_blockwise_fast_response__1.csv"
print(f"\nLoading data from: {data_file}")
df = pd.read_csv(data_file)
print(f"Total rows loaded: {len(df)}")

# Filter for test phase only
test_df = df[df['phase'] == 'test_0'].copy()
print(f"Test phase rows: {len(test_df)}")

# Filter out timeout trials
test_df = test_df[test_df['is_timeout'] == False].copy()
print(f"Test phase rows (no timeout): {len(test_df)}")

# Filter for easy and hard trials only
easy_hard_df = test_df[test_df['actual_difficulty_level'].isin(['easy', 'hard'])].copy()

print(f"\nEasy trials: {len(easy_hard_df[easy_hard_df['actual_difficulty_level'] == 'easy'])}")
print(f"Hard trials: {len(easy_hard_df[easy_hard_df['actual_difficulty_level'] == 'hard'])}")

# Calculate statistics
easy_rts = easy_hard_df[easy_hard_df['actual_difficulty_level'] == 'easy']['rt_choice'].values
hard_rts = easy_hard_df[easy_hard_df['actual_difficulty_level'] == 'hard']['rt_choice'].values

print(f"\nEasy trials RT: Mean = {np.mean(easy_rts):.3f}s, SD = {np.std(easy_rts):.3f}s")
print(f"Hard trials RT: Mean = {np.mean(hard_rts):.3f}s, SD = {np.std(hard_rts):.3f}s")

# Statistical test
if len(easy_rts) > 0 and len(hard_rts) > 0:
    t_stat, p_value = stats.ttest_ind(easy_rts, hard_rts)
    print(f"\nIndependent t-test: t = {t_stat:.3f}, p = {p_value:.4f}")

print("\nCreating plots...")

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor('white')

# 1. Box plot
print("Creating box plot...")
ax1 = axes[0]
bp = ax1.boxplot([easy_rts, hard_rts], 
                   labels=['Easy\n(80% target)', 'Hard\n(60% target)'],
                   patch_artist=True,
                   widths=0.6)
bp['boxes'][0].set_facecolor('lightgreen')
bp['boxes'][0].set_alpha(0.7)
bp['boxes'][1].set_facecolor('salmon')
bp['boxes'][1].set_alpha(0.7)

ax1.set_ylabel('Reaction Time (seconds)', fontsize=12)
ax1.set_title('RT Distribution by Difficulty', fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

means = [np.mean(easy_rts), np.mean(hard_rts)]
ax1.plot([1, 2], means, 'D', color='darkblue', markersize=8, label='Mean', zorder=3)
ax1.legend()

# 2. Bar plot with error bars (simpler than violin)
print("Creating bar plot...")
ax2 = axes[1]
x_pos = [1, 2]
means = [np.mean(easy_rts), np.mean(hard_rts)]
sems = [np.std(easy_rts) / np.sqrt(len(easy_rts)), 
        np.std(hard_rts) / np.sqrt(len(hard_rts))]

bars = ax2.bar(x_pos, means, yerr=sems, 
               capsize=10, color=['lightgreen', 'salmon'],
               alpha=0.7, edgecolor=['darkgreen', 'darkred'], linewidth=2)

ax2.set_xticks(x_pos)
ax2.set_xticklabels(['Easy\n(80% target)', 'Hard\n(60% target)'])
ax2.set_ylabel('Reaction Time (seconds)', fontsize=12)
ax2.set_title('Mean RT ± SEM', fontsize=13, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# 3. Individual data points
print("Creating scatter plot...")
ax3 = axes[2]

np.random.seed(42)
easy_x = np.random.normal(1, 0.04, size=len(easy_rts))
hard_x = np.random.normal(2, 0.04, size=len(hard_rts))

ax3.scatter(easy_x, easy_rts, alpha=0.5, s=60, color='green', 
            edgecolors='darkgreen', linewidth=1, label='Easy trials')
ax3.scatter(hard_x, hard_rts, alpha=0.5, s=60, color='red', 
            edgecolors='darkred', linewidth=1, label='Hard trials')

# Add means
ax3.plot([0.85, 1.15], [np.mean(easy_rts), np.mean(easy_rts)], 
         'k-', linewidth=3, zorder=10)
ax3.plot([1.85, 2.15], [np.mean(hard_rts), np.mean(hard_rts)], 
         'k-', linewidth=3, zorder=10)

ax3.set_xlim([0.5, 2.5])
ax3.set_xticks([1, 2])
ax3.set_xticklabels(['Easy\n(80% target)', 'Hard\n(60% target)'])
ax3.set_ylabel('Reaction Time (seconds)', fontsize=12)
ax3.set_title('Individual RTs', fontsize=13, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
ax3.legend(loc='upper right', fontsize=9)

# Add significance annotation
if len(easy_rts) > 0 and len(hard_rts) > 0:
    y_max = max(easy_rts.max(), hard_rts.max())
    y_pos = y_max + 0.3
    ax3.plot([1, 2], [y_pos, y_pos], 'k-', linewidth=1.5)
    
    if p_value < 0.001:
        sig_text = '***'
    elif p_value < 0.01:
        sig_text = '**'
    elif p_value < 0.05:
        sig_text = '*'
    else:
        sig_text = 'n.s.'
    
    ax3.text(1.5, y_pos + 0.1, f'p = {p_value:.4f} {sig_text}', 
             ha='center', va='bottom', fontsize=10)

plt.tight_layout()

# Save figure
print("\nSaving figure...")
output_dir = Path(__file__).parent.parent / "Main_Experiment" / "data" / "quest_group_analysis"
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "rt_comparison_easy_vs_hard.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close(fig)  # Explicitly close the figure

print(f"✓ Plot saved to: {output_file}")
print("✓ Script completed successfully!")


