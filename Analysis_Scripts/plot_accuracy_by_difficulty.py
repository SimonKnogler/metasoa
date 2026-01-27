"""
Analysis Script: Accuracy by Difficulty Level
Generates bar plot showing mean accuracy for easy, medium, and hard trials
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob

# Paths
script_dir = Path(__file__).parent
data_dir = script_dir.parent / "Main_Experiment" / "data" / "subjects"
output_dir = script_dir.parent / "Main_Experiment" / "data" / "quest_group_analysis"
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Accuracy by Difficulty Level Analysis")
print("=" * 60)

# Load all participant data files (exclude kinematics)
csv_files = glob.glob(str(data_dir / "CDT_v2_blockwise_fast_response_*.csv"))
csv_files = [f for f in csv_files if '_kinematics' not in f]

print(f"\nFound {len(csv_files)} participant data files")

# Load and concatenate all data
all_data = []
for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file)
        all_data.append(df)
        print(f"  Loaded: {Path(csv_file).name}")
    except Exception as e:
        print(f"  Error loading {csv_file}: {e}")

if not all_data:
    print("No data files found!")
    exit(1)

df_all = pd.concat(all_data, ignore_index=True)
print(f"\nTotal trials loaded: {len(df_all)}")

# Filter test phase trials only
df_test = df_all[df_all['phase'].str.contains('test', na=False)].copy()
print(f"Test phase trials: {len(df_test)}")

# Exclude timeout trials
df_test_valid = df_test[df_test['is_timeout'] == False].copy()
print(f"Valid (non-timeout) test trials: {len(df_test_valid)}")

# Check for required columns
required_cols = ['actual_difficulty_level', 'accuracy']
missing_cols = [col for col in required_cols if col not in df_test_valid.columns]
if missing_cols:
    print(f"ERROR: Missing required columns: {missing_cols}")
    exit(1)

# Group by difficulty level and calculate statistics
difficulty_stats = df_test_valid.groupby('actual_difficulty_level')['accuracy'].agg([
    ('mean', 'mean'),
    ('std', 'std'),
    ('count', 'count'),
    ('sem', lambda x: x.std() / np.sqrt(len(x)))  # Standard error
]).reset_index()

print("\n" + "=" * 60)
print("Accuracy by Difficulty Level:")
print("=" * 60)
for _, row in difficulty_stats.iterrows():
    print(f"{row['actual_difficulty_level']:8s}: M = {row['mean']:.3f}, SD = {row['std']:.3f}, "
          f"SEM = {row['sem']:.3f}, N = {int(row['count'])}")

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))

# Define order and colors
difficulty_order = ['easy', 'medium', 'hard']
colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red

# Filter and sort data
plot_data = difficulty_stats[difficulty_stats['actual_difficulty_level'].isin(difficulty_order)].copy()
plot_data['difficulty_level'] = pd.Categorical(
    plot_data['actual_difficulty_level'], 
    categories=difficulty_order, 
    ordered=True
)
plot_data = plot_data.sort_values('difficulty_level')

# Create bar plot
x_pos = np.arange(len(plot_data))
bars = ax.bar(x_pos, plot_data['mean'], yerr=plot_data['sem'], 
               capsize=5, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Customize plot
ax.set_xlabel('Difficulty Level', fontsize=14, fontweight='bold')
ax.set_ylabel('Mean Accuracy', fontsize=14, fontweight='bold')
ax.set_title('Accuracy by Difficulty Level (Test Phase)', fontsize=16, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([d.capitalize() for d in plot_data['actual_difficulty_level']], fontsize=12)
ax.set_ylim(0, 1.0)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance (50%)')
ax.grid(axis='y', alpha=0.3)
ax.legend(fontsize=11)

# Add value labels on bars
for i, (bar, row) in enumerate(zip(bars, plot_data.itertuples())):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{row.mean:.2%}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()

# Save figure
output_path = output_dir / "accuracy_by_difficulty.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")

plt.close()

print("\n" + "=" * 60)
print("Analysis complete!")
print("=" * 60)

