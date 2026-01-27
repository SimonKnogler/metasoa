"""
Analysis Script: Confidence & Agency Ratings Distribution
Generates two-panel histogram showing distribution of confidence and agency ratings
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
print("Confidence & Agency Ratings Distribution Analysis")
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

# Filter test phase trials only (where ratings are collected)
df_test = df_all[df_all['phase'].str.contains('test', na=False)].copy()
print(f"Test phase trials: {len(df_test)}")

# Exclude timeout trials
df_test_valid = df_test[df_test['is_timeout'] == False].copy()
print(f"Valid (non-timeout) test trials: {len(df_test_valid)}")

# Check for required columns
required_cols = ['confidence_rating', 'agency_rating']
missing_cols = [col for col in required_cols if col not in df_test_valid.columns]
if missing_cols:
    print(f"ERROR: Missing required columns: {missing_cols}")
    exit(1)

# Remove NaN values
df_ratings = df_test_valid[['confidence_rating', 'agency_rating']].dropna()
print(f"Trials with valid ratings: {len(df_ratings)}")

# Calculate statistics
conf_stats = {
    'mean': df_ratings['confidence_rating'].mean(),
    'std': df_ratings['confidence_rating'].std(),
    'median': df_ratings['confidence_rating'].median(),
    'count': df_ratings['confidence_rating'].count()
}

agency_stats = {
    'mean': df_ratings['agency_rating'].mean(),
    'std': df_ratings['agency_rating'].std(),
    'median': df_ratings['agency_rating'].median(),
    'count': df_ratings['agency_rating'].count()
}

print("\n" + "=" * 60)
print("Rating Statistics:")
print("=" * 60)
print(f"Confidence Rating: M = {conf_stats['mean']:.2f}, SD = {conf_stats['std']:.2f}, "
      f"Mdn = {conf_stats['median']:.1f}, N = {int(conf_stats['count'])}")
print(f"Agency Rating:     M = {agency_stats['mean']:.2f}, SD = {agency_stats['std']:.2f}, "
      f"Mdn = {agency_stats['median']:.1f}, N = {int(agency_stats['count'])}")

# Create two-panel figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Define bins (1-7 scale)
bins = np.arange(0.5, 8.5, 1)  # Bins centered on 1, 2, 3, 4, 5, 6, 7

# Left panel: Confidence ratings
ax1 = axes[0]
counts_conf, _, bars_conf = ax1.hist(df_ratings['confidence_rating'], bins=bins, 
                                      color='#3498db', alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.axvline(conf_stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean = {conf_stats['mean']:.2f}")
ax1.set_xlabel('Confidence Rating (1-7)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=13, fontweight='bold')
ax1.set_title('Confidence Rating Distribution', fontsize=14, fontweight='bold')
ax1.set_xticks([1, 2, 3, 4, 5, 6, 7])
ax1.grid(axis='y', alpha=0.3)
ax1.legend(fontsize=11)

# Add statistics text
stats_text_conf = f"M = {conf_stats['mean']:.2f}\nSD = {conf_stats['std']:.2f}\nN = {int(conf_stats['count'])}"
ax1.text(0.98, 0.97, stats_text_conf, transform=ax1.transAxes, 
         fontsize=11, va='top', ha='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Right panel: Agency ratings
ax2 = axes[1]
counts_agency, _, bars_agency = ax2.hist(df_ratings['agency_rating'], bins=bins, 
                                          color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.axvline(agency_stats['mean'], color='darkred', linestyle='--', linewidth=2, label=f"Mean = {agency_stats['mean']:.2f}")
ax2.set_xlabel('Agency Rating (1-7)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=13, fontweight='bold')
ax2.set_title('Agency Rating Distribution', fontsize=14, fontweight='bold')
ax2.set_xticks([1, 2, 3, 4, 5, 6, 7])
ax2.grid(axis='y', alpha=0.3)
ax2.legend(fontsize=11)

# Add statistics text
stats_text_agency = f"M = {agency_stats['mean']:.2f}\nSD = {agency_stats['std']:.2f}\nN = {int(agency_stats['count'])}"
ax2.text(0.98, 0.97, stats_text_agency, transform=ax2.transAxes, 
         fontsize=11, va='top', ha='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Save figure
output_path = output_dir / "ratings_distribution.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")

plt.close()

# Additional analysis: Check scale usage
print("\n" + "=" * 60)
print("Scale Usage Analysis:")
print("=" * 60)
print("\nConfidence Rating Distribution:")
for rating in range(1, 8):
    count = (df_ratings['confidence_rating'] == rating).sum()
    percentage = (count / len(df_ratings)) * 100
    print(f"  Rating {rating}: {count:4d} ({percentage:5.1f}%)")

print("\nAgency Rating Distribution:")
for rating in range(1, 8):
    count = (df_ratings['agency_rating'] == rating).sum()
    percentage = (count / len(df_ratings)) * 100
    print(f"  Rating {rating}: {count:4d} ({percentage:5.1f}%)")

print("\n" + "=" * 60)
print("Analysis complete!")
print("=" * 60)

