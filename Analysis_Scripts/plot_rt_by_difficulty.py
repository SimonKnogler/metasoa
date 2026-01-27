"""
Plot reaction times for easy vs hard trials in the test phase
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy import stats

try:
    print("Starting script...")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    
    # Load data
    data_file = Path(__file__).parent.parent / "Main_Experiment" / "data" / "subjects" / "CDT_v2_blockwise_fast_response__1.csv"
    print(f"Loading data from: {data_file}")
    df = pd.read_csv(data_file)
    print(f"Total rows loaded: {len(df)}")
    
    # Filter for test phase only
    test_df = df[df['phase'] == 'test_0'].copy()
    print(f"Test phase rows: {len(test_df)}")
    
    # Filter out timeout trials (they have NaN RT)
    test_df = test_df[test_df['is_timeout'] == False].copy()
    print(f"Test phase rows (no timeout): {len(test_df)}")
    
    # Filter for easy and hard trials only
    easy_hard_df = test_df[test_df['actual_difficulty_level'].isin(['easy', 'hard'])].copy()
    
    print(f"\nTotal test trials (non-timeout): {len(test_df)}")
    print(f"Easy trials: {len(easy_hard_df[easy_hard_df['actual_difficulty_level'] == 'easy'])}")
    print(f"Hard trials: {len(easy_hard_df[easy_hard_df['actual_difficulty_level'] == 'hard'])}")
    
    # Calculate statistics
    easy_rts = easy_hard_df[easy_hard_df['actual_difficulty_level'] == 'easy']['rt_choice']
    hard_rts = easy_hard_df[easy_hard_df['actual_difficulty_level'] == 'hard']['rt_choice']
    
    print(f"\nEasy trials RT: Mean = {easy_rts.mean():.3f}s, SD = {easy_rts.std():.3f}s")
    print(f"Hard trials RT: Mean = {hard_rts.mean():.3f}s, SD = {hard_rts.std():.3f}s")
    
    # Statistical test
    if len(easy_rts) > 0 and len(hard_rts) > 0:
        t_stat, p_value = stats.ttest_ind(easy_rts, hard_rts)
        print(f"\nIndependent t-test: t = {t_stat:.3f}, p = {p_value:.4f}")
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Box plot
    ax1 = axes[0]
    bp = ax1.boxplot([easy_rts, hard_rts], 
                       labels=['Easy\n(80% target)', 'Hard\n(60% target)'],
                       patch_artist=True,
                       widths=0.6)
    # Color the boxes
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('salmon')
    bp['boxes'][1].set_alpha(0.7)
    
    ax1.set_ylabel('Reaction Time (seconds)', fontsize=12)
    ax1.set_title('RT Distribution by Difficulty', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add mean markers
    means = [easy_rts.mean(), hard_rts.mean()]
    ax1.plot([1, 2], means, 'D', color='darkblue', markersize=8, label='Mean', zorder=3)
    ax1.legend()
    
    # 2. Violin plot
    ax2 = axes[1]
    parts = ax2.violinplot([easy_rts, hard_rts], 
                            positions=[1, 2],
                            showmeans=True,
                            showmedians=True)
    
    # Color the violin plots
    colors = ['lightgreen', 'salmon']
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(['Easy\n(80% target)', 'Hard\n(60% target)'])
    ax2.set_ylabel('Reaction Time (seconds)', fontsize=12)
    ax2.set_title('RT Density by Difficulty', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Individual data points with mean and error bars
    ax3 = axes[2]
    
    # Plot individual points with jitter
    np.random.seed(42)
    easy_x = np.random.normal(1, 0.04, size=len(easy_rts))
    hard_x = np.random.normal(2, 0.04, size=len(hard_rts))
    
    ax3.scatter(easy_x, easy_rts, alpha=0.5, s=60, color='green', edgecolors='darkgreen', linewidth=1, label='Easy trials')
    ax3.scatter(hard_x, hard_rts, alpha=0.5, s=60, color='red', edgecolors='darkred', linewidth=1, label='Hard trials')
    
    # Add mean and SEM error bars
    easy_mean = easy_rts.mean()
    easy_sem = easy_rts.sem()
    hard_mean = hard_rts.mean()
    hard_sem = hard_rts.sem()
    
    ax3.errorbar([1], [easy_mean], yerr=[easy_sem], 
                 fmt='D', color='darkgreen', markersize=10, capsize=8, 
                 capthick=2, linewidth=2, label='Mean ± SEM', zorder=10)
    ax3.errorbar([2], [hard_mean], yerr=[hard_sem], 
                 fmt='D', color='darkred', markersize=10, capsize=8, 
                 capthick=2, linewidth=2, zorder=10)
    
    ax3.set_xlim([0.5, 2.5])
    ax3.set_xticks([1, 2])
    ax3.set_xticklabels(['Easy\n(80% target)', 'Hard\n(60% target)'])
    ax3.set_ylabel('Reaction Time (seconds)', fontsize=12)
    ax3.set_title('Individual RTs by Difficulty', fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    ax3.legend(loc='upper right', fontsize=9)
    
    # Add statistical annotation
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
    output_dir = Path(__file__).parent.parent / "Main_Experiment" / "data" / "quest_group_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "rt_comparison_easy_vs_hard.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_file}")
    print("✓ Script completed successfully!")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
