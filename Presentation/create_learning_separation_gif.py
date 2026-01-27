"""
Create animated GIF showing learning level separation in logit space
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
import os

# Parameters from the experiment code - use smaller separation for more realistic range
LEARNING_LOGIT_SEPARATION = 0.6  # Reduced from 1.2 to keep within reasonable accuracy range
GAMMA = 0.5  # 2AFC chance level

def logit(x):
    """Logit transform matching experiment code"""
    x = float(np.clip(x, 1e-6, 1-1e-6))
    return float(np.log(x/(1-x)))

def inv_logit(z):
    """Inverse logit transform"""
    return float(1.0/(1.0 + np.exp(-z)))

def psychometric(s, alpha, beta, lapse=0.02):
    """Psychometric function"""
    sigmoid = 1.0 / (1.0 + np.exp(-beta * (s - alpha)))
    return GAMMA + (1.0 - GAMMA - lapse) * sigmoid

# More realistic psychometric curve parameters (shallower slope, centered better)
alpha = 0.0  # Threshold at 75% (in logit space, corresponds to s=0.5)
beta = 3.5   # Shallower slope for more realistic human performance
lapse = 0.02

# Calculate thresholds for 60% and 80%
# Find stimulus values that give 60% and 80% accuracy
s_range = np.linspace(0.1, 0.9, 1000)
accuracies = [psychometric(logit(s), alpha, beta, lapse) for s in s_range]

# Find closest to 60% and 80%
s_60 = s_range[np.argmin(np.abs(np.array(accuracies) - 0.60))]
s_80 = s_range[np.argmin(np.abs(np.array(accuracies) - 0.80))]

# Logit values
z_60 = logit(s_60)
z_80 = logit(s_80)
z_mid = 0.5 * (z_60 + z_80)

# Separated learning levels
z_hard_learn = z_mid - LEARNING_LOGIT_SEPARATION
z_easy_learn = z_mid + LEARNING_LOGIT_SEPARATION

# Convert back to proportion space
s_hard_learn = inv_logit(z_hard_learn)
s_easy_learn = inv_logit(z_easy_learn)

print(f"60% threshold: s={s_60:.3f}, logit={z_60:.3f}")
print(f"80% threshold: s={s_80:.3f}, logit={z_80:.3f}")
print(f"Midpoint: logit={z_mid:.3f}")
print(f"Hard learning: s={s_hard_learn:.3f}, logit={z_hard_learn:.3f}")
print(f"Easy learning: s={s_easy_learn:.3f}, logit={z_easy_learn:.3f}")

# Animation parameters
n_frames_curve = 30
n_frames_transform = 30
n_frames_separate = 30
n_frames_pause = 20
total_frames = n_frames_curve + n_frames_transform + n_frames_separate + n_frames_pause

frames = []

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for frame_idx in range(total_frames):
    ax1.clear()
    ax2.clear()
    
    # Determine animation phase
    if frame_idx < n_frames_curve:
        # Phase 1: Show initial points
        phase = 'initial'
        progress = frame_idx / n_frames_curve
    elif frame_idx < n_frames_curve + n_frames_transform:
        # Phase 2: Transform (not used anymore, skip to separation)
        phase = 'separation'
        progress = (frame_idx - n_frames_curve) / n_frames_transform
    elif frame_idx < n_frames_curve + n_frames_transform + n_frames_separate:
        # Phase 3: Separate
        phase = 'separation'
        progress = (frame_idx - n_frames_curve - n_frames_transform) / n_frames_separate
    else:
        # Phase 4: Pause
        phase = 'pause'
        progress = 1.0
    
    # === LEFT PANEL: Logit Space (Symmetric Separation) ===
    ax1.axvline(z_mid, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Midpoint')
    ax1.set_xlabel('Stimulus Intensity (logit)', fontsize=12)
    ax1.set_ylabel('Position', fontsize=12)
    ax1.set_title('Symmetric Separation in Logit Space', fontsize=14, fontweight='bold')
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-0.5, 2)
    ax1.grid(alpha=0.2)
    
    if phase == 'initial':
        # Show initial positions
        ax1.scatter([z_60], [0.5], s=300, c='blue', edgecolors='darkblue', 
                   linewidths=3, zorder=5, label='60% Threshold (Hard)')
        ax1.scatter([z_80], [0.5], s=300, c='green', edgecolors='darkgreen', 
                   linewidths=3, zorder=5, label='80% Threshold (Easy)')
        ax1.scatter([z_mid], [0.5], s=300, c='orange', marker='s', 
                   edgecolors='darkorange', linewidths=3, zorder=5, label='Midpoint (Medium)')
    else:
        # Animate separation
        if phase == 'separation':
            current_z_hard = z_60 + progress * (z_hard_learn - z_60)
            current_z_easy = z_80 + progress * (z_easy_learn - z_80)
        else:  # pause
            current_z_hard = z_hard_learn
            current_z_easy = z_easy_learn
        
        ax1.scatter([current_z_hard], [0.5], s=300, c='blue', edgecolors='darkblue', 
                   linewidths=3, zorder=5, label='Hard Learning Level')
        ax1.scatter([z_mid], [0.5], s=300, c='orange', marker='s', 
                   edgecolors='darkorange', linewidths=3, zorder=5, label='Medium (Test)')
        ax1.scatter([current_z_easy], [0.5], s=300, c='green', edgecolors='darkgreen', 
                   linewidths=3, zorder=5, label='Easy Learning Level')
        
        if progress >= 0.99 or phase == 'pause':
            # Show separation distance annotations
            ax1.annotate('', xy=(z_mid, 1.0), xytext=(current_z_hard, 1.0),
                       arrowprops=dict(arrowstyle='<->', color='red', lw=2))
            ax1.text((z_mid + current_z_hard)/2, 1.1, f'D = {LEARNING_LOGIT_SEPARATION}',
                    ha='center', fontsize=11, color='red', fontweight='bold')
            
            ax1.annotate('', xy=(z_mid, 1.4), xytext=(current_z_easy, 1.4),
                       arrowprops=dict(arrowstyle='<->', color='red', lw=2))
            ax1.text((z_mid + current_z_easy)/2, 1.5, f'D = {LEARNING_LOGIT_SEPARATION}',
                    ha='center', fontsize=11, color='red', fontweight='bold')
            
            # Add text labels
            ax1.text(current_z_hard, 0.2, 'Hard\n(Low Cue)', ha='center', fontsize=10, 
                    color='blue', fontweight='bold')
            ax1.text(z_mid, 0.2, 'Medium\n(Test)', ha='center', fontsize=10, 
                    color='orange', fontweight='bold')
            ax1.text(current_z_easy, 0.2, 'Easy\n(High Cue)', ha='center', fontsize=10, 
                    color='green', fontweight='bold')
    
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_yticks([])
    
    # === RIGHT PANEL: Psychometric Curve ===
    s_plot = np.linspace(0.05, 0.95, 200)
    acc_plot = [psychometric(logit(s), alpha, beta, lapse) for s in s_plot]
    
    ax2.plot(s_plot, acc_plot, 'k-', linewidth=2, label='Psychometric Curve')
    ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax2.set_xlabel('Stimulus Intensity (proportion)', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Movement on Psychometric Function', fontsize=14, fontweight='bold')
    ax2.set_ylim(0.45, 1.0)
    ax2.set_xlim(0.05, 0.95)
    ax2.grid(alpha=0.2)
    
    if phase == 'initial':
        # Show initial thresholds
        ax2.scatter([s_60], [0.60], s=300, c='blue', edgecolors='darkblue', 
                   linewidths=3, zorder=5, label='60% Threshold (Hard)')
        ax2.scatter([s_80], [0.80], s=300, c='green', edgecolors='darkgreen', 
                   linewidths=3, zorder=5, label='80% Threshold (Easy)')
        # Medium point at midpoint accuracy
        s_mid_prob = inv_logit(z_mid)
        acc_mid = psychometric(z_mid, alpha, beta, lapse)
        ax2.scatter([s_mid_prob], [acc_mid], s=300, c='orange', marker='s',
                   edgecolors='darkorange', linewidths=3, zorder=5, label='Midpoint (Medium)')
    else:
        # Calculate current positions (synchronized with left panel)
        if phase == 'separation':
            current_z_hard = z_60 + progress * (z_hard_learn - z_60)
            current_z_easy = z_80 + progress * (z_easy_learn - z_80)
        else:  # pause
            current_z_hard = z_hard_learn
            current_z_easy = z_easy_learn
        
        # Convert back to probability space
        current_s_hard = inv_logit(current_z_hard)
        current_s_easy = inv_logit(current_z_easy)
        s_mid_prob = inv_logit(z_mid)
        
        # Get accuracies on psychometric curve
        acc_hard = psychometric(current_z_hard, alpha, beta, lapse)
        acc_easy = psychometric(current_z_easy, alpha, beta, lapse)
        acc_mid = psychometric(z_mid, alpha, beta, lapse)
        
        # Draw dots on psychometric curve
        ax2.scatter([current_s_hard], [acc_hard], s=300, c='blue', edgecolors='darkblue', 
                   linewidths=3, zorder=5, label='Hard Learning Level')
        ax2.scatter([s_mid_prob], [acc_mid], s=300, c='orange', marker='s',
                   edgecolors='darkorange', linewidths=3, zorder=5, label='Medium (Test)')
        ax2.scatter([current_s_easy], [acc_easy], s=300, c='green', edgecolors='darkgreen', 
                   linewidths=3, zorder=5, label='Easy Learning Level')
        
        # Add horizontal lines to show accuracy levels
        if progress >= 0.99 or phase == 'pause':
            ax2.axhline(acc_hard, color='blue', linestyle=':', alpha=0.3, linewidth=1)
            ax2.axhline(acc_mid, color='orange', linestyle=':', alpha=0.3, linewidth=1)
            ax2.axhline(acc_easy, color='green', linestyle=':', alpha=0.3, linewidth=1)
            
            # Add text labels showing accuracy
            ax2.text(0.06, acc_hard + 0.02, f'{acc_hard:.0%}', fontsize=9, color='blue', fontweight='bold')
            ax2.text(0.06, acc_mid + 0.02, f'{acc_mid:.0%}', fontsize=9, color='orange', fontweight='bold')
            ax2.text(0.06, acc_easy + 0.02, f'{acc_easy:.0%}', fontsize=9, color='green', fontweight='bold')
    
    ax2.legend(loc='lower right', fontsize=9)
    
    plt.tight_layout()
    
    # Save frame with fixed size (no bbox_inches='tight' to ensure consistent dimensions)
    frame_path = f'Presentation/diagrams/temp_frame_{frame_idx:03d}.png'
    plt.savefig(frame_path, dpi=100)
    frames.append(frame_path)
    print(f'Generated frame {frame_idx + 1}/{total_frames}')

plt.close()

# Create GIF using imageio
print("\nCreating GIF...")
import imageio

images = []
for frame_path in frames:
    images.append(imageio.imread(frame_path))

# Add pause at the end (repeat last frame)
for _ in range(20):
    images.append(imageio.imread(frames[-1]))

output_path = 'Presentation/diagrams/learning_separation_visual.gif'
imageio.mimsave(output_path, images, duration=0.05, loop=0)

print(f"\nGIF saved to: {output_path}")

# Clean up temporary frames
print("Cleaning up temporary frames...")
for frame_path in frames:
    os.remove(frame_path)

print("Done!")

