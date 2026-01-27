import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Utilities
# ----------------------------

def logit(p: float) -> float:
	p = np.clip(p, 1e-9, 1 - 1e-9)
	return float(np.log(p / (1.0 - p)))


def inv_logit(z: float) -> float:
	return float(1.0 / (1.0 + np.exp(-z)))


def clamp_prop(s: float) -> float:
	# Match experiment bounds (updated to 0.90)
	return float(np.clip(s, 0.02, 0.90))


# ----------------------------
# Replayed QUEST+ matching experiment settings
# ----------------------------

class ReplayedQuestPlus:
	"""
	Minimal reimplementation of SimpleQuestPlus used in the experiment, to replay
	training data and compute posterior-predictive curves and thresholds.
	- s_grid and alpha_grid go up to 0.90
	- alpha prior mean depends on p_target: >=0.75 -> 0.70; else -> 0.30
	- Excludes timeout trials from analysis (consistent with experiment behavior)
	"""
	def __init__(self, p_target: float):
		self.s_grid = np.linspace(logit(0.02), logit(0.90), 61)
		self.alpha_grid = np.linspace(logit(0.03), logit(0.90), 51)
		self.beta_grid = np.geomspace(1.5, 10.0, 19)
		self.lambda_grid = np.array([0.00, 0.01, 0.02])
		self.gamma = 0.5
		self.p_target = float(p_target)

		# Priors
		if self.p_target >= 0.75:
			alpha_mu = logit(0.70)
		else:
			alpha_mu = logit(0.30)
		alpha_sd = 1.2
		self.prior_alpha = np.exp(-0.5 * ((self.alpha_grid - alpha_mu) / alpha_sd) ** 2)
		self.prior_alpha /= self.prior_alpha.sum()

		beta_mu = 3.0
		beta_gsd = 1.8
		ln_beta_mu = np.log(beta_mu)
		ln_beta_sd = np.log(beta_gsd)
		self.prior_beta = np.exp(-0.5 * ((np.log(self.beta_grid) - ln_beta_mu) / ln_beta_sd) ** 2)
		self.prior_beta /= self.prior_beta.sum()

		self.prior_lambda = np.ones_like(self.lambda_grid) / len(self.lambda_grid)

		self.post_alpha = self.prior_alpha.copy()
		self.post_beta = self.prior_beta.copy()
		self.post_lambda = self.prior_lambda.copy()

		self.alpha_mean_history: List[float] = []
		self.alpha_entropy_history: List[float] = []

	def psychometric(self, s_logit, a, b, lmb):
		F = 1.0 / (1.0 + np.exp(-(s_logit - a) * b))
		return self.gamma + (1.0 - self.gamma - lmb) * F

	def _entropy(self, p: np.ndarray) -> float:
		p = np.clip(p, 1e-12, 1.0)
		p = p / p.sum()
		return float(-(p * np.log(p)).sum())

	def update(self, s_phys: float, correct: int):
		sL = logit(clamp_prop(s_phys))
		old_alpha = self.post_alpha.copy()
		old_beta = self.post_beta.copy()
		old_lambda = self.post_lambda.copy()

		# Likelihoods for alpha
		like_alpha = np.zeros_like(self.alpha_grid)
		Wb = old_beta[:, None]
		Wl = old_lambda[None, :]
		for i, a in enumerate(self.alpha_grid):
			P = self.psychometric(sL, a, self.beta_grid[:, None], self.lambda_grid[None, :])
			W = Wb * Wl
			p = (P * W).sum() / W.sum()
			like_alpha[i] = p if correct else (1.0 - p)

		# Likelihoods for beta
		like_beta = np.zeros_like(self.beta_grid)
		Wa = old_alpha[:, None]
		Wl = old_lambda[None, :]
		for j, b in enumerate(self.beta_grid):
			P = self.psychometric(sL, self.alpha_grid[:, None], b, self.lambda_grid[None, :])
			W = Wa * Wl
			p = (P * W).sum() / W.sum()
			like_beta[j] = p if correct else (1.0 - p)

		# Likelihoods for lambda
		like_lambda = np.zeros_like(self.lambda_grid)
		Wa = old_alpha[:, None]
		Wb = old_beta[None, :]
		for k, lmb in enumerate(self.lambda_grid):
			P = self.psychometric(sL, self.alpha_grid[:, None], self.beta_grid[None, :], lmb)
			W = Wa * Wb
			p = (P * W).sum() / W.sum()
			like_lambda[k] = p if correct else (1.0 - p)

		# Simultaneous update
		self.post_alpha = old_alpha * like_alpha + 1e-12
		self.post_beta = old_beta * like_beta + 1e-12
		self.post_lambda = old_lambda * like_lambda + 1e-12
		self.post_alpha /= self.post_alpha.sum()
		self.post_beta /= self.post_beta.sum()
		self.post_lambda /= self.post_lambda.sum()

		# Track
		a_mean = float((self.alpha_grid * self.post_alpha).sum())
		self.alpha_mean_history.append(a_mean)
		self.alpha_entropy_history.append(self._entropy(self.post_alpha))

	def posterior_predictive(self, s_values: np.ndarray) -> np.ndarray:
		s_values = np.asarray(s_values, dtype=float)
		s_values = np.clip(s_values, 0.02, 0.90)
		s_logit = np.array([logit(s) for s in s_values])
		pa = self.post_alpha[:, None, None]
		pb = self.post_beta[None, :, None]
		pl = self.post_lambda[None, None, :]
		W = pa * pb * pl
		W /= W.sum()
		preds = []
		for sL in s_logit:
			P = self.psychometric(sL, self.alpha_grid[:, None, None], self.beta_grid[None, :, None], self.lambda_grid[None, None, :])
			preds.append(float((W * P).sum()))
		return np.array(preds)

	def threshold_for_target(self, p_target: float) -> float:
		sL_grid = self.s_grid
		pa = self.post_alpha[:, None, None]
		pb = self.post_beta[None, :, None]
		pl = self.post_lambda[None, None, :]
		W = pa * pb * pl
		W /= W.sum()
		diffs = []
		for sL in sL_grid:
			P_s = self.psychometric(sL, self.alpha_grid[:, None, None], self.beta_grid[None, :, None], self.lambda_grid[None, None, :])
			p_hat = float((W * P_s).sum())
			diffs.append(abs(p_hat - p_target))
		idx = int(np.argmin(diffs))
		return clamp_prop(inv_logit(sL_grid[idx]))


# ----------------------------
# Data helpers
# ----------------------------

def find_data_files(data_dir: Path) -> List[Path]:
	files = []
	for p in data_dir.glob("CDT_v2_blockwise*.csv"):
		if not p.name.endswith("_kinematics.csv"):
			files.append(p)
	return sorted(files)


def extract_mapping(df: pd.DataFrame) -> Tuple[str, str]:
	candidates_low = [c for c in df.columns if "low_precision_colour" in c]
	candidates_high = [c for c in df.columns if "high_precision_colour" in c]
	low = None
	high = None
	for c in candidates_low:
		val = df[c].dropna()
		if len(val):
			low = str(val.iloc[0])
			break
	for c in candidates_high:
		val = df[c].dropna()
		if len(val):
			high = str(val.iloc[0])
			break
	if low is None or high is None:
		# Fallback: infer from accuracy (not ideal)
		low, high = "green", "blue"
	return low, high


def extract_participant_id(df: pd.DataFrame, default_id: str = "unknown") -> str:
	for col in df.columns:
		if col == "participant" or col.endswith("participant"):
			vals = df[col].dropna()
			if len(vals):
				return str(vals.iloc[0])
	return default_id


# ----------------------------
# Group analysis
# ----------------------------

def analyze_calibration_group(data_dir: Path) -> None:
	"""
	Group analysis specifically for calibration data.
	During calibration, all cues are black and we have one staircase per angle.
	"""
	out_dir = data_dir / "quest_group_analysis"
	out_dir.mkdir(exist_ok=True)

	files = find_data_files(data_dir)
	if not files:
		print(f"No data files found in {data_dir}")
		return

	# Storage for combined summary across all angles
	all_thresholds: List[Dict[str, object]] = []
	
	# Process each angle separately
	for angle in [0, 90]:
		print(f"Processing calibration {angle}° rotation...")
		
		# Storage for this angle
		per_part_curves: Dict[str, np.ndarray] = {}
		per_part_thresholds: List[Dict[str, object]] = []
		entropy_histories: Dict[str, List[float]] = {}
		
		# For group overlays
		s_plot = np.linspace(0.02, 0.90, 120)
		group_data: pd.DataFrame = pd.DataFrame()

		for f in files:
			df = pd.read_csv(f)
			if 'phase' not in df.columns:
				continue
			
			# Filter to calibration phase
			calib = df[df['phase'].astype(str).str.contains('calibration', case=False)].copy()
			if calib.empty:
				continue
			
			# Filter by angle
			if 'angle_bias' not in calib.columns:
				continue
			calib = calib[calib['angle_bias'] == angle].copy()
			if calib.empty:
				continue
			
			participant_id = extract_participant_id(df, default_id=f.stem)
			
			# Exclude timeout trials
			if 'resp_shape' in calib.columns:
				timeout_count = (calib['resp_shape'] == 'timeout').sum()
				if timeout_count > 0:
					print(f"  Excluding {timeout_count} timeout trials for participant {participant_id}")
				calib = calib[calib['resp_shape'] != 'timeout']
			
			# Prepare data: use 'prop_used' as stimulus and 'accuracy' as response
			if 'prop_used' not in calib.columns or 'accuracy' not in calib.columns:
				continue
			
			calib_clean = calib[['prop_used', 'accuracy']].dropna()
			if calib_clean.empty:
				continue
			
			calib_clean = calib_clean.rename(columns={'prop_used': 's', 'accuracy': 'correct'})
			calib_clean['participant'] = participant_id
			group_data = pd.concat([group_data, calib_clean], ignore_index=True)
			
			# Replay QUEST+ with neutral prior (p_target doesn't matter for calibration, we use full curve)
			qp = ReplayedQuestPlus(p_target=0.625)  # neutral prior
			for _, row in calib_clean.iterrows():
				qp.update(float(row['s']), int(row['correct']))
			
			preds = qp.posterior_predictive(s_plot)
			thresh_60 = qp.threshold_for_target(0.60)
			thresh_80 = qp.threshold_for_target(0.80)
			
			# Calculate final alpha SD (standard deviation of alpha posterior)
			alpha_sd = float(np.sqrt((qp.alpha_grid**2 * qp.post_alpha).sum() - 
			                         ((qp.alpha_grid * qp.post_alpha).sum())**2))
			
			per_part_curves[participant_id] = preds
			entropy_histories[participant_id] = qp.alpha_entropy_history
			
			threshold_entry = {
				'participant': participant_id,
				'angle_bias': angle,
				'threshold_60': thresh_60,
				'threshold_80': thresh_80,
				'alpha_sd': alpha_sd,
				'n_trials': len(calib_clean),
			}
			per_part_thresholds.append(threshold_entry)
			all_thresholds.append(threshold_entry)
		
		# Skip if no data for this angle
		if not per_part_thresholds:
			print(f"No calibration data found for {angle}° rotation")
			continue
		
		# Print convergence summary
		thr_df = pd.DataFrame(per_part_thresholds)
		print(f"\n  Convergence Summary for {angle}°:")
		print(f"    Mean alpha SD: {thr_df['alpha_sd'].mean():.4f} ± {thr_df['alpha_sd'].std():.4f}")
		print(f"    Range: [{thr_df['alpha_sd'].min():.4f}, {thr_df['alpha_sd'].max():.4f}]")
		converged_count = (thr_df['alpha_sd'] < 0.20).sum()
		print(f"    Converged (α_SD < 0.20): {converged_count}/{len(thr_df)} participants")
		
		# 1) Overlay curves per participant + group mean
		plt.figure(figsize=(6, 4))
		for pid, preds in per_part_curves.items():
			plt.plot(s_plot, preds, linewidth=1.0, alpha=0.35)
		if per_part_curves:
			all_preds = np.vstack(list(per_part_curves.values()))
			mean_preds = all_preds.mean(axis=0)
			plt.plot(s_plot, mean_preds, color='black', linewidth=2.0, label='group mean')
		plt.axhline(0.60, color='orange', linestyle='--', linewidth=1.0, alpha=0.6, label='60% target')
		plt.axhline(0.80, color='blue', linestyle='--', linewidth=1.0, alpha=0.6, label='80% target')
		plt.title(f'Calibration posterior-predictive curves – {angle}° rotation')
		plt.xlabel('s (self-motion proportion)')
		plt.ylabel('P(correct)')
		plt.ylim(0.45, 1.02)
		plt.grid(alpha=0.3)
		plt.legend()
		plt.tight_layout()
		plt.savefig(out_dir / f'calibration_curves_{angle}deg.png', dpi=150)
		plt.close()
		
		# 2) Threshold distributions (boxplot with jitter)
		if not thr_df.empty:
			plt.figure(figsize=(5.5, 4))
			data_60 = thr_df['threshold_60']
			data_80 = thr_df['threshold_80']
			plt.boxplot([data_60.dropna(), data_80.dropna()], labels=['60% accuracy', '80% accuracy'])
			rng = np.random.default_rng(0)
			for idx, series in enumerate([data_60, data_80], start=1):
				xs = idx + (rng.random(len(series)) - 0.5) * 0.2
				plt.scatter(xs, series, alpha=0.6, s=12)
			plt.ylabel('threshold s')
			plt.title(f'Calibration threshold distributions – {angle}° rotation')
			plt.tight_layout()
			plt.savefig(out_dir / f'calibration_thresholds_boxplot_{angle}deg.png', dpi=150)
			plt.close()
			
			# Save threshold CSV with alpha SD included
			thr_df.to_csv(out_dir / f'calibration_thresholds_{angle}deg.csv', index=False)
			
			# 2b) Alpha SD distribution plot
			plt.figure(figsize=(5.5, 4))
			plt.boxplot([thr_df['alpha_sd'].dropna()], labels=[f'{angle}° rotation'])
			xs = 1 + (rng.random(len(thr_df)) - 0.5) * 0.2
			plt.scatter(xs, thr_df['alpha_sd'], alpha=0.6, s=12, color='tab:orange')
			plt.axhline(0.20, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Convergence criterion')
			plt.ylabel('Alpha SD (posterior uncertainty)')
			plt.title(f'QUEST+ Convergence (Alpha SD) – {angle}° rotation')
			plt.legend()
			plt.grid(alpha=0.3, axis='y')
			plt.tight_layout()
			plt.savefig(out_dir / f'calibration_alpha_sd_boxplot_{angle}deg.png', dpi=150)
			plt.close()
		
		# 3) Group-level posterior predictive fit (pooled trials)
		if not group_data.empty:
			qp = ReplayedQuestPlus(p_target=0.625)
			for _, row in group_data.iterrows():
				qp.update(float(row['s']), int(row['correct']))
			preds = qp.posterior_predictive(s_plot)
			
			# Binned observed accuracy
			bins = np.linspace(0.02, 0.90, 14)
			group_data['s_bin'] = pd.cut(group_data['s'], bins=bins, include_lowest=True)
			by_bin = group_data.groupby('s_bin')
			bin_centers = by_bin['s'].mean()
			bin_acc = by_bin['correct'].mean()
			bin_n = by_bin['correct'].count()
			
			plt.figure(figsize=(6, 4))
			plt.plot(s_plot, preds, color='black', label='group posterior-predictive')
			plt.scatter(bin_centers, bin_acc, s=np.clip(bin_n * 5, 10, 120), color='tab:blue', alpha=0.7, label='observed (binned)')
			plt.axhline(0.60, color='orange', linestyle='--', linewidth=1.0, alpha=0.6, label='60% target')
			plt.axhline(0.80, color='blue', linestyle='--', linewidth=1.0, alpha=0.6, label='80% target')
			plt.title(f'Calibration group fit – {angle}° rotation')
			plt.xlabel('s (self-motion proportion)')
			plt.ylabel('P(correct)')
			plt.ylim(0.45, 1.02)
			plt.grid(alpha=0.3)
			plt.legend()
			plt.tight_layout()
			plt.savefig(out_dir / f'calibration_group_fit_{angle}deg.png', dpi=150)
			plt.close()
		
		# 4) Entropy over trials
		if entropy_histories:
			histories = list(entropy_histories.values())
			max_len = max(len(h) for h in histories)
			padded = np.full((len(histories), max_len), np.nan, dtype=float)
			for i, h in enumerate(histories):
				padded[i, :len(h)] = h
			mean_entropy = np.nanmean(padded, axis=0)
			std_entropy = np.nanstd(padded, axis=0)
			n_eff = np.sum(~np.isnan(padded), axis=0)
			sem = std_entropy / np.sqrt(np.maximum(n_eff, 1))
			low = mean_entropy - 1.96 * sem
			high = mean_entropy + 1.96 * sem
			trials = np.arange(1, len(mean_entropy) + 1)
			
			plt.figure(figsize=(6, 3.6))
			plt.plot(trials, mean_entropy, color='black', label='mean entropy')
			plt.fill_between(trials, low, high, color='gray', alpha=0.2, label='95% CI')
			plt.title(f'Calibration alpha posterior entropy – {angle}° rotation')
			plt.xlabel('trial')
			plt.ylabel('entropy (nats)')
			plt.grid(alpha=0.3)
			plt.tight_layout()
			plt.savefig(out_dir / f'calibration_entropy_{angle}deg.png', dpi=150)
			plt.close()
		
		# 5) Combined summary panel
		from matplotlib.gridspec import GridSpec
		fig = plt.figure(figsize=(12, 8))
		gs = GridSpec(2, 2, figure=fig, wspace=0.28, hspace=0.32)
		
		# Top-left: Curves
		ax1 = fig.add_subplot(gs[0, 0])
		for pid, preds in per_part_curves.items():
			ax1.plot(s_plot, preds, linewidth=1.0, alpha=0.35)
		if per_part_curves:
			all_preds = np.vstack(list(per_part_curves.values()))
			mean_preds = all_preds.mean(axis=0)
			ax1.plot(s_plot, mean_preds, color='mediumpurple', linewidth=2.5, label='Group mean')
		ax1.axhline(0.60, color='orange', linestyle='--', linewidth=1.0, alpha=0.6)
		ax1.axhline(0.80, color='blue', linestyle='--', linewidth=1.0, alpha=0.6)
		ax1.set_title(f'Calibration curves – {angle}° rotation')
		ax1.set_xlabel('Self-motion proportion (s)')
		ax1.set_ylabel('P(correct)')
		ax1.set_ylim(0.45, 1.02)
		ax1.grid(alpha=0.3)
		ax1.legend(loc='upper left')
		
		# Top-right: Threshold distributions
		ax2 = fig.add_subplot(gs[0, 1])
		if not thr_df.empty:
			data_60 = thr_df['threshold_60']
			data_80 = thr_df['threshold_80']
			ax2.boxplot([data_60.dropna(), data_80.dropna()], labels=['s@60%', 's@80%'])
			rng = np.random.default_rng(0)
			for idx, series in enumerate([data_60, data_80], start=1):
				xs = idx + (rng.random(len(series)) - 0.5) * 0.2
				ax2.scatter(xs, series, alpha=0.6, s=12)
		ax2.set_ylabel('s (self-motion proportion)')
		ax2.set_title(f'Threshold distributions – {angle}° rotation')
		ax2.grid(alpha=0.3, axis='y')
		
		# Bottom-left: Group fit
		ax3 = fig.add_subplot(gs[1, 0])
		if not group_data.empty:
			ax3.plot(s_plot, preds, color='black', label='group fit')
			ax3.scatter(bin_centers, bin_acc, s=np.clip(bin_n * 5, 10, 120), color='tab:blue', alpha=0.7, label='observed')
			ax3.axhline(0.60, color='orange', linestyle='--', linewidth=1.0, alpha=0.6)
			ax3.axhline(0.80, color='blue', linestyle='--', linewidth=1.0, alpha=0.6)
			ax3.set_xlabel('Self-motion proportion (s)')
			ax3.set_ylabel('P(correct)')
			ax3.set_ylim(0.45, 1.02)
			ax3.grid(alpha=0.3)
			ax3.legend(loc='upper left')
		ax3.set_title(f'Group fit – {angle}° rotation')
		
		# Bottom-right: Entropy
		ax4 = fig.add_subplot(gs[1, 1])
		if entropy_histories:
			ax4.plot(trials, mean_entropy, color='black', label='mean entropy')
			ax4.fill_between(trials, low, high, color='gray', alpha=0.2, label='95% CI')
			ax4.set_xlabel('Trial')
			ax4.set_ylabel('Entropy (a.u.)')
			ax4.grid(alpha=0.3)
			ax4.legend()
		ax4.set_title(f'Posterior uncertainty – {angle}° rotation')
		
		fig.tight_layout()
		fig_path = out_dir / f'calibration_summary_panels_{angle}deg.png'
		fig.savefig(fig_path, dpi=150)
		plt.close(fig)
		
		print(f"Completed calibration analysis for {angle}° rotation")
	
	# Save combined summary table with both angles
	if all_thresholds:
		combined_df = pd.DataFrame(all_thresholds)
		# Sort by participant then angle for readability
		combined_df = combined_df.sort_values(['participant', 'angle_bias'])
		combined_path = out_dir / 'calibration_thresholds_combined.csv'
		combined_df.to_csv(combined_path, index=False)
		print(f"\n✓ Combined summary saved: {combined_path}")
		print(f"  Total entries: {len(combined_df)} ({len(combined_df['participant'].unique())} participants × {len(combined_df['angle_bias'].unique())} angles)")


def analyze_group(data_dir: Path) -> None:
	out_dir = data_dir / "quest_group_analysis"
	out_dir.mkdir(exist_ok=True)

	files = find_data_files(data_dir)
	if not files:
		print(f"No data files found in {data_dir}")
		return

	# Process each angle separately
	for angle in [0, 90]:
		print(f"Processing {angle}° rotation...")
		
		# Storage for this angle
		per_part_curves: Dict[str, Dict[str, np.ndarray]] = {"blue": {}, "green": {}}
		per_part_thresholds: List[Dict[str, object]] = []
		entropy_histories: Dict[str, Dict[str, List[float]]] = {"blue": {}, "green": {}}
		group_bins: Dict[str, Dict[str, np.ndarray]] = {}

		# For group overlays
		s_plot = np.linspace(0.02, 0.90, 120)
		group_data: Dict[str, pd.DataFrame] = {"blue": pd.DataFrame(), "green": pd.DataFrame()}

		for f in files:
			df = pd.read_csv(f)
			if 'phase' not in df.columns:
				continue
			# Support both practice phases AND calibration phases
			train = df[df['phase'].astype(str).str.contains('practice_|calibration', case=False, regex=True)].copy()
			if train.empty:
				continue
			
			# Filter by angle
			if 'angle_bias' in train.columns:
				train = train[train['angle_bias'] == angle].copy()
				if train.empty:
					continue
			else:
				# If no angle_bias column, skip this angle
				continue
			
			# For calibration data, cue_color is always 'black', so we need to infer colors from low/high precision mapping
			if 'cue_color' not in train.columns:
				continue
				
			low_col, high_col = extract_mapping(train)
			participant_id = extract_participant_id(df, default_id=f.stem)

			# Append to group pool per color (exclude timeout trials)
			for color in ['green', 'blue']:
				cd = train[train['cue_color'] == color].copy()
				# Count and exclude timeout trials
				if 'chosen_by' in cd.columns:
					timeout_count = (cd['chosen_by'] == 'timeout').sum()
					if timeout_count > 0:
						print(f"  Excluding {timeout_count} timeout trials for {color} color")
					cd = cd[cd['chosen_by'] != 'timeout']
				cd = cd[['stimulus_intensity_s', 'response_correct']].dropna()
				if not cd.empty:
					cd = cd.rename(columns={'stimulus_intensity_s': 's', 'response_correct': 'correct'})
					cd['participant'] = participant_id
					group_data[color] = pd.concat([group_data[color], cd], ignore_index=True)

			for color, p_target in [(low_col, 0.60), (high_col, 0.80)]:
				cd = train[train['cue_color'] == color].copy()
				if cd.empty:
					continue
				# Exclude timeout trials from QUEST replay
				if 'chosen_by' in cd.columns:
					timeout_count = (cd['chosen_by'] == 'timeout').sum()
					if timeout_count > 0:
						print(f"  Excluding {timeout_count} timeout trials from QUEST replay for {color} color")
					cd = cd[cd['chosen_by'] != 'timeout']
				qp = ReplayedQuestPlus(p_target=p_target)
				cd = cd[['stimulus_intensity_s', 'response_correct']].dropna()
				cd = cd.reset_index(drop=True)
				for _, row in cd.iterrows():
					qp.update(float(row['stimulus_intensity_s']), int(row['response_correct']))

				preds = qp.posterior_predictive(s_plot)
				thresh = qp.threshold_for_target(p_target)
				per_part_curves[color][participant_id] = preds
				entropy_histories[color][participant_id] = qp.alpha_entropy_history
				per_part_thresholds.append({
					'participant': participant_id,
					'color': color,
					'p_target': p_target,
					'threshold_s': thresh,
					'angle_bias': angle,
				})

		# Skip if no data for this angle
		if not per_part_thresholds:
			print(f"No data found for {angle}° rotation")
			continue

		# 1) Overlay curves per participant per color + group mean
		for color in ['blue', 'green']:
			plt.figure(figsize=(6, 4))
			# Thin lines for each participant
			for pid, preds in per_part_curves[color].items():
				plt.plot(s_plot, preds, linewidth=1.0, alpha=0.35)
			# Group mean
			if per_part_curves[color]:
				all_preds = np.vstack(list(per_part_curves[color].values()))
				mean_preds = all_preds.mean(axis=0)
				plt.plot(s_plot, mean_preds, color='black', linewidth=2.0, label='group mean')
			plt.title(f'Posterior-predictive curves – {color} – {angle}° rotation')
			plt.xlabel('s (self-motion proportion)')
			plt.ylabel('P(correct)')
			plt.ylim(0.45, 1.02)
			plt.grid(alpha=0.3)
			plt.legend()
			plt.tight_layout()
			plt.savefig(out_dir / f'group_curves_{color}_{angle}deg.png', dpi=150)
			plt.close()

		# 2) Threshold distributions (boxplot with jitter)
		thr_df = pd.DataFrame(per_part_thresholds)
		if not thr_df.empty:
			plt.figure(figsize=(5.5, 4))
			data_blue = thr_df[thr_df['color'] == 'blue']['threshold_s']
			data_green = thr_df[thr_df['color'] == 'green']['threshold_s']
			plt.boxplot([data_blue.dropna(), data_green.dropna()], labels=['blue (0.80)', 'green (0.60)'])
			# Jitter scatter
			rng = np.random.default_rng(0)
			for idx, series in enumerate([data_blue, data_green], start=1):
				xs = idx + (rng.random(len(series)) - 0.5) * 0.2
				plt.scatter(xs, series, alpha=0.6, s=12)
			plt.ylabel('threshold s at p_target')
			plt.title(f'Threshold distributions across participants – {angle}° rotation')
			plt.tight_layout()
			plt.savefig(out_dir / f'group_thresholds_boxplot_{angle}deg.png', dpi=150)
			plt.close()

			# Save CSV summary
			thr_df.to_csv(out_dir / f'group_thresholds_{angle}deg.csv', index=False)

		# 3) Group-level posterior predictive fit (pooled trials)
		for color, p_target in [('blue', 0.80), ('green', 0.60)]:
			pool = group_data[color]
			if pool.empty:
				continue
			qp = ReplayedQuestPlus(p_target=p_target)
			for _, row in pool.iterrows():
				qp.update(float(row['s']), int(row['correct']))
			preds = qp.posterior_predictive(s_plot)

			# Binned observed accuracy
			bins = np.linspace(0.02, 0.90, 14)
			pool['s_bin'] = pd.cut(pool['s'], bins=bins, include_lowest=True)
			by_bin = pool.groupby('s_bin')
			bin_centers = by_bin['s'].mean()
			bin_acc = by_bin['correct'].mean()
			bin_n = by_bin['correct'].count()

			plt.figure(figsize=(6, 4))
			plt.plot(s_plot, preds, color='black', label='group posterior-predictive')
			plt.scatter(bin_centers, bin_acc, s=np.clip(bin_n * 5, 10, 120), color='tab:blue', alpha=0.7, label='observed (binned)')
			plt.title(f'Group fit – {color} (p_target={p_target:.2f}) – {angle}° rotation')
			plt.xlabel('s (self-motion proportion)')
			plt.ylabel('P(correct)')
			plt.ylim(0.45, 1.02)
			plt.grid(alpha=0.3)
			plt.legend()
			plt.tight_layout()
			plt.savefig(out_dir / f'group_fit_{color}_{angle}deg.png', dpi=150)
			plt.close()

		# 4) Entropy over trials (alpha posterior), averaged across participants
		entropy_summary = {}
		for color in ['blue', 'green']:
			histories = list(entropy_histories[color].values())
			if not histories:
				continue
			max_len = max(len(h) for h in histories)
			# Pad with NaN to compute nanmean across varying lengths
			padded = np.full((len(histories), max_len), np.nan, dtype=float)
			for i, h in enumerate(histories):
				padded[i, :len(h)] = h
			mean_entropy = np.nanmean(padded, axis=0)
			# Simple 95% CI via SEM assuming approx normal
			std_entropy = np.nanstd(padded, axis=0)
			n_eff = np.sum(~np.isnan(padded), axis=0)
			sem = std_entropy / np.sqrt(np.maximum(n_eff, 1))
			low = mean_entropy - 1.96 * sem
			high = mean_entropy + 1.96 * sem
			trials = np.arange(1, len(mean_entropy) + 1)

			plt.figure(figsize=(6, 3.6))
			plt.plot(trials, mean_entropy, color='black', label='mean entropy')
			plt.fill_between(trials, low, high, color='gray', alpha=0.2, label='95% CI')
			plt.title(f'Alpha posterior entropy over trials – {color} – {angle}° rotation')
			plt.xlabel('trial (training)')
			plt.ylabel('entropy (nats)')
			plt.grid(alpha=0.3)
			plt.tight_layout()
			plt.savefig(out_dir / f'group_entropy_{color}_{angle}deg.png', dpi=150)
			plt.close()

			entropy_summary[color] = (trials, mean_entropy)

		# 5) Combined 2x2 summary figure for this angle
		from matplotlib.gridspec import GridSpec
		fig = plt.figure(figsize=(12, 8))
		gs = GridSpec(2, 2, figure=fig, wspace=0.28, hspace=0.32)

		# Top-left: High control curves
		ax1 = fig.add_subplot(gs[0, 0])
		for pid, preds in per_part_curves['blue'].items():
			ax1.plot(s_plot, preds, linewidth=1.0, alpha=0.35)
		if per_part_curves['blue']:
			all_preds = np.vstack(list(per_part_curves['blue'].values()))
			mean_preds = all_preds.mean(axis=0)
			ax1.plot(s_plot, mean_preds, color='mediumpurple', linewidth=2.5, label='Group mean')
		ax1.axhline(0.80, color='tab:blue', linestyle='--', linewidth=1.2, alpha=0.8)
		ax1.set_title(f'All participant curves – Expected HIGH control (target ~80%) – {angle}° rotation')
		ax1.set_xlabel('Self-motion proportion (s)')
		ax1.set_ylabel('P(correct)')
		ax1.set_ylim(0.45, 1.02)
		ax1.grid(alpha=0.3)
		ax1.legend(loc='upper left')

		# Bottom-left: Low control curves
		ax3 = fig.add_subplot(gs[1, 0])
		for pid, preds in per_part_curves['green'].items():
			ax3.plot(s_plot, preds, linewidth=1.0, alpha=0.35)
		if per_part_curves['green']:
			all_preds_g = np.vstack(list(per_part_curves['green'].values()))
			mean_preds_g = all_preds_g.mean(axis=0)
			ax3.plot(s_plot, mean_preds_g, color='mediumpurple', linewidth=2.5, label='Group mean')
		ax3.axhline(0.60, color='tab:blue', linestyle='--', linewidth=1.2, alpha=0.8)
		ax3.set_title(f'All participant curves – Expected LOW control (target ~60%) – {angle}° rotation')
		ax3.set_xlabel('Self-motion proportion (s)')
		ax3.set_ylabel('P(correct)')
		ax3.set_ylim(0.45, 1.02)
		ax3.grid(alpha=0.3)
		ax3.legend(loc='upper left')

		# Top-right: Threshold distributions boxplot
		ax2 = fig.add_subplot(gs[0, 1])
		if not thr_df.empty:
			data_blue = thr_df[thr_df['color'] == 'blue']['threshold_s']
			data_green = thr_df[thr_df['color'] == 'green']['threshold_s']
			bp = ax2.boxplot([data_blue.dropna(), data_green.dropna()], labels=['High control s@80%', 'Low control s@60%'])
			# Light jitter scatter
			rng = np.random.default_rng(0)
			for idx, series in enumerate([data_blue, data_green], start=1):
				xs = idx + (rng.random(len(series)) - 0.5) * 0.2
				ax2.scatter(xs, series, alpha=0.6, s=12)
		ax2.set_ylabel('s (self-motion proportion)')
		ax2.set_title(f'Threshold distributions across participants – {angle}° rotation')
		ax2.grid(alpha=0.3, axis='y')

		# Bottom-right: Entropy over trials for both colors
		ax4 = fig.add_subplot(gs[1, 1])
		if 'blue' in entropy_summary:
			tr_b, m_b = entropy_summary['blue']
			ax4.plot(tr_b, m_b, label='High control', color='tab:blue')
		if 'green' in entropy_summary:
			tr_g, m_g = entropy_summary['green']
			ax4.plot(tr_g, m_g, label='Low control', color='tab:orange')
		ax4.set_title(f'Posterior uncertainty over trials (entropy) – {angle}° rotation')
		ax4.set_xlabel('Trial')
		ax4.set_ylabel('Entropy (a.u.)')
		ax4.grid(alpha=0.3)
		ax4.legend()

		fig.tight_layout()
		fig_path = out_dir / f'group_summary_panels_{angle}deg.png'
		fig.savefig(fig_path, dpi=150)
		plt.close(fig)

		print(f"Completed analysis for {angle}° rotation")

	# Save a JSON index of outputs
	index = {
		'files': [str(f.name) for f in files],
		'outputs': [
			'group_curves_blue_0deg.png', 'group_curves_green_0deg.png',
			'group_thresholds_boxplot_0deg.png', 'group_thresholds_0deg.csv',
			'group_fit_blue_0deg.png', 'group_fit_green_0deg.png',
			'group_entropy_blue_0deg.png', 'group_entropy_green_0deg.png',
			'group_summary_panels_0deg.png',
			'group_curves_blue_90deg.png', 'group_curves_green_90deg.png',
			'group_thresholds_boxplot_90deg.png', 'group_thresholds_90deg.csv',
			'group_fit_blue_90deg.png', 'group_fit_green_90deg.png',
			'group_entropy_blue_90deg.png', 'group_entropy_green_90deg.png',
			'group_summary_panels_90deg.png',
		],
	}
	with open(out_dir / 'index.json', 'w', encoding='utf-8') as f:
		json.dump(index, f, indent=2)


if __name__ == '__main__':
	root = Path(__file__).parent.parent  # Go up to package root
	data_dir = root / 'Main_Experiment' / 'data'
	print(f"Analyzing group data in: {data_dir}")
	analyze_group(data_dir)
