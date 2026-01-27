# CDT Experiment Package

This is a complete, standalone package for running the Control Detection Task experiment.

## 📁 Folder Structure

### `Main_Experiment/`
- **`CDT_windows_blockwise_fast_response.py`** - Main experiment script
- **`CDT_pilot_quest_convergence.py`** - Pilot script for testing QUEST+ convergence

### `Motion_Library/`
- **`core_pool.npy`** - Filtered motion trajectories (1600 trajectories, 800 target + 800 distractor)
- **`core_pool_feats.npy`** - Feature vectors for trajectories
- **`core_pool_labels.npy`** - Labels (0=target, 1=distractor)
- **`cluster_centroids.json`** - Cluster centroids for trajectory classification
- **`scaler_params.json`** - Scaling parameters for features
- **`*_original_backup.*`** - Backup of original unfiltered library
- **`continuous_movement_data_*.csv`** - Raw movement data from collection sessions

### `Analysis_Scripts/`
- **`analyze_group_posteriors.py`** - Generate group summary panels for QUEST analysis
- **`analyze_quest_posteriors.py`** - Individual QUEST posterior analysis
- **`entropy_benchmarks.py`** - QUEST entropy benchmarks

### `Data/`
- **`quest_group_analysis/`** - Group analysis results and summary panels
- **Individual participant data files will be stored here**

### `Plots/`
- **Analysis plots and figures will be stored here**

## 🚀 Quick Start

### Running the Main Experiment:
```bash
cd Main_Experiment
python CDT_windows_blockwise_fast_response.py
```

### Running the Pilot (QUEST Convergence Test):
```bash
cd Main_Experiment
python CDT_pilot_quest_convergence.py
```

### Running Analysis:
```bash
cd Analysis_Scripts
python analyze_group_posteriors.py
```

## 📊 Key Features

- **Filtered Motion Library**: 1600 high-quality trajectories (800 target, 800 distractor)
- **QUEST+ Adaptive Algorithm**: Converges to participant-specific thresholds
- **Two Rotation Conditions**: 0° and 90° blocks
- **Fast Response Mode**: Early responses allowed during 5-second trials
- **Comprehensive Analysis**: Group-level convergence monitoring

## ⚙️ System Requirements

- Python 3.7+
- PsychoPy
- NumPy
- Pandas
- Matplotlib (for analysis)

## 📝 Notes

- Experiment duration: ~60 minutes (full) or ~20 minutes (pilot)
- Motion library paths are automatically detected relative to script location
- Data files include kinematics tracking and QUEST convergence metrics
- All original trajectory processing and quality controls preserved

## 🔬 Pilot Testing

Use the pilot script to validate QUEST+ convergence before running full experiments:
- Same practice trial counts as main experiment (50 per condition)
- Enhanced convergence monitoring and reporting
- Checks algorithm stability and threshold estimation

## 📈 Analysis Pipeline

1. **Data Collection**: Run experiments (main or pilot)
2. **QUEST Analysis**: Use `analyze_quest_posteriors.py` for individual analysis
3. **Group Analysis**: Use `analyze_group_posteriors.py` for summary panels
4. **Convergence Assessment**: Review plots in `Data/quest_group_analysis/`
