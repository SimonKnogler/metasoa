# CDT Experiment Package

## Project Overview

This repository contains the **Control Detection Task (CDT)** experiment package, designed to investigate the **sense of agency** — the subjective experience of controlling one's actions and their effects in the world.

### Scientific Background

The sense of agency is increasingly understood as an **inferential, context-sensitive process** rather than a binary signal. Our perception of control depends not only on current sensorimotor signals but also on **prior expectations**. This project investigates how learned cue-based expectations influence control detection and subjective agency judgments.

### Research Questions

1. **Expectation effects on control detection**: Do learned cues that predict task difficulty bias control detection performance?
2. **Cue-congruency effects**: Does performance differ when the actual difficulty matches vs. mismatches the cued expectation?
3. **Rotation disruption**: How do visuomotor rotations (0° vs 90°) affect control detection and learning?

### The Task

Participants view two shapes (square and circle) moving on screen while controlling a mouse. One shape's movement direction is influenced by the participant's mouse movements (the "target"), while the other shape follows a pre-recorded trajectory. Participants must identify which shape they controlled by pressing a key.

### Experimental Design

The experiment uses a **cue-based expectation learning paradigm**:

1. **Calibration Phase**: Adaptive staircase procedures determine each participant's detection threshold
2. **Learning Phase**: Colored cues (e.g., red/blue) are paired with different difficulty levels (easy/hard), allowing participants to learn cue-difficulty associations
3. **Test Phase**: Cues are presented but difficulty is held constant (medium), testing whether learned expectations bias control detection

Two rotation conditions (0° and 90° visuomotor rotation) are tested in separate blocks with counterbalanced order and color-difficulty mappings.

---

This is a complete, standalone package for running the Control Detection Task experiment.

## 📁 Folder Structure

### `Main_Experiment/`
- **`CDT_windows_blockwise_fast_response.py`** - Main experiment script with cue-based expectation learning
- **`documents/`** - Participant information sheets, consent forms, and experimenter protocols
- **`data/subjects/`** - Participant data files

### `Multi Timescale Inference/`
- **`MT Inference.py`** - Subproject: Multi-timescale inference study (2-shape vs 4-shape conditions)
- **`MT_Inference_Analysis.py`** - Analysis script for history effects and temporal integration
- **`data/subjects/`** - Participant data files

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

- **Cue-Based Expectation Learning**: Colored cues signal different difficulty levels
- **Adaptive Calibration**: Staircase procedures find individual detection thresholds
- **Counterbalanced Design**: Learning order, color-difficulty mappings, and rotation conditions fully counterbalanced
- **Two Rotation Conditions**: 0° (congruent) and 90° (rotated) visuomotor mapping
- **Fast Response Mode**: Early responses allowed during motion phase
- **Filtered Motion Library**: 1600 high-quality pre-recorded trajectories

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
