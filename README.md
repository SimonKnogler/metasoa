# Control Detection Task (CDT) Experiment

A PsychoPy-based experiment investigating the **sense of agency** — the subjective experience of controlling one's actions and their effects in the world.

## Scientific Background

The sense of agency is increasingly understood as an **inferential, context-sensitive process** rather than a binary signal. Our perception of control depends not only on current sensorimotor signals but also on **prior expectations**. This project investigates how learned cue-based expectations influence control detection and subjective agency judgments.

### Research Questions

1. **Expectation effects on control detection**: Do learned cues that predict task difficulty bias control detection performance?
2. **Cue-congruency effects**: Does performance differ when the actual difficulty matches vs. mismatches the cued expectation?
3. **Rotation disruption**: How do visuomotor rotations (0° vs 90°) affect control detection and learning?

### The Task

Participants view two shapes (square and circle) moving on screen while controlling a mouse. One shape's movement direction is influenced by the participant's mouse movements (the "target"), while the other shape follows a pre-recorded trajectory. Participants must identify which shape they controlled.

### Experimental Design

The experiment uses a **cue-based expectation learning paradigm**:

1. **Calibration Phase**: Adaptive staircase procedures determine each participant's detection threshold
2. **Learning Phase**: Colored cues are paired with different difficulty levels, allowing participants to learn cue-difficulty associations
3. **Test Phase**: Cues are presented but difficulty is held constant, testing whether learned expectations bias control detection

Two rotation conditions (0° and 90° visuomotor rotation) are tested in separate blocks with counterbalanced order.

---

## Repository Structure

```
metasoa/
├── Main_Experiment/
│   ├── CDT_windows_blockwise_fast_response.py   # Main experiment script
│   ├── data/
│   │   ├── subjects/              # Participant data files
│   │   ├── analysis_output/       # Generated figures
│   │   └── quest_group_analysis/  # QUEST calibration analysis
│   └── documents/                 # Consent forms, protocols, etc.
│
├── Analysis_Scripts/              # Data analysis scripts
│   ├── analyze_group_posteriors.py
│   ├── ddm_analysis.py
│   ├── mixed_model_desenderlab.py
│   └── ...
│
├── Motion_Library/                # Pre-recorded motion trajectories
│   ├── core_pool.npy              # Filtered trajectories (1600 total)
│   ├── core_pool_feats.npy        # Feature vectors
│   ├── core_pool_labels.npy       # Labels
│   └── *.json                     # Cluster/scaling parameters
│
└── docs/
    ├── COUNTERBALANCING_SCHEME.md
    ├── Presentation/              # Diagrams and presentation materials
    └── references/                # Reference papers and guides
```

## Quick Start

### Requirements

- Python 3.8+
- PsychoPy
- NumPy, Pandas, Matplotlib, SciPy

### Installation

```bash
pip install psychopy numpy pandas matplotlib scipy scikit-learn
```

### Running the Experiment

```bash
cd Main_Experiment
python CDT_windows_blockwise_fast_response.py
```

### Running Analysis

```bash
cd Analysis_Scripts
python analyze_group_posteriors.py
```

## Key Features

- **Cue-Based Expectation Learning**: Colored cues signal different difficulty levels
- **Adaptive Calibration**: QUEST+ staircase procedures find individual detection thresholds
- **Counterbalanced Design**: Learning order, color-difficulty mappings, and rotation conditions fully counterbalanced across 8 conditions
- **Two Rotation Conditions**: 0° (congruent) and 90° (rotated) visuomotor mapping
- **Fast Response Mode**: Early responses allowed during motion phase
- **Filtered Motion Library**: 1600 high-quality pre-recorded trajectories

## Data Output

Each participant generates:
- Main data file (`CDT_*.csv`) with trial-by-trial responses
- Kinematics file (`*_kinematics.csv`) with frame-by-frame movement data

## Analysis Pipeline

1. **Data Collection**: Run experiments
2. **QUEST Analysis**: `analyze_group_posteriors.py` for calibration analysis
3. **Main Analysis**: `mixed_model_desenderlab.py` for hypothesis testing
4. **DDM Analysis**: `ddm_analysis.py` for drift-diffusion modeling

## Documentation

- [Counterbalancing Scheme](docs/COUNTERBALANCING_SCHEME.md) - Details of the 8-condition counterbalancing
- [Experimenter Protocol](Main_Experiment/documents/experimenter_protocol.md) - Step-by-step running instructions

## License

This project is part of academic research. Please contact the authors before using.
