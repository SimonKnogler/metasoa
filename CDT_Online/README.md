# CDT Online - Control Detection Task for Pavlovia

Online version of the Control Detection Task (CDT) experiment for deployment on Pavlovia.

## Overview

This experiment measures the sense of agency and metacognitive confidence in a control detection paradigm. Participants control one of two shapes via mouse movements and must identify which shape they control.

### Key Features

- **Real-time mouse tracking**: Shapes respond to mouse movements
- **Trajectory mixing**: Target shape mixes participant's input with pre-recorded trajectories
- **Angular rotation**: Optional rotation of mouse-to-movement mapping
- **Confidence & Agency ratings**: Collected after each trial

## Quick Start

1. **Convert motion library** (if not already done):
   ```bash
   python scripts/convert_motion_library.py
   ```

2. **Open in PsychoPy Builder**:
   - Open `CDT_online.psyexp`
   - Review experiment structure

3. **Sync to Pavlovia**:
   - Experiment > Sync with Pavlovia
   - Follow prompts to create/update project

4. **Test**:
   - Set project to PILOTING mode
   - Run through complete session
   - Check data output

See [PAVLOVIA_SETUP.md](PAVLOVIA_SETUP.md) for detailed instructions.

## File Structure

```
CDT_Online/
├── CDT_online.psyexp      # Main experiment file (PsychoPy Builder)
├── conditions.csv         # Trial conditions
├── index.html             # Web entry point
├── js/
│   └── cdt_utils.js       # JavaScript utilities for online version
├── resources/
│   ├── motion_library.json      # Converted trajectories (3.5 MB)
│   ├── trajectory_pairs.json    # Pre-computed pairs
│   ├── cluster_centroids.json   # Trajectory clustering data
│   └── scaler_params.json       # Normalization parameters
├── scripts/
│   └── convert_motion_library.py  # NPY to JSON converter
├── PAVLOVIA_SETUP.md      # Deployment guide
└── README.md              # This file
```

## Experiment Design

### Conditions (2x2 factorial)

| Factor | Levels |
|--------|--------|
| Expected Control | High, Low (cue color) |
| Angular Bias | 0°, 90° rotation |

### Trial Structure

1. **Fixation** (1s) - Colored cue indicates expected control
2. **Motion phase** (4s) - Move mouse, shapes respond
3. **Response** - Press A (square) or S (circle)
4. **Confidence** - Rate 1-4
5. **Agency** - Rate 1-7

### Data Columns

| Column | Description |
|--------|-------------|
| `participant` | Participant ID |
| `target_shape` | Which shape was controlled |
| `response_shape` | Participant's response |
| `accuracy` | 1 = correct, 0 = incorrect |
| `rt_choice` | Response time (seconds) |
| `prop_self` | Proportion of self-control |
| `angle_bias` | Rotation angle (degrees) |
| `mean_evidence` | Average sensory evidence |
| `confidence_rating` | 1-4 scale |
| `agency_rating` | 1-7 scale |

## Technical Notes

### Browser Compatibility

- **Recommended**: Chrome, Firefox
- **Supported**: Safari, Edge
- **Required**: WebGL, JavaScript enabled

### Performance Considerations

- Motion library is ~3.5 MB (optimized from full library)
- Pre-computed trajectory pairs reduce runtime computation
- Frame rate monitoring built-in for data quality checks

### Differences from Desktop Version

| Feature | Desktop | Online |
|---------|---------|--------|
| Frame rate | Consistent 60 Hz | Variable (browser-dependent) |
| Trajectories | 1600 | 800 (optimized subset) |
| Trajectory matching | Dynamic | Pre-computed |
| Kinematics logging | Every frame | Summary statistics |

## Requirements

- PsychoPy 2024.1.1 or later
- Pavlovia account
- Modern web browser for participants

## Citation

If you use this experiment, please cite:

```
[Your citation here]
```

## License

[Your license here]

## Contact

Simon Knogler - [your email]
