# Counterbalancing Scheme - CDT Experiment

## Overview

The experiment uses **systematic counterbalancing** (not randomization) across 8 conditions. Each participant is deterministically assigned to one of 8 conditions based on their participant ID.

## Why Counterbalancing > Randomization?

**Advantages:**
- ✓ **Guaranteed balance** - Exactly equal numbers across all conditions
- ✓ **Systematic control** - All known confounds are controlled
- ✓ **Analyzable** - Can statistically test for order/color effects
- ✓ **Deterministic** - Same participant ID always gets same condition (reproducible)

**What's still randomized:**
- Trial order within each block (uses participant-specific seed)
- Which specific trajectories are used (uses participant-specific seed)

## 8 Counterbalancing Conditions

The counterbalancing index is calculated as: `participant_number % 8`

For non-numeric IDs (like "SIM", "test"), the ID is hashed to get a consistent number.

### Factor 1: Learning Order (Bit 0)
- **Even CB indices (0,2,4,6):** 0° first, then 90°
- **Odd CB indices (1,3,5,7):** 90° first, then 0°

### Factor 2: Color Palette Assignment (Bit 1)
- **CB indices 0,1,4,5:** First angle gets Blue/Green palette
- **CB indices 2,3,6,7:** First angle gets Red/Yellow palette

### Factor 3: Color-Difficulty Mapping (Bit 2)
- **CB indices 0,1,2,3:** Original color order (blue=hard, green=easy OR red=hard, yellow=easy)
- **CB indices 4,5,6,7:** Flipped color order (green=hard, blue=easy OR yellow=hard, red=easy)

**Note:** Second angle always uses the OTHER palette and FLIPPED color mapping for variety.

---

## Complete Counterbalancing Table

| Participant # | CB Index | Learning Order | First Angle Palette | First Angle Low (Hard) | First Angle High (Easy) | Second Angle Palette | Second Angle Low (Hard) | Second Angle High (Easy) |
|---------------|----------|----------------|---------------------|------------------------|-------------------------|----------------------|-------------------------|--------------------------|
| 1, 9, 17, ... | 1 | 90°→0° | Blue/Green | Blue | Green | Yellow/Red | Yellow | Red |
| 2, 10, 18, ... | 2 | 0°→90° | Red/Yellow | Red | Yellow | Green/Blue | Green | Blue |
| 3, 11, 19, ... | 3 | 90°→0° | Red/Yellow | Red | Yellow | Green/Blue | Green | Blue |
| 4, 12, 20, ... | 4 | 0°→90° | Blue/Green | Green | Blue | Red/Yellow | Red | Yellow |
| 5, 13, 21, ... | 5 | 90°→0° | Blue/Green | Green | Blue | Red/Yellow | Red | Yellow |
| 6, 14, 22, ... | 6 | 0°→90° | Red/Yellow | Yellow | Red | Blue/Green | Blue | Green |
| 7, 15, 23, ... | 7 | 90°→0° | Red/Yellow | Yellow | Red | Blue/Green | Blue | Green |
| 8, 16, 24, ... | 0 | 0°→90° | Blue/Green | Blue | Green | Yellow/Red | Yellow | Red |

---

## Examples

### Participant "1"
- CB Index: 1
- Learning order: [90, 0] (90° first, then 0°)
- First angle (90°): Blue (hard), Green (easy)
- Second angle (0°): Yellow (hard), Red (easy)

### Participant "2"
- CB Index: 2
- Learning order: [0, 90] (0° first, then 90°)
- First angle (0°): Red (hard), Yellow (easy)
- Second angle (90°): Green (hard), Blue (easy)

### Participant "test" (hashed)
- Will hash to a consistent number, then % 8
- Same assignment every time you run with "test"

---

## Testing Different Conditions

To quickly test different conditions during development:

```python
# Participant 1 → CB Index 1
# Participant 2 → CB Index 2
# Participant 3 → CB Index 3
# ...
# Participant 8 → CB Index 0
```

Simply use participant IDs 1-8 to see all 8 conditions!

---

## What You'll See When Running

```
============================================================
COUNTERBALANCING ASSIGNMENT:
  Participant: 1 → CB Index: 1/8
  Learning order: [90, 0]
  First angle (90°): ('blue', 'green') (low=hard, high=easy)
  Second angle (0°): ('yellow', 'red') (low=hard, high=easy)
============================================================
```

This is logged at startup and saved to your data file.

