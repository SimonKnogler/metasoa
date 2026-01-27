# Experimenter Protocol

## Control Detection Task (CDT) - Blockwise Fast Response Version

**Version**: 1.0  
**Last Updated**: [INSERT DATE]

---

## Table of Contents

1. [Overview](#overview)
2. [Pre-Session Checklist](#pre-session-checklist)
3. [Participant Arrival and Setup](#participant-arrival-and-setup)
4. [Running the Experiment](#running-the-experiment)
5. [During the Experiment](#during-the-experiment)
6. [Post-Session Procedures](#post-session-procedures)
7. [Data Backup](#data-backup)
8. [Troubleshooting](#troubleshooting)
9. [Emergency Procedures](#emergency-procedures)

---

## Overview

### Experiment Structure

| Block | Phase | Angle Condition | Approximate Trials | Duration |
|:-----:|:------|:----------------|:------------------:|:--------:|
| 1 | Calibration | Both 0° and 90° interleaved | ~80-100 | ~15 min |
| 2 | Learning | First angle (counterbalanced) | ~60 | ~10 min |
| 3 | Test | First angle | ~150 | ~20 min |
| 4 | Learning | Second angle | ~60 | ~10 min |
| 5 | Test | Second angle | ~150 | ~20 min |

**Total Duration**: ~45-60 minutes (full mode) or ~15-20 minutes (check mode)

### Counterbalancing

The experiment automatically counterbalances based on participant ID number:
- **Learning order**: 0° first vs. 90° first
- **Color palette**: Blue/Green vs. Red/Yellow for first angle
- **Color-difficulty mapping**: Which color = easy vs. hard

Participant IDs 1-8 cycle through all 8 counterbalancing conditions. Assign participant IDs sequentially to ensure balanced groups.

---

## Pre-Session Checklist

### Day Before / Start of Testing Day

| ☐ | Task |
|:-:|:-----|
| ☐ | Check computer is functioning and has Python/PsychoPy installed |
| ☐ | Verify mouse is working properly (standard wired mouse recommended) |
| ☐ | Ensure monitor is set to correct resolution (1920×1080 recommended) |
| ☐ | Print sufficient copies of: Consent forms, Demographics questionnaires, Debrief sheets |
| ☐ | Prepare participant payment/credits (if applicable) |
| ☐ | Check available disk space for data storage |

### Before Each Session

| ☐ | Task |
|:-:|:-----|
| ☐ | Close unnecessary applications on testing computer |
| ☐ | Disable notifications, screen savers, and auto-updates |
| ☐ | Set room lighting to comfortable level (consistent across participants) |
| ☐ | Position chair and mouse at comfortable height |
| ☐ | Have blank consent form and demographics questionnaire ready |
| ☐ | Determine next participant ID number (check data folder) |
| ☐ | Note scheduled participant name and time in log book |

---

## Participant Arrival and Setup

### Step 1: Welcome and Seating (~2 minutes)

1. Greet participant warmly
2. Seat them comfortably at the testing computer
3. Adjust chair height so arm rests comfortably for mouse use
4. Ask if they need water or to use the restroom before starting

### Step 2: Information and Consent (~5 minutes)

1. Provide **Participant Information Sheet**
2. Allow time to read (do not rush)
3. Ask: *"Do you have any questions about the study?"*
4. Address any questions fully
5. If participant agrees:
   - Provide **Consent Form**
   - Have them initial each box and sign
   - Researcher signs and dates
   - Give participant a copy of both documents to keep

### Step 3: Demographics Questionnaire (~3 minutes)

1. Assign **Participant ID** (next sequential number)
2. Write Participant ID on consent form and questionnaire
3. Have participant complete demographics questionnaire
4. Review responses for eligibility:
   - Age 18+
   - Normal/corrected vision
   - Can use mouse with dominant hand
   - No exclusionary conditions

### Step 4: Eligibility Check

**If participant is eligible**: Proceed to experiment

**If participant is NOT eligible**:
1. Thank them for their time
2. Explain (sensitively) that they don't meet the criteria for this particular study
3. Provide any promised compensation for their time
4. Do NOT run the experiment

---

## Running the Experiment

### Step 1: Launch the Experiment

1. Open command prompt or terminal in the experiment directory:
   ```
   cd C:\Users\knogl\Desktop\CDT_Experiment_Package\Main_Experiment
   ```

2. Run the experiment script:
   ```
   python CDT_windows_blockwise_fast_response.py
   ```

### Step 2: Participant Dialog

When the dialog box appears:

| Field | Action |
|:------|:-------|
| **participant** | Enter participant ID number (e.g., "1", "2", "15") |
| **session** | Leave as "001" unless running multiple sessions |
| **simulate** | Leave UNCHECKED (simulation mode is for testing only) |
| **check_mode** | Leave UNCHECKED for full experiment; CHECK for quick test run |

Click **OK** to start.

### Step 3: Verbal Instructions

Before the participant starts reading on-screen instructions, provide a brief verbal overview:

> *"In this experiment, you'll see two shapes moving on the screen—a square and a circle. Your job is to figure out which one is following your mouse movements. Press 'A' if you think it's the square, or 'S' if you think it's the circle. Try to respond as quickly and accurately as you can. There will be breaks throughout. Do you have any questions before we begin?"*

### Step 4: Monitor the Start

- Let participant read through on-screen instructions at their own pace
- Confirm they press SPACE to advance through instruction screens
- Be available for questions but avoid hovering

---

## During the Experiment

### Monitoring Guidelines

| Do | Don't |
|:---|:------|
| Stay nearby but unobtrusive | Hover over participant's shoulder |
| Be available for questions | Interrupt during trials |
| Monitor for technical issues | Comment on performance |
| Note any unusual events | Leave the room completely |

### Break Screens

- Automatic 30-second breaks occur every 100 trials
- Longer 60-second countdown breaks between blocks
- Participant can take additional time at break screens if needed
- If participant requests additional break, note approximate time

### Handling Questions During Experiment

**For task-related questions**:
> *"Just do your best to identify which shape you're controlling. If you're unsure, make your best guess."*

**For questions about ratings**:
> *"Use whatever interpretation feels natural to you. There's no right or wrong answer."*

**Avoid**: Providing hints about the manipulation or expected results

### Signs of Distress or Fatigue

If participant appears distressed, uncomfortable, or excessively fatigued:
1. Pause at next break screen
2. Ask: *"How are you feeling? Would you like to take a longer break or stop?"*
3. Remind them they can withdraw at any time
4. If they continue, note the event in the log

### Emergency Quit

If you need to stop the experiment immediately:
- Press **ESCAPE** key at any time
- Data collected up to that point will be auto-saved

---

## Post-Session Procedures

### Step 1: Experiment Completion

When the final "Thank you" screen appears:
1. Let participant press SPACE to close
2. If window doesn't close, press ESCAPE

### Step 2: Debrief (~5 minutes)

1. Provide **Debrief Sheet**
2. Briefly explain the study purpose verbally:
   > *"This study looked at how we detect when we're in control of things. The colored cues were associated with different difficulty levels, and we were interested in whether those expectations affected your sense of control."*
3. Ask: *"Do you have any questions about the study?"*
4. Give participant the debrief sheet to keep

### Step 3: Compensation

1. Provide payment/credits as appropriate
2. Have participant sign receipt (if required)
3. Thank them for their participation

### Step 4: Documentation

Complete the following for each participant:

| ☐ | Task |
|:-:|:-----|
| ☐ | Record participant ID in master log |
| ☐ | Note any unusual events or comments |
| ☐ | File consent form securely (separate from data) |
| ☐ | File demographics questionnaire |
| ☐ | Verify data files were created |

---

## Data Backup

### Data File Locations

After each session, verify these files exist in:
`Main_Experiment/data/subjects/`

| File | Description |
|:-----|:------------|
| `CDT_v2_blockwise_fast_response_[ID].csv` | Main trial data |
| `CDT_v2_blockwise_fast_response_[ID]_kinematics.csv` | Frame-by-frame movement data |

### Backup Procedure

**After each participant**:

1. Verify data files exist and have reasonable file sizes:
   - Main CSV: ~100-500 KB
   - Kinematics CSV: ~5-20 MB

2. Copy to backup location:
   ```
   xcopy "data\subjects\CDT_v2_blockwise_fast_response_[ID]*" "[BACKUP_LOCATION]\"
   ```

3. Log backup completion in master log

**End of testing day**:

1. Backup entire `data/subjects/` folder to:
   - External hard drive
   - Institutional cloud storage (if approved for data storage)
   - Both if possible

2. Verify backup integrity by spot-checking file sizes

---

## Troubleshooting

### Common Issues and Solutions

#### Experiment won't start

| Symptom | Solution |
|:--------|:---------|
| "ModuleNotFoundError: psychopy" | Use correct Python environment (PsychoPy standalone or conda env) |
| "No module named numpy/pandas" | Install missing packages: `pip install numpy pandas` |
| Dialog appears but nothing happens | Check for errors in terminal, restart script |
| Screen flashes and closes | Check terminal for error messages |

#### During experiment

| Symptom | Solution |
|:--------|:---------|
| Screen freezes | Wait 10 seconds; if frozen, press ESCAPE, note trial number, restart |
| Mouse not responding | Check USB connection, try different port |
| Shapes not moving | Participant may not be moving mouse—remind them to move continuously |
| Display looks wrong | Check resolution is 1920×1080 |

#### Data issues

| Symptom | Solution |
|:--------|:---------|
| No data file created | Check terminal for save errors; data should auto-save on quit |
| File exists but is empty | Experiment may have crashed before first trial; check for error messages |
| Duplicate participant ID | Files are auto-numbered (e.g., `_1`, `_2`); note which is valid in log |

### Error Messages

| Error | Meaning | Action |
|:------|:--------|:-------|
| "Motion pool not found" | Missing stimulus files | Check `Motion_Library/` folder exists with `.npy` files |
| "core.quit()" | Normal experiment exit | Not an error |
| "Escape pressed" | Experimenter/participant quit | Normal if intentional; note in log |

---

## Emergency Procedures

### Fire Alarm / Building Evacuation

1. Press ESCAPE to save data
2. Close laptop/turn off monitor (do not wait for shutdown)
3. Guide participant to nearest exit
4. Follow building evacuation procedures
5. Do not return until all-clear given

### Participant Medical Emergency

1. Press ESCAPE to save data
2. Call emergency services if needed
3. Contact building security/first aiders
4. Stay with participant until help arrives
5. Document incident fully afterward

### Participant Distress

If participant becomes upset:
1. Pause experiment at next convenient point
2. Offer to stop the session
3. Remind them participation is voluntary
4. If appropriate, provide information about support services
5. Do not pressure them to continue

---

## Appendix: Quick Reference Card

### Key Commands

| Key | Function |
|:---:|:---------|
| SPACE | Advance through instructions / Continue after break |
| A | Respond "Square" |
| S | Respond "Circle" |
| 1-4 | Confidence rating |
| 1-7 | Agency rating |
| ESCAPE | Quit and save data |

### Experimenter Checklist (One Page)

**BEFORE**:
- [ ] Computer ready, notifications off
- [ ] Next participant ID known
- [ ] Consent form and questionnaire printed
- [ ] Run script: `python CDT_windows_blockwise_fast_response.py`

**DURING**:
- [ ] Enter participant ID when prompted
- [ ] check_mode = UNCHECKED for real data
- [ ] Monitor unobtrusively
- [ ] Note any issues in log

**AFTER**:
- [ ] Give debrief sheet
- [ ] Provide compensation
- [ ] Verify data saved
- [ ] Backup data
- [ ] File consent form securely

---

*Version: 1.0*  
*Date: [INSERT DATE]*

