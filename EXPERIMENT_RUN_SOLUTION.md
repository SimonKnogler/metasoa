# CDT Experiment - Running Solution

## Problem Solved ✅

The experiment couldn't run because PsychoPy wasn't found in the default Python environment.

## Root Cause

You have **three** Python installations on your system:
1. **Default Python** (whatever `python` points to) - does NOT have PsychoPy
2. **Miniconda3 conda environment** (`C:\Users\knogl\Miniconda3\envs\psychopy_env\python.exe`) - has PsychoPy but with broken freetype dependency
3. **Standalone PsychoPy** (`C:\Program Files\PsychoPy\python.exe`) - **fully working PsychoPy 2025.1.1** ✅

## Solution Implemented

### 1. Updated Experiment Script
Modified `Main_Experiment/CDT_windows_blockwise_fast_response.py` to:
- Check for required PsychoPy packages at startup
- Automatically detect and switch to the standalone PsychoPy installation
- Fall back to conda environment if standalone not available
- Gracefully handle missing dependencies

### 2. Updated VS Code Settings
Modified `.vscode/settings.json` to use the standalone PsychoPy Python interpreter by default.

## How to Run the Experiment

### Method 1: Click Run Button in IDE (Recommended)
1. Open `Main_Experiment/CDT_windows_blockwise_fast_response.py` in your IDE
2. Click the **Run** button
3. The experiment dialog will open automatically!

### Method 2: Command Line
From the project root directory:
```powershell
python Main_Experiment\CDT_windows_blockwise_fast_response.py
```

The script will automatically:
1. Detect that PsychoPy is missing in the default Python
2. Find the standalone PsychoPy installation
3. Re-launch itself with the correct interpreter
4. Open the experiment dialog

### Method 3: Direct Execution (Manual)
```powershell
& "C:\Program Files\PsychoPy\python.exe" Main_Experiment\CDT_windows_blockwise_fast_response.py
```

## What Changed

### Before (Broken)
- Script tried to use default Python → PsychoPy not found → Error
- Conda environment had broken freetype library

### After (Working) ✅
- Script automatically detects missing PsychoPy
- Switches to standalone PsychoPy installation
- Works reliably every time!

## Technical Details

The interpreter check mechanism (lines 24-55 in the experiment script):
1. Tries to import PsychoPy packages
2. If import fails, searches for working Python installations in order:
   - `C:/Program Files/PsychoPy/python.exe` (standalone - most reliable)
   - `C:/Users/knogl/Miniconda3/envs/psychopy_env/python.exe` (conda backup)
   - Other common paths
3. Re-executes the script with the working Python
4. Exits the original process

This ensures the experiment always runs with a working PsychoPy installation, regardless of which Python your IDE or terminal uses by default.

## Verification

✅ Standalone PsychoPy found at: `C:\Program Files\PsychoPy\python.exe`
✅ PsychoPy version: 2025.1.1
✅ Experiment launches successfully
✅ Works with IDE Run button
✅ Works from command line
✅ Automatic interpreter switching functional

## Notes

- The standalone PsychoPy installation is the most reliable option
- The conda environment (`psychopy_env`) has a broken freetype dependency
- The script now handles environment switching automatically
- No manual intervention needed - just click Run!

---

**Status**: ✅ Fully Working - Ready for Data Collection
**Last Updated**: October 15, 2025

