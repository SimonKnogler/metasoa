# Quick Start: Running the Evidence-Accuracy Analysis

## ✅ Confirmed Working Methods

### Method 1: Double-click the Batch File (EASIEST)
1. Open File Explorer
2. Navigate to: `C:\Users\knogl\Desktop\CDT_Experiment_Package`
3. Double-click `run_evidence_analysis.bat`
4. Analysis will run and show results in a window
5. Press any key to close when done

### Method 2: Run from Terminal
```bash
cd C:\Users\knogl\Desktop\CDT_Experiment_Package
python Analysis_Scripts\evidence_accuracy_correlation.py
```

## 🔧 Fixing the IDE Run Button

Your IDE is using a different Python interpreter than the one where packages are installed.

### If Using VS Code:
1. **Close and reopen VS Code** (important!)
2. The `.vscode/settings.json` has been configured to use the correct Python
3. Try the run button again - it should work now

**OR manually select interpreter:**
1. Press `Ctrl+Shift+P`
2. Type: "Python: Select Interpreter"
3. Choose: `C:\Users\knogl\Miniconda3\python.exe`
4. Try running again

### If Using PyCharm:
1. File → Settings → Project: CDT_Experiment_Package → Python Interpreter
2. Click gear icon → Add
3. Select "System Interpreter"
4. Browse to: `C:\Users\knogl\Miniconda3\python.exe`
5. Click OK
6. Try running again

## 📊 What the Analysis Does

The script analyzes the relationship between:
- **Cumulative evidence** (how much evidence accumulated before response)
- **Trial accuracy** (correct vs incorrect responses)

**Outputs:**
- `evidence_accuracy_correlation.png` - Visualization with scatter plot and logistic regression
- `evidence_accuracy_correlation_stats.txt` - Statistical summary

**Location:** `Main_Experiment/data/quest_group_analysis/`

## ✓ Verification

The script is **confirmed working** from:
- ✅ Terminal/Command Prompt
- ✅ Batch file launcher
- ⚠️ IDE run button (needs interpreter configuration)

**Current Python:** `C:\Users\knogl\Miniconda3\python.exe`  
**Packages installed:** ✅ pandas, numpy, matplotlib, scipy, scikit-learn

If the IDE run button still doesn't work after configuring the interpreter, it's purely an IDE configuration issue - the script itself is fully functional!

