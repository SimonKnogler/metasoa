# Analysis Scripts Setup Guide

## Required Python Packages

To run the analysis scripts in this project, you need to install the following packages:

### Installation Methods

**Option 1: Install from requirements.txt (Recommended)**
```bash
pip install -r requirements.txt
```

**Option 2: Install packages individually**
```bash
pip install pandas numpy matplotlib scipy scikit-learn
```

### For PsychoPy Experiment Scripts
If you're running the main experiment (`CDT_windows_blockwise_fast_response.py`), you also need:
```bash
pip install psychopy
```

## Running Analysis Scripts

### Evidence-Accuracy Correlation Analysis
```bash
python Analysis_Scripts/evidence_accuracy_correlation.py
```

This will:
- Analyze the relationship between cumulative evidence and trial accuracy
- Generate a visualization with scatter plot and logistic regression
- Save outputs to `Main_Experiment/data/quest_group_analysis/`

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'sklearn'"

**Solution:** Install scikit-learn:
```bash
pip install scikit-learn
```

### Issue: IDE's run button doesn't work

**Possible causes:**
1. Your IDE is using a different Python environment
2. Packages are installed in a different environment

**Solutions:**

**Option 1: Use the batch file (Windows - Easiest)**
- Double-click `run_evidence_analysis.bat` in the project root
- This uses the correct Python environment automatically

**Option 2: Configure VS Code (If using VS Code)**
1. The `.vscode/settings.json` file has been created with the correct Python path
2. Reload VS Code (close and reopen)
3. Press F5 or click the run button
4. It should now use `C:\Users\knogl\Miniconda3\python.exe`

**Option 3: Manual Interpreter Selection**

### VS Code Setup
1. Open Command Palette (Ctrl+Shift+P / Cmd+Shift+P)
2. Type "Python: Select Interpreter"
3. Choose `C:\Users\knogl\Miniconda3\python.exe`
4. Try running again

### PyCharm Setup
1. File → Settings → Project → Python Interpreter
2. Click the gear icon → Add
3. Select "System Interpreter"
4. Browse to `C:\Users\knogl\Miniconda3\python.exe`
5. Click OK and try running again

**Verification:**
The script works from terminal, so if the run button still fails, it's definitely an IDE interpreter configuration issue.

