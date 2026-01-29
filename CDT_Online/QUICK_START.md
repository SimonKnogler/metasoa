# CDT Online - Quick Start Guide

## 🚀 Fast Track to Pavlovia (30 minutes)

### Prerequisites Check (2 min)
- [ ] PsychoPy 2024.1.1+ installed
- [ ] Pavlovia account at [pavlovia.org](https://pavlovia.org)
- [ ] Files ready: `CDT_online.psyexp`, `js/cdt_utils.js`, `resources/motion_library.json`

### Step 1: Open in PsychoPy Builder (2 min)
1. Launch PsychoPy
2. File > Open > `CDT_Online/CDT_online.psyexp`
3. Verify experiment loads without errors

### Step 2: Configure Online Settings (5 min)
1. Click gear icon (Experiment Settings)
2. **Online tab:**
   - Output path: `html`
   - Export HTML: `on Sync`
3. **Resources section:**
   - Click "+" and add:
     - `resources/motion_library.json`
     - `resources/trajectory_pairs.json`
   - **OR** ensure `resources/` folder is in same directory as `.psyexp`

### Step 3: Add JavaScript Utilities (5 min)
**Option A (Recommended):** Include as resource
1. In Experiment Settings > Online > Resources
2. Add `js/cdt_utils.js`
3. PsychoPy will auto-include it

**Option B:** Manual inclusion
1. After sync, edit generated `index.html` on Pavlovia
2. Add before `CDT_online.js`: `<script src="js/cdt_utils.js"></script>`

### Step 4: Sync to Pavlovia (10 min)
1. Experiment > Sync with Pavlovia
2. Log in with Pavlovia/GitLab credentials
3. Create new project: `CDT_online`
4. Wait for upload (5-10 min for 3.5 MB motion library)

### Step 5: Test in Piloting Mode (10 min)
1. On Pavlovia dashboard, set status to **PILOTING**
2. Click **Pilot** button
3. Complete full test session:
   - [ ] Mouse tracking works
   - [ ] Shapes move correctly
   - [ ] Responses work (A, S keys)
   - [ ] Ratings work (1-4, 1-7)
4. Check browser console (F12) for errors
5. Download test data and verify columns

### Step 6: Go Live (1 min)
1. Set status to **RUNNING**
2. Copy experiment URL
3. Share with participants!

## 🔧 Troubleshooting

**"cdt_utils.js not found"**
→ Add as resource in PsychoPy Builder, or manually edit `index.html` after sync

**"Resources failed to load"**
→ Re-upload `motion_library.json` in Resources section

**"Mouse tracking laggy"**
→ Test in Chrome (best performance), check frame rate in console

## 📋 Full Checklist

See `DEPLOYMENT_CHECKLIST.md` for detailed step-by-step instructions.

## 💰 Cost

- **Piloting**: FREE (unlimited testing)
- **Running**: £0.20 per participant
- **100 participants**: ~£20

## 🔗 Important URLs

- **Pavlovia**: [pavlovia.org](https://pavlovia.org)
- **PsychoPy Forum**: [discourse.psychopy.org](https://discourse.psychopy.org)
- **PsychoJS Docs**: [psychopy.github.io/psychojs](https://psychopy.github.io/psychojs)

---

**Ready?** Start with Step 1! 🎯
