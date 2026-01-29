# CDT Online - Pavlovia Deployment Checklist

## Pre-Deployment Checklist

Before you start, verify you have:

- [ ] PsychoPy 2024.1.1 or later installed
- [ ] Pavlovia account created at [pavlovia.org](https://pavlovia.org)
- [ ] All resources converted (motion_library.json exists in `resources/` folder)
- [ ] `cdt_utils.js` file is in the `js/` folder
- [ ] `CDT_online.psyexp` opens correctly in PsychoPy Builder

## Step-by-Step Deployment

### Step 1: Verify Resources (5 minutes)

1. Check that these files exist in `CDT_Online/resources/`:
   ```
   ✅ motion_library.json (~3.5 MB)
   ✅ trajectory_pairs.json (~6 KB)
   ✅ cluster_centroids.json (optional)
   ✅ scaler_params.json (optional)
   ```

2. Verify `js/cdt_utils.js` exists and contains all functions

3. Open `CDT_online.psyexp` in PsychoPy Builder to ensure it loads

### Step 2: Configure PsychoPy Builder (10 minutes)

1. **Open the experiment:**
   - Launch PsychoPy
   - File > Open > `CDT_Online/CDT_online.psyexp`

2. **Set Experiment Settings:**
   - Click the gear icon (Experiment Settings)
   - **General tab:**
     - Units: `pix` (pixels) ✅
     - Window size: `1920, 1080` ✅
     - Full-screen: `True` ✅
   
   - **Online tab:**
     - Output path: `html`
     - Export HTML: `on Sync`
     - Completed URL: (leave blank for now, or add Prolific completion link)
     - Incomplete URL: (leave blank for now)

3. **Add Resources:**
   - In Experiment Settings > Online tab
   - Click "Resources" section
   - Click "+" to add files
   - Add ALL files from `resources/` folder:
     - `motion_library.json`
     - `trajectory_pairs.json`
     - `cluster_centroids.json` (if you want it)
     - `scaler_params.json` (if you want it)
   
   **OR** ensure the `resources/` folder is in the same directory as `CDT_online.psyexp` - PsychoPy will auto-detect it.

4. **Verify JavaScript is included:**
   - The experiment should reference `js/cdt_utils.js`
   - Check that the "Before JS Experiment" code component includes:
     ```javascript
     // Functions from cdt_utils.js are loaded globally
     ```
   - Make sure `index.html` includes: `<script src="js/cdt_utils.js"></script>`

### Step 3: Sync to Pavlovia (15-20 minutes)

1. **Connect to Pavlovia:**
   - In PsychoPy Builder: **Experiment > Sync with Pavlovia**
   - If first time: Log in with your Pavlovia/GitLab credentials
   - Authorize PsychoPy to access your Pavlovia account

2. **Create/Select Project:**
   - Choose "Create new project" (recommended for first time)
   - Project name: `CDT_online` (or your preferred name)
   - Description: "Control Detection Task - Online Version"
   - Click "Sync"

3. **Wait for Upload:**
   - Upload may take 5-10 minutes due to motion_library.json (3.5 MB)
   - Watch the progress bar
   - Don't close PsychoPy during upload

4. **Verify Sync Success:**
   - You should see "Sync successful" message
   - Check Pavlovia dashboard - your project should appear

### Step 4: Configure Project on Pavlovia (10 minutes)

1. **Go to Pavlovia Dashboard:**
   - Visit [pavlovia.org](https://pavlovia.org)
   - Log in
   - Find your project (`CDT_online`)

2. **Check Project Settings:**
   - Click on your project
   - Verify all files uploaded:
     - `CDT_online.js` (auto-generated)
     - `index.html`
     - `conditions.csv`
     - `js/cdt_utils.js`
     - `resources/motion_library.json`
     - `resources/trajectory_pairs.json`

3. **Set Project Status:**
   - Status dropdown: Select **"PILOTING"** (for testing)
   - This allows unlimited free testing

### Step 5: Test in Piloting Mode (30-60 minutes)

1. **Open Pilot Version:**
   - In Pavlovia project page, click **"Pilot"** button
   - Experiment opens in new browser tab

2. **Complete Full Test Session:**
   - Go through entire experiment:
     - [ ] Loading screen appears
     - [ ] Instructions display
     - [ ] Mouse tracking works smoothly
     - [ ] Shapes respond to mouse movement
     - [ ] Target shape follows mouse (with prop_self mixing)
     - [ ] Distractor moves with trajectory + linear bias
     - [ ] Response keys (A, S) work
     - [ ] Early response during motion works
     - [ ] Timeout after 5 seconds works
     - [ ] Confidence rating (1-4) works
     - [ ] Agency rating (1-7) works
     - [ ] End screen appears

3. **Check Browser Console:**
   - Press F12 (or Cmd+Option+I on Mac)
   - Look for errors in Console tab
   - Should see: "CDT utilities loaded - EXACT REPLICA of offline experiment"

4. **Test Different Conditions:**
   - Run multiple test sessions
   - Test both 0° and 90° angle conditions
   - Test both high and low expectation conditions
   - Verify ±90° randomization works (check applied_angle_bias in data)

### Step 6: Verify Data Output (10 minutes)

1. **Download Test Data:**
   - In Pavlovia project page, go to **"Data"** tab
   - Download the CSV file from your test session

2. **Check Data Columns:**
   Verify these columns exist (matching offline):
   ```
   ✅ participant
   ✅ target_snippet_id
   ✅ distractor_snippet_id
   ✅ phase
   ✅ angle_bias
   ✅ applied_angle_bias (should be ±90 when angle_bias=90)
   ✅ expect_level
   ✅ true_shape
   ✅ resp_shape
   ✅ accuracy
   ✅ rt_choice
   ✅ prop_used
   ✅ early_response
   ✅ mean_evidence
   ✅ sum_evidence
   ✅ var_evidence
   ✅ rt_frame
   ✅ num_frames_preRT
   ✅ mean_evidence_preRT
   ✅ sum_evidence_preRT
   ✅ var_evidence_preRT
   ✅ max_cum_evidence_preRT
   ✅ min_cum_evidence_preRT
   ✅ max_abs_cum_evidence_preRT
   ✅ prop_positive_evidence_preRT
   ✅ confidence_rating
   ✅ agency_rating
   ```

3. **Compare with Offline Data:**
   - Run one trial offline
   - Run same trial online
   - Compare evidence values (should be very similar, accounting for frame rate differences)

### Step 7: Fix Any Issues (if needed)

**Common Issues:**

1. **"Resources failed to load"**
   - Solution: Re-upload resources in PsychoPy Builder > Experiment Settings > Online > Resources
   - Or ensure `resources/` folder is in project root

2. **"cdt_utils.js not found"**
   - Solution: Check `index.html` includes: `<script src="js/cdt_utils.js"></script>`
   - Or manually add in PsychoPy Builder > Experiment Settings > Online > Resources

3. **Mouse tracking laggy**
   - Solution: Test in Chrome (best performance)
   - Check frame rate in browser console
   - Recommend participants use Chrome/Firefox

4. **Evidence values don't match offline**
   - Solution: Check browser console for errors
   - Verify `cdt_utils.js` loaded correctly
   - Compare frame-by-frame: offline vs online

### Step 8: Prepare for Production (15 minutes)

1. **Set Completion URLs:**
   - In PsychoPy Builder > Experiment Settings > Online:
     - **Completed URL**: Your Prolific completion link (if using Prolific)
     - **Incomplete URL**: Your Prolific incomplete link

2. **Test with Participant ID:**
   - Add `?participant=TEST123` to Pavlovia URL
   - Verify participant ID appears in data

3. **Set Project Status:**
   - Change from "PILOTING" to **"RUNNING"** when ready
   - ⚠️ This will use credits - make sure you have enough!

### Step 9: Launch (5 minutes)

1. **Get Experiment URL:**
   - In Pavlovia project page, copy the experiment URL
   - Format: `https://pavlovia.org/run/YOUR_USERNAME/CDT_online`

2. **Share with Participants:**
   - **Prolific**: Add URL to study, use `?participant={{%PROLIFIC_PID%}}`
   - **MTurk**: Add URL to HIT, use `?participant={{workerId}}`
   - **Direct**: Share URL directly

3. **Monitor Data:**
   - Check Pavlovia Dashboard regularly
   - Download data periodically
   - Monitor for errors in project logs

## Post-Deployment

### Daily Checks:
- [ ] Download new data
- [ ] Check for error messages in Pavlovia logs
- [ ] Verify participant completion rates
- [ ] Monitor credit usage

### Weekly Checks:
- [ ] Compare online vs offline data quality
- [ ] Check frame rate statistics (if logging)
- [ ] Review participant feedback (if collected)

## Cost Management

- **Pavlovia Credits**: £0.20 per participant
- **For 100 participants**: ~£20
- **Check institutional license**: Many universities have site licenses

## Support Resources

- **PsychoPy Forum**: [discourse.psychopy.org](https://discourse.psychopy.org)
- **Pavlovia Support**: [pavlovia.org/support](https://pavlovia.org/support)
- **PsychoJS Docs**: [psychopy.github.io/psychojs](https://psychopy.github.io/psychojs)

## Quick Reference

**Experiment URL Format:**
```
https://pavlovia.org/run/YOUR_USERNAME/CDT_online?participant=PARTICIPANT_ID
```

**Prolific Integration:**
```
https://pavlovia.org/run/YOUR_USERNAME/CDT_online?participant={{%PROLIFIC_PID%}}
```

**Status Levels:**
- **INACTIVE**: Project not running
- **PILOTING**: Free testing (unlimited)
- **RUNNING**: Data collection (uses credits)

---

**Ready to deploy?** Start with Step 1 and work through each step systematically!
