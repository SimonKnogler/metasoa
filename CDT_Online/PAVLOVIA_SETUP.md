# Pavlovia Setup Guide for CDT Online

This guide walks you through deploying the Control Detection Task (CDT) experiment online via Pavlovia.

## Prerequisites

1. **PsychoPy installed** (version 2024.1.1 or later recommended)
2. **Pavlovia account** - Create one at [pavlovia.org](https://pavlovia.org)
3. **GitLab account** - Pavlovia uses GitLab for project management

## Step 1: Prepare Your Pavlovia Account

1. Go to [pavlovia.org](https://pavlovia.org)
2. Click "Sign In" and create an account (or sign in with GitLab)
3. Once logged in, go to Dashboard > Settings
4. Note your username - you'll need it for syncing

## Step 2: Open the Experiment in PsychoPy Builder

1. Launch PsychoPy
2. Open `CDT_Online/CDT_online.psyexp`
3. Review the experiment structure:
   - `init_resources` - Loads motion library
   - `instructions` - Shows task instructions
   - `trial` - Main trial with mouse tracking
   - `confidence` - Confidence rating
   - `agency` - Agency rating
   - `end` - Thank you screen

## Step 3: Configure Pavlovia Settings

In PsychoPy Builder:

1. Go to **Experiment Settings** (gear icon)
2. Click the **Online** tab
3. Set:
   - **Output path**: `html`
   - **Export HTML**: `on Sync`
   - **Completed URL**: Your completion URL (e.g., Prolific completion link)
   - **Incomplete URL**: Your incomplete URL

## Step 4: Add Resources

The experiment needs these resource files uploaded to Pavlovia:

1. Go to **Experiment Settings > Online > Resources**
2. Add these files from the `resources/` folder:
   - `motion_library.json` (3.5 MB)
   - `trajectory_pairs.json` (6 KB)
   - `cluster_centroids.json` (optional)
   - `scaler_params.json` (optional)

Or, in the experiment folder, ensure the `resources/` folder contains these files.

## Step 5: Sync to Pavlovia

1. In PsychoPy Builder, click **Experiment > Sync with Pavlovia**
2. If prompted, log in to your Pavlovia account
3. Choose to create a **new project** or sync to an existing one
4. Wait for the upload to complete (may take a few minutes due to motion library size)

## Step 6: Configure the Pavlovia Project

On Pavlovia:

1. Go to your Dashboard
2. Find your project (e.g., `CDT_online`)
3. Click on the project to open settings
4. Set the **Status**:
   - `INACTIVE` - Project is not running
   - `PILOTING` - Free testing (no credits used)
   - `RUNNING` - Data collection (uses credits)

## Step 7: Test in Piloting Mode

1. Set status to **PILOTING**
2. Click **Pilot** to open the experiment in a new tab
3. Test thoroughly:
   - Check mouse tracking works
   - Verify shapes move correctly
   - Test all response keys
   - Complete a full session

### Testing Checklist

- [ ] Loading screen appears and disappears
- [ ] Instructions display correctly
- [ ] Mouse tracking is responsive
- [ ] Shapes follow mouse (target) and trajectory (distractor)
- [ ] Response keys (A, S) work
- [ ] Confidence rating (1-4) works
- [ ] Agency rating (1-7) works
- [ ] Data saves correctly

## Step 8: Check Data Output

1. After completing a test session, go to your Pavlovia Dashboard
2. Click on your project
3. Go to **Data** tab
4. Download the CSV file
5. Verify columns:
   - `participant`
   - `target_shape`
   - `response_shape`
   - `accuracy`
   - `rt_choice`
   - `prop_self`
   - `angle_bias`
   - `mean_evidence`
   - `confidence_rating`
   - `agency_rating`

## Step 9: Run the Experiment

1. Purchase credits on Pavlovia (or use institutional license)
2. Set status to **RUNNING**
3. Share the experiment URL with participants

### Recruitment Integration

**Prolific:**
1. Create a study on Prolific
2. Use your Pavlovia URL as the study link
3. Add `?participant={{%PROLIFIC_PID%}}` to the URL
4. Set completion URL in Pavlovia settings

**MTurk:**
1. Create a HIT on MTurk
2. Use Pavlovia URL with `?participant={{workerId}}`
3. Configure completion codes

## Troubleshooting

### "Resources failed to load"
- Check that all JSON files are in the `resources/` folder
- Verify file sizes aren't too large (motion_library.json should be ~3.5 MB)
- Try clearing browser cache

### "Mouse tracking is laggy"
- Recommend participants use Chrome or Firefox
- Suggest closing other browser tabs
- Frame rate issues are logged - check console

### "Data not saving"
- Ensure experiment status is RUNNING (not PILOTING for real data)
- Check browser console for errors
- Verify Pavlovia credits are available

### "Experiment won't start"
- Check browser console for JavaScript errors
- Verify PsychoJS library is loading
- Test in a different browser

## File Structure

```
CDT_Online/
├── CDT_online.psyexp      # PsychoPy Builder file
├── CDT_online.js          # Auto-generated JavaScript (after sync)
├── index.html             # Entry point
├── conditions.csv         # Trial conditions
├── js/
│   └── cdt_utils.js       # Custom JavaScript utilities
├── resources/
│   ├── motion_library.json
│   ├── trajectory_pairs.json
│   ├── cluster_centroids.json
│   └── scaler_params.json
└── scripts/
    └── convert_motion_library.py  # Conversion script
```

## Cost Estimation

Pavlovia charges per participant session:
- £0.20 per participant (as of 2024)
- For 100 participants: ~£20

Many institutions have Pavlovia site licenses - check with your university!

## Support

- **PsychoPy Forum**: [discourse.psychopy.org](https://discourse.psychopy.org)
- **Pavlovia Support**: [pavlovia.org/support](https://pavlovia.org/support)
- **PsychoJS Documentation**: [psychopy.github.io/psychojs](https://psychopy.github.io/psychojs)
