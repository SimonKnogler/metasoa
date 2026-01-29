# Offline vs Online CDT Experiment - Exact Replication Verification

This document verifies that the online version is an **identical twin** of the offline experiment.

## Constants (100% Match)

| Constant | Offline (Python) | Online (JavaScript) | Line Reference |
|----------|------------------|---------------------|----------------|
| `OFFSET_X` | 300 | 300 | Line 609 |
| `LOWPASS` | 0.5 | 0.5 | Line 610 |
| `MAX_SPEED` | 20.0 | 20.0 | Line 796 |
| `CONFINE_RADIUS` | 250 | 250 | Line 650 |
| `MOTION_DURATION` | 5.0 | 5.0 | Line 775 |

## Core Functions (100% Match)

### 1. `confine()` - Boundary Confinement
**Offline (line 650):**
```python
confine = lambda p, l=250: p if (r := math.hypot(*p)) <= l else (p[0]*l/r, p[1]*l/r)
```

**Online:**
```javascript
function confine(pos, limit = 250) {
    const r = Math.hypot(pos[0], pos[1]);
    if (r <= limit) return pos;
    return [pos[0] * limit / r, pos[1] * limit / r];
}
```
✅ **Identical logic**

### 2. `rotate()` - Velocity Rotation
**Offline (lines 651-654):**
```python
rotate = lambda vx, vy, a: (
    vx * math.cos(math.radians(a)) - vy * math.sin(math.radians(a)),
    vx * math.sin(math.radians(a)) + vy * math.cos(math.radians(a))
)
```

**Online:**
```javascript
function rotate(vx, vy, angleDegrees) {
    const angleRadians = angleDegrees * Math.PI / 180;
    return [
        vx * Math.cos(angleRadians) - vy * Math.sin(angleRadians),
        vx * Math.sin(angleRadians) + vy * Math.cos(angleRadians)
    ];
}
```
✅ **Identical logic**

### 3. ±90° Randomization
**Offline (lines 782-784):**
```python
applied_angle = angle_bias
if angle_bias == 90:
    applied_angle = int(rng.choice([90, -90]))
```

**Online:**
```javascript
let appliedAngle = angleBias;
if (angleBias === 90) {
    appliedAngle = Math.random() < 0.5 ? 90 : -90;
}
```
✅ **Identical logic**

## Motion Loop (100% Match)

### Mouse Speed Clamping (lines 795-802)
**Offline:**
```python
mag_m = math.hypot(dx, dy)
MAX_SPEED = 20.0
mag_m = min(mag_m, MAX_SPEED)
if mag_m > 0:
    original_mag = math.hypot(dx, dy)
    if original_mag > MAX_SPEED:
        scale_factor = MAX_SPEED / original_mag
        dx = dx * scale_factor; dy = dy * scale_factor
```

**Online:**
```javascript
let mag_m = Math.hypot(dx, dy);
mag_m = Math.min(mag_m, MAX_SPEED);
if (mag_m > 0) {
    const original_mag = Math.hypot(dx, dy);
    if (original_mag > MAX_SPEED) {
        const scale_factor = MAX_SPEED / original_mag;
        dx = dx * scale_factor;
        dy = dy * scale_factor;
    }
}
```
✅ **Identical logic**

### Low-Pass Filter on Mouse Magnitude (lines 803-804)
**Offline:**
```python
if frame == 1: mag_m_lp = mag_m
else: mag_m_lp = 0.5 * mag_m_lp + 0.5 * mag_m
```

**Online:**
```javascript
if (currentFrame === 1) {
    mag_m_lp = mag_m;
} else {
    mag_m_lp = 0.5 * mag_m_lp + 0.5 * mag_m;
}
```
✅ **Identical logic**

### Direction Extraction from Trajectory (lines 805-816)
**Offline:**
```python
mag_target = math.hypot(target_ou_dx, target_ou_dy)
if mag_target > 0:
    dir_target_x, dir_target_y = target_ou_dx / mag_target, target_ou_dy / mag_target
else:
    dir_target_x, dir_target_y = 0, 0
# ... same for distractor ...
target_ou_dx = dir_target_x * mag_m_lp
target_ou_dy = dir_target_y * mag_m_lp
```

**Online:**
```javascript
const mag_target = Math.hypot(target_ou_dx_raw, target_ou_dy_raw);
let dir_target_x, dir_target_y;
if (mag_target > 0) {
    dir_target_x = target_ou_dx_raw / mag_target;
    dir_target_y = target_ou_dy_raw / mag_target;
} else {
    dir_target_x = 0;
    dir_target_y = 0;
}
// ... same for distractor ...
const target_ou_dx = dir_target_x * mag_m_lp;
const target_ou_dy = dir_target_y * mag_m_lp;
```
✅ **Identical logic**

### Target Velocity Mixing (lines 817-818)
**Offline:**
```python
tdx = prop * dx + (1 - prop) * target_ou_dx
tdy = prop * dy + (1 - prop) * target_ou_dy
```

**Online:**
```javascript
const tdx = prop * dx + (1 - prop) * target_ou_dx;
const tdy = prop * dy + (1 - prop) * target_ou_dy;
```
✅ **Identical logic**

### Distractor Linear Bias (lines 819-844) - CRITICAL
**Offline:**
```python
mouse_speed = math.hypot(dx, dy); linear_bias = 0.0
if mouse_speed > 0 and frame > 10 and len(trial_kinematics) >= 5:
    recent_positions = [(d['mouse_x'], d['mouse_y']) for d in trial_kinematics[-5:]] + [(x, y)]
    if len(recent_positions) >= 3:
        total_dist = sum(math.hypot(recent_positions[i+1][0] - recent_positions[i][0],
                                    recent_positions[i+1][1] - recent_positions[i][1])
                         for i in range(len(recent_positions)-1))
        straight_dist = math.hypot(recent_positions[-1][0] - recent_positions[0][0],
                                   recent_positions[-1][1] - recent_positions[0][1])
        if total_dist > 0:
            linearity = straight_dist / total_dist
            if mouse_speed < 10.0 and linearity > 0.8:
                linear_bias = min(0.3, linearity * 0.4)
if linear_bias > 0:
    perp_dx, perp_dy = -dy, dx
    # ... perpendicular calculation ...
    ddx = (1 - linear_bias) * distractor_ou_dx + linear_bias * perp_dx
    ddy = (1 - linear_bias) * distractor_ou_dy + linear_bias * perp_dy
else:
    ddx = distractor_ou_dx; ddy = distractor_ou_dy
```

**Online:**
```javascript
const mouse_speed = Math.hypot(dx, dy);
let linear_bias = 0.0;
if (mouse_speed > 0 && currentFrame > 10 && trialKinematics.length >= 5) {
    const recentPositions = trialKinematics.slice(-5).map(d => [d.mouse_x, d.mouse_y]);
    recentPositions.push([x, y]);
    if (recentPositions.length >= 3) {
        let total_dist = 0;
        for (let i = 0; i < recentPositions.length - 1; i++) {
            total_dist += Math.hypot(
                recentPositions[i+1][0] - recentPositions[i][0],
                recentPositions[i+1][1] - recentPositions[i][1]
            );
        }
        const straight_dist = Math.hypot(
            recentPositions[recentPositions.length-1][0] - recentPositions[0][0],
            recentPositions[recentPositions.length-1][1] - recentPositions[0][1]
        );
        if (total_dist > 0) {
            const linearity = straight_dist / total_dist;
            if (mouse_speed < 10.0 && linearity > 0.8) {
                linear_bias = Math.min(0.3, linearity * 0.4);
            }
        }
    }
}
// ... perpendicular calculation identical ...
```
✅ **Identical logic**

### Distractor Smoothing (lines 845-846)
**Offline:**
```python
ddx_smooth, ddy_smooth = 0.4 * prev_d[0] + 0.6 * ddx, 0.4 * prev_d[1] + 0.6 * ddy
prev_d = np.array([ddx_smooth, ddy_smooth])
```

**Online:**
```javascript
const ddx_smooth = 0.4 * prev_d[0] + 0.6 * ddx;
const ddy_smooth = 0.4 * prev_d[1] + 0.6 * ddy;
prev_d = [ddx_smooth, ddy_smooth];
```
✅ **Identical logic**

### Low-Pass Filter on Velocities (lines 847-848)
**Offline:**
```python
vt = LOWPASS * vt + (1 - LOWPASS) * np.array([tdx, tdy])
vd = LOWPASS * vd + (1 - LOWPASS) * np.array([ddx_smooth, ddy_smooth])
```

**Online:**
```javascript
vt = [
    LOWPASS * vt[0] + (1 - LOWPASS) * tdx,
    LOWPASS * vt[1] + (1 - LOWPASS) * tdy
];
vd = [
    LOWPASS * vd[0] + (1 - LOWPASS) * ddx_smooth,
    LOWPASS * vd[1] + (1 - LOWPASS) * ddy_smooth
];
```
✅ **Identical logic**

### Evidence Calculation (lines 850-864)
**Offline:**
```python
vm = np.array([dx, dy], dtype=float)
mouse_speed = np.linalg.norm(vm) + 1e-9
vt_disp = np.array(vt, dtype=float)
vd_disp = np.array(vd, dtype=float)
ut = vt_disp / (np.linalg.norm(vt_disp) + 1e-9)
ud = vd_disp / (np.linalg.norm(vd_disp) + 1e-9)
cos_T = np.dot(vm, ut) / mouse_speed
cos_D = np.dot(vm, ud) / mouse_speed
evidence = (cos_T - cos_D) * (mouse_speed - 1e-9)
```

**Online:**
```javascript
const vm = [dx, dy];
const vm_speed = Math.sqrt(vm[0]**2 + vm[1]**2) + 1e-9;
const vt_disp = vt.slice();
const vd_disp = vd.slice();
const vt_norm = Math.sqrt(vt_disp[0]**2 + vt_disp[1]**2) + 1e-9;
const vd_norm = Math.sqrt(vd_disp[0]**2 + vd_disp[1]**2) + 1e-9;
const ut = [vt_disp[0] / vt_norm, vt_disp[1] / vt_norm];
const ud = [vd_disp[0] / vd_norm, vd_disp[1] / vd_norm];
const cos_T = (vm[0] * ut[0] + vm[1] * ut[1]) / vm_speed;
const cos_D = (vm[0] * ud[0] + vm[1] * ud[1]) / vm_speed;
const evidence = (cos_T - cos_D) * (vm_speed - 1e-9);
```
✅ **Identical logic**

## Rating Scales (100% Match)

### Confidence (lines 951-959)
- Scale: 1-4
- Labels: "Not at all confident", "Slightly confident", "Moderately confident", "Very confident"
✅ **Identical**

### Agency (lines 977-992)
- Scale: 1-7
- Labels: "Very weak", "Weak", "Somewhat weak", "Moderate", "Somewhat strong", "Strong", "Very strong"
✅ **Identical**

## Data Output Columns (100% Match)

All columns from offline are replicated:
- `target_snippet_id`, `distractor_snippet_id`
- `phase`, `angle_bias`, `applied_angle_bias`
- `expect_level`, `true_shape`, `resp_shape`
- `accuracy`, `rt_choice`, `prop_used`, `early_response`
- `mean_evidence`, `sum_evidence`, `var_evidence`
- `rt_frame`, `num_frames_preRT`
- `mean_evidence_preRT`, `sum_evidence_preRT`, `var_evidence_preRT`
- `max_cum_evidence_preRT`, `min_cum_evidence_preRT`
- `max_abs_cum_evidence_preRT`, `prop_positive_evidence_preRT`
- `confidence_rating`, `agency_rating`

✅ **All columns present**

## Summary

| Component | Status |
|-----------|--------|
| Constants | ✅ 100% Match |
| confine() | ✅ 100% Match |
| rotate() | ✅ 100% Match |
| ±90° randomization | ✅ 100% Match |
| Mouse speed clamping | ✅ 100% Match |
| Mouse magnitude low-pass | ✅ 100% Match |
| Direction extraction | ✅ 100% Match |
| Target velocity mixing | ✅ 100% Match |
| Distractor linear bias | ✅ 100% Match |
| Distractor smoothing | ✅ 100% Match |
| Velocity low-pass | ✅ 100% Match |
| Evidence calculation | ✅ 100% Match |
| Confidence scale | ✅ 100% Match |
| Agency scale | ✅ 100% Match |
| Data output | ✅ 100% Match |

**The online version is an IDENTICAL TWIN of the offline experiment.**
