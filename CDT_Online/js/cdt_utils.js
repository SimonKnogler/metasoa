/**
 * CDT Online Experiment - JavaScript Utilities
 * EXACT REPLICA of the offline PsychoPy experiment
 * 
 * This file replicates ALL logic from:
 * Main_Experiment/CDT_windows_blockwise_fast_response.py
 * 
 * Every function, constant, and algorithm matches the offline version exactly.
 */

// =============================================================================
// Constants - EXACTLY matching offline experiment (lines 609-610)
// =============================================================================

const OFFSET_X = 300;           // Initial shape offset in pixels (line 609)
const LOWPASS = 0.5;            // Low-pass filter coefficient (line 610)
const MAX_SPEED = 20.0;         // Maximum mouse speed clamp (line 796-797)
const CONFINE_RADIUS = 250;     // Boundary radius for shape confinement (line 650)
const MOTION_DURATION = 5.0;    // Total motion duration in seconds (line 775)

// =============================================================================
// Global State Variables
// =============================================================================

var motionLibrary = null;
var trajectoryPairs = null;

// Trial state - matches offline variables (lines 764-768)
var currentFrame = 0;
var lastMousePos = [0, 0];
var vt = [0, 0];                // Target velocity (low-passed)
var vd = [0, 0];                // Distractor velocity (low-passed)
var mag_m_lp = 0;               // Low-passed mouse magnitude
var prev_d = [0, 0];            // Previous distractor velocity for smoothing
var trialKinematics = [];       // Frame-by-frame kinematics

// =============================================================================
// Core Math Functions - EXACT replicas from offline (lines 650-654)
// =============================================================================

/**
 * Confine position to circular boundary
 * EXACT replica of line 650:
 * confine = lambda p, l=250: p if (r := math.hypot(*p)) <= l else (p[0]*l/r, p[1]*l/r)
 */
function confine(pos, limit = CONFINE_RADIUS) {
    const r = Math.hypot(pos[0], pos[1]);
    if (r <= limit) {
        return pos;
    } else {
        return [pos[0] * limit / r, pos[1] * limit / r];
    }
}

/**
 * Rotate velocity vector by angle
 * EXACT replica of lines 651-654:
 * rotate = lambda vx, vy, a: (
 *     vx * math.cos(math.radians(a)) - vy * math.sin(math.radians(a)),
 *     vx * math.sin(math.radians(a)) + vy * math.cos(math.radians(a))
 * )
 */
function rotate(vx, vy, angleDegrees) {
    const angleRadians = angleDegrees * Math.PI / 180;
    const cos = Math.cos(angleRadians);
    const sin = Math.sin(angleRadians);
    return [
        vx * cos - vy * sin,
        vx * sin + vy * cos
    ];
}

// =============================================================================
// Resource Loading
// =============================================================================

async function loadAllResources() {
    console.log('Loading CDT experiment resources...');
    
    try {
        const motionResponse = await fetch('resources/motion_library.json');
        if (!motionResponse.ok) throw new Error('Failed to load motion library');
        motionLibrary = await motionResponse.json();
        console.log(`Loaded motion library: ${motionLibrary.metadata.total_selected} trajectories`);
        
        const pairsResponse = await fetch('resources/trajectory_pairs.json');
        if (!pairsResponse.ok) throw new Error('Failed to load trajectory pairs');
        trajectoryPairs = await pairsResponse.json();
        console.log(`Loaded trajectory pairs: ${trajectoryPairs.n_pairs} pairs`);
        
        return true;
    } catch (error) {
        console.error('Error loading resources:', error);
        throw error;
    }
}

function resourcesLoaded() {
    return motionLibrary !== null && trajectoryPairs !== null;
}

// =============================================================================
// Trajectory Selection - Matching offline find_matched_trajectory_pair()
// =============================================================================

function getTrajectoryPair() {
    if (!trajectoryPairs || !motionLibrary) {
        console.error('Resources not loaded!');
        return null;
    }
    
    const pairIdx = Math.floor(Math.random() * trajectoryPairs.pairs.length);
    const pair = trajectoryPairs.pairs[pairIdx];
    
    return {
        targetTrajectory: motionLibrary.trajectories[pair[0]],
        distractorTrajectory: motionLibrary.trajectories[pair[1]],
        targetIndex: pair[0],
        distractorIndex: pair[1]
    };
}

/**
 * Apply consistent smoothing to trajectories
 * EXACT replica of apply_consistent_smoothing() (lines 580-597)
 */
function applyConsistentSmoothing(trajectory1, trajectory2, windowSize = 3) {
    function smoothTrajectory(traj) {
        if (traj.length < windowSize) return traj;
        
        const smoothed = [];
        for (let i = 0; i < traj.length; i++) {
            const start = Math.max(0, i - Math.floor(windowSize / 2));
            const end = Math.min(traj.length, i + Math.floor(windowSize / 2) + 1);
            
            let sumX = 0, sumY = 0;
            for (let j = start; j < end; j++) {
                sumX += traj[j][0];
                sumY += traj[j][1];
            }
            const count = end - start;
            smoothed.push([sumX / count, sumY / count]);
        }
        return smoothed;
    }
    
    // Convert velocities to positions (cumsum)
    function cumsum(velocities) {
        const positions = [[0, 0]];
        for (let i = 0; i < velocities.length; i++) {
            positions.push([
                positions[i][0] + velocities[i][0],
                positions[i][1] + velocities[i][1]
            ]);
        }
        return positions.slice(1);
    }
    
    // Convert positions back to velocities (diff)
    function diff(positions) {
        const velocities = [];
        for (let i = 1; i < positions.length; i++) {
            velocities.push([
                positions[i][0] - positions[i-1][0],
                positions[i][1] - positions[i-1][1]
            ]);
        }
        return velocities;
    }
    
    const pos1 = cumsum(trajectory1);
    const pos2 = cumsum(trajectory2);
    const smoothPos1 = smoothTrajectory(pos1);
    const smoothPos2 = smoothTrajectory(pos2);
    const vel1 = diff([[0, 0], ...smoothPos1]);
    const vel2 = diff([[0, 0], ...smoothPos2]);
    
    return [vel1, vel2];
}

// =============================================================================
// Trial Initialization - Matching offline run_trial() setup (lines 676-768)
// =============================================================================

/**
 * Initialize a new trial
 * @param {number} propSelf - Proportion of self-control (prop_used)
 * @param {number} angleBias - Rotation angle in degrees (0 or 90)
 * @param {string} expectLevel - "high" or "low"
 * @param {string} phase - "calibration", "practice", or "test"
 * @returns {Object} Trial configuration
 */
function initializeTrial(propSelf, angleBias, expectLevel = "high", phase = "test") {
    // Get trajectory pair (lines 720-761)
    const pair = getTrajectoryPair();
    if (!pair) {
        console.error('Failed to get trajectory pair');
        return null;
    }
    
    // Apply consistent smoothing (line 762)
    const [smoothedTarget, smoothedDistractor] = applyConsistentSmoothing(
        pair.targetTrajectory,
        pair.distractorTrajectory
    );
    
    // Reset trial state variables (lines 764-768)
    currentFrame = 0;
    lastMousePos = [0, 0];
    vt = [0, 0];
    vd = [0, 0];
    mag_m_lp = 0;
    prev_d = [0, 0];
    trialKinematics = [];
    
    // Determine target shape - random choice (line 719)
    const targetShape = Math.random() < 0.5 ? 'square' : 'dot';
    
    // Determine initial positions - random left/right (lines 703-707)
    const leftShape = Math.random() < 0.5 ? 'square' : 'dot';
    
    // CRITICAL: Randomize ±90° when angle_bias == 90 (lines 782-784)
    let appliedAngle = angleBias;
    if (angleBias === 90) {
        appliedAngle = Math.random() < 0.5 ? 90 : -90;
    }
    
    return {
        targetTrajectory: smoothedTarget,
        distractorTrajectory: smoothedDistractor,
        targetSnippetIdx: pair.targetIndex,
        distractorSnippetIdx: pair.distractorIndex,
        targetShape: targetShape,
        leftShape: leftShape,
        propSelf: propSelf,
        angleBias: angleBias,
        appliedAngle: appliedAngle,  // The actual rotation applied (±90 or 0)
        expectLevel: expectLevel,
        phase: phase,
        startTime: performance.now()
    };
}

// =============================================================================
// Frame Processing - EXACT replica of the main loop (lines 787-871)
// =============================================================================

/**
 * Process one frame of the trial
 * This is an EXACT replica of the offline motion loop (lines 787-871)
 * 
 * @param {Object} config - Trial configuration from initializeTrial
 * @param {Array} mousePos - Current mouse position [x, y] in pixels
 * @param {Object} shapePositions - Current shape positions {square: [x,y], dot: [x,y]}
 * @returns {Object} Updated positions and evidence
 */
function processFrame(config, mousePos, shapePositions) {
    // Get mouse displacement (lines 788-790)
    const x = mousePos[0];
    const y = mousePos[1];
    let dx = x - lastMousePos[0];
    let dy = y - lastMousePos[1];
    lastMousePos = [x, y];
    
    // Apply rotation (line 791)
    [dx, dy] = rotate(dx, dy, config.appliedAngle);
    
    // Get trajectory velocities for this frame (lines 792-793)
    const trajFrame = currentFrame % config.targetTrajectory.length;
    const target_ou_dx_raw = config.targetTrajectory[trajFrame][0];
    const target_ou_dy_raw = config.targetTrajectory[trajFrame][1];
    const distractor_ou_dx_raw = config.distractorTrajectory[trajFrame][0];
    const distractor_ou_dy_raw = config.distractorTrajectory[trajFrame][1];
    
    currentFrame++;
    
    // Calculate mouse magnitude with speed clamping (lines 795-802)
    let mag_m = Math.hypot(dx, dy);
    mag_m = Math.min(mag_m, MAX_SPEED);
    
    // Clamp actual displacement if needed
    if (mag_m > 0) {
        const original_mag = Math.hypot(dx, dy);
        if (original_mag > MAX_SPEED) {
            const scale_factor = MAX_SPEED / original_mag;
            dx = dx * scale_factor;
            dy = dy * scale_factor;
        }
    }
    
    // Low-pass filter mouse magnitude (lines 803-804)
    if (currentFrame === 1) {
        mag_m_lp = mag_m;
    } else {
        mag_m_lp = 0.5 * mag_m_lp + 0.5 * mag_m;
    }
    
    // Extract direction from target trajectory and scale by mouse magnitude (lines 805-816)
    const mag_target = Math.hypot(target_ou_dx_raw, target_ou_dy_raw);
    let dir_target_x, dir_target_y;
    if (mag_target > 0) {
        dir_target_x = target_ou_dx_raw / mag_target;
        dir_target_y = target_ou_dy_raw / mag_target;
    } else {
        dir_target_x = 0;
        dir_target_y = 0;
    }
    
    const mag_distractor = Math.hypot(distractor_ou_dx_raw, distractor_ou_dy_raw);
    let dir_distractor_x, dir_distractor_y;
    if (mag_distractor > 0) {
        dir_distractor_x = distractor_ou_dx_raw / mag_distractor;
        dir_distractor_y = distractor_ou_dy_raw / mag_distractor;
    } else {
        dir_distractor_x = 0;
        dir_distractor_y = 0;
    }
    
    // Scale trajectory by low-passed mouse magnitude (lines 815-816)
    const target_ou_dx = dir_target_x * mag_m_lp;
    const target_ou_dy = dir_target_y * mag_m_lp;
    let distractor_ou_dx = dir_distractor_x * mag_m_lp;
    let distractor_ou_dy = dir_distractor_y * mag_m_lp;
    
    // Mix mouse input with trajectory for target (lines 817-818)
    const prop = config.propSelf;
    const tdx = prop * dx + (1 - prop) * target_ou_dx;
    const tdy = prop * dy + (1 - prop) * target_ou_dy;
    
    // === DISTRACTOR LINEAR BIAS LOGIC (lines 819-844) ===
    // This makes the distractor move perpendicular when participant moves linearly
    const mouse_speed = Math.hypot(dx, dy);
    let linear_bias = 0.0;
    
    if (mouse_speed > 0 && currentFrame > 10 && trialKinematics.length >= 5) {
        // Get recent positions
        const recentPositions = trialKinematics.slice(-5).map(d => [d.mouse_x, d.mouse_y]);
        recentPositions.push([x, y]);
        
        if (recentPositions.length >= 3) {
            // Calculate total path distance
            let total_dist = 0;
            for (let i = 0; i < recentPositions.length - 1; i++) {
                total_dist += Math.hypot(
                    recentPositions[i+1][0] - recentPositions[i][0],
                    recentPositions[i+1][1] - recentPositions[i][1]
                );
            }
            
            // Calculate straight-line distance
            const straight_dist = Math.hypot(
                recentPositions[recentPositions.length-1][0] - recentPositions[0][0],
                recentPositions[recentPositions.length-1][1] - recentPositions[0][1]
            );
            
            if (total_dist > 0) {
                const linearity = straight_dist / total_dist;
                // Apply bias when moving slowly and linearly (lines 830-831)
                if (mouse_speed < 10.0 && linearity > 0.8) {
                    linear_bias = Math.min(0.3, linearity * 0.4);
                }
            }
        }
    }
    
    // Apply linear bias to distractor (lines 832-844)
    let ddx, ddy;
    if (linear_bias > 0) {
        // Perpendicular direction
        let perp_dx = -dy;
        let perp_dy = dx;
        const perp_mag = Math.hypot(perp_dx, perp_dy);
        
        if (perp_mag > 0) {
            perp_dx /= perp_mag;
            perp_dy /= perp_mag;
            const cursor_mag = Math.hypot(dx, dy);
            perp_dx *= cursor_mag;
            perp_dy *= cursor_mag;
        } else {
            perp_dx = 0;
            perp_dy = 0;
        }
        
        ddx = (1 - linear_bias) * distractor_ou_dx + linear_bias * perp_dx;
        ddy = (1 - linear_bias) * distractor_ou_dy + linear_bias * perp_dy;
    } else {
        ddx = distractor_ou_dx;
        ddy = distractor_ou_dy;
    }
    
    // Smooth distractor velocity (lines 845-846)
    const ddx_smooth = 0.4 * prev_d[0] + 0.6 * ddx;
    const ddy_smooth = 0.4 * prev_d[1] + 0.6 * ddy;
    prev_d = [ddx_smooth, ddy_smooth];
    
    // Apply low-pass filter to both velocities (lines 847-848)
    vt = [
        LOWPASS * vt[0] + (1 - LOWPASS) * tdx,
        LOWPASS * vt[1] + (1 - LOWPASS) * tdy
    ];
    vd = [
        LOWPASS * vd[0] + (1 - LOWPASS) * ddx_smooth,
        LOWPASS * vd[1] + (1 - LOWPASS) * ddy_smooth
    ];
    
    // === EVIDENCE CALCULATION (lines 850-864) ===
    // Uses DISPLAYED velocities (after low-pass filtering)
    const vm = [dx, dy];
    const vm_speed = Math.sqrt(vm[0]**2 + vm[1]**2) + 1e-9;
    
    const vt_disp = vt.slice();  // Displayed target velocity
    const vd_disp = vd.slice();  // Displayed distractor velocity
    
    const vt_norm = Math.sqrt(vt_disp[0]**2 + vt_disp[1]**2) + 1e-9;
    const vd_norm = Math.sqrt(vd_disp[0]**2 + vd_disp[1]**2) + 1e-9;
    
    const ut = [vt_disp[0] / vt_norm, vt_disp[1] / vt_norm];
    const ud = [vd_disp[0] / vd_norm, vd_disp[1] / vd_norm];
    
    const cos_T = (vm[0] * ut[0] + vm[1] * ut[1]) / vm_speed;
    const cos_D = (vm[0] * ud[0] + vm[1] * ud[1]) / vm_speed;
    
    // Evidence formula (line 864)
    const evidence = (cos_T - cos_D) * (vm_speed - 1e-9);
    
    // Update shape positions (lines 866-871)
    let newSquarePos, newDotPos;
    if (config.targetShape === 'square') {
        newSquarePos = confine([
            shapePositions.square[0] + vt[0],
            shapePositions.square[1] + vt[1]
        ]);
        newDotPos = confine([
            shapePositions.dot[0] + vd[0],
            shapePositions.dot[1] + vd[1]
        ]);
    } else {
        newDotPos = confine([
            shapePositions.dot[0] + vt[0],
            shapePositions.dot[1] + vt[1]
        ]);
        newSquarePos = confine([
            shapePositions.square[0] + vd[0],
            shapePositions.square[1] + vd[1]
        ]);
    }
    
    // Store kinematics (lines 873-877)
    trialKinematics.push({
        frame: currentFrame,
        mouse_x: x,
        mouse_y: y,
        square_x: newSquarePos[0],
        square_y: newSquarePos[1],
        dot_x: newDotPos[0],
        dot_y: newDotPos[1],
        evidence: evidence
    });
    
    return {
        squarePos: newSquarePos,
        dotPos: newDotPos,
        evidence: evidence,
        vt: vt.slice(),
        vd: vd.slice(),
        frame: currentFrame
    };
}

// =============================================================================
// Trial Finalization - Matching offline evidence aggregation (lines 1006-1047)
// =============================================================================

/**
 * Finalize trial and calculate all evidence metrics
 * EXACT replica of lines 1006-1047
 */
function finalizeTrial(config, responseKey, responseTime, earlyResponse = true) {
    // Determine response shape
    const respShape = responseKey === 'a' ? 'square' : 
                      responseKey === 's' ? 'dot' : 'timeout';
    
    // Calculate accuracy
    const correct = respShape === config.targetShape ? 1 : 0;
    
    // Aggregate evidence across trial (lines 1007-1010)
    const frameEvidence = trialKinematics.map(d => d.evidence);
    const meanEvidence = frameEvidence.reduce((a, b) => a + b, 0) / frameEvidence.length;
    const sumEvidence = frameEvidence.reduce((a, b) => a + b, 0);
    
    // Variance calculation
    const varEvidence = frameEvidence.reduce((acc, val) => 
        acc + (val - meanEvidence) ** 2, 0) / frameEvidence.length;
    
    // Pre-response evidence metrics (lines 1012-1034)
    let preRTMetrics = {};
    if (earlyResponse && trialKinematics.length > 0) {
        const preRTEvidence = frameEvidence;
        
        // Cumulative sum
        const cumsum = [];
        let running = 0;
        for (const e of preRTEvidence) {
            running += e;
            cumsum.push(running);
        }
        
        preRTMetrics = {
            mean_evidence_preRT: preRTEvidence.reduce((a, b) => a + b, 0) / preRTEvidence.length,
            sum_evidence_preRT: preRTEvidence.reduce((a, b) => a + b, 0),
            var_evidence_preRT: preRTEvidence.reduce((acc, val, _, arr) => {
                const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
                return acc + (val - mean) ** 2;
            }, 0) / preRTEvidence.length,
            max_cum_evidence_preRT: Math.max(...cumsum),
            min_cum_evidence_preRT: Math.min(...cumsum),
            max_abs_cum_evidence_preRT: Math.max(...cumsum.map(Math.abs)),
            prop_positive_evidence_preRT: preRTEvidence.filter(e => e > 0).length / preRTEvidence.length,
            rt_frame: trialKinematics[trialKinematics.length - 1].frame,
            num_frames_preRT: preRTEvidence.length
        };
    } else {
        preRTMetrics = {
            mean_evidence_preRT: NaN,
            sum_evidence_preRT: NaN,
            var_evidence_preRT: NaN,
            max_cum_evidence_preRT: NaN,
            min_cum_evidence_preRT: NaN,
            max_abs_cum_evidence_preRT: NaN,
            prop_positive_evidence_preRT: NaN,
            rt_frame: NaN,
            num_frames_preRT: NaN
        };
    }
    
    return {
        // Core trial data
        target_snippet_id: config.targetSnippetIdx,
        distractor_snippet_id: config.distractorSnippetIdx,
        phase: config.phase,
        angle_bias: config.angleBias,
        applied_angle_bias: config.appliedAngle,
        expect_level: config.expectLevel,
        true_shape: config.targetShape,
        resp_shape: respShape,
        accuracy: correct,
        rt_choice: responseTime,
        prop_used: config.propSelf,
        early_response: earlyResponse,
        
        // Evidence metrics
        mean_evidence: meanEvidence,
        sum_evidence: sumEvidence,
        var_evidence: varEvidence,
        
        // Pre-RT metrics
        ...preRTMetrics,
        
        // Kinematics (for detailed analysis)
        kinematics: trialKinematics
    };
}

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Get initial shape positions based on left/right randomization
 * Matches lines 703-707
 */
function getInitialPositions(leftShape) {
    if (leftShape === 'square') {
        return {
            square: [-OFFSET_X, 0],
            dot: [OFFSET_X, 0]
        };
    } else {
        return {
            square: [OFFSET_X, 0],
            dot: [-OFFSET_X, 0]
        };
    }
}

/**
 * Clamp prop_self to valid range
 * Matches clamp_prop() line 1060-1061
 */
function clampProp(s) {
    return Math.max(0.02, Math.min(0.90, s));
}

/**
 * Logit function
 * Matches logit() lines 1053-1055
 */
function logit(x) {
    x = Math.max(1e-6, Math.min(1 - 1e-6, x));
    return Math.log(x / (1 - x));
}

/**
 * Inverse logit function
 * Matches inv_logit() lines 1057-1058
 */
function invLogit(z) {
    return 1.0 / (1.0 + Math.exp(-z));
}

// =============================================================================
// Frame Rate Monitoring
// =============================================================================

var frameTimestamps = [];
var frameRateHistory = [];

function recordFrameTimestamp() {
    const now = performance.now();
    frameTimestamps.push(now);
    if (frameTimestamps.length > 60) {
        frameTimestamps.shift();
    }
}

function calculateFrameRate() {
    if (frameTimestamps.length < 2) return 60;
    
    const totalTime = frameTimestamps[frameTimestamps.length - 1] - frameTimestamps[0];
    const frameRate = (frameTimestamps.length - 1) / (totalTime / 1000);
    
    frameRateHistory.push(frameRate);
    if (frameRateHistory.length > 100) frameRateHistory.shift();
    
    return frameRate;
}

function getFrameRateStats() {
    if (frameRateHistory.length === 0) {
        return { mean: 60, min: 60, max: 60, std: 0 };
    }
    
    const mean = frameRateHistory.reduce((a, b) => a + b, 0) / frameRateHistory.length;
    const min = Math.min(...frameRateHistory);
    const max = Math.max(...frameRateHistory);
    const variance = frameRateHistory.reduce((acc, val) => 
        acc + (val - mean) ** 2, 0) / frameRateHistory.length;
    
    return { mean, min, max, std: Math.sqrt(variance) };
}

// =============================================================================
// Export confirmation
// =============================================================================

console.log('CDT utilities loaded - EXACT REPLICA of offline experiment');
console.log('Constants: OFFSET_X=' + OFFSET_X + ', LOWPASS=' + LOWPASS + 
            ', MAX_SPEED=' + MAX_SPEED + ', CONFINE_RADIUS=' + CONFINE_RADIUS);
