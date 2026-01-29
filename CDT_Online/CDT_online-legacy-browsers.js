/******************* 
 * Cdt_Online *
 *******************/


// store info about the experiment session:
let expName = 'CDT_online';  // from the Builder filename that created this script
let expInfo = {
    'participant': '',
    'session': '001',
};

// Start code blocks for 'Before Experiment'
// Global variables matching offline experiment
var motionLibrary = null;
var trajectoryPairs = null;
var currentPropSelf = 0.4;
var currentAngleBias = 0;
var currentExpectLevel = 'high';
// Constants matching offline experiment EXACTLY
const OFFSET_X = 300;
const LOWPASS = 0.5;
const MAX_SPEED = 20.0;
const CONFINE_RADIUS = 250;
const MOTION_DURATION = 5.0;

// Trial state variables
var trialConfig = null;
var shapePositions = null;
var responseKey = null;
var responseRT = null;
var earlyResponse = false;
var trialStartTime = 0;

// Functions from cdt_utils.js are loaded globally
// init psychoJS:
const psychoJS = new PsychoJS({
  debug: true
});

// open window:
psychoJS.openWindow({
  fullscr: true,
  color: new util.Color([0.5, 0.5, 0.5]),
  units: 'pix',
  waitBlanking: true,
  backgroundImage: '',
  backgroundFit: 'none',
});
// schedule the experiment:
psychoJS.schedule(psychoJS.gui.DlgFromDict({
  dictionary: expInfo,
  title: expName
}));

const flowScheduler = new Scheduler(psychoJS);
const dialogCancelScheduler = new Scheduler(psychoJS);
psychoJS.scheduleCondition(function() { return (psychoJS.gui.dialogComponent.button === 'OK'); },flowScheduler, dialogCancelScheduler);

// flowScheduler gets run if the participants presses OK
flowScheduler.add(updateInfo); // add timeStamp
flowScheduler.add(experimentInit);
flowScheduler.add(init_resourcesRoutineBegin());
flowScheduler.add(init_resourcesRoutineEachFrame());
flowScheduler.add(init_resourcesRoutineEnd());
flowScheduler.add(instructionsRoutineBegin());
flowScheduler.add(instructionsRoutineEachFrame());
flowScheduler.add(instructionsRoutineEnd());
const trialsLoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(trialsLoopBegin(trialsLoopScheduler));
flowScheduler.add(trialsLoopScheduler);
flowScheduler.add(trialsLoopEnd);





flowScheduler.add(endRoutineBegin());
flowScheduler.add(endRoutineEachFrame());
flowScheduler.add(endRoutineEnd());
flowScheduler.add(quitPsychoJS, '', true);

// quit if user presses Cancel in dialog box:
dialogCancelScheduler.add(quitPsychoJS, '', false);

psychoJS.start({
  expName: expName,
  expInfo: expInfo,
  resources: [
    // resources:
    {'name': 'conditions.csv', 'path': 'conditions.csv'},
  ]
});

psychoJS.experimentLogger.setLevel(core.Logger.ServerLevel.EXP);

async function updateInfo() {
  currentLoop = psychoJS.experiment;  // right now there are no loops
  expInfo['date'] = util.MonotonicClock.getDateStr();  // add a simple timestamp
  expInfo['expName'] = expName;
  expInfo['psychopyVersion'] = '2024.2.4';
  expInfo['OS'] = window.navigator.platform;


  // store frame rate of monitor if we can measure it successfully
  expInfo['frameRate'] = psychoJS.window.getActualFrameRate();
  if (typeof expInfo['frameRate'] !== 'undefined')
    frameDur = 1.0 / Math.round(expInfo['frameRate']);
  else
    frameDur = 1.0 / 60.0; // couldn't get a reliable measure so guess

  // add info from the URL:
  util.addInfoFromUrl(expInfo);
  

  
  psychoJS.experiment.dataFileName = (("." + "/") + `data/${expInfo["participant"]}_${expName}_${expInfo["date"]}`);
  psychoJS.experiment.field_separator = '\t';


  return Scheduler.Event.NEXT;
}

async function experimentInit() {
  // Initialize components for Routine "init_resources"
  init_resourcesClock = new util.Clock();
  loading_text = new visual.TextStim({
    win: psychoJS.window,
    name: 'loading_text',
    text: 'Loading experiment resources...\\n\\nPlease wait.',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 30.0,  wrapWidth: 1000.0, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -1.0 
  });
  
  // Initialize components for Routine "instructions"
  instructionsClock = new util.Clock();
  instr_text = new visual.TextStim({
    win: psychoJS.window,
    name: 'instr_text',
    text: /*Control Detection Task
  
  In this experiment, you will control one of two shapes
  using your mouse movements.
  
  Your task is to identify which shape you are controlling.
  
  Move your mouse to see the shapes respond.
  The shape that follows your movement is the one you control.
  
  Press A for Square, S for Circle
  
  Press SPACE to continue.*/
  ,
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 26.0,  wrapWidth: 1000.0, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  instr_key = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "trial"
  trialClock = new util.Clock();
  square = new visual.Rect ({
    win: psychoJS.window, name: 'square', 
    width: [40, 40][0], height: [40, 40][1],
    ori: 0.0, 
    pos: [0, 0], 
    draggable: false, 
    anchor: 'center', 
    lineWidth: 2.0, 
    lineColor: new util.Color('black'), 
    fillColor: new util.Color('black'), 
    colorSpace: 'rgb', 
    opacity: undefined, 
    depth: -1, 
    interpolate: true, 
  });
  
  dot = new visual.Polygon({
    win: psychoJS.window, name: 'dot', 
    edges: 100, size:[40, 40],
    ori: 0.0, 
    pos: [0, 0], 
    draggable: false, 
    anchor: 'center', 
    lineWidth: 2.0, 
    lineColor: new util.Color('black'), 
    fillColor: new util.Color('black'), 
    colorSpace: 'rgb', 
    opacity: undefined, 
    depth: -2, 
    interpolate: true, 
  });
  
  mouse = new core.Mouse({
    win: psychoJS.window,
  });
  mouse.mouseClock = new util.Clock();
  // Initialize components for Routine "timeout_feedback"
  timeout_feedbackClock = new util.Clock();
  timeout_text = new visual.TextStim({
    win: psychoJS.window,
    name: 'timeout_text',
    text: 'Too slow!\\n\\nPlease respond faster next time.',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 26.0,  wrapWidth: 1000.0, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -1.0 
  });
  
  // Initialize components for Routine "confidence"
  confidenceClock = new util.Clock();
  conf_text = new visual.TextStim({
    win: psychoJS.window,
    name: 'conf_text',
    text: 'How confident are you in your choice?\\n\\n1 = Not at all confident\\n2 = Slightly confident\\n3 = Moderately confident\\n4 = Very confident',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 26.0,  wrapWidth: 1000.0, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -1.0 
  });
  
  conf_key = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "agency"
  agencyClock = new util.Clock();
  agency_text = new visual.TextStim({
    win: psychoJS.window,
    name: 'agency_text',
    text: "How much control did you feel over the shape's movement?",
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 50], draggable: false, height: 26.0,  wrapWidth: 1000.0, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -1.0 
  });
  
  agency_scale = new visual.TextStim({
    win: psychoJS.window,
    name: 'agency_scale',
    text: '1          2          3          4          5          6          7\\nVery      Weak    Somewhat   Moderate  Somewhat   Strong    Very\\nweak                weak                  strong              strong',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, (- 100)], draggable: false, height: 18.0,  wrapWidth: 1200.0, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -2.0 
  });
  
  agency_key = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "end"
  endClock = new util.Clock();
  end_text = new visual.TextStim({
    win: psychoJS.window,
    name: 'end_text',
    text: 'Thank you for participating!\\n\\nYour data has been saved.',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 30.0,  wrapWidth: 1000.0, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  // Create some handy timers
  globalClock = new util.Clock();  // to track the time since experiment started
  routineTimer = new util.CountdownTimer();  // to track time remaining of each (non-slip) routine
  
  return Scheduler.Event.NEXT;
}

function init_resourcesRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'init_resources' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    init_resourcesClock.reset();
    routineTimer.reset();
    init_resourcesMaxDurationReached = false;
    // update component parameters for each repeat
    // Load resources using functions from cdt_utils.js
    loadAllResources();
    psychoJS.experiment.addData('init_resources.started', globalClock.getTime());
    init_resourcesMaxDuration = null
    // keep track of which components have finished
    init_resourcesComponents = [];
    init_resourcesComponents.push(loading_text);
    
    init_resourcesComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}

function init_resourcesRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'init_resources' ---
    // get current time
    t = init_resourcesClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    // Wait for resources to load
    if (!resourcesLoaded()) {
        // Keep waiting
    } else {
        continueRoutine = false;
    }
    
    // *loading_text* updates
    if (t >= 0.0 && loading_text.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      loading_text.tStart = t;  // (not accounting for frame time here)
      loading_text.frameNStart = frameN;  // exact frame index
      
      loading_text.setAutoDraw(true);
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    init_resourcesComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}

function init_resourcesRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'init_resources' ---
    init_resourcesComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('init_resources.stopped', globalClock.getTime());
    // the Routine "init_resources" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}

function instructionsRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'instructions' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    instructionsClock.reset();
    routineTimer.reset();
    instructionsMaxDurationReached = false;
    // update component parameters for each repeat
    instr_key.keys = undefined;
    instr_key.rt = undefined;
    _instr_key_allKeys = [];
    psychoJS.experiment.addData('instructions.started', globalClock.getTime());
    instructionsMaxDuration = null
    // keep track of which components have finished
    instructionsComponents = [];
    instructionsComponents.push(instr_text);
    instructionsComponents.push(instr_key);
    
    instructionsComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}

function instructionsRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'instructions' ---
    // get current time
    t = instructionsClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *instr_text* updates
    if (t >= 0.0 && instr_text.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      instr_text.tStart = t;  // (not accounting for frame time here)
      instr_text.frameNStart = frameN;  // exact frame index
      
      instr_text.setAutoDraw(true);
    }
    
    
    // *instr_key* updates
    if (t >= 0.0 && instr_key.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      instr_key.tStart = t;  // (not accounting for frame time here)
      instr_key.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { instr_key.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { instr_key.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { instr_key.clearEvents(); });
    }
    
    if (instr_key.status === PsychoJS.Status.STARTED) {
      let theseKeys = instr_key.getKeys({keyList: ['space'], waitRelease: false});
      _instr_key_allKeys = _instr_key_allKeys.concat(theseKeys);
      if (_instr_key_allKeys.length > 0) {
        instr_key.keys = _instr_key_allKeys[_instr_key_allKeys.length - 1].name;  // just the last key pressed
        instr_key.rt = _instr_key_allKeys[_instr_key_allKeys.length - 1].rt;
        instr_key.duration = _instr_key_allKeys[_instr_key_allKeys.length - 1].duration;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    instructionsComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}

function instructionsRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'instructions' ---
    instructionsComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('instructions.stopped', globalClock.getTime());
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(instr_key.corr, level);
    }
    psychoJS.experiment.addData('instr_key.keys', instr_key.keys);
    if (typeof instr_key.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('instr_key.rt', instr_key.rt);
        psychoJS.experiment.addData('instr_key.duration', instr_key.duration);
        routineTimer.reset();
        }
    
    instr_key.stop();
    // the Routine "instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}

function trialsLoopBegin(trialsLoopScheduler, snapshot) {
  return async function() {
    TrialHandler.fromSnapshot(snapshot); // update internal variables (.thisN etc) of the loop
    
    // set up handler to look after randomisation of conditions etc
    trials = new TrialHandler({
      psychoJS: psychoJS,
      nReps: 1, method: TrialHandler.Method.RANDOM,
      extraInfo: expInfo, originPath: undefined,
      trialList: 'conditions.csv',
      seed: random, name: 'trials'
    });
    psychoJS.experiment.addLoop(trials); // add the loop to the experiment
    currentLoop = trials;  // we're now the current loop
    
    // Schedule all the trials in the trialList:
    trials.forEach(function() {
      snapshot = trials.getSnapshot();
    
      trialsLoopScheduler.add(importConditions(snapshot));
      trialsLoopScheduler.add(trialRoutineBegin(snapshot));
      trialsLoopScheduler.add(trialRoutineEachFrame());
      trialsLoopScheduler.add(trialRoutineEnd(snapshot));
      trialsLoopScheduler.add(timeout_feedbackRoutineBegin(snapshot));
      trialsLoopScheduler.add(timeout_feedbackRoutineEachFrame());
      trialsLoopScheduler.add(timeout_feedbackRoutineEnd(snapshot));
      trialsLoopScheduler.add(confidenceRoutineBegin(snapshot));
      trialsLoopScheduler.add(confidenceRoutineEachFrame());
      trialsLoopScheduler.add(confidenceRoutineEnd(snapshot));
      trialsLoopScheduler.add(agencyRoutineBegin(snapshot));
      trialsLoopScheduler.add(agencyRoutineEachFrame());
      trialsLoopScheduler.add(agencyRoutineEnd(snapshot));
      trialsLoopScheduler.add(trialsLoopEndIteration(trialsLoopScheduler, snapshot));
    });
    
    return Scheduler.Event.NEXT;
  }
}

async function trialsLoopEnd() {
  // terminate loop
  psychoJS.experiment.removeLoop(trials);
  // update the current loop from the ExperimentHandler
  if (psychoJS.experiment._unfinishedLoops.length>0)
    currentLoop = psychoJS.experiment._unfinishedLoops.at(-1);
  else
    currentLoop = psychoJS.experiment;  // so we use addData from the experiment
  return Scheduler.Event.NEXT;
}

function trialsLoopEndIteration(scheduler, snapshot) {
  // ------Prepare for next entry------
  return async function () {
    if (typeof snapshot !== 'undefined') {
      // ------Check if user ended loop early------
      if (snapshot.finished) {
        // Check for and save orphaned data
        if (psychoJS.experiment.isEntryEmpty()) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        scheduler.stop();
      } else {
        psychoJS.experiment.nextEntry(snapshot);
      }
    return Scheduler.Event.NEXT;
    }
  };
}

function trialRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'trial' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    trialClock.reset();
    routineTimer.reset();
    trialMaxDurationReached = false;
    // update component parameters for each repeat
    // Initialize trial using EXACT replica functions
    trialConfig = initializeTrial(
        currentPropSelf,      // From conditions file
        currentAngleBias,     // From conditions file
        currentExpectLevel,   // From conditions file
        'test'                // Phase
    );
    
    // Get initial positions (matches offline lines 703-707)
    shapePositions = getInitialPositions(trialConfig.leftShape);
    
    // Set shape positions
    square.pos = shapePositions.square;
    dot.pos = shapePositions.dot;
    
    // Reset response variables
    responseKey = null;
    responseRT = null;
    earlyResponse = false;
    trialStartTime = t;
    
    // Set mouse position to center
    mouse.setPos([0, 0]);
    lastMousePos = [0, 0];
    // setup some python lists for storing info about the mouse
    // current position of the mouse:
    mouse.x = [];
    mouse.y = [];
    mouse.leftButton = [];
    mouse.midButton = [];
    mouse.rightButton = [];
    mouse.time = [];
    mouse.clicked_name = [];
    gotValidClick = false; // until a click is received
    psychoJS.experiment.addData('trial.started', globalClock.getTime());
    trialMaxDuration = null
    // keep track of which components have finished
    trialComponents = [];
    trialComponents.push(square);
    trialComponents.push(dot);
    trialComponents.push(mouse);
    
    trialComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}

function trialRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'trial' ---
    // get current time
    t = trialClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    // Record frame for frame rate monitoring
    recordFrameTimestamp();
    
    // Get mouse position
    let mousePos = mouse.getPos();
    
    // Process frame using EXACT replica of offline motion loop
    let frameResult = processFrame(trialConfig, mousePos, {
        square: square.pos,
        dot: dot.pos
    });
    
    // Update shape positions
    square.pos = frameResult.squarePos;
    dot.pos = frameResult.dotPos;
    
    // Check for response (early response during motion, matching offline)
    let keys = psychoJS.eventManager.getKeys({keyList: ['a', 's'], timeStamped: true});
    if (keys.length > 0) {
        responseKey = keys[0].name;
        responseRT = keys[0].rt - trialStartTime;
        earlyResponse = true;
        continueRoutine = false;
    }
    
    // Check for timeout (5 seconds, matching offline line 775)
    if (t - trialStartTime >= MOTION_DURATION && responseKey === null) {
        responseKey = 'timeout';
        responseRT = NaN;
        earlyResponse = false;
        continueRoutine = false;
    }
    
    if (square.status === PsychoJS.Status.STARTED){ // only update if being drawn
      square.setPos([(- 300), 0], false);
    }
    
    // *square* updates
    if (t >= 0.0 && square.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      square.tStart = t;  // (not accounting for frame time here)
      square.frameNStart = frameN;  // exact frame index
      
      square.setAutoDraw(true);
    }
    
    
    if (dot.status === PsychoJS.Status.STARTED){ // only update if being drawn
      dot.setPos([300, 0], false);
    }
    
    // *dot* updates
    if (t >= 0.0 && dot.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      dot.tStart = t;  // (not accounting for frame time here)
      dot.frameNStart = frameN;  // exact frame index
      
      dot.setAutoDraw(true);
    }
    
    // *mouse* updates
    if (t >= 0.0 && mouse.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      mouse.tStart = t;  // (not accounting for frame time here)
      mouse.frameNStart = frameN;  // exact frame index
      
      mouse.status = PsychoJS.Status.STARTED;
      mouse.mouseClock.reset();
      prevButtonState = mouse.getPressed();  // if button is down already this ISN'T a new click
      }
    if (mouse.status === PsychoJS.Status.STARTED) {  // only update if started and not finished!
      _mouseButtons = mouse.getPressed();
      if (!_mouseButtons.every( (e,i,) => (e == prevButtonState[i]) )) { // button state changed?
        prevButtonState = _mouseButtons;
        if (_mouseButtons.reduce( (e, acc) => (e+acc) ) > 0) { // state changed to a new click
          // check if the mouse was inside our 'clickable' objects
          gotValidClick = false;
          mouse.clickableObjects = eval(any click)
          ;// make sure the mouse's clickable objects are an array
          if (!Array.isArray(mouse.clickableObjects)) {
              mouse.clickableObjects = [mouse.clickableObjects];
          }
          // iterate through clickable objects and check each
          for (const obj of mouse.clickableObjects) {
              if (obj.contains(mouse)) {
                  gotValidClick = true;
                  mouse.clicked_name.push(obj.name);
              }
          }
          if (!gotValidClick) {
              mouse.clicked_name.push(null);
          }
          _mouseXYs = mouse.getPos();
          mouse.x.push(_mouseXYs[0]);
          mouse.y.push(_mouseXYs[1]);
          mouse.leftButton.push(_mouseButtons[0]);
          mouse.midButton.push(_mouseButtons[1]);
          mouse.rightButton.push(_mouseButtons[2]);
          mouse.time.push(mouse.mouseClock.getTime());
        }
      }
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    trialComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}

function trialRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'trial' ---
    trialComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('trial.stopped', globalClock.getTime());
    // Finalize trial and get all metrics
    let trialResults = finalizeTrial(trialConfig, responseKey, responseRT, earlyResponse);
    
    // Save all data (matching offline output columns)
    psychoJS.experiment.addData('target_snippet_id', trialResults.target_snippet_id);
    psychoJS.experiment.addData('distractor_snippet_id', trialResults.distractor_snippet_id);
    psychoJS.experiment.addData('phase', trialResults.phase);
    psychoJS.experiment.addData('angle_bias', trialResults.angle_bias);
    psychoJS.experiment.addData('applied_angle_bias', trialResults.applied_angle_bias);
    psychoJS.experiment.addData('expect_level', trialResults.expect_level);
    psychoJS.experiment.addData('true_shape', trialResults.true_shape);
    psychoJS.experiment.addData('resp_shape', trialResults.resp_shape);
    psychoJS.experiment.addData('accuracy', trialResults.accuracy);
    psychoJS.experiment.addData('rt_choice', trialResults.rt_choice);
    psychoJS.experiment.addData('prop_used', trialResults.prop_used);
    psychoJS.experiment.addData('early_response', trialResults.early_response);
    psychoJS.experiment.addData('mean_evidence', trialResults.mean_evidence);
    psychoJS.experiment.addData('sum_evidence', trialResults.sum_evidence);
    psychoJS.experiment.addData('var_evidence', trialResults.var_evidence);
    psychoJS.experiment.addData('rt_frame', trialResults.rt_frame);
    psychoJS.experiment.addData('num_frames_preRT', trialResults.num_frames_preRT);
    psychoJS.experiment.addData('mean_evidence_preRT', trialResults.mean_evidence_preRT);
    psychoJS.experiment.addData('sum_evidence_preRT', trialResults.sum_evidence_preRT);
    psychoJS.experiment.addData('var_evidence_preRT', trialResults.var_evidence_preRT);
    psychoJS.experiment.addData('max_cum_evidence_preRT', trialResults.max_cum_evidence_preRT);
    psychoJS.experiment.addData('min_cum_evidence_preRT', trialResults.min_cum_evidence_preRT);
    psychoJS.experiment.addData('max_abs_cum_evidence_preRT', trialResults.max_abs_cum_evidence_preRT);
    psychoJS.experiment.addData('prop_positive_evidence_preRT', trialResults.prop_positive_evidence_preRT);
    
    // Store for timeout handling
    window.lastTrialResults = trialResults;
    // store data for psychoJS.experiment (ExperimentHandler)
    psychoJS.experiment.addData('mouse.x', mouse.x);
    psychoJS.experiment.addData('mouse.y', mouse.y);
    psychoJS.experiment.addData('mouse.leftButton', mouse.leftButton);
    psychoJS.experiment.addData('mouse.midButton', mouse.midButton);
    psychoJS.experiment.addData('mouse.rightButton', mouse.rightButton);
    psychoJS.experiment.addData('mouse.time', mouse.time);
    psychoJS.experiment.addData('mouse.clicked_name', mouse.clicked_name);
    
    // the Routine "trial" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}

function timeout_feedbackRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'timeout_feedback' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    timeout_feedbackClock.reset(routineTimer.getTime());
    routineTimer.add(2.000000);
    timeout_feedbackMaxDurationReached = false;
    // update component parameters for each repeat
    // Skip this routine if response was given
    if (window.lastTrialResults && window.lastTrialResults.resp_shape !== 'timeout') {
        continueRoutine = false;
    }
    psychoJS.experiment.addData('timeout_feedback.started', globalClock.getTime());
    timeout_feedbackMaxDuration = null
    // keep track of which components have finished
    timeout_feedbackComponents = [];
    timeout_feedbackComponents.push(timeout_text);
    
    timeout_feedbackComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}

function timeout_feedbackRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'timeout_feedback' ---
    // get current time
    t = timeout_feedbackClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *timeout_text* updates
    if (t >= 0.0 && timeout_text.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      timeout_text.tStart = t;  // (not accounting for frame time here)
      timeout_text.frameNStart = frameN;  // exact frame index
      
      timeout_text.setAutoDraw(true);
    }
    
    frameRemains = 0.0 + 2.0 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
    if (timeout_text.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      timeout_text.setAutoDraw(false);
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    timeout_feedbackComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}

function timeout_feedbackRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'timeout_feedback' ---
    timeout_feedbackComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('timeout_feedback.stopped', globalClock.getTime());
    if (timeout_feedbackMaxDurationReached) {
        timeout_feedbackClock.add(timeout_feedbackMaxDuration);
    } else {
        timeout_feedbackClock.add(2.000000);
    }
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}

function confidenceRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'confidence' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    confidenceClock.reset();
    routineTimer.reset();
    confidenceMaxDurationReached = false;
    // update component parameters for each repeat
    // Skip confidence for timeout trials (matching offline lines 946-960)
    if (window.lastTrialResults && window.lastTrialResults.resp_shape === 'timeout') {
        continueRoutine = false;
    }
    conf_key.keys = undefined;
    conf_key.rt = undefined;
    _conf_key_allKeys = [];
    psychoJS.experiment.addData('confidence.started', globalClock.getTime());
    confidenceMaxDuration = null
    // keep track of which components have finished
    confidenceComponents = [];
    confidenceComponents.push(conf_text);
    confidenceComponents.push(conf_key);
    
    confidenceComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}

function confidenceRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'confidence' ---
    // get current time
    t = confidenceClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *conf_text* updates
    if (t >= 0.0 && conf_text.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      conf_text.tStart = t;  // (not accounting for frame time here)
      conf_text.frameNStart = frameN;  // exact frame index
      
      conf_text.setAutoDraw(true);
    }
    
    
    // *conf_key* updates
    if (t >= 0.0 && conf_key.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      conf_key.tStart = t;  // (not accounting for frame time here)
      conf_key.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { conf_key.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { conf_key.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { conf_key.clearEvents(); });
    }
    
    if (conf_key.status === PsychoJS.Status.STARTED) {
      let theseKeys = conf_key.getKeys({keyList: ['1', '2', '3', '4'], waitRelease: false});
      _conf_key_allKeys = _conf_key_allKeys.concat(theseKeys);
      if (_conf_key_allKeys.length > 0) {
        conf_key.keys = _conf_key_allKeys[_conf_key_allKeys.length - 1].name;  // just the last key pressed
        conf_key.rt = _conf_key_allKeys[_conf_key_allKeys.length - 1].rt;
        conf_key.duration = _conf_key_allKeys[_conf_key_allKeys.length - 1].duration;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    confidenceComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}

function confidenceRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'confidence' ---
    confidenceComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('confidence.stopped', globalClock.getTime());
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(conf_key.corr, level);
    }
    psychoJS.experiment.addData('conf_key.keys', conf_key.keys);
    if (typeof conf_key.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('conf_key.rt', conf_key.rt);
        psychoJS.experiment.addData('conf_key.duration', conf_key.duration);
        routineTimer.reset();
        }
    
    conf_key.stop();
    // Save confidence rating
    let confRating = parseInt(conf_key.keys);
    psychoJS.experiment.addData('confidence_rating', confRating);
    // the Routine "confidence" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}

function agencyRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'agency' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    agencyClock.reset();
    routineTimer.reset();
    agencyMaxDurationReached = false;
    // update component parameters for each repeat
    // Skip agency for timeout trials (matching offline lines 969-993)
    if (window.lastTrialResults && window.lastTrialResults.resp_shape === 'timeout') {
        continueRoutine = false;
    }
    agency_key.keys = undefined;
    agency_key.rt = undefined;
    _agency_key_allKeys = [];
    psychoJS.experiment.addData('agency.started', globalClock.getTime());
    agencyMaxDuration = null
    // keep track of which components have finished
    agencyComponents = [];
    agencyComponents.push(agency_text);
    agencyComponents.push(agency_scale);
    agencyComponents.push(agency_key);
    
    agencyComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}

function agencyRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'agency' ---
    // get current time
    t = agencyClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *agency_text* updates
    if (t >= 0.0 && agency_text.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      agency_text.tStart = t;  // (not accounting for frame time here)
      agency_text.frameNStart = frameN;  // exact frame index
      
      agency_text.setAutoDraw(true);
    }
    
    
    // *agency_scale* updates
    if (t >= 0.0 && agency_scale.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      agency_scale.tStart = t;  // (not accounting for frame time here)
      agency_scale.frameNStart = frameN;  // exact frame index
      
      agency_scale.setAutoDraw(true);
    }
    
    
    // *agency_key* updates
    if (t >= 0.0 && agency_key.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      agency_key.tStart = t;  // (not accounting for frame time here)
      agency_key.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { agency_key.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { agency_key.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { agency_key.clearEvents(); });
    }
    
    if (agency_key.status === PsychoJS.Status.STARTED) {
      let theseKeys = agency_key.getKeys({keyList: ['1', '2', '3', '4', '5', '6', '7'], waitRelease: false});
      _agency_key_allKeys = _agency_key_allKeys.concat(theseKeys);
      if (_agency_key_allKeys.length > 0) {
        agency_key.keys = _agency_key_allKeys[_agency_key_allKeys.length - 1].name;  // just the last key pressed
        agency_key.rt = _agency_key_allKeys[_agency_key_allKeys.length - 1].rt;
        agency_key.duration = _agency_key_allKeys[_agency_key_allKeys.length - 1].duration;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    agencyComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}

function agencyRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'agency' ---
    agencyComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('agency.stopped', globalClock.getTime());
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(agency_key.corr, level);
    }
    psychoJS.experiment.addData('agency_key.keys', agency_key.keys);
    if (typeof agency_key.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('agency_key.rt', agency_key.rt);
        psychoJS.experiment.addData('agency_key.duration', agency_key.duration);
        routineTimer.reset();
        }
    
    agency_key.stop();
    // Save agency rating
    let agencyRating = parseInt(agency_key.keys);
    psychoJS.experiment.addData('agency_rating', agencyRating);
    // the Routine "agency" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}

function endRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'end' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    endClock.reset(routineTimer.getTime());
    routineTimer.add(5.000000);
    endMaxDurationReached = false;
    // update component parameters for each repeat
    psychoJS.experiment.addData('end.started', globalClock.getTime());
    endMaxDuration = null
    // keep track of which components have finished
    endComponents = [];
    endComponents.push(end_text);
    
    endComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}

function endRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'end' ---
    // get current time
    t = endClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *end_text* updates
    if (t >= 0.0 && end_text.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      end_text.tStart = t;  // (not accounting for frame time here)
      end_text.frameNStart = frameN;  // exact frame index
      
      end_text.setAutoDraw(true);
    }
    
    frameRemains = 0.0 + 5.0 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
    if (end_text.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      end_text.setAutoDraw(false);
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    endComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}

function endRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'end' ---
    endComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('end.stopped', globalClock.getTime());
    if (endMaxDurationReached) {
        endClock.add(endMaxDuration);
    } else {
        endClock.add(5.000000);
    }
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}

function importConditions(currentLoop) {
  return async function () {
    psychoJS.importAttributes(currentLoop.getCurrentTrial());
    return Scheduler.Event.NEXT;
    };
}

async function quitPsychoJS(message, isCompleted) {
  // Check for and save orphaned data
  if (psychoJS.experiment.isEntryEmpty()) {
    psychoJS.experiment.nextEntry();
  }
  psychoJS.window.close();
  psychoJS.quit({message: message, isCompleted: isCompleted});
  
  return Scheduler.Event.QUIT;
}
