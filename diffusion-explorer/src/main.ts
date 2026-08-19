import './style.css';
import { DATASET_LABELS } from './datasets';
import type { ArchConfig, HyperParams, Activation, WorkerOutbound, DatasetType } from './types';
import { SceneRenderer } from './viz/renderer';
import { LossChart } from './viz/losschart';
import { ScheduleChart } from './viz/schedulechart';
import { makeSchedule } from './diffusion';
import { DEFAULT_TRAIL_COUNT } from './constants';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
type Sampler = 'ancestral' | 'ddim';

interface Checkpoint {
  step: number;
  loss: number;
  distance: number;
  trajectory: { ancestral: number[][][]; ddim: number[][][] };
  scoreField: number[][][];
}

const state = {
  dataset: 'moons' as DatasetType,
  arch: { hiddenLayers: 2, hiddenWidth: 128, activation: 'swish' as Activation, timeEmbedDim: 16 } as ArchConfig,
  hyper: {
    T: 40, lr: 0.02, batchSize: 64, totalSteps: 3000,
    checkpointInterval: 40, numTrackedPoints: 350, datasetSize: 4096,
    betaEnd: 0.02, scheduleType: 'linear' as 'linear' | 'cosine',
  } as HyperParams,
  trueData: [] as number[][],
  checkpoints: [] as Checkpoint[],
  trainingIdx: 0,
  diffusionIdx: 0,
  liveFollow: true,
  isRunning: false,
  isInitialized: false,
  playTimer: null as number | null,
  showTrails: true,
  trailCount: DEFAULT_TRAIL_COUNT as number | null, // null = draw all tracked points' trails
  showScoreField: false,
  selectedIndex: null as number | null,
  sampler: 'ancestral' as Sampler, // visualization-only choice, reuses whichever trajectory the worker already computed
};

// current checkpoint's trajectory for whichever sampler is selected
function currentTrajectory(ckpt: Checkpoint | undefined): number[][][] | null {
  if (!ckpt) return null;
  return state.sampler === 'ddim' ? ckpt.trajectory.ddim : ckpt.trajectory.ancestral;
}

// ---------------------------------------------------------------------------
// DOM scaffold
// ---------------------------------------------------------------------------
const app = document.getElementById('app')!;
app.innerHTML = `
  <aside class="sidebar">
    <div class="brand"><span class="brand-mark">&nabla;</span> Diffusion Explorer</div>

    <div>
      <p class="section-label">Dataset</p>
      <div class="pill-group" id="dataset-group"></div>
      <div class="field" style="margin-top: 0.85rem;">
        <div class="row-between"><label>Training points</label></div>
        <select id="dataset-size-select">
          <option value="256">256</option>
          <option value="1024">1,024</option>
          <option value="4096" selected>4,096</option>
          <option value="16000">16,000</option>
        </select>
      </div>
    </div>

    <div>
      <p class="section-label">Architecture</p>
      <div class="field">
        <div class="row-between"><label>Hidden layers</label></div>
        <div class="stepper">
          <button id="layers-minus" type="button">&minus;</button>
          <span class="stepper-value" id="layers-value">2</span>
          <button id="layers-plus" type="button">+</button>
        </div>
      </div>
      <div class="field">
        <label>Hidden width</label>
        <select id="width-select">
          <option value="16">16</option>
          <option value="32">32</option>
          <option value="64">64</option>
          <option value="128" selected>128</option>
        </select>
      </div>
      <div class="field">
        <label>Activation</label>
        <div class="pill-group" id="activation-group"></div>
      </div>
    </div>

    <div>
      <p class="section-label">Training</p>
      <div class="field">
        <div class="row-between"><label>Learning rate</label></div>
        <select id="lr-select">
          <option value="0.02" selected>0.02</option>
          <option value="0.01">0.01</option>
          <option value="0.005">0.005</option>
          <option value="0.002">0.002</option>
        </select>
      </div>
      <div class="field">
        <label>Diffusion steps (T)</label>
        <select id="T-select">
          <option value="20">20</option>
          <option value="40" selected>40</option>
          <option value="60">60</option>
        </select>
      </div>
      <div class="field">
        <label>Noise schedule shape</label>
        <div class="pill-group" id="schedule-type-group"></div>
      </div>
      <div class="field">
        <div class="row-between">
          <label id="beta-end-label">Noise variance (&beta;<sub>T</sub>)</label>
        </div>
        <select id="beta-end-select">
          <option value="0.01">0.01 &mdash; gentle</option>
          <option value="0.02" selected>0.02 &mdash; default</option>
          <option value="0.05">0.05 &mdash; strong</option>
          <option value="0.1">0.1 &mdash; aggressive</option>
        </select>
      </div>
      <div class="field">
        <label>Schedule preview</label>
        <div style="height:52px;"><canvas id="schedule-canvas" style="width:100%;height:100%;"></canvas></div>
        <p class="hint" style="margin-top:4px;">
          <span style="color:#e8593c">&#9632;</span> &beta;<sub>t</sub> &nbsp;
          <span style="color:#37d6b3">&#9632;</span> &alpha;&#772;<sub>t</sub>
        </p>
      </div>
      <div class="field">
        <label>Total training steps</label>
        <select id="steps-select">
          <option value="800">800</option>
          <option value="1500">1500</option>
          <option value="3000" selected>3000</option>
          <option value="6000">6000</option>
        </select>
      </div>
      <div class="btn-row">
        <button class="btn primary" id="start-btn">Start</button>
        <button class="btn" id="pause-btn" disabled>Pause</button>
        <button class="btn icon-only" id="reset-btn" title="Reset">&#8635;</button>
      </div>
      <p class="hint">Changing dataset, architecture, or training settings resets the run.</p>
    </div>
  </aside>

  <main class="main">
    <div class="panel">
      <div class="panel-header">
        <h2>Learned distribution</h2>
        <span class="status-chip" id="status-chip">idle</span>
      </div>
      <div class="canvas-wrap"><canvas id="scene-canvas"></canvas></div>
      <div class="legend">
        <span class="legend-item"><span class="legend-dot" style="background:rgba(231,233,238,0.5)"></span>True data</span>
        <span class="legend-item"><span class="legend-dot" style="background:#f0a73b"></span>Model samples</span>
        <label class="legend-item" style="cursor:pointer;">
          <input type="checkbox" id="trails-toggle" checked style="width:auto;" />
          Trails
        </label>
        <label class="legend-item" style="cursor:pointer;">
          <input type="checkbox" id="scorefield-toggle" style="width:auto;" />
          Score field
        </label>
      </div>
      <div class="scrubber-row" style="margin-top: 8px; gap: 14px; flex-wrap: wrap;">
        <div style="display:flex;align-items:center;gap:8px;">
          <label style="font-size:12.5px;color:var(--text-dim);">Sampler</label>
          <div class="pill-group" id="sampler-group" style="grid-template-columns: 1fr 1fr; width: 160px;"></div>
        </div>
        <div style="display:flex;align-items:center;gap:8px;">
          <label for="trail-count-select" style="font-size:12.5px;color:var(--text-dim);">Trail count</label>
          <select id="trail-count-select" style="width:auto;">
            <option value="20">20</option>
            <option value="50">50</option>
            <option value="70" selected>70</option>
            <option value="150">150</option>
            <option value="all">All (${state.hyper.numTrackedPoints})</option>
          </select>
        </div>
      </div>
      <p class="hint" id="selection-hint">Click a sample dot to isolate its trajectory. Trails are colored by curvature — cool = straight, warm = curved.</p>

      <div class="scrubber">
        <label>Training step</label>
        <input type="range" id="training-slider" min="0" max="0" value="0" step="1" disabled />
        <span class="readout" id="training-readout">&mdash;</span>
      </div>
      <div class="scrubber">
        <label>Diffusion step t</label>
        <div class="scrubber-row" style="grid-column: 2 / 3;">
          <input type="range" id="diffusion-slider" min="0" max="40" value="40" step="1" disabled />
        </div>
        <span class="readout" id="diffusion-readout">&mdash;</span>
      </div>
      <div class="scrubber-row" style="margin-top: 6px;">
        <button class="play-btn" id="play-diffusion-btn" title="Play reverse diffusion" disabled>&#9654;</button>
        <label style="display:flex;align-items:center;gap:6px;font-size:12.5px;color:var(--text-dim);cursor:pointer;">
          <input type="checkbox" id="live-follow" checked style="width:auto;" />
          Follow latest checkpoint
        </label>
      </div>
    </div>

    <div class="panel">
      <div class="panel-header">
        <h2>Training loss &amp; distance to target</h2>
      </div>
      <div class="loss-wrap"><canvas id="loss-canvas"></canvas></div>
      <p class="hint" style="margin-top:6px;">
        <span style="color:#37d6b3">&#9632;</span> Loss &nbsp;
        <span style="color:#e8593c">&#9632;</span> Distance to target (sliced-Wasserstein)
      </p>
      <div class="stat-row">
        <div class="stat"><span class="label">Step</span><span class="val" id="stat-step">0 / 0</span></div>
        <div class="stat"><span class="label">Loss</span><span class="val" id="stat-loss">&mdash;</span></div>
        <div class="stat"><span class="label">Distance</span><span class="val" id="stat-distance">&mdash;</span></div>
        <div class="stat"><span class="label">Checkpoints</span><span class="val" id="stat-ckpts">0</span></div>
      </div>
    </div>
  </main>
`;

// ---------------------------------------------------------------------------
// Pill groups (dataset + activation + schedule-type + sampler pickers)
// ---------------------------------------------------------------------------
function buildPillGroup(
  containerId: string,
  options: { value: string; label: string }[],
  active: string,
  onSelect: (value: string) => void
) {
  const el = document.getElementById(containerId)!;
  el.innerHTML = '';
  for (const opt of options) {
    const pill = document.createElement('div');
    pill.className = 'pill' + (opt.value === active ? ' active' : '');
    pill.textContent = opt.label;
    pill.onclick = () => {
      onSelect(opt.value);
      buildPillGroup(containerId, options, opt.value, onSelect);
    };
    el.appendChild(pill);
  }
}

buildPillGroup(
  'dataset-group',
  (Object.keys(DATASET_LABELS) as DatasetType[]).map(k => ({ value: k, label: DATASET_LABELS[k] })),
  state.dataset,
  v => { state.dataset = v as DatasetType; resetRun(); }
);

buildPillGroup(
  'activation-group',
  [{ value: 'relu', label: 'ReLU' }, { value: 'tanh', label: 'tanh' }, { value: 'swish', label: 'swish' }],
  state.arch.activation,
  v => { state.arch.activation = v as Activation; resetRun(); }
);

buildPillGroup(
  'schedule-type-group',
  [{ value: 'linear', label: 'Linear' }, { value: 'cosine', label: 'Cosine' }],
  state.hyper.scheduleType,
  v => {
    state.hyper.scheduleType = v as 'linear' | 'cosine';
    betaEndSelect.disabled = v === 'cosine';
    betaEndLabel.style.opacity = v === 'cosine' ? '0.45' : '1';
    updateScheduleChart();
    resetRun();
  }
);

buildPillGroup(
  'sampler-group',
  [{ value: 'ancestral', label: 'Ancestral' }, { value: 'ddim', label: 'DDIM' }],
  state.sampler,
  v => { state.sampler = v as Sampler; redrawScene(); }
);

// ---------------------------------------------------------------------------
// Element refs
// ---------------------------------------------------------------------------
const layersMinus = document.getElementById('layers-minus') as HTMLButtonElement;
const layersPlus = document.getElementById('layers-plus') as HTMLButtonElement;
const layersValue = document.getElementById('layers-value')!;
const widthSelect = document.getElementById('width-select') as HTMLSelectElement;
const datasetSizeSelect = document.getElementById('dataset-size-select') as HTMLSelectElement;
const lrSelect = document.getElementById('lr-select') as HTMLSelectElement;
const tSelect = document.getElementById('T-select') as HTMLSelectElement;
const betaEndSelect = document.getElementById('beta-end-select') as HTMLSelectElement;
const betaEndLabel = document.getElementById('beta-end-label') as HTMLLabelElement;
const stepsSelect = document.getElementById('steps-select') as HTMLSelectElement;
const startBtn = document.getElementById('start-btn') as HTMLButtonElement;
const pauseBtn = document.getElementById('pause-btn') as HTMLButtonElement;
const resetBtn = document.getElementById('reset-btn') as HTMLButtonElement;
const statusChip = document.getElementById('status-chip')!;
const trainingSlider = document.getElementById('training-slider') as HTMLInputElement;
const trainingReadout = document.getElementById('training-readout')!;
const diffusionSlider = document.getElementById('diffusion-slider') as HTMLInputElement;
const diffusionReadout = document.getElementById('diffusion-readout')!;
const playDiffusionBtn = document.getElementById('play-diffusion-btn') as HTMLButtonElement;
const liveFollowCheckbox = document.getElementById('live-follow') as HTMLInputElement;
const statStep = document.getElementById('stat-step')!;
const statLoss = document.getElementById('stat-loss')!;
const statDistance = document.getElementById('stat-distance')!;
const statCkpts = document.getElementById('stat-ckpts')!;
const trailsToggle = document.getElementById('trails-toggle') as HTMLInputElement;
const trailCountSelect = document.getElementById('trail-count-select') as HTMLSelectElement;
const scoreFieldToggle = document.getElementById('scorefield-toggle') as HTMLInputElement;
const selectionHint = document.getElementById('selection-hint')!;

const sceneCanvas = document.getElementById('scene-canvas') as HTMLCanvasElement;
const lossCanvas = document.getElementById('loss-canvas') as HTMLCanvasElement;
const scheduleCanvas = document.getElementById('schedule-canvas') as HTMLCanvasElement;
const renderer = new SceneRenderer(sceneCanvas);
const lossChart = new LossChart(lossCanvas);
const scheduleChart = new ScheduleChart(scheduleCanvas);

function updateScheduleChart() {
  const sched = makeSchedule(state.hyper.T, 1e-4, state.hyper.betaEnd, state.hyper.scheduleType);
  scheduleChart.draw(sched);
}

layersMinus.onclick = () => { state.arch.hiddenLayers = Math.max(1, state.arch.hiddenLayers - 1); layersValue.textContent = String(state.arch.hiddenLayers); resetRun(); };
layersPlus.onclick = () => { state.arch.hiddenLayers = Math.min(4, state.arch.hiddenLayers + 1); layersValue.textContent = String(state.arch.hiddenLayers); resetRun(); };
widthSelect.onchange = () => { state.arch.hiddenWidth = Number(widthSelect.value); resetRun(); };
datasetSizeSelect.onchange = () => { state.hyper.datasetSize = Number(datasetSizeSelect.value); resetRun(); };
lrSelect.onchange = () => { state.hyper.lr = Number(lrSelect.value); resetRun(); };
tSelect.onchange = () => {
  state.hyper.T = Number(tSelect.value);
  diffusionSlider.max = String(state.hyper.T);
  updateScheduleChart();
  resetRun();
};
betaEndSelect.onchange = () => { state.hyper.betaEnd = Number(betaEndSelect.value); updateScheduleChart(); resetRun(); };
stepsSelect.onchange = () => { state.hyper.totalSteps = Number(stepsSelect.value); resetRun(); };

updateScheduleChart(); // draw the initial linear-schedule preview before any run starts

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------
let worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });

function attachWorkerHandlers() {
  worker.onmessage = (e: MessageEvent<WorkerOutbound>) => {
    const msg = e.data;
    if (msg.type === 'ready') {
      state.trueData = msg.trueData;
      state.isInitialized = true;
      redrawScene();
    } else if (msg.type === 'checkpoint') {
      state.checkpoints.push({
        step: msg.step, loss: msg.loss, distance: msg.distance,
        trajectory: msg.trajectory, scoreField: msg.scoreField,
      });
      statCkpts.textContent = String(state.checkpoints.length);
      trainingSlider.max = String(state.checkpoints.length - 1);
      trainingSlider.disabled = false;
      diffusionSlider.disabled = false;
      playDiffusionBtn.disabled = false;
      if (state.liveFollow) {
        state.trainingIdx = state.checkpoints.length - 1;
        state.diffusionIdx = Number(diffusionSlider.max);
        trainingSlider.value = String(state.trainingIdx);
        diffusionSlider.value = String(state.diffusionIdx);
      }
      statStep.textContent = `${msg.step} / ${msg.totalSteps}`;
      statLoss.textContent = msg.loss.toFixed(4);
      statDistance.textContent = msg.distance.toFixed(4);
      redrawScene();
      redrawLoss();
    } else if (msg.type === 'paused' || msg.type === 'done') {
      state.isRunning = false;
      startBtn.disabled = false;
      startBtn.textContent = msg.type === 'done' ? 'Completed' : 'Resume';
      pauseBtn.disabled = true;
      statusChip.textContent = msg.type === 'done' ? 'done' : 'paused';
      statusChip.classList.remove('training');
      setConfigDisabled(false);
      if (msg.type === 'done' && state.selectedIndex === null) {
        selectionHint.textContent = 'Training complete — click any generated point to trace how it formed, from noise to data.';
      }
    }
  };
}
attachWorkerHandlers();

function sendInit() {
  worker.postMessage({ type: 'init', dataset: state.dataset, arch: state.arch, hyper: state.hyper });
}
sendInit();

// ---------------------------------------------------------------------------
// Controls: start / pause / reset
// ---------------------------------------------------------------------------
function setConfigDisabled(disabled: boolean) {
  document.querySelectorAll<HTMLElement>('.sidebar select, .sidebar .pill, .sidebar .stepper button')
    .forEach(el => {
      if (disabled) el.setAttribute('data-locked', '1'); else el.removeAttribute('data-locked');
      (el as any).style.pointerEvents = disabled ? 'none' : '';
      (el as any).style.opacity = disabled ? '0.45' : '';
    });
}

startBtn.onclick = () => {
  state.isRunning = true;
  startBtn.disabled = true;
  startBtn.textContent = 'Training…';
  pauseBtn.disabled = false;
  statusChip.textContent = 'training';
  statusChip.classList.add('training');
  setConfigDisabled(true);
  worker.postMessage({ type: 'start' });
};

pauseBtn.onclick = () => {
  worker.postMessage({ type: 'pause' });
};

resetBtn.onclick = () => resetRun();

function resetRun() {
  worker.terminate();
  worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
  attachWorkerHandlers();

  state.checkpoints = [];
  state.trainingIdx = 0;
  state.diffusionIdx = Number(diffusionSlider.max);
  state.isRunning = false;
  state.isInitialized = false;
  state.selectedIndex = null;
  selectionHint.textContent = 'Click a sample dot to isolate its trajectory.';
  stopPlayback();

  trainingSlider.max = '0';
  trainingSlider.value = '0';
  trainingSlider.disabled = true;
  diffusionSlider.disabled = true;
  playDiffusionBtn.disabled = true;
  startBtn.disabled = false;
  startBtn.textContent = 'Start';
  pauseBtn.disabled = true;
  statusChip.textContent = 'idle';
  statusChip.classList.remove('training');
  statStep.textContent = `0 / ${state.hyper.totalSteps}`;
  statLoss.textContent = '—';
  statDistance.textContent = '—';
  statCkpts.textContent = '0';
  setConfigDisabled(false);

  sendInit();
  updateScheduleChart();
  redrawScene();
  redrawLoss();
}

// ---------------------------------------------------------------------------
// Scrubbers
// ---------------------------------------------------------------------------
trainingSlider.oninput = () => {
  state.trainingIdx = Number(trainingSlider.value);
  liveFollowCheckbox.checked = state.trainingIdx === state.checkpoints.length - 1;
  state.liveFollow = liveFollowCheckbox.checked;
  redrawScene();
  redrawLoss();
};

diffusionSlider.oninput = () => {
  state.diffusionIdx = Number(diffusionSlider.value);
  redrawScene();
};

liveFollowCheckbox.onchange = () => {
  state.liveFollow = liveFollowCheckbox.checked;
  if (state.liveFollow && state.checkpoints.length > 0) {
    state.trainingIdx = state.checkpoints.length - 1;
    trainingSlider.value = String(state.trainingIdx);
    redrawScene();
  }
};

trailsToggle.onchange = () => {
  state.showTrails = trailsToggle.checked;
  trailCountSelect.disabled = !trailsToggle.checked;
  redrawScene();
};
trailCountSelect.onchange = () => {
  state.trailCount = trailCountSelect.value === 'all' ? null : Number(trailCountSelect.value);
  redrawScene();
};
scoreFieldToggle.onchange = () => { state.showScoreField = scoreFieldToggle.checked; redrawScene(); };

sceneCanvas.onclick = (e: MouseEvent) => {
  const ckpt = state.checkpoints[state.trainingIdx];
  const traj = currentTrajectory(ckpt);
  if (!traj) return;
  const points = traj[Math.min(state.diffusionIdx, traj.length - 1)];
  const [dataX, dataY] = renderer.pixelToData(e.clientX, e.clientY);

  let nearest = -1;
  let nearestDist = Infinity;
  for (let i = 0; i < points.length; i++) {
    const d = Math.hypot(points[i][0] - dataX, points[i][1] - dataY);
    if (d < nearestDist) { nearestDist = d; nearest = i; }
  }
  // ~5% of the view range as a click tolerance, in data units
  const CLICK_TOLERANCE = 0.2;
  if (nearest >= 0 && nearestDist < CLICK_TOLERANCE && state.selectedIndex !== nearest) {
    state.selectedIndex = nearest;
    selectionHint.textContent = `Tracing sample #${nearest} from noise to data. Click empty space to deselect.`;
    // Jump back to pure noise and replay forward so picking a point
    // immediately shows how it got there, one diffusion step at a time.
    startDiffusionPlayback();
  } else {
    state.selectedIndex = null;
    selectionHint.textContent = state.checkpoints.length && state.trainingIdx === state.checkpoints.length - 1 && !state.isRunning
      ? 'Click any generated point to trace how it formed, from noise to data.'
      : 'Click a sample dot to isolate its trajectory.';
    redrawScene();
  }
};

function stopPlayback() {
  if (state.playTimer !== null) {
    window.clearInterval(state.playTimer);
    state.playTimer = null;
    playDiffusionBtn.innerHTML = '&#9654;';
  }
}

function startDiffusionPlayback() {
  stopPlayback();
  playDiffusionBtn.innerHTML = '&#10074;&#10074;';
  diffusionSlider.value = '0';
  state.diffusionIdx = 0;
  redrawScene();
  state.playTimer = window.setInterval(() => {
    const max = Number(diffusionSlider.max);
    state.diffusionIdx = Math.min(max, state.diffusionIdx + 1);
    diffusionSlider.value = String(state.diffusionIdx);
    redrawScene();
    if (state.diffusionIdx >= max) stopPlayback();
  }, 90);
}

playDiffusionBtn.onclick = () => {
  if (state.playTimer !== null) { stopPlayback(); return; }
  startDiffusionPlayback();
};

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------
function redrawScene() {
  const ckpt = state.checkpoints[state.trainingIdx];
  const traj = currentTrajectory(ckpt);
  const idx = traj ? Math.min(state.diffusionIdx, traj.length - 1) : 0;
  const points = traj ? traj[idx] : null;
  const fieldIdx = ckpt ? Math.min(state.diffusionIdx, ckpt.scoreField.length - 1) : 0;
  const field = ckpt ? ckpt.scoreField[fieldIdx] : null;

  renderer.draw({
    trueData: state.trueData,
    currentPoints: points,
    trailTrajectory: traj,
    trailUpToIndex: idx,
    showTrails: state.showTrails,
    trailCount: state.trailCount,
    scoreField: field,
    showScoreField: state.showScoreField,
    selectedIndex: state.selectedIndex,
  });

  if (ckpt) {
    trainingReadout.textContent = `${ckpt.step}`;
    const tRemaining = Number(diffusionSlider.max) - state.diffusionIdx;
    diffusionReadout.textContent = `t=${tRemaining}`;
  } else {
    trainingReadout.textContent = '—';
    diffusionReadout.textContent = '—';
  }
}

function redrawLoss() {
  const history = state.checkpoints.map(c => ({ step: c.step, loss: c.loss, distance: c.distance }));
  lossChart.draw(history, state.checkpoints.length > 0 ? state.trainingIdx : null);
}

window.addEventListener('resize', () => { redrawScene(); redrawLoss(); updateScheduleChart(); });

redrawScene();
redrawLoss();
