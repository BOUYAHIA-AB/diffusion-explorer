/// <reference lib="webworker" />
import { createMLP, cloneLayers, updateEma, type MLP, type Layer } from './nn';
import {
  makeSchedule, trainStep, reverseSampleTrajectory, reverseSampleTrajectoryDDIM,
  computeScoreFieldTrajectory, slicedWasserstein, mulberry32, type Schedule,
} from './diffusion';
import { generateDataset } from './datasets';
import { EMA_DECAY } from './constants';
import type { InitMessage, WorkerInbound, ArchConfig, HyperParams, DatasetType } from './types';

let mlp: MLP;
let emaLayers: Layer[];
let sched: Schedule;
let hyper: HyperParams;
let arch: ArchConfig;
let dataset: number[][] = [];
let step = 0;
let paused = true;
let running = false;
const NOISE_SEED = 42;      // fixed so every checkpoint's reverse trajectory starts from the same noise
const DISTANCE_RNG_SEED = 43; // separate fixed seed for the sliced-Wasserstein projection directions

function initRun(msg: InitMessage) {
  arch = msg.arch;
  hyper = msg.hyper;
  const datasetType: DatasetType = msg.dataset;

  dataset = generateDataset(datasetType, hyper.datasetSize);
  const hiddenDims = new Array(arch.hiddenLayers).fill(arch.hiddenWidth);
  mlp = createMLP(2 + arch.timeEmbedDim, hiddenDims, 2, arch.activation);
  emaLayers = cloneLayers(mlp.layers); // EMA starts equal to the (random) initial weights
  sched = makeSchedule(hyper.T, 1e-4, hyper.betaEnd, hyper.scheduleType);
  step = 0;
  paused = true;
  running = false;

  (postMessage as (m: unknown) => void)({ type: 'ready', trueData: dataset.slice(0, 1500) });
}

async function trainLoop() {
  if (running) return;
  running = true;
  paused = false;

  while (step < hyper.totalSteps && !paused) {
    const batch: number[][] = new Array(hyper.batchSize);
    for (let i = 0; i < hyper.batchSize; i++) {
      batch[i] = dataset[Math.floor(Math.random() * dataset.length)];
    }
    const loss = trainStep(mlp, batch, sched, arch.timeEmbedDim, hyper.lr);
    step++;
    updateEma(emaLayers, mlp.layers, EMA_DECAY);

    if (step % hyper.checkpointInterval === 0 || step === hyper.totalSteps) {
      // sample from the EMA-smoothed weights, not the raw live weights —
      // standard diffusion-training practice, since the live weights are
      // noisier step-to-step than what you actually want to generate from
      const emaMlp = { layers: emaLayers } as unknown as MLP;
      const ancestral = reverseSampleTrajectory(emaMlp, sched, arch.timeEmbedDim, hyper.numTrackedPoints, NOISE_SEED);
      const ddim = reverseSampleTrajectoryDDIM(emaMlp, sched, arch.timeEmbedDim, hyper.numTrackedPoints, NOISE_SEED);
      const scoreField = computeScoreFieldTrajectory(emaMlp, sched, arch.timeEmbedDim);

      // distance metric compares generated samples (ancestral final step, the
      // canonical "what did the model actually produce") to a subsample of
      // the true training pool
      const generatedFinal = ancestral[ancestral.length - 1];
      const trueSubsample = dataset.slice(0, Math.min(dataset.length, 500));
      const distanceRng = mulberry32(DISTANCE_RNG_SEED);
      const distance = slicedWasserstein(generatedFinal, trueSubsample, 24, distanceRng);

      (postMessage as (m: unknown) => void)({
        type: 'checkpoint',
        step,
        loss,
        totalSteps: hyper.totalSteps,
        trajectory: { ancestral, ddim },
        scoreField,
        distance,
      });
      // yield so pause/reset messages and postMessage delivery aren't starved
      await new Promise(r => setTimeout(r, 0));
    } else if (step % 10 === 0) {
      await new Promise(r => setTimeout(r, 0));
    }
  }

  running = false;
  (postMessage as (m: unknown) => void)({ type: step >= hyper.totalSteps ? 'done' : 'paused', step });
}

self.onmessage = (e: MessageEvent<WorkerInbound>) => {
  const msg = e.data;
  if (msg.type === 'init') initRun(msg);
  else if (msg.type === 'start') trainLoop();
  else if (msg.type === 'pause') paused = true;
  else if (msg.type === 'reset') { step = 0; paused = true; running = false; }
};
