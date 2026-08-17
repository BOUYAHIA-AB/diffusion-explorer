/// <reference lib="webworker" />
import { createMLP, type MLP } from './nn';
import { makeSchedule, trainStep, reverseSampleTrajectory, computeScoreFieldTrajectory, type Schedule } from './diffusion';
import { generateDataset } from './datasets';
import type { InitMessage, WorkerInbound, ArchConfig, HyperParams, DatasetType } from './types';

let mlp: MLP;
let sched: Schedule;
let hyper: HyperParams;
let arch: ArchConfig;
let dataset: number[][] = [];
let step = 0;
let paused = true;
let running = false;
const NOISE_SEED = 42; // fixed so every checkpoint's reverse trajectory starts from the same noise

function initRun(msg: InitMessage) {
  arch = msg.arch;
  hyper = msg.hyper;
  const datasetType: DatasetType = msg.dataset;

  dataset = generateDataset(datasetType, hyper.datasetSize);
  const hiddenDims = new Array(arch.hiddenLayers).fill(arch.hiddenWidth);
  mlp = createMLP(2 + arch.timeEmbedDim, hiddenDims, 2, arch.activation);
  sched = makeSchedule(hyper.T, 1e-4, hyper.betaEnd);
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

    if (step % hyper.checkpointInterval === 0 || step === hyper.totalSteps) {
      const trajectory = reverseSampleTrajectory(mlp, sched, arch.timeEmbedDim, hyper.numTrackedPoints, NOISE_SEED);
      const scoreField = computeScoreFieldTrajectory(mlp, sched, arch.timeEmbedDim);
      (postMessage as (m: unknown) => void)({
        type: 'checkpoint',
        step,
        loss,
        totalSteps: hyper.totalSteps,
        trajectory,
        scoreField,
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
