import { createMLP } from '../src/nn';
import { makeSchedule, trainStep, reverseSampleTrajectory, computeScoreFieldTrajectory } from '../src/diffusion';
import { generateDataset } from '../src/datasets';
import { SCORE_GRID_RES } from '../src/constants';

const dataset = generateDataset('gaussians', 4096);
const arch = { hiddenLayers: 2, hiddenWidth: 64, activation: 'swish' as const, timeEmbedDim: 16 };
const mlp = createMLP(2 + arch.timeEmbedDim, [arch.hiddenWidth, arch.hiddenWidth], 2, arch.activation);
const sched = makeSchedule(40);

const losses: number[] = [];
for (let step = 1; step <= 1200; step++) {
  const batch: number[][] = [];
  for (let i = 0; i < 64; i++) batch.push(dataset[Math.floor(Math.random() * dataset.length)]);
  const loss = trainStep(mlp, batch, sched, arch.timeEmbedDim, 0.01);
  losses.push(loss);
  if (step % 200 === 0) {
    const avgRecent = losses.slice(-200).reduce((a, b) => a + b, 0) / 200;
    console.log(`step ${step}\tavg loss (last 200) = ${avgRecent.toFixed(4)}`);
  }
}

const early = losses.slice(0, 100).reduce((a, b) => a + b, 0) / 100;
const late = losses.slice(-100).reduce((a, b) => a + b, 0) / 100;
console.log(`\nearly avg loss: ${early.toFixed(4)}   late avg loss: ${late.toFixed(4)}`);
if (late >= early) {
  console.error('FAIL: loss did not decrease');
  process.exit(1);
}

// sample from the trained model and check the samples land near the 8 ring modes
const trajectory = reverseSampleTrajectory(mlp, sched, arch.timeEmbedDim, 500, 42);
const finalSamples = trajectory[trajectory.length - 1];
const radii = finalSamples.map(p => Math.sqrt(p[0] * p[0] + p[1] * p[1]));
const meanRadius = radii.reduce((a, b) => a + b, 0) / radii.length;
const stdRadius = Math.sqrt(radii.reduce((a, b) => a + (b - meanRadius) ** 2, 0) / radii.length);
console.log(`\nfinal sample radius: mean=${meanRadius.toFixed(3)} std=${stdRadius.toFixed(3)} (true ring radius = 1.25)`);

if (Math.abs(meanRadius - 1.25) > 0.35) {
  console.error('FAIL: generated samples are not converging near the true ring radius');
  process.exit(1);
}

console.log('\nPASS: loss decreased and samples converged toward the target distribution');

// score field: verify shape, index alignment with the trajectory, and that
// it isn't full of NaNs or all-zero vectors
const field = computeScoreFieldTrajectory(mlp, sched, arch.timeEmbedDim);
const expectedGridPoints = SCORE_GRID_RES * SCORE_GRID_RES;

if (field.length !== trajectory.length) {
  console.error(`FAIL: score field has ${field.length} steps, trajectory has ${trajectory.length} (must match for the scrubber to stay in sync)`);
  process.exit(1);
}
if (field[0].length !== expectedGridPoints) {
  console.error(`FAIL: expected ${expectedGridPoints} grid points per step, got ${field[0].length}`);
  process.exit(1);
}

let anyNaN = false;
let totalMag = 0;
for (const row of field[0]) {
  const [, , dx, dy] = row;
  if (!Number.isFinite(dx) || !Number.isFinite(dy)) anyNaN = true;
  totalMag += Math.hypot(dx, dy);
}
const avgMagHighNoise = totalMag / field[0].length;
console.log(`score field: ${field.length} steps x ${field[0].length} grid points, avg magnitude at t=T: ${avgMagHighNoise.toFixed(3)}`);

if (anyNaN) {
  console.error('FAIL: score field contains NaN or non-finite values');
  process.exit(1);
}
if (avgMagHighNoise < 1e-4) {
  console.error('FAIL: score field is essentially all zero — network is not producing a meaningful signal');
  process.exit(1);
}

console.log('PASS: score field shape and index alignment are correct, values are finite and non-trivial');

// dataset size control: verify every dataset type generates the exact requested
// count at every size offered in the UI, including the smallest (256) where
// checkerboard's rejection sampling could plausibly misbehave
import { generateDataset as genDS } from '../src/datasets';
const sizesToCheck = [256, 1024, 4096, 16000];
const typesToCheck = ['moons', 'gaussians', 'spiral', 'checkerboard'] as const;
for (const size of sizesToCheck) {
  for (const type of typesToCheck) {
    const start = Date.now();
    const pts = genDS(type, size);
    const elapsed = Date.now() - start;
    if (pts.length !== size) {
      console.error(`FAIL: ${type} at size ${size} returned ${pts.length} points`);
      process.exit(1);
    }
    if (elapsed > 2000) {
      console.error(`FAIL: ${type} at size ${size} took ${elapsed}ms — rejection sampling may be too slow at this size`);
      process.exit(1);
    }
  }
}
console.log(`PASS: all 4 dataset types generate correct counts at all 4 configurable sizes (256-16000)`);

// noise variance (betaEnd): check the schedule is well-formed at every value
// offered in the UI, and that training doesn't blow up (NaN/Inf) at the most
// aggressive setting, where large noise steps are more likely to destabilize
// a naive hand-written optimizer
import { makeSchedule as makeSched2 } from '../src/diffusion';
for (const betaEnd of [0.01, 0.02, 0.05, 0.1]) {
  const s = makeSched2(40, 1e-4, betaEnd);
  const lastBeta = s.betas[s.betas.length - 1];
  const lastAlphaBar = s.alphaBars[s.alphaBars.length - 1];
  if (Math.abs(lastBeta - betaEnd) > 1e-9) {
    console.error(`FAIL: betaEnd=${betaEnd} did not reach the requested final beta (got ${lastBeta})`);
    process.exit(1);
  }
  if (!(lastAlphaBar > 0 && lastAlphaBar < 1)) {
    console.error(`FAIL: betaEnd=${betaEnd} produced an invalid final alphaBar=${lastAlphaBar}`);
    process.exit(1);
  }
}
console.log('PASS: noise schedule is well-formed at every configurable betaEnd (0.01-0.1)');

const mlpHighNoise = createMLP(2 + arch.timeEmbedDim, [arch.hiddenWidth, arch.hiddenWidth], 2, arch.activation);
const schedHighNoise = makeSched2(40, 1e-4, 0.1); // most aggressive UI setting
let sawNaN = false;
for (let step = 1; step <= 300; step++) {
  const batch: number[][] = [];
  for (let i = 0; i < 64; i++) batch.push(dataset[Math.floor(Math.random() * dataset.length)]);
  const loss = trainStep(mlpHighNoise, batch, schedHighNoise, arch.timeEmbedDim, 0.01);
  if (!Number.isFinite(loss)) { sawNaN = true; break; }
}
if (sawNaN) {
  console.error('FAIL: training produced NaN/Inf loss at betaEnd=0.1 — the most aggressive noise setting is numerically unstable');
  process.exit(1);
}
console.log('PASS: training stays numerically stable (no NaN) at the highest configurable noise variance');

// Edge case now that dataset size is user-controllable: a dataset smaller
// than the batch size should still train fine (sampling is with replacement).
console.log('\n--- small dataset size edge case ---');
const smallDataset = generateDataset('moons', 32);
const smallMlp = createMLP(2 + arch.timeEmbedDim, [32], 2, arch.activation);
const smallSched = makeSchedule(20);
let smallLossOk = true;
for (let step = 1; step <= 300; step++) {
  const batch: number[][] = [];
  for (let i = 0; i < 64; i++) batch.push(smallDataset[Math.floor(Math.random() * smallDataset.length)]);
  const loss = trainStep(smallMlp, batch, smallSched, arch.timeEmbedDim, 0.01);
  if (!Number.isFinite(loss)) { smallLossOk = false; break; }
}
if (!smallLossOk) {
  console.error('FAIL: training on a dataset smaller than batch size produced non-finite loss');
  process.exit(1);
}
console.log('PASS: small dataset size (32 points, batch size 64) trains without NaN/crash');
