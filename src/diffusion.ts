import { adamStep, backward, forward, type MLP } from './nn';
import { VIEW_RANGE, SCORE_GRID_RES } from './constants';

export interface Schedule {
  T: number;
  betas: number[];      // index i corresponds to timestep t = i + 1
  alphas: number[];
  alphaBars: number[];
}

export function makeSchedule(T: number, betaStart = 1e-4, betaEnd = 0.02): Schedule {
  const betas: number[] = [];
  for (let i = 0; i < T; i++) betas.push(betaStart + ((betaEnd - betaStart) * i) / (T - 1));
  const alphas = betas.map(b => 1 - b);
  const alphaBars: number[] = [];
  let acc = 1;
  for (const a of alphas) { acc *= a; alphaBars.push(acc); }
  return { T, betas, alphas, alphaBars };
}

// Seeded PRNG (mulberry32) so the same noise draws can be reused across checkpoints
export function mulberry32(seed: number) {
  let s = seed >>> 0;
  return function () {
    s |= 0; s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussian(rng: () => number): number {
  const u1 = Math.max(rng(), 1e-9);
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

export function timeEmbedding(t: number, dim: number): number[] {
  const half = Math.floor(dim / 2);
  const emb: number[] = new Array(dim);
  for (let i = 0; i < half; i++) {
    const freq = Math.exp((-Math.log(10000) * i) / half);
    const arg = t * freq;
    emb[i] = Math.sin(arg);
    emb[half + i] = Math.cos(arg);
  }
  if (dim % 2 === 1) emb[dim - 1] = 0;
  return emb;
}

export function qSample(x0: [number, number], t: number, noise: [number, number], sched: Schedule): [number, number] {
  const ab = sched.alphaBars[t - 1];
  const sqrtAb = Math.sqrt(ab);
  const sqrt1mAb = Math.sqrt(1 - ab);
  return [sqrtAb * x0[0] + sqrt1mAb * noise[0], sqrtAb * x0[1] + sqrt1mAb * noise[1]];
}

export function trainStep(
  mlp: MLP,
  batch: number[][],
  sched: Schedule,
  timeDim: number,
  lr: number
): number {
  const gradsAccum = mlp.layers.map(l => ({
    dW: l.W.map(row => row.map(() => 0)),
    db: l.b.map(() => 0),
  }));
  let totalLoss = 0;

  for (const x0 of batch) {
    const t = 1 + Math.floor(Math.random() * sched.T);
    const noise: [number, number] = [gaussian(Math.random), gaussian(Math.random)];
    const xt = qSample(x0 as [number, number], t, noise, sched);
    const emb = timeEmbedding(t, timeDim);
    const input = [xt[0], xt[1], ...emb];

    const cache = forward(mlp, input);
    const pred = cache.acts[cache.acts.length - 1];
    const dOut = [pred[0] - noise[0], pred[1] - noise[1]];
    totalLoss += dOut[0] * dOut[0] + dOut[1] * dOut[1];

    const grads = backward(mlp, cache, dOut);
    for (let l = 0; l < grads.length; l++) {
      for (let o = 0; o < grads[l].dW.length; o++) {
        for (let i = 0; i < grads[l].dW[o].length; i++) gradsAccum[l].dW[o][i] += grads[l].dW[o][i];
        gradsAccum[l].db[o] += grads[l].db[o];
      }
    }
  }

  const B = batch.length;
  for (const g of gradsAccum) {
    for (const row of g.dW) for (let i = 0; i < row.length; i++) row[i] /= B;
    for (let i = 0; i < g.db.length; i++) g.db[i] /= B;
  }
  adamStep(mlp, gradsAccum, lr);
  return totalLoss / B;
}

// Ancestral DDPM sampling starting from a fixed seed, recording every intermediate
// step so the same underlying noise draws can be replayed at any training checkpoint.
export function reverseSampleTrajectory(
  mlp: MLP,
  sched: Schedule,
  timeDim: number,
  numPoints: number,
  seed: number
): number[][][] {
  const rng = mulberry32(seed);
  let points: number[][] = Array.from({ length: numPoints }, () => [gaussian(rng), gaussian(rng)]);
  const trajectory: number[][][] = [points.map(p => p.slice())];

  for (let t = sched.T; t >= 1; t--) {
    const alpha = sched.alphas[t - 1];
    const alphaBar = sched.alphaBars[t - 1];
    const beta = sched.betas[t - 1];
    const emb = timeEmbedding(t, timeDim);
    const coef = beta / Math.sqrt(1 - alphaBar);
    const invSqrtAlpha = 1 / Math.sqrt(alpha);
    const sqrtBeta = Math.sqrt(beta);

    const next: number[][] = new Array(points.length);
    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      const input = [p[0], p[1], ...emb];
      const { acts } = forward(mlp, input);
      const predNoise = acts[acts.length - 1];
      let nx = (p[0] - coef * predNoise[0]) * invSqrtAlpha;
      let ny = (p[1] - coef * predNoise[1]) * invSqrtAlpha;
      if (t > 1) {
        nx += sqrtBeta * gaussian(rng);
        ny += sqrtBeta * gaussian(rng);
      }
      next[i] = [nx, ny];
    }
    points = next;
    trajectory.push(points.map(p => p.slice()));
  }
  return trajectory; // trajectory[0] = pure noise (t=T) ... trajectory[T] = generated sample (t=0)
}

// One row per grid point: [x, y, dx, dy]. dx/dy is the network's predicted
// denoising direction (negative predicted noise) — proportional to the score
// function ∇x log p(x) at that point and noise level. Indexed the same way
// as reverseSampleTrajectory's output (index k -> timestep t = T - k), so a
// UI can drive the trajectory scrubber and the field scrubber together.
export function computeScoreFieldTrajectory(
  mlp: MLP,
  sched: Schedule,
  timeDim: number
): number[][][] {
  const gridPositions: [number, number][] = [];
  for (let gy = 0; gy < SCORE_GRID_RES; gy++) {
    for (let gx = 0; gx < SCORE_GRID_RES; gx++) {
      const x = -VIEW_RANGE + ((gx + 0.5) / SCORE_GRID_RES) * (2 * VIEW_RANGE);
      const y = -VIEW_RANGE + ((gy + 0.5) / SCORE_GRID_RES) * (2 * VIEW_RANGE);
      gridPositions.push([x, y]);
    }
  }

  const fields: number[][][] = [];
  for (let t = sched.T; t >= 1; t--) {
    const emb = timeEmbedding(t, timeDim);
    const rows: number[][] = new Array(gridPositions.length);
    for (let i = 0; i < gridPositions.length; i++) {
      const [x, y] = gridPositions[i];
      const { acts } = forward(mlp, [x, y, ...emb]);
      const predNoise = acts[acts.length - 1];
      rows[i] = [x, y, -predNoise[0], -predNoise[1]];
    }
    fields.push(rows);
  }
  fields.push(fields[fields.length - 1]); // pad to length T+1 to match trajectory indexing
  return fields;
}
