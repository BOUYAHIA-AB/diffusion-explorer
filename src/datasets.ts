import type { DatasetType } from './types';

// All datasets are generated to roughly fill [-1.6, 1.6]^2 so a single
// fixed view range works for every option in the picker.

function gaussianPair(): [number, number] {
  // Box-Muller
  const u1 = Math.max(Math.random(), 1e-9);
  const u2 = Math.random();
  const r = Math.sqrt(-2 * Math.log(u1));
  return [r * Math.cos(2 * Math.PI * u2), r * Math.sin(2 * Math.PI * u2)];
}

function twoMoons(n: number): number[][] {
  const pts: number[][] = [];
  for (let i = 0; i < n; i++) {
    const upper = Math.random() < 0.5;
    const t = Math.random() * Math.PI;
    let x: number, y: number;
    if (upper) {
      x = Math.cos(t);
      y = Math.sin(t);
    } else {
      x = 1 - Math.cos(t);
      y = 1 - Math.sin(t) - 0.5;
    }
    const [nx, ny] = gaussianPair();
    pts.push([x * 1.3 + nx * 0.06, y * 1.3 + ny * 0.06]);
  }
  return pts;
}

function gaussianMixture(n: number, k = 8): number[][] {
  const pts: number[][] = [];
  for (let i = 0; i < n; i++) {
    const mode = Math.floor(Math.random() * k);
    const angle = (mode / k) * Math.PI * 2;
    const cx = Math.cos(angle) * 1.25;
    const cy = Math.sin(angle) * 1.25;
    const [nx, ny] = gaussianPair();
    pts.push([cx + nx * 0.13, cy + ny * 0.13]);
  }
  return pts;
}

function spiral(n: number): number[][] {
  const pts: number[][] = [];
  for (let i = 0; i < n; i++) {
    const sign = Math.random() < 0.5 ? 1 : -1;
    const t = Math.sqrt(Math.random()) * 2.6 * Math.PI;
    const r = t / (2.6 * Math.PI);
    const [nx, ny] = gaussianPair();
    const x = sign * r * Math.cos(t) * 1.5 + nx * 0.03;
    const y = sign * r * Math.sin(t) * 1.5 + ny * 0.03;
    pts.push([x, y]);
  }
  return pts;
}

function checkerboard(n: number): number[][] {
  const pts: number[][] = [];
  const cell = 0.8;
  while (pts.length < n) {
    const x = (Math.random() * 2 - 1) * 2;
    const y = (Math.random() * 2 - 1) * 2;
    const cx = Math.floor(x / cell);
    const cy = Math.floor(y / cell);
    if ((cx + cy) % 2 === 0) pts.push([x, y]);
  }
  return pts;
}

export function generateDataset(type: DatasetType, n: number): number[][] {
  switch (type) {
    case 'moons': return twoMoons(n);
    case 'gaussians': return gaussianMixture(n);
    case 'spiral': return spiral(n);
    case 'checkerboard': return checkerboard(n);
  }
}

export const DATASET_LABELS: Record<DatasetType, string> = {
  moons: 'Two moons',
  gaussians: '8 Gaussians',
  spiral: 'Spiral',
  checkerboard: 'Checkerboard',
};
