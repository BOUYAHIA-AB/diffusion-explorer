import type { Activation } from './types';

export interface Layer {
  W: number[][];   // [outDim][inDim]
  b: number[];      // [outDim]
  activation: Activation | 'linear';
}

export interface MLP {
  layers: Layer[];
  mW: number[][][]; vW: number[][][]; // Adam moment estimates, mirrors layer.W shape
  mb: number[][]; vb: number[][];
  t: number; // Adam timestep
}

function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z));
}

function applyAct(act: Activation | 'linear', z: number): number {
  switch (act) {
    case 'relu': return z > 0 ? z : 0;
    case 'tanh': return Math.tanh(z);
    case 'swish': return z * sigmoid(z);
    case 'linear': return z;
  }
}

// derivative with respect to the pre-activation z
function actDeriv(act: Activation | 'linear', z: number): number {
  switch (act) {
    case 'relu': return z > 0 ? 1 : 0;
    case 'tanh': { const th = Math.tanh(z); return 1 - th * th; }
    case 'swish': { const s = sigmoid(z); return s + z * s * (1 - s); }
    case 'linear': return 1;
  }
}

function randInit(fanIn: number, fanOut: number): number {
  // small Xavier-ish init
  const scale = Math.sqrt(2 / (fanIn + fanOut));
  return (Math.random() * 2 - 1) * scale;
}

export function createMLP(
  inputDim: number,
  hiddenDims: number[],
  outputDim: number,
  activation: Activation
): MLP {
  const dims = [inputDim, ...hiddenDims, outputDim];
  const layers: Layer[] = [];
  for (let l = 0; l < dims.length - 1; l++) {
    const inD = dims[l];
    const outD = dims[l + 1];
    const isOutput = l === dims.length - 2;
    const W: number[][] = Array.from({ length: outD }, () =>
      Array.from({ length: inD }, () => randInit(inD, outD))
    );
    const b: number[] = new Array(outD).fill(0);
    layers.push({ W, b, activation: isOutput ? 'linear' : activation });
  }
  return {
    layers,
    mW: layers.map(l => l.W.map(r => r.map(() => 0))),
    vW: layers.map(l => l.W.map(r => r.map(() => 0))),
    mb: layers.map(l => l.b.map(() => 0)),
    vb: layers.map(l => l.b.map(() => 0)),
    t: 0,
  };
}

export function cloneLayers(layers: Layer[]): Layer[] {
  return layers.map(l => ({ W: l.W.map(row => row.slice()), b: l.b.slice(), activation: l.activation }));
}

// Exponential moving average of weights, in place on emaLayers. Standard
// practice in diffusion training: sampling from the raw, noisy step-to-step
// SGD weights tends to look worse than sampling from a smoothed average of
// recent weights, since each individual training step only sees one random
// (x, t, noise) draw and is inherently a bit noisy.
export function updateEma(emaLayers: Layer[], liveLayers: Layer[], decay: number) {
  for (let l = 0; l < liveLayers.length; l++) {
    const live = liveLayers[l], ema = emaLayers[l];
    for (let o = 0; o < live.W.length; o++) {
      for (let i = 0; i < live.W[o].length; i++) {
        ema.W[o][i] = decay * ema.W[o][i] + (1 - decay) * live.W[o][i];
      }
      ema.b[o] = decay * ema.b[o] + (1 - decay) * live.b[o];
    }
  }
}

export interface ForwardCache {
  acts: number[][];   // acts[0] = input, acts[l+1] = output of layer l (post-activation)
  preActs: number[][]; // preActs[l] = pre-activation z of layer l
}

export function forward(mlp: MLP, x: number[]): ForwardCache {
  const acts: number[][] = [x];
  const preActs: number[][] = [];
  let a = x;
  for (const layer of mlp.layers) {
    const z: number[] = new Array(layer.b.length);
    for (let o = 0; o < layer.W.length; o++) {
      let sum = layer.b[o];
      const row = layer.W[o];
      for (let i = 0; i < row.length; i++) sum += row[i] * a[i];
      z[o] = sum;
    }
    preActs.push(z);
    const out = z.map(v => applyAct(layer.activation, v));
    acts.push(out);
    a = out;
  }
  return { acts, preActs };
}

export interface LayerGrad { dW: number[][]; db: number[]; }

// dLoss/dOutput of the final layer (post-activation, which is linear so == pre-activation grad)
export function backward(mlp: MLP, cache: ForwardCache, dOutFinal: number[]): LayerGrad[] {
  const grads: LayerGrad[] = [];
  let dA = dOutFinal;
  for (let l = mlp.layers.length - 1; l >= 0; l--) {
    const layer = mlp.layers[l];
    const z = cache.preActs[l];
    const aPrev = cache.acts[l];
    const dZ = z.map((zi, idx) => dA[idx] * actDeriv(layer.activation, zi));
    const dW: number[][] = layer.W.map((row, o) => row.map((_, i) => dZ[o] * aPrev[i]));
    const db: number[] = dZ.slice();
    grads.unshift({ dW, db });

    const dAprev = new Array(aPrev.length).fill(0);
    for (let o = 0; o < layer.W.length; o++) {
      const row = layer.W[o];
      const dz = dZ[o];
      for (let i = 0; i < row.length; i++) dAprev[i] += dz * row[i];
    }
    dA = dAprev;
  }
  return grads;
}

export function adamStep(mlp: MLP, grads: LayerGrad[], lr: number, beta1 = 0.9, beta2 = 0.999, eps = 1e-8) {
  mlp.t += 1;
  const bc1 = 1 - Math.pow(beta1, mlp.t);
  const bc2 = 1 - Math.pow(beta2, mlp.t);
  for (let l = 0; l < mlp.layers.length; l++) {
    const layer = mlp.layers[l];
    const g = grads[l];
    for (let o = 0; o < layer.W.length; o++) {
      for (let i = 0; i < layer.W[o].length; i++) {
        const gr = g.dW[o][i];
        mlp.mW[l][o][i] = beta1 * mlp.mW[l][o][i] + (1 - beta1) * gr;
        mlp.vW[l][o][i] = beta2 * mlp.vW[l][o][i] + (1 - beta2) * gr * gr;
        const mHat = mlp.mW[l][o][i] / bc1;
        const vHat = mlp.vW[l][o][i] / bc2;
        layer.W[o][i] -= (lr * mHat) / (Math.sqrt(vHat) + eps);
      }
      const gb = g.db[o];
      mlp.mb[l][o] = beta1 * mlp.mb[l][o] + (1 - beta1) * gb;
      mlp.vb[l][o] = beta2 * mlp.vb[l][o] + (1 - beta2) * gb * gb;
      const mHatB = mlp.mb[l][o] / bc1;
      const vHatB = mlp.vb[l][o] / bc2;
      layer.b[o] -= (lr * mHatB) / (Math.sqrt(vHatB) + eps);
    }
  }
}
