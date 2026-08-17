export type DatasetType = 'moons' | 'gaussians' | 'spiral' | 'checkerboard';
export type Activation = 'relu' | 'tanh' | 'swish';

export interface ArchConfig {
  hiddenLayers: number;   // number of hidden layers, 1-4
  hiddenWidth: number;    // units per hidden layer
  activation: Activation;
  timeEmbedDim: number;   // fixed at 16, kept configurable for clarity
}

export interface HyperParams {
  T: number;                 // diffusion steps
  lr: number;                // learning rate
  batchSize: number;
  totalSteps: number;        // training steps
  checkpointInterval: number;
  numTrackedPoints: number;  // points carried through the reverse trajectory for viz
  datasetSize: number;       // number of points sampled from the target distribution for training
  betaEnd: number;           // noise variance at the final diffusion step (beta_T) — how aggressively the forward process noises the data
}

export interface InitMessage {
  type: 'init';
  dataset: DatasetType;
  arch: ArchConfig;
  hyper: HyperParams;
}

export interface ControlMessage {
  type: 'start' | 'pause' | 'reset';
}

export type WorkerInbound = InitMessage | ControlMessage;

export interface ReadyMessage {
  type: 'ready';
  trueData: number[][];
}

export interface CheckpointMessage {
  type: 'checkpoint';
  step: number;
  loss: number;
  totalSteps: number;
  trajectory: number[][][]; // [T+1][numTrackedPoints][2], index 0 = pure noise, last = t=0 sample
  scoreField: number[][][]; // [T+1][gridPoints][4] rows of [x, y, dx, dy], same indexing as trajectory
}

export interface StatusMessage {
  type: 'paused' | 'done';
  step: number;
}

export type WorkerOutbound = ReadyMessage | CheckpointMessage | StatusMessage;
