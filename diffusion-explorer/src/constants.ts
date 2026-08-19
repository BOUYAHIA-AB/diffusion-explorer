export const VIEW_RANGE = 2.0;       // world space is [-VIEW_RANGE, VIEW_RANGE]^2
export const DENSITY_GRID_RES = 56;   // resolution of the density heatmap grid
export const SCORE_GRID_RES = 14;      // resolution of the score-field arrow grid (kept low so arrows stay legible)
export const DEFAULT_TRAIL_COUNT = 70; // initial value for the user-facing trail-count control
export const GRAD_CLIP_NORM = 5.0;      // global L2 gradient-norm clip; makes the tiny hand-rolled optimizer noticeably more stable
export const EMA_DECAY = 0.995;          // exponential moving average of weights, used for sampling (standard diffusion-training practice)
