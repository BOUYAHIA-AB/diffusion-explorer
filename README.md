# Diffusion Explorer

An in-browser playground for diffusion models: pick a 2D toy dataset, pick an
architecture, train it live, and watch the learned distribution converge —
scrubbing independently through **training time** (how much the model has
learned) and **generation time** (where a sample is in its own reverse
diffusion trajectory).

Everything trains entirely in the browser. No backend, no GPU, no API key.

## Run it

```bash
npm install
npm run dev       # http://localhost:5173
```

```bash
npm run build      # outputs to dist/, a static site you can host anywhere
npm run preview
```

## What you can see

- **Sample trails** — every tracked point's full path, not just its current
  position, drawn as a curved line. Toggle with the "Trails" checkbox, or
  **click any sample dot to trace it**: the view jumps back to pure noise and
  automatically plays forward through that sample's own reverse-diffusion
  trajectory, one step at a time, until it reaches the point you clicked
  (click empty space to deselect). This works especially well once training
  has finished — the hint text above the canvas invites you to try it as
  soon as a run completes. A selected sample's trail always draws in full,
  regardless of the trail count setting below. Use the **"Trail count"**
  dropdown to control how many trails are drawn when nothing is selected
  (20 / 50 / 70 / 150 / all tracked points) — fewer trails read as individual
  paths, more trails read as a field showing the overall flow. Trails are
  colored by **curvature**: cool/dark for paths close to a straight line,
  warm/gold for paths that wander — a direct, visual answer to "why do some
  samplers need fewer steps than others?" (see
  `/papers/diffusion-inference-acceleration` on the companion blog).
- **Score field** — toggle "Score field" to replace the density heatmap with
  a quiver plot of the network's learned score function
  ($\nabla_x \log p(x)$, shown as $-\epsilon_\theta / \sqrt{1-\bar\alpha_t}$)
  evaluated on a fixed grid at the current diffusion timestep. Early in
  training, or at high noise levels, expect a mostly uniform, low-magnitude
  field. As training progresses and noise decreases, the arrows should
  visibly organize into flows converging on the data manifold.

## What's actually happening

- **Training set size is configurable** ("Training points" in the sidebar:
  256 / 1,024 / 4,096 / 16,000). This is the number of points sampled from
  the chosen 2D shape that the model actually trains on — every batch is
  drawn with replacement from this pool. Smaller pools train faster and let
  you see overfitting-ish behavior on toy data (the model tracking the exact
  sampled points rather than the underlying shape); larger pools give a
  smoother target distribution. This is independent of the always-1,500-point
  reference cloud drawn faintly in the background, which is just a display
  cap for render performance.
- **Noise variance is configurable** ("Noise variance (β_T)" in the sidebar:
  0.01 / 0.02 (default) / 0.05 / 0.1). This sets the endpoint of the linear
  beta schedule — how much noise the forward process injects at the final
  diffusion step. Higher values push the data toward pure noise faster and
  more aggressively; lower values noise it more gently across the same
  number of steps. `betaStart` stays fixed at `1e-4` (its effect is small
  relative to `betaEnd`, so exposing one clear knob is more legible than two
  that interact). Watch the score field with this turned up — a higher
  `betaEnd` visibly washes out fine structure faster as the diffusion-step
  scrubber moves toward `t = T`.

- **The neural net is hand-written** (`src/nn.ts`) — a small dense MLP with a
  manual forward pass, manual backpropagation, and an Adam optimizer, no
  TensorFlow.js or other ML framework. At the sizes used here (a few hundred
  parameters, batch size 64) this is fast enough to train in real time and
  keeps every gradient computation visible and auditable.
- **The diffusion mechanics** (`src/diffusion.ts`) follow the standard DDPM
  formulation: a linear beta schedule, the closed-form forward noising
  shortcut, a noise-prediction training objective, and ancestral reverse
  sampling. If you've read a "math behind diffusion models" explainer, every
  function here maps directly onto that notation.
- **Training runs in a Web Worker** (`src/worker.ts`), never on the main
  thread, so the UI stays responsive while a run is in progress.
- **Every checkpoint replays the same noise.** The reverse-sampling trajectory
  at each checkpoint starts from an identical seeded batch of Gaussian noise
  (`NOISE_SEED = 42` in `worker.ts`). That's what makes the training-step
  scrubber meaningful — you're watching the *same* underlying noise get
  mapped to increasingly better samples as training progresses, not a
  different random draw each time.
- **The visualization** (`src/viz/`) bins the current sample batch into a
  small density grid, applies a couple of box-blur passes, maps it through a
  hand-picked colormap, and upscales it onto the canvas with smoothing
  enabled — a cheap way to get a soft heatmap look without a real KDE or a
  charting library.

## Verifying it actually works

`test/sanity.ts` is a headless (no browser) training run you can use to
sanity-check the core math after making changes:

```bash
node_modules/.bin/esbuild test/sanity.ts --bundle --platform=node --format=cjs --outfile=/tmp/sanity.cjs
node /tmp/sanity.cjs
```

It trains on the 8-Gaussians dataset for 1200 steps and asserts that (a) loss
decreases and (b) generated samples land near the true ring radius. This is
worth re-running after any change to `nn.ts` or `diffusion.ts` — hand-written
backprop is exactly the kind of code where a sign error compiles fine and
just silently fails to learn.

## Project structure

```
src/
  types.ts            Shared types + the main-thread <-> worker message protocol
  nn.ts                Dense MLP: forward, backward, Adam — no framework dependency
  diffusion.ts          Noise schedule, training step, seeded reverse sampling
  datasets.ts            2D toy dataset generators (moons, 8-Gaussians, spiral, checkerboard)
  constants.ts            Shared view-range and grid-resolution constants
  worker.ts              Owns the training loop; also computes each checkpoint's
                           score field (see computeScoreFieldTrajectory in diffusion.ts)
  viz/
    density.ts            Histogram binning + blur + colormap -> ImageData;
                            the same colormap doubles as the trail-curvature scale
    renderer.ts            Canvas 2D scene renderer: density heatmap OR score field,
                             curvature-colored trails, true-data reference, sample dots
    losschart.ts            Small hand-rolled loss sparkline
  main.ts                 UI construction, worker orchestration, dual-scrubber logic
  style.css                 Dark "observatory" theme
test/
  sanity.ts               Headless correctness check (see above)
```

## Extending it

- **New dataset**: add a generator function to `datasets.ts`, add it to
  `DatasetType` in `types.ts` and `DATASET_LABELS`, and it appears in the
  picker automatically.
- **New architecture knob** (e.g. residual connections, a second activation
  choice per layer): extend `ArchConfig` in `types.ts`, thread it through
  `createMLP` in `nn.ts`, and add the corresponding control in `main.ts`.
- **Hand-drawn custom datasets**: let the user paint a target distribution
  with the mouse instead of picking from the preset list — capture strokes on
  a canvas, sample points along them, and feed the result into
  `generateDataset`'s `'custom'` case (add it to `DatasetType` in `types.ts`
  first). This is the single most-requested extension based on similar tools
  in the wild.
- **Image-scale datasets**: this architecture (in-browser MLP + Web Worker)
  is intentionally scoped to 2D toy data, where the full density is directly
  visualizable. Moving to image data is a different system — see the
  "Phase 3/4" notes from the original planning discussion for what that
  requires (a real backend, job queue, and sample-grid-based visualization
  instead of a density heatmap).
