# Engineering notes

The README covers what the app does and how to run it. This is the deeper
material: how it's actually built, and an honest write-up of a real
convergence investigation — kept here instead of deleted, because it's the
kind of thing worth knowing if you're extending this project.

## What's actually happening under the hood

- **The neural net is hand-written** (`src/nn.ts`) — a small dense MLP with a
  manual forward pass, manual backpropagation, and an Adam optimizer, no
  TensorFlow.js or other ML framework. At the sizes used here (a few hundred
  parameters, batch size 64) this is fast enough to train in real time and
  keeps every gradient computation visible and auditable.
- **The diffusion mechanics** (`src/diffusion.ts`) follow the standard DDPM
  formulation: linear or cosine beta schedules, the closed-form forward
  noising shortcut, a noise-prediction training objective, and both
  ancestral and DDIM reverse sampling.
- **Training runs in a Web Worker** (`src/worker.ts`), never on the main
  thread, so the UI stays responsive while a run is in progress.
- **Every checkpoint replays the same noise.** The reverse-sampling
  trajectory at each checkpoint starts from an identical seeded batch of
  Gaussian noise (`NOISE_SEED = 42` in `worker.ts`). That's what makes the
  training-step scrubber meaningful — you're watching the *same* underlying
  noise get mapped to increasingly better samples as training progresses,
  not a different random draw each time.
- **Sampling uses EMA-smoothed weights**, not the raw live training weights
  (`nn.ts` / `worker.ts`, decay 0.995) — standard diffusion-training
  practice, since per-step SGD weights are noisier than what you actually
  want to generate from.
- **Gradients are clipped** to a global L2 norm of 5.0 before every Adam
  step (`diffusion.ts`) — the training loop is otherwise a fully manual,
  from-scratch optimizer with no other stabilization tricks.
- **The visualization** (`src/viz/`) bins the current sample batch into a
  small density grid, applies a couple of box-blur passes, maps it through a
  hand-picked colormap, and upscales it onto the canvas with smoothing
  enabled — a cheap way to get a soft heatmap look without a real KDE or a
  charting library.

### The noise-schedule caveat

At the default `T=40` and `betaEnd=0.02`, the **linear** schedule leaves
`alphaBar_T ≈ 0.67` — the data is *not* fully noised by the final step
(`betaEnd=0.02` was calibrated for the original DDPM paper's `T=1000`;
reused at a much shorter `T`, it under-noises rather than over-noises).
Training and sampling still converge fine (every check in `test/sanity.ts`
passes), but it's a real, mild train/sample mismatch: technically `x_T`
isn't quite pure noise, so starting the reverse process from `N(0, I)` is a
slight approximation. The **cosine** schedule doesn't have this problem at
any `T` (`alphaBar_T ≈ 0` by construction), and the "aggressive" (`0.1`)
linear preset gets much closer to full noising at `T=40`.

## A known limitation: convergence quality on harder shapes

If you train on `moons` or `checkerboard` and the result looks noticeably
worse than `gaussians` or `spiral`, that's real, not your imagination. Here's
what an actual investigation found (not a guess):

- **The hand-written backprop is mathematically exact** — verified with a
  finite-difference gradient check (~1.8e-9 relative error, essentially
  machine precision). Not a bug there.
- **A calibrated "noise floor"** — the distance-to-target metric compared
  against two independent draws from the *same* true distribution — sits
  around 0.04–0.06 for all four datasets. A well-converged model should land
  close to that. `moons` at default settings was landing around 0.19–0.23,
  roughly 4x off, and — critically — **more training steps alone didn't
  close that gap**; the distance metric plateaued rather than continuing
  toward the floor.
- Several standard interventions were tested in isolation: the cosine
  schedule (rules out the under-noising issue above as the cause),
  EMA-smoothed weights, a wider network, a higher learning rate, and
  gradient-norm clipping. Each helped a little; none was a single fix.

**What shipped as a result**: gradient clipping and EMA-smoothed weights
(both described above, and both standard practice regardless of whether they
single-handedly fix a given run), plus bumped defaults — architecture width
(64 → 128), learning rate (0.01 → 0.02), and training step ceiling (added a
6000-step option, default 800 → 3000) — since more capacity and more steps
did show real, if gradual, continued improvement in testing.

**What this doesn't claim to be**: a fix that makes `moons` and
`checkerboard` converge as cleanly as `gaussians` and `spiral` do. A
genuinely from-scratch, dependency-free, small MLP has a real capacity/
compute ceiling for asymmetric, multi-cluster 2D shapes within a few
thousand browser-interactive steps. That's a legitimate, visible thing to
let people discover by turning the width and step-count dials themselves,
not a defect to paper over. If closing this gap further matters to you, the
next real lever (not yet implemented) would be a learning-rate schedule
(warmup + cosine decay) rather than a flat rate for the whole run — usually
the next thing that matters after clipping and EMA in small-model training,
and a bigger change than a config-value tweak.

## Verifying it actually works

`test/sanity.ts` is a headless (no browser) training run you can use to
sanity-check the core math after making changes:

```bash
node_modules/.bin/esbuild test/sanity.ts --bundle --platform=node --format=cjs --outfile=/tmp/sanity.cjs
node /tmp/sanity.cjs
```

It trains on the 8-Gaussians dataset and checks, among other things: loss
decreases, generated samples land near the true ring radius, the
finite-difference gradient check passes, the score field is well-formed, all
four dataset generators produce correct counts at every configurable size,
the noise schedule is valid at every configurable setting, DDIM sampling and
the distance metric both behave correctly, and training stays numerically
stable at the most aggressive noise setting.

This is worth re-running after any change to `nn.ts` or `diffusion.ts` —
hand-written backprop is exactly the kind of code where a sign error
compiles fine and just silently fails to learn.

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
  visualizable. Moving to image data is a different system entirely — a real
  backend, a job queue, and sample-grid-based visualization instead of a
  density heatmap, since pixel-space density isn't directly renderable the
  way a 2D point cloud's is.
- **Learning-rate schedule**: see the convergence write-up above — this is
  the most likely next lever for training quality, and isn't implemented yet.
