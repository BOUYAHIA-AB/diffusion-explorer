# Diffusion Explorer

An in-browser playground for diffusion models. Pick a 2D toy dataset, pick an
architecture, train it live, and watch the learned distribution converge —
scrubbing independently through **training time** (how much the model has
learned) and **generation time** (where a sample is in its own reverse
diffusion trajectory).

Everything — training included — runs entirely in the browser. No backend,
no GPU, no API key.

**[Live demo](#)** — add your GitHub Pages URL here after deploying (see below).

## Features

- 4 toy datasets (two moons, 8 Gaussians, spiral, checkerboard), with a
  configurable training-set size
- Configurable architecture (hidden layers, width, activation) trained by a
  small hand-written MLP — no ML framework dependency, every gradient is
  auditable
- Linear vs. cosine noise schedule, with a configurable noise variance and a
  live β_t / ᾱ_t preview chart
- Ancestral vs. DDIM sampler toggle — same trained network, watch the
  deterministic path vs. the stochastic one
- Per-sample trajectory trails, colored by curvature, with click-to-trace on
  any generated point
- A score-field (∇ log p(x)) quiver-plot view, computed from the same
  network doing the sampling
- Loss and distance-to-target (sliced-Wasserstein) plotted side by side, so
  you can see the difference between "the network is optimizing" and "the
  distribution is actually converging"

## Run it locally

```bash
npm install
npm run dev       # http://localhost:5173
```

```bash
npm run build      # outputs a static site to dist/
npm run preview
```

## Deploy to GitHub Pages

This repo includes a GitHub Actions workflow (`.github/workflows/deploy.yml`)
that builds and deploys automatically — no manual steps after the initial
setup.

1. **Push this project to a GitHub repository** (create one on GitHub, then
   from this folder):
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```
2. **Enable Pages via Actions**: on GitHub, go to your repo's
   **Settings → Pages**, and under "Build and deployment", set **Source** to
   **GitHub Actions**.
3. **Push to `main`** (the commit above already did this). Check the
   **Actions** tab — the "Deploy to GitHub Pages" workflow runs
   automatically, and your site will be live at
   `https://<your-username>.github.io/<your-repo>/` a minute or two later.

That's it — every future push to `main` redeploys automatically. No asset-path
configuration was needed: the Vite config already builds with relative paths
(`base: './'` in `vite.config.ts`), so it works correctly whether it's served
from a custom domain, a user-page root, or a project subpath like
`/your-repo/`, without editing anything.

**Manual alternative**, if you'd rather not use Actions: run `npm run build`,
then push the contents of `dist/` to a `gh-pages` branch (the
[`gh-pages`](https://www.npmjs.com/package/gh-pages) npm package automates
this: `npx gh-pages -d dist`), and set Pages' source to that branch instead.

## Project structure

```
src/
  types.ts          Shared types + the main-thread <-> worker message protocol
  nn.ts              Dense MLP: forward, backward, Adam — no framework dependency
  diffusion.ts        Noise schedule, training step, seeded reverse sampling
  datasets.ts          2D toy dataset generators
  constants.ts          Shared view-range and grid-resolution constants
  worker.ts            Owns the training loop, off the main thread
  viz/                  Canvas renderer, density colormap, loss/schedule charts
  main.ts               UI construction, worker orchestration, dual-scrubber logic
  style.css               Dark "observatory" theme
test/
  sanity.ts            Headless correctness check — see docs/NOTES.md
```

## More detail

`docs/NOTES.md` has the deeper material: exactly what's happening under the
hood, an honest write-up of a real convergence investigation (what was
tested, what shipped, what's still a known limitation), and pointers for
extending the project further.
