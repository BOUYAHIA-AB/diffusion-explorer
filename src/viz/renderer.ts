import { densityImageData, colormap } from './density';
import { VIEW_RANGE, DENSITY_GRID_RES } from '../constants';

export interface DrawOptions {
  trueData: number[][];
  currentPoints: number[][] | null;
  // full per-checkpoint trajectory + how far along it we are, for trails
  trailTrajectory?: number[][][] | null;
  trailUpToIndex?: number;
  showTrails?: boolean;
  trailCount?: number | null; // null/undefined = draw all tracked points' trails
  // one row per grid point: [x, y, dx, dy], for the current diffusion step
  scoreField?: number[][] | null;
  showScoreField?: boolean;
  selectedIndex?: number | null;
}

export class SceneRenderer {
  private ctx: CanvasRenderingContext2D;
  private offscreen: HTMLCanvasElement;
  private offCtx: CanvasRenderingContext2D;

  constructor(private canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext('2d')!;
    this.offscreen = document.createElement('canvas');
    this.offscreen.width = DENSITY_GRID_RES;
    this.offscreen.height = DENSITY_GRID_RES;
    this.offCtx = this.offscreen.getContext('2d')!;
  }

  private resizeToDisplay() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = this.canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width * dpr));
    const h = Math.max(1, Math.round(rect.height * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
  }

  private toPixel(x: number, y: number, w: number, h: number): [number, number] {
    const px = ((x + VIEW_RANGE) / (2 * VIEW_RANGE)) * w;
    const py = h - ((y + VIEW_RANGE) / (2 * VIEW_RANGE)) * h;
    return [px, py];
  }

  /** Convert a mouse/pointer client position into data-space [x, y], for click-to-select. */
  pixelToData(clientX: number, clientY: number): [number, number] {
    const rect = this.canvas.getBoundingClientRect();
    const xFrac = (clientX - rect.left) / rect.width;
    const yFrac = (clientY - rect.top) / rect.height;
    const x = xFrac * 2 * VIEW_RANGE - VIEW_RANGE;
    const y = (1 - yFrac) * 2 * VIEW_RANGE - VIEW_RANGE;
    return [x, y];
  }

  private drawScoreField(rows: number[][], w: number, h: number) {
    const ctx = this.ctx;
    const scale = w / 500;
    let maxMag = 1e-6;
    for (const r of rows) {
      const mag = Math.hypot(r[2], r[3]);
      if (mag > maxMag) maxMag = mag;
    }
    const maxLenPx = 15 * scale;
    for (const [x, y, dx, dy] of rows) {
      const mag = Math.hypot(dx, dy);
      const norm = Math.min(1, mag / maxMag);
      const [px, py] = this.toPixel(x, y, w, h);
      const ux = dx / (mag || 1), uy = dy / (mag || 1);
      const len = maxLenPx * (0.25 + 0.75 * norm);
      const tx = px + ux * len;
      const ty = py - uy * len; // flip y for canvas space

      const alpha = 0.18 + 0.72 * norm;
      ctx.strokeStyle = `rgba(55, 214, 179, ${alpha})`;
      ctx.lineWidth = 1.1 * scale;
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(tx, ty);
      ctx.stroke();

      // small arrowhead
      const headLen = 3.2 * scale;
      const angle = Math.atan2(ty - py, tx - px);
      ctx.beginPath();
      ctx.moveTo(tx, ty);
      ctx.lineTo(tx - headLen * Math.cos(angle - 0.4), ty - headLen * Math.sin(angle - 0.4));
      ctx.moveTo(tx, ty);
      ctx.lineTo(tx - headLen * Math.cos(angle + 0.4), ty - headLen * Math.sin(angle + 0.4));
      ctx.stroke();
    }
  }

  private drawSingleTrail(
    trajectory: number[][][], upToIndex: number, w: number, h: number, i: number, emphasized: boolean
  ) {
    const ctx = this.ctx;
    const scale = w / 500;

    // curvature: ratio of actual path length to straight-line distance from start to now
    let pathLen = 0;
    for (let k = 1; k <= upToIndex; k++) {
      const a = trajectory[k - 1][i], b = trajectory[k][i];
      pathLen += Math.hypot(b[0] - a[0], b[1] - a[1]);
    }
    const start = trajectory[0][i], end = trajectory[upToIndex][i];
    const straight = Math.hypot(end[0] - start[0], end[1] - start[1]);
    const curviness = pathLen > 1e-6 ? Math.min(1, 1 - straight / pathLen) : 0;
    const [r, g, b] = colormap(curviness);

    ctx.beginPath();
    for (let k = 0; k <= upToIndex; k++) {
      const [px, py] = this.toPixel(trajectory[k][i][0], trajectory[k][i][1], w, h);
      if (k === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${emphasized ? 0.95 : 0.22})`;
    ctx.lineWidth = (emphasized ? 2.2 : 1) * scale;
    ctx.stroke();
  }

  private drawTrails(
    trajectory: number[][][], upToIndex: number, w: number, h: number,
    selectedIndex: number | null, trailCount: number | null
  ) {
    if (upToIndex < 1) return;
    const numPoints = trajectory[0].length;

    // A selected sample always gets its trail drawn in full, regardless of
    // the count control — it wouldn't reliably land on the stride below.
    if (selectedIndex !== null) {
      this.drawSingleTrail(trajectory, upToIndex, w, h, selectedIndex, true);
      return;
    }

    const count = trailCount && trailCount > 0 ? Math.min(trailCount, numPoints) : numPoints;
    const stride = Math.max(1, Math.round(numPoints / count));
    for (let i = 0; i < numPoints; i += stride) {
      this.drawSingleTrail(trajectory, upToIndex, w, h, i, false);
    }
  }

  draw(opts: DrawOptions) {
    this.resizeToDisplay();
    const { ctx, canvas } = this;
    const w = canvas.width, h = canvas.height;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#05070b';
    ctx.fillRect(0, 0, w, h);

    const showField = !!opts.showScoreField && !!opts.scoreField;

    // density heatmap of the model's current samples, unless the score field
    // is showing instead — the two are visually busy together
    if (!showField && opts.currentPoints && opts.currentPoints.length > 0) {
      const img = densityImageData(opts.currentPoints, DENSITY_GRID_RES, VIEW_RANGE);
      this.offCtx.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.globalAlpha = 0.95;
      ctx.drawImage(this.offscreen, 0, 0, w, h);
      ctx.globalAlpha = 1;
    }

    if (showField) {
      this.drawScoreField(opts.scoreField!, w, h);
    }

    // sample trajectory trails, drawn under the current points
    if (opts.showTrails && opts.trailTrajectory && opts.trailUpToIndex !== undefined) {
      this.drawTrails(opts.trailTrajectory, opts.trailUpToIndex, w, h, opts.selectedIndex ?? null, opts.trailCount ?? null);
    }

    // faint reference cloud of the true data distribution
    ctx.fillStyle = 'rgba(231, 233, 238, 0.22)';
    for (const p of opts.trueData) {
      const [x, y] = this.toPixel(p[0], p[1], w, h);
      ctx.beginPath();
      ctx.arc(x, y, 1.1 * (w / 500), 0, Math.PI * 2);
      ctx.fill();
    }

    // current sample points, small glowing dots on top of everything else
    if (opts.currentPoints) {
      const selected = opts.selectedIndex ?? null;
      ctx.shadowColor = 'rgba(240, 167, 59, 0.55)';
      for (let i = 0; i < opts.currentPoints.length; i++) {
        const isSelected = selected === i;
        const dim = selected !== null && !isSelected;
        const [x, y] = this.toPixel(opts.currentPoints[i][0], opts.currentPoints[i][1], w, h);
        ctx.fillStyle = isSelected ? '#f0a73b' : dim ? 'rgba(244,246,250,0.25)' : '#f4f6fa';
        ctx.shadowBlur = isSelected ? 8 * (w / 500) : 4 * (w / 500);
        ctx.beginPath();
        ctx.arc(x, y, (isSelected ? 2.6 : 1.6) * (w / 500), 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.shadowBlur = 0;
    }
  }
}
