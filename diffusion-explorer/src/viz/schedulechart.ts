import type { Schedule } from '../diffusion';

export class ScheduleChart {
  private ctx: CanvasRenderingContext2D;

  constructor(private canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext('2d')!;
  }

  private resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = this.canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width * dpr));
    const h = Math.max(1, Math.round(rect.height * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
  }

  draw(sched: Schedule) {
    this.resize();
    const { ctx, canvas } = this;
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    if (sched.T < 2) return;

    const pad = 6 * (w / 500);
    const xFor = (i: number) => pad + (i / (sched.T - 1)) * (w - 2 * pad);

    // beta_t, normalized to its own max so its shape is visible regardless
    // of the absolute noise-variance setting
    const maxBeta = Math.max(...sched.betas, 1e-9);
    ctx.strokeStyle = 'rgba(232, 89, 60, 0.85)'; // coral
    ctx.lineWidth = 1.4 * (w / 500);
    ctx.beginPath();
    sched.betas.forEach((b, i) => {
      const x = xFor(i);
      const y = h - pad - (b / maxBeta) * (h - 2 * pad);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // alphaBar_t, naturally in [0, 1] already
    ctx.strokeStyle = 'rgba(55, 214, 179, 0.9)'; // teal
    ctx.lineWidth = 1.4 * (w / 500);
    ctx.beginPath();
    sched.alphaBars.forEach((ab, i) => {
      const x = xFor(i);
      const y = h - pad - ab * (h - 2 * pad);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
}
