export interface LossPoint { step: number; loss: number; distance: number }

export class LossChart {
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

  private drawSeries(
    history: LossPoint[], key: 'loss' | 'distance', color: string,
    w: number, h: number, pad: number, highlightIndex: number | null
  ) {
    const ctx = this.ctx;
    const values = history.map(p => p[key]);
    const max = Math.max(...values, 1e-6);
    const xForI = (i: number) => pad + (i / (history.length - 1)) * (w - 2 * pad);
    const yForV = (v: number) => h - pad - (v / max) * (h - 2 * pad);

    ctx.strokeStyle = color;
    ctx.lineWidth = 1.4 * (w / 500);
    ctx.beginPath();
    history.forEach((p, i) => {
      const x = xForI(i), y = yForV(p[key]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();

    if (highlightIndex !== null && history[highlightIndex]) {
      const x = xForI(highlightIndex);
      const y = yForV(history[highlightIndex][key]);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, 2.4 * (w / 500), 0, Math.PI * 2);
      ctx.fill();
    }
  }

  draw(history: LossPoint[], highlightIndex: number | null) {
    this.resize();
    const { ctx, canvas } = this;
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    if (history.length < 2) return;

    const pad = 6 * (w / 500);
    // each series is independently normalized to its own max, since loss and
    // distance live on very different scales — the point is to see the shape
    // of each curve (both trending down), not compare absolute magnitudes
    this.drawSeries(history, 'distance', 'rgba(232, 89, 60, 0.8)', w, h, pad, highlightIndex);
    this.drawSeries(history, 'loss', 'rgba(55, 214, 179, 0.85)', w, h, pad, highlightIndex);
  }
}
