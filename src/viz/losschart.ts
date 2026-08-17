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

  draw(history: { step: number; loss: number }[], highlightIndex: number | null) {
    this.resize();
    const { ctx, canvas } = this;
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    if (history.length < 2) return;

    const pad = 6 * (w / 500);
    const maxLoss = Math.max(...history.map(p => p.loss), 1e-6);
    const minLoss = 0;
    const xForI = (i: number) => pad + (i / (history.length - 1)) * (w - 2 * pad);
    const yForLoss = (l: number) => h - pad - ((l - minLoss) / (maxLoss - minLoss || 1)) * (h - 2 * pad);

    ctx.strokeStyle = 'rgba(55, 214, 179, 0.75)';
    ctx.lineWidth = 1.4 * (w / 500);
    ctx.beginPath();
    history.forEach((p, i) => {
      const x = xForI(i), y = yForLoss(p.loss);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // filled area under the curve, very subtle
    ctx.lineTo(xForI(history.length - 1), h - pad);
    ctx.lineTo(xForI(0), h - pad);
    ctx.closePath();
    ctx.fillStyle = 'rgba(55, 214, 179, 0.08)';
    ctx.fill();

    if (highlightIndex !== null && history[highlightIndex]) {
      const x = xForI(highlightIndex);
      const y = yForLoss(history[highlightIndex].loss);
      ctx.fillStyle = '#f0a73b';
      ctx.beginPath();
      ctx.arc(x, y, 2.6 * (w / 500), 0, Math.PI * 2);
      ctx.fill();
    }
  }
}
