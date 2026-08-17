// A small, hand-picked colormap (dark indigo -> magenta -> amber -> pale gold)
// used to render learned-density heat without pulling in a charting library.
// Reused for trail curvature coloring too, so "warm" reads as one consistent
// visual language across the whole scene: cool = calm/aligned, warm = far/curved.
const STOPS: [number, number, number, number][] = [
  [0.0, 10, 12, 20],
  [0.28, 42, 24, 74],
  [0.55, 122, 44, 94],
  [0.78, 214, 91, 60],
  [1.0, 247, 201, 72],
];

function lerp(a: number, b: number, t: number) { return a + (b - a) * t; }

export function colormap(v: number): [number, number, number] {
  const t = Math.min(1, Math.max(0, v));
  for (let i = 0; i < STOPS.length - 1; i++) {
    const [t0, r0, g0, b0] = STOPS[i];
    const [t1, r1, g1, b1] = STOPS[i + 1];
    if (t >= t0 && t <= t1) {
      const f = t1 === t0 ? 0 : (t - t0) / (t1 - t0);
      return [lerp(r0, r1, f), lerp(g0, g1, f), lerp(b0, b1, f)];
    }
  }
  const last = STOPS[STOPS.length - 1];
  return [last[1], last[2], last[3]];
}

/**
 * Bins points into a gridRes x gridRes histogram over [-viewRange, viewRange]^2,
 * applies a couple of box-blur passes for a soft look, and returns an ImageData
 * that the renderer can upscale onto the main canvas with smoothing enabled.
 */
export function densityImageData(points: number[][], gridRes: number, viewRange: number): ImageData {
  const grid = new Float32Array(gridRes * gridRes);
  for (const p of points) {
    const gx = Math.floor(((p[0] + viewRange) / (2 * viewRange)) * gridRes);
    const gy = Math.floor(((p[1] + viewRange) / (2 * viewRange)) * gridRes);
    if (gx >= 0 && gx < gridRes && gy >= 0 && gy < gridRes) {
      grid[gy * gridRes + gx] += 1;
    }
  }

  // two box-blur passes (separable, cheap) for a smoother heatmap
  const blur = (src: Float32Array): Float32Array => {
    const out = new Float32Array(src.length);
    for (let y = 0; y < gridRes; y++) {
      for (let x = 0; x < gridRes; x++) {
        let sum = 0, count = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const nx = x + dx, ny = y + dy;
            if (nx >= 0 && nx < gridRes && ny >= 0 && ny < gridRes) {
              sum += src[ny * gridRes + nx];
              count++;
            }
          }
        }
        out[y * gridRes + x] = sum / count;
      }
    }
    return out;
  };
  const smoothed = blur(blur(grid));

  let max = 0;
  for (const v of smoothed) if (v > max) max = v;
  const norm = max > 0 ? 1 / max : 1;

  const img = new ImageData(gridRes, gridRes);
  for (let i = 0; i < smoothed.length; i++) {
    // gentle gamma so mid-density regions stay visible, not just the peak
    const v = Math.pow(smoothed[i] * norm, 0.55);
    const [r, g, b] = colormap(v);
    const alpha = v < 0.03 ? 0 : Math.min(1, v * 1.3) * 255;
    const px = i * 4;
    img.data[px] = r;
    img.data[px + 1] = g;
    img.data[px + 2] = b;
    img.data[px + 3] = alpha;
  }
  return img;
}
