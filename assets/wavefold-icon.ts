import type { G } from "@svgdotjs/svg.js";
import { SvgRenderer, roundedRectPoints, strokeToFillPath, type Point } from "./svg.ts";

interface Wave {
  frequency: number;
  amplitude: number;
  phase: number; // radians
}

// Catppuccin Mocha palette (catppuccin.com/palette)
const MOCHA = {
  base: "#1e1e2e",
  crust: "#11111b",
  surface0: "#313244",
  surface1: "#45475a",
  surface2: "#585b70",
  green: "#a6e3a1", // classic oscilloscope-trace green
};

// Target waveform
const W_CONTROL_POINTS: Point[] = [
  [0, 1],
  [0.25, -1],
  [0.5, 0.5],
  [0.75, -1],
  [1, 1],
];

function targetW(t: number): number {
  for (let i = 0; i < W_CONTROL_POINTS.length - 1; i++) {
    const [t0, y0] = W_CONTROL_POINTS[i]!;
    const [t1, y1] = W_CONTROL_POINTS[i + 1]!;
    if (t >= t0 && t <= t1) {
      const frac = (t - t0) / (t1 - t0);
      return y0 + (y1 - y0) * frac;
    }
  }
  return W_CONTROL_POINTS[W_CONTROL_POINTS.length - 1]![1];
}

function projectHarmonic(
  target: (t: number) => number,
  frequency: number,
  samples: number,
): { amplitude: number; phase: number } {
  let a = 0;
  let b = 0;
  for (let i = 0; i < samples; i++) {
    const t = i / samples;
    const theta = frequency * 2 * Math.PI * t;
    const value = target(t);
    a += value * Math.cos(theta);
    b += value * Math.sin(theta);
  }
  a = (2 / samples) * a;
  b = (2 / samples) * b;
  return { amplitude: Math.sqrt(a * a + b * b), phase: Math.atan2(a, b) };
}

export class WavefoldIcon extends SvgRenderer {
  private static readonly BORDER_WIDTH = 32;
  private static readonly CORNER_RADIUS = 96;
  private static readonly GRID_DIVISIONS = 4; // oscilloscope graticule
  private static readonly GRID_LINE_WIDTH = 16;
  private static readonly GRID_CENTER_WIDTH = 32;
  private static readonly MINOR_LINE_OFFSET = 8;
  private static readonly W_HEIGHT_FRACTION = 0.34; // peak-to-center height, of size
  private static readonly HARMONIC_COUNT = 4;
  private static readonly FOURIER_PROJECTION_SAMPLES = 1000;
  private static readonly TRACE_SAMPLES = 200;

  private readonly size: number;
  private readonly centerY: number;
  private readonly waves: Wave[];

  constructor(size = 512) {
    super(size);
    this.size = size;
    this.centerY = size / 2;
    const wHeight = size * WavefoldIcon.W_HEIGHT_FRACTION;
    this.waves = Array.from({ length: WavefoldIcon.HARMONIC_COUNT }, (_, i) => {
      const frequency = i + 1;
      const { amplitude, phase } = projectHarmonic(
        targetW,
        frequency,
        WavefoldIcon.FOURIER_PROJECTION_SAMPLES,
      );
      return { frequency, amplitude: amplitude * wHeight, phase };
    });
  }

  protected async draw(): Promise<void> {
    this.drawBackground();

    const waveGroup = this.canvas.group();
    waveGroup.clipWith(this.contentRect());
    await this.drawGrid(waveGroup);
    await this.drawTrace(waveGroup);

    await this.drawBorder();
  }

  private contentRect() {
    const { BORDER_WIDTH, CORNER_RADIUS } = WavefoldIcon;
    return this.canvas
      .rect(this.size - BORDER_WIDTH * 2, this.size - BORDER_WIDTH * 2)
      .move(BORDER_WIDTH, BORDER_WIDTH)
      .radius(CORNER_RADIUS - BORDER_WIDTH);
  }

  private drawBackground(): void {
    this.contentRect().fill(MOCHA.crust);
  }

  private async drawGrid(waveGroup: G): Promise<void> {
    const { GRID_DIVISIONS, GRID_LINE_WIDTH, GRID_CENTER_WIDTH, MINOR_LINE_OFFSET } = WavefoldIcon;
    const gridCenter = GRID_DIVISIONS / 2;

    for (let i = 1; i < GRID_DIVISIONS; i++) {
      if (i === gridCenter) continue;
      const basePos = (this.size / GRID_DIVISIONS) * i;
      const pos = basePos + (basePos < this.centerY ? -MINOR_LINE_OFFSET : MINOR_LINE_OFFSET);
      await this.gridLine(waveGroup, pos, MOCHA.base, GRID_LINE_WIDTH);
    }

    await this.gridLine(waveGroup, this.centerY, MOCHA.surface0, GRID_CENTER_WIDTH);
  }

  // Filled outlines instead of stroked lines everywhere below — exported
  // paths carry no `stroke` attribute, just `fill`.
  private async gridLine(waveGroup: G, pos: number, color: string, width: number): Promise<void> {
    const horizontal = await strokeToFillPath([[0, pos], [this.size, pos]], width);
    const vertical = await strokeToFillPath([[pos, 0], [pos, this.size]], width);
    waveGroup.path(horizontal).fill(color);
    waveGroup.path(vertical).fill(color);
  }

  private async drawTrace(waveGroup: G): Promise<void> {
    const points: Point[] = [];
    for (let i = 0; i <= WavefoldIcon.TRACE_SAMPLES; i++) {
      const t = i / WavefoldIcon.TRACE_SAMPLES;
      const offset = this.waves.reduce(
        (acc, wave) => acc + Math.sin(t * wave.frequency * 2 * Math.PI + wave.phase) * wave.amplitude,
        0,
      );
      points.push([t * this.size, this.centerY - offset]);
    }

    const fill = await strokeToFillPath(points, WavefoldIcon.BORDER_WIDTH);
    waveGroup.path(fill).fill(MOCHA.green);
  }

  private async drawBorder(): Promise<void> {
    const { BORDER_WIDTH, CORNER_RADIUS } = WavefoldIcon;
    const inset = BORDER_WIDTH / 2;

    const points = roundedRectPoints(inset, inset, this.size - inset * 2, this.size - inset * 2, CORNER_RADIUS);
    const ring = await strokeToFillPath(points, BORDER_WIDTH, { closed: true });
    this.canvas.path(ring).fill({ color: MOCHA.surface1, rule: "evenodd" });
  }
}
