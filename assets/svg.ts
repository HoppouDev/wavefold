import { createSVGWindow } from "svgdom";
import { SVG, registerWindow } from "@svgdotjs/svg.js";
import type { Svg } from "@svgdotjs/svg.js";
import * as clipperLib from "js-angusj-clipper";
import opentype from "opentype.js";
import { writeFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";

export type Point = readonly [x: number, y: number];

// Clipper works in integers; this scales our sub-pixel float coordinates
// up before offsetting and back down after, per the library's own
// recommended pattern for preserving precision.
const CLIPPER_SCALE = 1000;

let clipperPromise: Promise<clipperLib.ClipperLibWrapper> | undefined;
function getClipper(): Promise<clipperLib.ClipperLibWrapper> {
  clipperPromise ??= clipperLib.loadNativeClipperLibInstanceAsync(
    clipperLib.NativeClipperLibRequestedFormat.WasmWithAsmJsFallback,
  );
  return clipperPromise;
}

function toClipperPath(points: readonly Point[]): clipperLib.Path {
  return points.map(([x, y]) => ({
    x: Math.round(x * CLIPPER_SCALE),
    y: Math.round(y * CLIPPER_SCALE),
  }));
}

function fromClipperPaths(paths: clipperLib.Paths): Point[][] {
  return paths.map((path) => path.map((p): Point => [p.x / CLIPPER_SCALE, p.y / CLIPPER_SCALE]));
}

function subpathsToD(subpaths: readonly Point[][]): string {
  return subpaths
    .map((points) => `M${points.map(([x, y]) => `${x},${y}`).join("L")}Z`)
    .join(" ");
}

export interface StrokeToFillOptions {
  // Stroking a CLOSED polygon (e.g. a rect border) needs both the outward
  // and inward offset boundary to form a ring — an open path (a line, a
  // wandering trace) only needs one round-capped offset to fully enclose it.
  closed?: boolean;
}

// Converts what would be a stroked polyline into an equivalent filled
// outline path, via Clipper2 (js-angusj-clipper) — real, self-intersection
// -safe polygon offsetting. This replaced an earlier attempt built on
// maker.js (`svg-path-outline`): maker.js's offsetter produced visible
// self-intersection notches at the trace's sharper bends (confirmed by
// rendering), which Clipper2 — the industry-standard algorithm for exactly
// this — doesn't. `width` is the full stroke width, halved internally to
// match SVG's own stroke convention (straddles the path, half each side).
export async function strokeToFillPath(
  points: readonly Point[],
  width: number,
  options: StrokeToFillOptions = {},
): Promise<string> {
  const clipper = await getClipper();
  const delta = (width / 2) * CLIPPER_SCALE;
  const subject = toClipperPath(points);

  const offset = (d: number, endType: clipperLib.EndType) =>
    clipper.offsetToPaths({
      delta: d,
      offsetInputs: [{ data: subject, joinType: clipperLib.JoinType.Round, endType }],
    }) ?? [];

  const subpaths = options.closed
    ? [...offset(delta, clipperLib.EndType.ClosedPolygon), ...offset(-delta, clipperLib.EndType.ClosedPolygon)]
    : offset(delta, clipperLib.EndType.OpenRound);

  return subpathsToD(fromClipperPaths(subpaths));
}

// Rounded-rect boundary as sample points (clockwise from top-left),
// `cornerSegments` points per 90-degree corner — for feeding into
// strokeToFillPath, which needs a polyline, not native rect/radius sugar.
export function roundedRectPoints(
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
  cornerSegments = 16,
): Point[] {
  const corners: [cx: number, cy: number, startAngle: number][] = [
    [x + w - r, y + r, -Math.PI / 2],
    [x + w - r, y + h - r, 0],
    [x + r, y + h - r, Math.PI / 2],
    [x + r, y + r, Math.PI],
  ];

  const points: Point[] = [];
  for (const [cx, cy, startAngle] of corners) {
    for (let i = 0; i <= cornerSegments; i++) {
      const angle = startAngle + (Math.PI / 2) * (i / cornerSegments);
      points.push([cx + Math.cos(angle) * r, cy + Math.sin(angle) * r]);
    }
  }
  return points;
}

// Glyph outlines for `text` in `font`, as a single SVG path "d" string —
// opentype.js parses the font and walks its glyph curves; no reasonable
// way to hand-roll this (it needs the actual font's outline data).
export function textToPath(fontPath: string, text: string, x: number, y: number, fontSize: number): string {
  const font = opentype.loadSync(fontPath);
  return font.getPath(text, x, y, fontSize).toPathData(2);
}

// Base class for code-generated SVG images
export abstract class SvgRenderer {
  protected readonly canvas: Svg;

  constructor(size: number) {
    const window = createSVGWindow();
    registerWindow(window, window.document);
    this.canvas = SVG(window.document.documentElement) as Svg;
    this.canvas.size(size, size).viewbox(0, 0, size, size);

    // svg.js sets root xmlns via plain setAttribute but svgdom 0.1.28's
    // serializer rejects that and needs the XMLNS namespace.
    this.canvas.node.removeAttribute("xmlns");
    this.canvas.node.setAttributeNS(
      "http://www.w3.org/2000/xmlns/",
      "xmlns",
      "http://www.w3.org/2000/svg",
    );
  }

  protected abstract draw(): Promise<void>;

  async render(): Promise<string> {
    await this.draw();
    return this.canvas.svg();
  }

  async writeTo(path: string): Promise<void> {
    const svg = await this.render();
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, svg);
  }
}
