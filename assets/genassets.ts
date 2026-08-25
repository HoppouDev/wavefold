import { Resvg } from "@resvg/resvg-js";
import pngToIco from "png-to-ico";
import { writeAsset } from "./svg.ts";
import { WavefoldIcon } from "./wavefold-icon.ts";

// Single source of truth for every place the icon shows up: the README's
// embedded SVG, the Windows exe/installer .ico, and the iced window icon
// PNG main.rs embeds at compile time. Regenerate all of them from the one
// WavefoldIcon renderer instead of hand-exporting each format separately.
//
// These three paths are also hardcoded (as plain strings, not something
// importable) in README.md and .github/workflows/release.yml — if you
// rename any of them here, grep for the old path in both of those too.
const README_SVG = "docs/icon.svg";
const WINDOW_ICON_PNG = "assets/windows/icon.png";
const WINDOW_ICON_ICO = "assets/windows/icon.ico";

// Standard Windows multi-resolution icon set (matches the hand-authored
// icon.ico this replaced: 16/24/32/48/64/128/256). Also doubles as the
// source for the window-icon PNG (its largest, 256px, entry) — one render
// per size instead of rendering 256 twice.
const ICO_SIZES = [16, 24, 32, 48, 64, 128, 256];

function renderPng(svg: string, size: number): Buffer {
  return new Resvg(svg, { fitTo: { mode: "width", value: size } }).render().asPng();
}

// Render everything up front, before writing anything — a failure partway
// through (e.g. resvg or pngToIco throwing) would otherwise leave some of
// the three outputs updated and others stale, silently breaking the
// "single source of truth" this script exists to guarantee.
const svg = await new WavefoldIcon().render();
const icoPngs = ICO_SIZES.map((size) => renderPng(svg, size));
const ico = await pngToIco(icoPngs);
const windowIconPng = icoPngs[icoPngs.length - 1]!; // 256px, same as ICO_SIZES' last entry

writeAsset(README_SVG, svg);
writeAsset(WINDOW_ICON_PNG, windowIconPng);
writeAsset(WINDOW_ICON_ICO, ico);
