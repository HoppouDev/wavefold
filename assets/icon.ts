import { WavefoldIcon } from "./wavefold-icon.ts";

const outPath = process.argv[2] ?? "assets/generated/icon.svg";
await new WavefoldIcon().writeTo(outPath);
