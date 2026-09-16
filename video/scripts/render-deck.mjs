// Render every slide of the VoxHandsDeck composition to a PNG.
//
// Usage: node scripts/render-deck.mjs <outputDir>
//
// Bundles the Remotion project once and then renders one still per slide,
// which is far faster than invoking the CLI ten times.

import {bundle} from "@remotion/bundler";
import {renderStill, selectComposition} from "@remotion/renderer";
import {mkdirSync} from "node:fs";
import path from "node:path";
import {fileURLToPath} from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const videoRoot = path.resolve(here, "..");
const outDir = process.argv[2] ? path.resolve(process.argv[2]) : path.join(videoRoot, "out", "deck");

const main = async () => {
  mkdirSync(outDir, {recursive: true});
  console.log("[deck] bundling Remotion project…");
  const serveUrl = await bundle({entryPoint: path.join(videoRoot, "src", "index.ts")});
  const composition = await selectComposition({serveUrl, id: "VoxHandsDeck"});
  console.log(`[deck] ${composition.durationInFrames} slides at ${composition.width}x${composition.height}`);

  for (let frame = 0; frame < composition.durationInFrames; frame++) {
    const output = path.join(outDir, `slide-${String(frame + 1).padStart(2, "0")}.png`);
    await renderStill({composition, serveUrl, output, frame, imageFormat: "png"});
    console.log(`[deck] ${path.basename(output)}`);
  }
  console.log("[deck] done");
};

main().catch((error) => {
  console.error(`[deck] failed: ${error.message}`);
  process.exit(1);
});
